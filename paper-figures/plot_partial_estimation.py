""" Evaluate the pairwise comparison accuracy of the cost model on partial programs and complete programs

Example Usage:
python3 plot_partial_estimation.py --log-file ~/no-fine-tune.json  --load-network resnet-50 --eval
"""

import time
import argparse
import os
import logging
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from tvm import ansor

def get_eval_metric(model, task, states, costs):
    preds = model.predict(task, states)

    predicted_order = np.argsort(-preds)

    # pairwise accuracy
    correct_ct = wrong_ct = 0
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            idx1 = predicted_order[i]
            idx2 = predicted_order[j]

            if wkl_costs[idx1] < wkl_costs[idx2]:
                correct_ct += 1
            else:
                wrong_ct += 1
    accuracy = correct_ct * 1.0 / (correct_ct + wrong_ct)

    # top-10 recall
    TOP_K = 10
    real_top_k = set(np.argsort(costs)[:TOP_K])
    predicted_top_k = set(predicted_order[:TOP_K])
    recalled = real_top_k.intersection(predicted_top_k)

    return accuracy, 1.0 * len(recalled) / TOP_K


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str)
    parser.add_argument("--n-lines", type=int, default=-1)
    parser.add_argument("--load-model", action='store_true', default=True)
    parser.add_argument("--model-name", type=str)

    parser.add_argument("--load-network", type=str)
    parser.add_argument("--network-path", type=str, default=None, help="The path of tflite model")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--layout", type=str, default='NHWC')

    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--out-file", type=str, default='partial-evaluation.png')
    args = parser.parse_args()

    if args.eval:
        from common import to_str_round, load_network

        logging.basicConfig()
        logging.getLogger('ansor').setLevel(logging.DEBUG)
        os.environ['ANSOR_PROGRAM_COMPLETE_RATE'] = '1.0'

        if args.load_network:
            load_network(args.load_network, args.network_path, args.batch_size, args.layout)

        # load records from log file 
        inputs, results = ansor.LogReader(args.log_file).read_lines(args.n_lines)
        wkl_ct = OrderedDict()
        for inp, res in zip(inputs, results):
            wkl_key = inp.task.workload_key
            if wkl_key not in wkl_ct:
                wkl_ct[wkl_key] = 0
            wkl_ct[wkl_key] += 1

        # train the model
        model_name = args.model_name or args.log_file + ".xgb"
        model = ansor.XGBModel()
        if args.load_model and os.path.exists(model_name):
            model.load(model_name)
            print("Pretrained cost model loaded.")
        else:
            print("====== Train a New Model ======")
            model.update(inputs, results)
            model.save(model_name)
            print("Saved.")

        n_steps = args.n_steps
        accuracy_list = []
        recall_list = []
        weight = []

        tic = time.time()
        for i, wkl_key in enumerate(wkl_ct):
            print("No: %d / %d\tkey: %s\t" % (i +1 , len(wkl_ct), wkl_key))

            wkl_inputs, wkl_results, wkl_costs = [], [], []
            for inp, res in zip(inputs, results):
                if inp.task.workload_key == wkl_key: 
                    wkl_inputs.append(inp)
                    wkl_results.append(res)
                    wkl_costs.append(ansor.utils.array_mean(res.costs))
            task = wkl_inputs[0].task
            task = ansor.SearchTask(ansor.workload_key_to_dag(task.workload_key),
                    task.workload_key, task.target, task.target_host)
            wkl_states = ansor.serialization.get_states_from_measure_inputs(wkl_inputs, task)

            # compute evaluation metric
            tmp_accuracy_list = []
            tmp_recall_list = []
            for step in range(n_steps+1):
                os.environ['ANSOR_PROGRAM_COMPLETE_RATE'] = "%.2f" % (1.0 * step / n_steps)
                acc, recall = get_eval_metric(model, task, wkl_states, wkl_costs)
                tmp_accuracy_list.append(acc)
                tmp_recall_list.append(recall)

            print("#training records: %d\t#test records: %d\ttime: %.2f" % (len(inputs), len(wkl_inputs), time.time() - tic))
            print("Accuracy curve:  %s" % to_str_round(tmp_accuracy_list, 3))
            print("Recall curve:  %s" % to_str_round(tmp_recall_list, 3))

            accuracy_list.append(tmp_accuracy_list)
            recall_list.append(tmp_recall_list)
            weight.append(len(wkl_inputs))

        weight = np.array(weight) / np.sum(weight)
        for i in range(len(weight)):
            accuracy_list[i] = np.array(accuracy_list[i]) * weight[i]
            recall_list[i] = np.array(recall_list[i]) * weight[i]

        # print averaged output
        accuracy_curve = np.sum(accuracy_list, axis=0)
        recall_curve = np.sum(recall_list, axis=0)
        print("Average accuracy: %s" % to_str_round(accuracy_curve, 4))
        print("Average reacall: %s" % to_str_round(recall_curve, 4))
    else:
        # data got by 
        # `python3 plot_partial_estimation.py --log-file ~/no-fine-tune.json  --load-network resnet-50 --eval`
        accuracy_curve = [0.4940, 0.5072, 0.5290, 0.4857, 0.5016, 0.5241, 0.5119, 0.5146, 0.5497, 0.5751, 0.6222, 0.6271, 0.6296, 0.6428, 0.6897, 0.6951, 0.6852, 0.7701, 0.8212, 0.8576, 0.8710]
        recall_curve = [0.0067, 0.0087, 0.0149, 0.0059, 0.0109, 0.0093, 0.0114, 0.0536, 0.0518, 0.0611, 0.0757, 0.0926, 0.1428, 0.1745, 0.1939, 0.1991, 0.2878, 0.4538, 0.6899, 0.7844, 0.8726]

        # `python3 plot_partial_estimation.py --log-file ~/full.json  --load-network resnet-50 --eval`
        #accuracy_curve = [0.5273, 0.5273, 0.4896, 0.4828, 0.5058, 0.5078, 0.5132, 0.5294, 0.5333, 0.5585, 0.5308, 0.5328, 0.5347, 0.5573, 0.6098, 0.5909, 0.6308, 0.6969, 0.7323, 0.7681, 0.7869]
        #recall_curve = [0.0004, 0.0004, 0.0090, 0.0036, 0.0055, 0.0068, 0.0055, 0.0062, 0.0019, 0.0019, 0.0012, 0.0004, 0.0016, 0.0062, 0.0134, 0.0198, 0.0198, 0.0283, 0.0542, 0.0875, 0.0999]

    n_steps = args.n_steps
    out_file = args.out_file
    fontsize = 17

    fig, ax = plt.subplots()
    gs = gridspec.GridSpec(1, 2, wspace=0.3)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    xs = np.arange(0, 1 + 1.0 / n_steps, 1.0 / n_steps)
    ax1.plot(xs, accuracy_curve)
    ax2.plot(xs, recall_curve)
    ax1.set_ylabel('Pairwise Accuracy', fontsize=fontsize)
    ax2.set_ylabel('Top-K Recall', fontsize=fontsize)
    ax1.set_xlabel('Completion Rate of Programs', fontsize=fontsize)
    ax2.set_xlabel('Completion Rate of Programs', fontsize=fontsize)

    ax1.set_xlim(left=0, right=1.0)
    ax1.set_ylim(top=1.0)
    ax1.set_xticklabels(["%.1f" % x for x in ax1.get_xticks()], fontsize=fontsize)
    ax1.set_yticklabels(["%.1f" % x for x in ax1.get_yticks()], fontsize=fontsize)

    ax2.set_xlim(left=0, right=1.0)
    ax2.set_ylim(top=1.0)
    ax2.set_xticklabels(["%.1f" % x for x in ax2.get_xticks()], fontsize=fontsize)
    ax2.set_yticklabels(["%.1f" % x for x in ax2.get_yticks()], fontsize=fontsize)

    fig.set_size_inches((11, 2.5))
    fig.savefig(out_file, bbox_inches='tight')
    print("Output the plot to %s" % out_file)

