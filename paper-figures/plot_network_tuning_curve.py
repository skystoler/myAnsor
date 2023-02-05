import os
from collections import defaultdict
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter

from common import show_name_replace_dict, method2color, geomean, max_curve, plot_max_curve
from common import run_cmd, to_str_round

def get_cost(network, target, log_file, log_n_lines, batch_size, n_trials=None, load_model=False):
    """Get the inference cost for one network"""
    tmp_out_file = "results.tsv"
    os.system("rm -rf %s" % tmp_out_file)

    # run evaluation
    cmd = ('python3.6 tune_network.py --network %s --batch-size %d --target "%s" --log-file "%s" --log-n-lines %d'
            % (network, batch_size, target, log_file, log_n_lines))
    if n_trials is None:
        cmd +=  " --tune false"
    else:
        cmd += " --n-trials %d" % n_trials
        if load_model:
            cmd += " --load-model"

    run_cmd(cmd)

    # parse log file
    cost = 1e10
    if os.path.exists(tmp_out_file):
        for line in open(tmp_out_file):
            item = line.split('\t')
            cost = eval(item[6])['costs']
        os.system("rm -rf %s" % tmp_out_file)

    return cost

def parse_data(log_names, log_file):
    """Parse the tsv file"""
    # data[lname][ct] = value
    data = defaultdict(lambda: defaultdict(lambda : 1e10))
    for line in open(log_file, 'r'):
        items = line.split('\t')
        name, n_trials, cost = items[:3]
        data[name][int(n_trials)] = np.mean((eval(cost)))

    tstamp_list = []
    gflops_list = []
    name_list = []
    color_list = []
    fmt_list = []

    log_files = log_names + ['AutoTVM.json']
    baseline = 'AutoTVM'
    show_name_dict =  {
        'full': show_name_replace_dict['ours'],
        'limit-space': 'Limited space',
        'no-fine-tune': 'No fine-tuning',
        'no-task-scheduler': 'No task scheduler',
    } 

    color_dict = {
        'full': method2color('ours'),
        'limit-space': 'C4',
        'no-fine-tune': 'C2',
        'no-task-scheduler': 'C3',
        'AutoTVM': method2color('AutoTVM')
    }

    for i, log_file in enumerate(log_files):
        name = log_file.split('.')[0]
        keys = [0,] + list(data[name].keys())
        keys.sort()
        tstamp_list.append(keys)
        gflops_list.append([data[baseline][0] / data[name][k] for k in keys])
        name_list.append(show_name_dict.get(name, name))
        color_list.append(color_dict[name])

        if name == baseline:
            fmt_list.append('--')
        else:
            fmt_list.append('')

    return tstamp_list, gflops_list, name_list, fmt_list, color_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--paper", action='store_true')
    parser.add_argument("--network", type=str, default="resnet-50")
    parser.add_argument("--target", type=str, default='llvm -mcpu=skylake-avx512')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--x-max", type=int, default=20000)
    parser.add_argument("--log-file", type=str, default="network_tuning_curve.tsv")
    parser.add_argument("--out-file", type=str, default="network-tuning-curve.png")
    args = parser.parse_args()

    #network = 'resnet-50'
    #network = 'tflite-mobilenet-v2'
    network = args.network

    log_names = ['full.json', 'no-task-scheduler.json', 'no-fine-tune.json', 'limit-space.json']
    #log_names = ['full.json']

    if args.eval:
        if "&" in network:
            network_names = network.split("&")
            for log_file in log_names:
                for log_n_lines in (range(3000, 20000, 5000)):
                    costs = []
                    reference_costs = [0.0060, 0.0014]
                    for network_name in network_names:
                        tmp_cost = get_cost(network_name, args.target, log_file, log_n_lines, args.batch_size)
                        costs.append(np.mean(tmp_cost))
                    cost = geomean(np.array(costs) / np.array(reference_costs))
                    with open(args.log_file, 'a') as fout:
                        name = log_file.split('.')[0]
                        fout.write("\t".join([name, str(log_n_lines), to_str_round(cost), "%.2f" % time.time()]) + "\n")

                    #time.sleep(5)
        else:
            for log_file in log_names:
                for log_n_lines in (range(2000, 20000, 2000)):
                    cost = get_cost(network, args.target, log_file, log_n_lines, args.batch_size)

                    with open(args.log_file, 'a') as fout:
                        name = log_file.split('.')[0]
                        fout.write("\t".join([name, str(log_n_lines), to_str_round(cost), "%.2f" % time.time()]) + "\n")

                    #time.sleep(5)

    if args.paper:
        # draw the plot for the paper
        fig, ax = plt.subplots()
        gs = gridspec.GridSpec(1, 2) #, width_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        fontsize = 19

        tstamp_list, gflops_list, name_list, fmt_list, color_list = \
                parse_data(log_names, "saved_logs/network-tuning-curve/2020-05-20-mobilenet.tsv")
        ax1.set_xlim(left=0, right=13000)
        ax1.set_ylim(bottom=0, top=1.79)
        for i, (xs, ys) in enumerate(zip(tstamp_list, gflops_list)):
            ax1.plot(xs, max_curve(ys), fmt_list[i], color=color_list[i])
        ax1.set_xticks(list(range(0, 13000, 3000)))
        ax1.set_xticklabels(["%d" % x for x in ax1.get_xticks()], fontsize=fontsize-2)
        ax1.set_yticklabels(ax1.get_yticks(), fontsize=fontsize)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax1.text(0.5, 0.15, 'Mobilenet V2', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=fontsize-2)

        tstamp_list, gflops_list, name_list, fmt_list, color_list = \
                parse_data(log_names, "saved_logs/network-tuning-curve/2020-05-23-double-net.tsv")
        ax2.set_xlim(left=0, right=26000)
        ax2.set_ylim(bottom=0, top=1.5)
        for i, (xs, ys) in enumerate(zip(tstamp_list, gflops_list)):
            ax2.plot(xs, max_curve(ys), fmt_list[i], color=color_list[i])
        ax2.set_xticks(list(range(0, 26000, 6000)))
        ax2.set_xticklabels(["%d" % x for x in ax2.get_xticks()], fontsize=fontsize-2)
        ax2.set_yticklabels(ax1.get_yticks(), fontsize=fontsize)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.text(0.55, 0.15, 'Mobilenet V2 + ResNet-50', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=fontsize-2)

        # label and legend
        ax1.set_ylabel("Relative Speedup", fontsize=fontsize)
        ax1.text(1.10, -0.15, '# Measurement trials', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=ax1.yaxis.label.get_size())
        ax2.legend(name_list,
                   fontsize=fontsize-1,
                   loc='upper center',
                   bbox_to_anchor=(-0.2, 1.27),
                   ncol=3,
                   handlelength=1.0,
                   handletextpad=0.5,
                   columnspacing=1.1)

        fig.set_size_inches((11, 5))
        fig.savefig(args.out_file, bbox_inches='tight')
    else:
        tstamp_list, gflops_list, name_list, fmt_list, color_list = \
                parse_data(log_names, args.log_file)
        plot_max_curve(tstamp_list, gflops_list, name_list, args.out_file,
                       fmts=fmt_list,
                       colors=color_list,
                       x_label='# Measurement trials', x_max=args.x_max,
                       y_label='Relative Speedup', title=None, figure_size=(11, 5))

