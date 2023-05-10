import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import tvm
from tvm import ansor

from common import plot_max_curve, show_name_replace_dict

def get_workload_key_obj(name):
    if name == 'single-op-ablation':
        func_name = 'conv2d_nhwc'
        shape = (16, 7, 7, 512, 512, 3, 1, 1)
        wkl_keys = [json.dumps((func_name,) + shape)]
        weights = [1 for _ in range(len(wkl_keys))]

        def objective_func(costs):
            return sum(c * w for c, w in zip(costs, weights))
    else:
        raise ValueError("Invalid workload: " + name)

    return wkl_keys, objective_func


def extract_data(log_files, workload_keys, objective_func):
    # data[log_file] = throughput lists
    data = defaultdict(list)

    workload_id_dict = {workload_keys[i] : i for i in range(len(workload_keys))}

    for log_file in log_files:
        best_costs = 1e10 * np.ones(len(workload_keys))
        throughputs = []

        for inp, res in ansor.load_from_file(log_file):
            wkl_key = inp.task.workload_key

            wkl_id = workload_id_dict[wkl_key]

            best_costs[wkl_id] = min(best_costs[wkl_id], ansor.utils.array_mean(res.costs))
            score = 1 / objective_func(best_costs)
            throughputs.append((res.timestamp, score))
        data[log_file] = throughputs

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wkl", type=str, required=True)
    parser.add_argument("--logs", nargs="+", type=str, required=True)
    parser.add_argument("--names", nargs="+", type=str)
    #parser.add_argument("--x-axis", choices=['#trial', 'time (second)'], default='time (second)')
    parser.add_argument("--x-axis", choices=['#trial', 'time (second)'], default='#trial')
    parser.add_argument("--x-max", type=int)
    parser.add_argument("--out-file", type=str)
    parser.add_argument("--baseline", type=str)
    parser.add_argument("--median", type=int)
    args = parser.parse_args()

    out_file = args.out_file or args.wkl + ".png"
    wkl_keys, objective_func = get_workload_key_obj(args.wkl)

    # build the filename -> showname dict
    name_replace_dict = {
        'op-full-single': show_name_replace_dict['ours'],
        'op-beam-single': 'Beam search',
        'op-no-fine-tune-single': 'No fine-tuning',
        'op-limit-space-single': 'Limited space'
    }

    filename_to_showname = {}
    for i, log_file in enumerate(args.logs):
        if args.names is not None and i < len(args.names):
            show_name = args.names[i]
        else:
            show_name = log_file[:-5]
        show_name = show_name.split('/')[-1]
        filename_to_showname[log_file] = name_replace_dict.get(show_name, show_name)

    name_list = []
    tstamp_list = []
    gflops_list = []
    fmt_list = []
    color_list = ['C0', 'C3', 'C2', 'C4']

    # Extract data
    if args.median is None:
        data = extract_data(args.logs, wkl_keys, objective_func)
    else:
        assert args.x_axis == '#trial'
        data = dict()
        for log_file in args.logs:
            tmp_log_names = []
            for i in range(args.median):
                filename = "%s.%d" % (log_file, i)
                if os.path.exists(filename):
                    tmp_log_names.append(filename)
                else:
                    print("WARNING: %s does not exist" % filename)
            tmp_data = extract_data(tmp_log_names, wkl_keys, objective_func)
            data[log_file] = np.median(list(tmp_data.values()), axis=0)
            print(data[log_file].shape)

    for i, log_file in enumerate(data):
        if "time" in args.x_axis:
            base_tstamp = data[log_file][0][0]
            tstamps = [x[0] - base_tstamp for x in data[log_file]]
        else:
            tstamps = list(range(len(data[log_file])))

        if args.x_max is not None:
            tstamps = [x for x in tstamps if x < args.x_max]

        tstamp_list.append(tstamps)
        gflops_list.append([x[1] for x in data[log_file]][:len(tstamps)])
        name_list.append(filename_to_showname[log_file])
        fmt_list.append('')

    if args.baseline:
        index = args.logs.index(args.baseline)
        if False:
            tstamp_list.append(tstamp_list[index])
            gflops_list.append([gflops_list[index][-1] for _ in range(len(tstamp_list[index]))])
            name_list.append(None)
            fmt_list.append(':')
            color_list.append(color_list[index])
            base = max(gflops_list[-1])
        else:
            base = gflops_list[index][-1]
        for i in range(len(gflops_list)):
            gflops_list[i] = np.array(gflops_list[i]) / base
        y_label = 'Normalized Performance'
    else:
        y_label = 'Throughput'

    if 'trial' in args.x_axis:
        args.x_axis = "# Measurement trials"

    # Make plot
    plot_max_curve(tstamp_list, gflops_list, name_list, out_file,
                   fmts=fmt_list, colors=color_list[:len(tstamp_list)],
                   x_label=args.x_axis, x_max = args.x_max,
                   y_label=y_label, title=None, figure_size=(10, 4.5))

#    plot_mean_curve(tstamp_list, gflops_list, name_list, out_file,
#                    fmts=fmt_list, colors=color_list,
#                    x_label=args.x_axis, x_max = args.x_max,
#                    y_label=y_label, title=None, figure_size=(11, 5),
#                    window_size=1)

