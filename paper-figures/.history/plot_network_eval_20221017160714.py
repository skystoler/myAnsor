"""Plot figures for network evaluation"""

import argparse
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from common import BaselineDatabase, LogFileDatabase, geomean, draw_grouped_bar_chart, to_str_round


def load_data(networks, baseline_file, device, backend, methods):
    # data[wkl][library] = cost
    data, _ = BaselineDatabase(baseline_file).get_data_dict(device, backend, networks)

    ret_data = OrderedDict()
    for wkl in networks:
        for method in data[wkl]:
            if method not in methods:
                continue
            cost = data[wkl][method]

            wkl_meta_name, batch_size = wkl.split('.B')
            if wkl_meta_name not in ret_data:
                ret_data[wkl_meta_name] = OrderedDict()
            ret_data[wkl_meta_name][method] = cost

            print("%20s\t%20s\t%.4f" % (wkl, method, cost))

    return ret_data

def two_d_dict_max(obj, min_value):
    return {k: {a : max(min_value, b) for a, b in v.items()} for k, v in data.items()}

#networks = ['resnet_50', 'mobilenet_v2', 'resnet3d_18', 'dcgan', 'bert']
networks = ['resnet_50']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-file", type=str, default="baseline/results.tsv")
    parser.add_argument("--device", type=str, required=True,
                        choices=['Intel-Platinum-8269CY-2.50GHz', 'Intel-Platinum-8124M-3.00GHz',
                                 'Intel-E5-2670-v3-2.30Ghz', 'Intel-i7-8750H-2.20Ghz',
                                 'Tesla V100-SXM2-16GB', 'aarch64-Cortex-A53-1.4Ghz','Intel-E5-2650-v4-2.20GHz'])
    parser.add_argument("--out-file", type=str)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    out_file = args.out_file or "%s_network.B%d.png" % (args.device, args.batch_size)

    if args.device in ['Tesla V100-SXM2-16GB']:
        methods = ['pytorch', 'tensorflow', 'tensorflow-tensorrt', 'AutoTVM', 'ours']
        backend = 'gpu'
    elif 'Intel' in args.device:
        methods = ['pytorch', 'tensorflow', 'AutoTVM', 'Ansor', 'ours']
        backend = 'cpu'
    else:
        methods = ['tflite', 'AutoTVM', 'ours']
        backend = 'cpu'

    yticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    yscale_log = False
    y_max = 1.4

    if args.batch_size > 0:
        networks = ["%s.B%d" % (name, args.batch_size) for name in networks]

        data = load_data(networks, args.baseline_file, args.device, backend, methods)
        fig, ax = plt.subplots()

        draw_grouped_bar_chart(data, legend_nrow=1, legend_bbox_to_anchor=(0.50, 1.30), figax=ax, yticks=yticks, y_max=y_max, draw_ylabel='Normalized Performance')
        ax.text(0.15, 0.85, 'Batch size = 1', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=ax.yaxis.label.get_size())

        fig.set_size_inches((11, 3))
        fig.savefig(out_file, bbox_inches='tight')
        print("Output the plot to %s" % out_file)
    else:
        fig, ax = plt.subplots()
        gs = gridspec.GridSpec(2, 1) #, width_ratios=[1, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        legend_nrow = 1

        if 'Intel' in args.device:
            legend_bbox_to_anchor = (0.50, 1.35)
        else:
            legend_bbox_to_anchor = (0.45, 1.35)

        tmp_networks = ["%s.B%d" % (name, 1) for name in networks]
        data = load_data(tmp_networks, args.baseline_file, args.device, backend, methods)
        draw_grouped_bar_chart(data, yticks=yticks, figax=ax1, legend_nrow=legend_nrow, legend_bbox_to_anchor=legend_bbox_to_anchor, draw_ylabel="", yscale_log=yscale_log, y_max=y_max)
        ax1.text(0.15, 0.85, 'Batch size = 1', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=ax1.yaxis.label.get_size())

        tmp_networks = ["%s.B%d" % (name, 16) for name in networks]
        data = load_data(tmp_networks, args.baseline_file, args.device, backend, methods)
        draw_grouped_bar_chart(data, yticks=yticks, figax=ax2, draw_legend=False, draw_ylabel="", yscale_log=yscale_log, y_max=y_max)
        ax2.text(0.15, 0.85, 'Batch size = 16', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=ax2.yaxis.label.get_size())
        ax2.text(-0.09, 1.0, 'Normalized Performance', horizontalalignment='center', verticalalignment='center', rotation=90, transform=ax2.transAxes, fontsize=ax2.yaxis.label.get_size())

        fig.set_size_inches((11, 5.5))
        fig.savefig(out_file, bbox_inches='tight')
        print("Output the plot to %s" % out_file)

