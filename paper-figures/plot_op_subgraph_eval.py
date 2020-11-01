"""Plot figures for single op and subgraph evaluation"""

import argparse
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from common import BaselineDatabase, LogFileDatabase, geomean, draw_grouped_bar_chart, to_str_round, throughput_to_cost
from shape_configs import shape_dict


def extract_data_for_device(mode, batch_size, device, backend, baseline_file, methods, baseline=None):
    assert baseline == None

    if mode == 'op':
        wkl_meta_names = ['C1D', 'C2D', 'C3D', 'GMM', 'GRP', 'DIL', 'DEP', 'T2D', 'CAP', 'NRM']
    elif mode == 'subgraph':
        #wkl_meta_names = ['conv2d_bn_relu', 'transpose_batch_matmul']
        wkl_meta_names = ['conv2d_bn_relu', 'transpose_batch_matmul_softmax']
    else:
        raise ValueError("Invalid mode")

    # Build workload names
    wkl_names = OrderedDict()
    for wkl_meta_name in wkl_meta_names:
        for shape in shape_dict[wkl_meta_name]:
            if shape[0] == 1:
                shape = list(shape)
                shape[0] = batch_size
            wkl_name = "%s%s" % (wkl_meta_name, tuple(shape))
            wkl_names[wkl_name] = True

    # data[wkl][library] = cost
    data, _ = BaselineDatabase(baseline_file).get_data_dict(device, backend, list(wkl_names.keys()))

    # Compute normalized performance relative to the best cost
    norm_data = OrderedDict()
    for wkl_name in wkl_names:
        if not wkl_name in data:
            norm_data[wkl_name] = {}
            continue
        best = np.min([data[wkl_name].get(method, 1e10) for method in methods])
        norm_data[wkl_name] = OrderedDict()
        for method in methods:
            if method not in data[wkl_name]:
                print("Data missing for `%s` on `%s`" % (wkl_name, method))
            else:
                norm_data[wkl_name][method] = best / data[wkl_name][method]
        print("%-40s " % wkl_name, to_str_round([norm_data[wkl_name].get(method, 0.0) for method in methods], 2))

    # Compute geo-mean of normalized performance and normalize the mean to the best.
    ret = OrderedDict()
    for wkl_meta_name in wkl_meta_names:
        wkl_names = []
        for shape in shape_dict[wkl_meta_name]:
            if shape[0] == 1:
                shape = list(shape)
                shape[0] = batch_size
            wkl_name = "%s%s" % (wkl_meta_name, tuple(shape))
            wkl_names.append(wkl_name)

        ret[wkl_meta_name] = OrderedDict()
        for method in methods:
            perf_number = []
            for wkl_name in wkl_names:
                if method in norm_data[wkl_name]:
                    perf_number.append(norm_data[wkl_name][method])
            ret[wkl_meta_name][method] = geomean(perf_number) if perf_number else 0

        best_perf = np.max([ret[wkl_meta_name][method] for method in methods])
        for method in methods:
            ret[wkl_meta_name][method] /= best_perf

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-file", type=str, default="baseline/results.tsv")
    parser.add_argument("--device", type=str,
                        choices=['Intel-Platinum-8269CY-2.50GHz', 'Intel-Platinum-8124M-3.00GHz',
                                 'Intel-E5-2670-v3-2.30Ghz', 'Intel-i7-8750H-2.20Ghz',
                                 'Tesla V100-SXM2-16GB'])
    parser.add_argument("--mode", choices=['subgraph', 'op'], default='subgraph')
    parser.add_argument("--out-file", type=str)
    args = parser.parse_args()

    out_file = args.out_file or args.device + "_" + args.mode + ".png"

    yscale_log = False
    yticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    y_max = 1.4
    legend_nrow = 1
    legend_bbox_to_anchor = (0.45, 1.35)

    if args.mode == 'op':
        fig, ax = plt.subplots()
        gs = gridspec.GridSpec(2, 1) #, width_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        baseline = None
        methods = ['AutoTVM', 'halide', 'pytorch', 'FlexTensor',  'ours']
        data = extract_data_for_device(args.mode, 1, args.device, 'cpu', args.baseline_file, methods)
        draw_grouped_bar_chart(throughput_to_cost(data), baseline='', draw_legend=True, figax=ax1, yticks=yticks, yscale_log=yscale_log,
                legend_nrow=legend_nrow, legend_bbox_to_anchor=legend_bbox_to_anchor, draw_ylabel="", y_max=y_max)
        ax1.text(0.15, 0.85, 'Batch size = 1', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=ax1.yaxis.label.get_size())

        data = extract_data_for_device(args.mode, 16, args.device, 'cpu', args.baseline_file, methods)
        draw_grouped_bar_chart(throughput_to_cost(data), baseline='', draw_legend=False, figax=ax2, yticks=yticks, yscale_log=yscale_log, 
                legend_nrow=legend_nrow, legend_bbox_to_anchor=legend_bbox_to_anchor, draw_ylabel="", y_max=y_max)
        ax2.text(0.15, 0.85, 'Batch size = 16', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=ax2.yaxis.label.get_size())
        ax2.text(-0.10, 1.15, 'Normalized Performance', horizontalalignment='center', verticalalignment='center', rotation=90, transform=ax2.transAxes, fontsize=ax2.yaxis.label.get_size())

        fig.set_size_inches((11, 5.5))
        fig.savefig(out_file, bbox_inches='tight')
        print("Output to %s ..." % out_file)
    elif args.mode == 'subgraph':
        assert args.device is None

        fig, ax = plt.subplots()
        gs = gridspec.GridSpec(2, 1) #, width_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        def extract_data_for_both_device(mode, batch_size, methods):
            cpu_data = extract_data_for_device(args.mode, batch_size, 'Intel-Platinum-8124M-3.00GHz', 'cpu', args.baseline_file, methods)
            gpu_data = extract_data_for_device(args.mode, batch_size, 'Tesla V100-SXM2-16GB', 'gpu', args.baseline_file, methods)

            data = {}
            replace_dict = {
                'conv2d_bn_relu' : 'ConvLayer',
                'transpose_batch_matmul_softmax': 'TBS'
            }

            # remove flextensor in transpose_batch_matmul_softmax
            del cpu_data['transpose_batch_matmul_softmax']['FlexTensor']
            del gpu_data['transpose_batch_matmul_softmax']['FlexTensor']

            for wkl, value in cpu_data.items():
                data[replace_dict[wkl] + " @ $\mathbf{C}$"] = value
                # remove halide in gpu data
                tmp_gpu_data = gpu_data[wkl]
                del tmp_gpu_data['halide']
                data[replace_dict[wkl] + " @ $\mathbf{G}$"] = tmp_gpu_data
            return data

        baseline = None
        methods = ['AutoTVM', 'halide', 'pytorch', 'FlexTensor',  'ours']
        data = extract_data_for_both_device(args.mode, 1, methods)
        draw_grouped_bar_chart(throughput_to_cost(data), baseline='', draw_legend=True, figax=ax1, yticks=yticks, yscale_log=yscale_log,
                legend_nrow=legend_nrow, legend_bbox_to_anchor=legend_bbox_to_anchor, draw_ylabel="", y_max=y_max)
        ax1.text(0.15, 0.85, 'Batch size = 1', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=ax1.yaxis.label.get_size())

        data = extract_data_for_both_device(args.mode, 16, methods)
        draw_grouped_bar_chart(throughput_to_cost(data), baseline='', draw_legend=False, figax=ax2, yticks=yticks, yscale_log=yscale_log, 
                legend_nrow=legend_nrow, legend_bbox_to_anchor=legend_bbox_to_anchor, draw_ylabel="", y_max=y_max)
        ax2.text(0.15, 0.85, 'Batch size = 16', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=ax2.yaxis.label.get_size())
        ax2.text(-0.10, 1.15, 'Normalized Performance', horizontalalignment='center', verticalalignment='center', rotation=90, transform=ax2.transAxes, fontsize=ax2.yaxis.label.get_size())

        fig.set_size_inches((11, 5.5))
        fig.savefig(out_file, bbox_inches='tight')
        print("Output to %s ..." % out_file)

