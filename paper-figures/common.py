"""Common utility for scripts"""
import argparse
import math
import os
import re
import time
from collections import defaultdict, namedtuple, OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

############################################################
#####################  Other Utilities  ####################
############################################################

def geomean(xs):
    """Compute geometric mean"""
    return math.exp(math.fsum(math.log(x) for x in xs) / len(xs))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def run_cmd(cmd):
    print(cmd)
    ret_code = os.system(cmd)
    if ret_code != 0:
        exit(ret_code)

############################################################
######################  I/O Utilities  #####################
############################################################

# The format for a line in resulst file
BenchmarkRecord = namedtuple("BenchmarkRecord", [
    'device', 'backend', 'workload_type', 'workload_name', 'library', 'algorithm', 'value',
    'time_stamp'
])


class BaselineDatabase:
    """A class for query records in baseline database"""
    def __init__(self, filename):
        self.filename = filename

        self.lines = []
        for line in open(filename):
            if line.startswith('#') or line.isspace():
                continue
            self.lines.append(line.split('\t'))

    def filter_records(self, devices=None, backends=None, wkl_names=None, libraries=None):
        ret = []
        for line in self.lines:
            line = BenchmarkRecord(*line)

            if devices is not None and line.device not in devices:
                continue
            if backends is not None and line.backend not in backends:
                continue
            if wkl_names is not None and line.workload_name not in wkl_names:
                continue
            if libraries is not None and line.library not in libraries:
                continue

            ret.append(line)
        return ret

    def get_data_dict(self, device, backend, wkl_names) -> Tuple[Dict, List]:
        """Return a data dict s.t.  data[wkl][library] = cost"""
        data = dict() 
        all_libraries = set()

        # Read costs for baselines
        records = self.filter_records(devices=[device], backends=[backend], wkl_names=wkl_names)
        for record in records:
            # use min over (possible) multiple algorithms
            all_libraries.add(record.library)

            if record.workload_name not in data:
                data[record.workload_name] = dict()
            if record.library not in data[record.workload_name]:
                data[record.workload_name][record.library] = 1e10

            data[record.workload_name][record.library] = \
                min(data[record.workload_name][record.library],
                    np.mean(eval(record.value)['costs']))

        return data, list(all_libraries)


class LogFileDatabase:
    """A class for indexing best records in a log file"""
    def __init__(self, filename: str, n_lines: int = -1):
        inputs, results = LogReader(filename).read_lines(n_lines)

        # best records, search by (target_key, workload_key).  e.g. ('gpu', 'conv2d...')
        self.best_by_targetkey = {}

        # best according to (model, workload_key).  e.g. ('1080ti', 'conv2d...'))
        self.best_by_model = {}

        # find best records and build the index
        for inp, res in zip(inputs, results):
            if res.error_no != 0:
                continue

            # use target keys in tvm target system as key to build best map
            for target_key in inp.task.target.keys:
                key = (target_key, inp.task.workload_key)
                if key not in self.best_by_targetkey:
                    self.best_by_targetkey[key] = (inp, res)
                else:
                    _, other_res = self.best_by_targetkey[key]
                    if np.mean([x.value for x in other_res.costs]) > \
                            np.mean([x.value for x in res.costs]):
                        self.best_by_targetkey[key] = (inp, res)

            # use model as key to build best map
            key = (inp.task.target.model, inp.task.workload_key)
            if key not in self.best_by_model:
                if inp.task.target.model != 'unknown':
                    self.best_by_model[key] = (inp, res)
            else:
                _, other_res = self.best_by_model[key]
                if np.mean([x.value for x in other_res.costs]) > \
                        np.mean([x.value for x in res.costs]):
                    self.best_by_model[key] = (inp, res)

    def write_best(self, filename: str):
        best_records = list(self.best_by_targetkey.values())
        inputs = [x[0] for x in best_records]
        results = [x[1] for x in best_records]
        write_measure_records(filename, inputs, results)


############################################################
######################  Plot Utilities  ####################
############################################################

############################## Curve
def max_curve(raw_curve):
    """Return b[i] = max(a[:i]) """
    ret = []
    cur_max = -np.inf
    for x in raw_curve:
        cur_max = max(cur_max, x)
        ret.append(cur_max)
    return ret

def min_curve(raw_curve):
    """Return b[i] = min(a[:i]) """
    ret = []
    cur_min = np.inf
    for x in raw_curve:
        cur_min = min(cur_min, x)
        ret.append(cur_min)
    return ret

def mean_curve(raw_curve, window_size=None):
    """Return b[i] = mean(a[:i]) """
    ret = []
    mean = 0
    if window_size is None:
        for i, x in enumerate(raw_curve):
            mean = (mean * i + x) / (i + 1)
            ret.append(mean)
    else:
        for i, x in enumerate(raw_curve):
            if i >= window_size:
                mean = (mean * window_size + x - raw_curve[i - window_size]) / window_size
            else:
                mean = (mean * i + x) / (i + 1)
            ret.append(mean)
    return ret

def plot_max_curve(xs_list, ys_list, legends, out_file, x_label, y_label="GFLOPS",
                   x_min=0, x_max=None, curve_type='max',
                   fmts=None, colors=None, title='Best Performance', figure_size=None):
    colors = colors or ['C%d' % i for i in range(10)]
    fig, ax = plt.subplots()
    fontsize = 19
    curve_func = max_curve if curve_type == 'max' else min_curve

    for i, (xs, ys) in enumerate(zip(xs_list, ys_list)):
        if fmts is None:
            plt.plot(xs, curve_func(ys), color=colors[i])
        else:
            plt.plot(xs, curve_func(ys), fmts[i], color=colors[i])

    plt.legend([x for x in legends if x], fontsize=fontsize)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    if title:
        plt.title(title, fontsize=fontsize)
    plt.xlim(left=x_min)
    if x_max is not None:
        plt.xlim(right=x_max)
    plt.ylim(bottom=0)
    if figure_size:
        fig.set_size_inches(figure_size)

    fig.savefig(out_file, bbox_inches='tight')
    print("Output to file %s" % out_file)


############################## Color
def enhance_color(color, h=1, l=1, s=1):
    """Make color looks better for pyplot"""
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = np.array(colorsys.rgb_to_hls(*mc.to_rgb(c)))

    h, l, s = h * c[0], l * c[1], s * c[2]
    h, l, s = [max(min(x, 1), 0) for x in [h, l, s]]

    return colorsys.hls_to_rgb(h, l, s)

method_color_dict = {
    #'ours': 'C0',
    'AutoTVM': '#CB997E',
    'tensorflow': '#B8B7A3',
    'Ansor': '#B8B7A3',
    'Ansor-DPC': '#6B705C',
    'tensorflow-tensorrt': 'C9',
    'tflite': 'C2',


    'queen-bee-x':'#A5A58D',
    'segmented-x':'#CB997E',
    'single-point-x':'#FDE8D5',
    'three-point-x':'#DDBEA9',



    'pytorch': enhance_color('#DDBEA9', l=1.1, s=0.9),

    'FlexTensor': enhance_color('C5'),
    'halide': enhance_color('teal', l=1.25),

    'Limit space': 'C7',
    'No fine-tuning': 'C8',
    'No task scheduler': 'C1',
}

def method2color(method):
    return method_color_dict[method]


############################## Order
method_order_list = [
    'pytorch', 'tensorflow', 'tensorflow-xla', 'tensorflow-tensorrt',
    'tflite', 'halide', 'FlexTensor',  'AutoTVM', 'Ansor','Ansor-DPC',
    'queen-bee-x','segmented-x','single-point-x','three-point-x',
    'Limit space', 'No fine-tuning',
    'ours',
]

def method2order(method):
    if '-batch-' in method:
        method, batch_size = method.split('-batch-')
        batch_size = int(batch_size)
        return method_order_list.index(method) + batch_size / 100
    else:
        return method_order_list.index(method)

############################## Name
show_name_replace_dict = {
    'pytorch': "PyTorch",
    'tensorflow-tensorrt': 'TensorRT-TF',
    'tensorflow': 'TensorFlow',

    'tflite': 'TensorFlow Lite',
    'halide': 'Halide',

    'Ansor': 'Ansor',
    'Ansor-DPC': 'Ansor-DPC',

    'queen-bee-x':'queen bee crossover',
    'segmented-x':'segemented crossover',
    'single-point-x':'single point crossover',
    'three-point-x':'3 point crossover',

    'batch-16': 'batch',

    'resnet_50': 'ResNet-50',
    'mobilenet_v2': 'Mobilenet V2',
    'resnet3d_18': '3D-ResNet',
    'dcgan': 'DCGAN',
    'dqn': 'DQN',
    'bert': 'BERT',
}

def show_name(name):
    for key, value in show_name_replace_dict.items():
        name = name.replace(key, value)

    return name

def to_str_round(x, decimal=6):
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)) or isinstance(x, np.ndarray):
        return "[" + ", ".join([to_str_round(y, decimal=decimal)
                                for y in x]) + "]"
    if isinstance(x, dict):
        return str({k: eval(to_str_round(v)) for k, v in x.items()})
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        format_str = "%%.%df" % decimal
        return format_str % x
    raise ValueError("Invalid value: " + str(x))


############################## Group bar chart
def throughput_to_cost(data):
    ret = OrderedDict()
    for key_1 in data:
        ret[key_1] = OrderedDict()
        for key_2 in data[key_1]:
            ret[key_1][key_2] = 1 / max(data[key_1][key_2], 1e-10)
    return ret

def draw_grouped_bar_chart(data, baseline=None, output='out.png',
        yscale_log=False, yticks=None, y_max=None,y_min=None,
        legend_bbox_to_anchor=None, legend_nrow=None,
        figure_size=None, figax=None, draw_ylabel=True, draw_legend=True):
    """
    Parameters
    data: OrderedDict[workload_name -> OrderedDict[method] -> cost]]
    """
    width = 1
    gap = 1.5
    fontsize = 19
    xticks_font_size = fontsize - 2

    figure_size = figure_size or (11, 4)
    legend_bbox_to_anchor = legend_bbox_to_anchor or (0.45, 1.35)

    all_methods = set()
    legend_set = {}

    if figax is None:
        fig, ax = plt.subplots()
        axes = []
        axes.append(ax)
    else:
        # for drawing subplot
        ax = figax

    x0 = 0
    xticks = []
    xlabels = []

    workloads = list(data.keys())
    for wkl in workloads:
        ys = []
        colors = []

        methods = list(data[wkl].keys())

        if baseline in data[wkl]:
            baseline_cost = data[wkl][baseline]
        else:
            # normalize to best library
            baseline_cost = 1e10
            for method in methods:
                if data[wkl][method] < baseline_cost:
                    baseline_cost = data[wkl][method]

        methods.sort(key=lambda x: method2order(x))
        for method in methods:
            relative_speedup = baseline_cost / data[wkl][method]
            if yticks is None:
                ys.append(relative_speedup)
            else:
                ys.append(max(relative_speedup, yticks[0] * 1.1))
            colors.append(method2color(method))

        # draw the bars
        xs = np.arange(x0, x0 + len(ys))
        bars = ax.bar(xs, ys, width=width, color=colors)

        for method, bar_obj in zip(methods, bars):
            all_methods.add(method)
            if method not in legend_set:
                legend_set[method] = bar_obj

        # tick and label
        x0 += len(ys) + gap

        xticks.append(x0 - gap - len(ys)*width/2.0 - width/2.0)
        xlabels.append(show_name(wkl))

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=xticks_font_size)
        plt.tick_params(axis='x', which='both', bottom='off', top='off')

        if draw_ylabel is True:
            ax.set_ylabel('Relative Speedup', fontsize=fontsize)
        elif isinstance(draw_ylabel, str):
            ax.set_ylabel(draw_ylabel, fontsize=fontsize)

        if yscale_log:
            ax.set_yscale('log', basey=2)
        if yticks is not None:
            ax.set_yticks(yticks)
        if y_max:
            ax.set_ylim(bottom=y_min,top=y_max)

        from matplotlib.ticker import FormatStrFormatter
        ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.grid(linewidth=0.4, linestyle='dotted') # draw grid line
        ax.set_axisbelow(True)  # grid lines are behind the rest
        ax.tick_params(bottom=False, top=False, right=False)

    # put legend outside the plot
    all_methods = list(all_methods)
    all_methods.sort(key=lambda x : method2order(x))

    if draw_legend:
        legend_nrow = legend_nrow or 2
        ncol = (len(all_methods) + legend_nrow - 1)// legend_nrow
        ax.legend([legend_set[x] for x in all_methods],
                  [show_name(x) for x in all_methods],
                  fontsize=fontsize-1,
                  loc='upper center',
                  #loc=(0.0,0.1),
                  bbox_to_anchor=legend_bbox_to_anchor,
                  ncol=ncol,
                  prop={'size':14},
                  handlelength=1.0,
                  handletextpad=0.5,
                  columnspacing=1.1)
 
    if figax is None:
        fig.set_size_inches(figure_size)
        fig.savefig(output, bbox_inches='tight')
        print("Output the plot to %s" % output)

