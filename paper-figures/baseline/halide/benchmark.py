"""
Benchmark operators using Halide's auto-scheduler (adam19)
"""

import argparse
import platform
import os
import time
from collections import namedtuple
from utils import log_line, BenchmarkRecord
from utils import shape_dict

def run_cmd(cmd):
    print(cmd)
    os.system(cmd)

Workload = namedtuple("Workload", ['workload_type', 'workload_name', 'generator_filename'])

wkl_list = [
    Workload("op", "GMM", "batch_matmul"),
    Workload("op", "C1D", "conv1d_nlc"),
    Workload("op", "C2D", "conv2d_nhwc"),
    Workload("op", "C3D", "conv3d_ndhwc"),
    Workload("op", "GRP", "conv2d_nhwc"),
    Workload("op", "DIL", "conv2d_nhwc"),
    Workload("op", "DEP", "depthwise_conv2d_nhwc"),
    Workload("op", "T2D", "conv2d_transpose_nhwc"),
    Workload("op", "CAP", "conv2d_capsule_nhwijc"),
    Workload("op", "NRM", "norm_bmn"),
    Workload("subgraph", "conv2d_bn_relu", "conv2d_bn_relu"),
    Workload("subgraph", "transpose_batch_matmul", "transpose_batch_matmul"),
    Workload("subgraph", "transpose_batch_matmul_softmax", "transpose_batch_matmul_softmax"),
]

HALIDE_HOME = os.environ['HALIDE_HOME']

def benchmark(target_wkl, batch_size, device, out_file, auto_tune):
    for wkl in wkl_list:
        if target_wkl is not None and wkl.workload_name != target_wkl:
            continue

        for shape in shape_dict[wkl.workload_name]:
            best_time = -1

            if shape[0] == 1:
                shape = list(shape)
                shape[0] = batch_size

            os.environ['HL_APP_ARGS'] = ', '.join(map(str, shape))

            workload_name = "%s%s" % (wkl.workload_name, tuple(shape))
            log_file_name = "%s_bench.txt" % (workload_name)

            run_cmd('rm -rf %s/apps/autoscheduler/bin/host' % HALIDE_HOME)
            run_cmd('rm -rf %s/apps/autoscheduler/samples' % HALIDE_HOME)
            run_cmd('cp %s/apps/autoscheduler/test_generators/%s.cpp  %s/apps/autoscheduler/demo_generator.cpp'
                    % (HALIDE_HOME, wkl.generator_filename, HALIDE_HOME))

            tic = time.time()
            if auto_tune:
                run_cmd('make -C %s/apps/autoscheduler autotune > "%s"' % (HALIDE_HOME, log_file_name))
            else:
                run_cmd('make -C %s/apps/autoscheduler demo > "%s"' % (HALIDE_HOME, log_file_name))
            used_time = time.time() - tic
 
            for line in open(log_file_name, "r"):
                if line.startswith('Benchmark for demo produces best case'):
                    line = line.split('of')[1].split('sec')[0]
                    cost = float(line)
                    break
                elif line.startswith('Best runtime is'):
                    line = line.split('is')[1].split('msec')[0]
                    cost = float(line) / 1e3
                else:
                    pass
 
            print("wkl: %s\tbest cost: %.4f ms\ttuning time: %.2f" % (workload_name, cost * 1e3, used_time))
            algorithm = 'adam19-tune' if auto_tune else 'adam19-no-tune'
            log_line(BenchmarkRecord(device, 'cpu', wkl.workload_type, workload_name,
                                    'halide', algorithm, {"costs": [cost]}, time.time()),
                                    out_file)

            # delete intermediate output
            #os.system("rm *.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=platform.processor())
    parser.add_argument("--wkl", type=str)  # default is None, which means testing all workloads
    parser.add_argument("--out-file", type=str, default='results.tsv')
    parser.add_argument("--no-tune", action='store_true')
    parser.add_argument("--num-tune-batch", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=-1)  # -1 means test both 1 and 16
    args = parser.parse_args()

    if 'TVM_NUM_THREADS' in os.environ:
        os.environ['HL_NUM_THREADS'] = os.environ['TVM_NUM_THREADS']
        print("Setting HL_NUM_THREADS to be the same as TVM_NUM_THREADS (%s)" % os.environ['TVM_NUM_THREADS'])
    os.environ['HL_TUNE_NUM_BATCHES'] = str(args.num_tune_batch)

    if args.batch_size > 0:
        batch_size_list = [args.batch_size]
    else:
        batch_size_list = [1, 16]

    for batch_size in batch_size_list:
        benchmark(args.wkl, batch_size, args.device, args.out_file, not args.no_tune)

