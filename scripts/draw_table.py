import argparse
from collections import defaultdict

import numpy as np

from common import BenchmarkRecord

libraries = ['pytorch', 'ours']
networks = ['resnet-50', 'mobilenet-v2', 'resnet3d-18', 'dcgan', 'bert']
batch_sizes = [1, 16]

def load_data(workload_names, libraries, backend, input_file):
    ret_data = defaultdict(lambda : defaultdict(lambda : 1e10))
    for line in open(input_file):
        fields = line.split('\t')
        record = BenchmarkRecord(*fields)

        if record.backend != backend:
            continue

        cost = np.mean(eval(record.value)['costs']) * 1000
        ret_data[record.workload_name][record.library] = cost

    return ret_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=['intel-cpu', 'nvidia-gpu'])
    parser.add_argument("--result-file", type=str, default='results.tsv')
    args = parser.parse_args()

    workload_names = []
    for network in networks:
        for batch_size in batch_sizes:
            workload_name = "%s.B%d" % (network, batch_size)
            workload_names.append(workload_name)

    backend = 'cpu' if 'cpu' in args.backend else 'gpu'
    data = load_data(workload_names, libraries, backend, args.result_file)

    print("-------------------------------------------------------------")
    print("       Inference Execution Time Evaluation (unit: ms)          ")
    print("-------------------------------------------------------------")
    print("   Network   | Batch size | PyTorch | Ansor (ous) | Speedup ")
    print("-------------------------------------------------------------")
    for network in networks:
        for batch_size in batch_sizes:
            workload_name = "%s.B%d" % (network, batch_size)
            t1, t2 = data[workload_name]['pytorch'], data[workload_name]['ours']
            print("%-12s |     %2d     | %7.2f |   %7.2f   | %6.2f X " %
                  (network, batch_size, t1, t2, t1 / t2))
    print("-------------------------------------------------------------")

