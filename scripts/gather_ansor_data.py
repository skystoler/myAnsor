import argparse

from common import run_cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['intel-cpu', 'nvidia-gpu'], required=True)
    args = parser.parse_args()

    if args.backend == 'intel-cpu':
        run_cmd('python3 tune_op_subgraph.py --wkl all --batch-size -1 --target "llvm -mcpu=core-avx2" --tune false --log-file op-subgraph-cpu-ansor-logs.json')
        run_cmd('python3 evaluate_all_networks.py --backend intel-cpu')
    elif args.backend == 'nvidia-gpu':
        run_cmd('python3 tune_op_subgraph.py --wkl subgraph --batch-size -1 --target "cuda" --tune false --log-file op-subgraph-gpu-ansor-logs.json')
        run_cmd('python3 evaluate_all_networks.py --backend nvidia-gpu')
    else:
        raise ValueError("Invalid backend: " + args.backend)
