import argparse

from utils import run_cmd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=['cpu', 'gpu'], default='cpu')
    args = parser.parse_args()

    if args.backend == 'cpu':
        # single op
        run_cmd("python3 benchmark_single_op.py --batch-size 1")
        run_cmd("python3 benchmark_single_op.py --batch-size 16")
        run_cmd("python3 benchmark_single_op.py --test flextensor_single_op.txt")

        # subgraph
        run_cmd("python3 optimize_conv2d_bn_relu.py         --target llvm --device 0 --parallel 1 --trials 1000 --timeout 10 --method q --log flextensor_conv2d_bn_relu.txt")
        run_cmd("python3 optimize_conv2d_bn_relu.py         --target llvm --device 0 --test flextensor_conv2d_bn_relu.txt")

        run_cmd("python3 optimize_transpose_batch_matmul.py --target llvm --device 0 --parallel 1 --trials 1000 --timeout 10 --method q --log flextensor_transpose_batch_matmul.txt")
        run_cmd("python3 optimize_transpose_batch_matmul.py --target llvm --device 0 --test flextensor_transpose_batch_matmul.txt")
    else:
        run_cmd("python3 optimize_conv2d_bn_relu.py         --target cuda --device 0 --parallel 1 --trials 1000 --timeout 10 --method q --log flextensor_conv2d_bn_relu.txt")
        run_cmd("python3 optimize_conv2d_bn_relu.py         --target cuda --device 0 --test flextensor_conv2d_bn_relu.txt")

        run_cmd("python3 optimize_transpose_batch_matmul.py --target cuda --device 0 --parallel 1 --trials 1000 --timeout 10 --method q --log flextensor_transpose_batch_matmul.txt")
        run_cmd("python3 optimize_transpose_batch_matmul.py --target cuda --device 0 --test flextensor_transpose_batch_matmul.txt")

