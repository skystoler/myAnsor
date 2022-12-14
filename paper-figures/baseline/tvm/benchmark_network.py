"""Benchmark the autotvm"""
import argparse

from utils import run_cmd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=['intel-cpu', 'nvidia-gpu', 'arm-cpu'], default='intel-cpu')
    args = parser.parse_args()

    if args.backend == 'intel-cpu':
        run_cmd('python3 tune_relay_x86.py --network resnet-50    --batch-size 1')
        run_cmd('python3 tune_relay_x86.py --network resnet-50    --batch-size 16')
        """
        run_cmd('python3.6 tune_relay_x86.py --network mobilenet_v2 --batch-size 1')
        run_cmd('python3.6 tune_relay_x86.py --network mobilenet_v2 --batch-size 16')
        run_cmd('python3.6 tune_relay_x86.py --network resnet3d-18  --batch-size 1  --graph-tuner false')
        run_cmd('python3.6 tune_relay_x86.py --network resnet3d-18  --batch-size 16 --graph-tuner false')
        run_cmd('python3.6 tune_relay_x86.py --network dcgan --batch-size 1  --graph-tuner false')
        run_cmd('python3.6 tune_relay_x86.py --network dcgan --batch-size 16 --graph-tuner false')
        run_cmd('python3.6 tune_relay_x86.py --network bert --batch-size 1  --graph-tuner false')
        run_cmd('python3.6 tune_relay_x86.py --network bert --batch-size 16 --graph-tuner false')
    elif args.backend == 'arm-cpu':
        run_cmd('python3.6 tune_relay_arm.py --rpc-n-parallel 16 --network resnet-50    --batch-size 1')
        run_cmd('python3.6 tune_relay_arm.py --rpc-n-parallel 16 --network mobilenet_v2 --batch-size 1')
        run_cmd('python3.6 tune_relay_arm.py --rpc-n-parallel 16 --network resnet3d-18  --batch-size 1')
        run_cmd('python3.6 tune_relay_arm.py --rpc-n-parallel 16 --network dcgan        --batch-size 1')
        run_cmd('python3.6 tune_relay_arm.py --rpc-n-parallel 16 --network bert         --batch-size 1')
    elif args.backend == 'nvidia-gpu':
        run_cmd('python3.6 tune_relay_cuda.py --network resnet-50    --batch-size 1')
        run_cmd('python3.6 tune_relay_cuda.py --network resnet-50    --batch-size 16')
        run_cmd('python3.6 tune_relay_cuda.py --network mobilenet_v2 --batch-size 1')
        run_cmd('python3.6 tune_relay_cuda.py --network mobilenet_v2 --batch-size 16')
        run_cmd('python3.6 tune_relay_cuda.py --network resnet3d-18  --batch-size 1')
        run_cmd('python3.6 tune_relay_cuda.py --network resnet3d-18  --batch-size 16')
        run_cmd('python3.6 tune_relay_cuda.py --network dcgan --batch-size 1')
        run_cmd('python3.6 tune_relay_cuda.py --network dcgan --batch-size 16')
        run_cmd('python3.6 tune_relay_cuda.py --network bert  --batch-size 1')
        run_cmd('python3.6 tune_relay_cuda.py --network bert  --batch-size 16')
    else:
        raise NotImplemented
    """
