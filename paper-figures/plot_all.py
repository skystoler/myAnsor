"""Produce all plots""" 
import os
import argparse

def run_cmd(cmd):
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", type=str, choices=['png', 'pdf'], default='png')
    args = parser.parse_args()

    # single op & subgraph eval
    run_cmd('python3 plot_op_subgraph_eval.py --device "Intel-Platinum-8124M-3.00GHz" --mode op --out-file figure_6.%s' % args.format)
    run_cmd('python3 plot_op_subgraph_eval.py --mode subgraph --out-file figure_8.%s' % args.format)

    # network eval
    run_cmd('python3 plot_network_eval.py --device "Intel-Platinum-8124M-3.00GHz" --batch-size -1 --out-file figure_9_a.%s' % args.format)
    run_cmd('python3 plot_network_eval.py --device "Tesla V100-SXM2-16GB" --batch-size -1  --out-file figure_9_b.%s' % args.format)
    run_cmd('python3 plot_network_eval.py --device "aarch64-Cortex-A53-1.4Ghz" --out-file figure_9_c.%s' % args.format)

