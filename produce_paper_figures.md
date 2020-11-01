## Instruction on Reproducing the Evaluation Figures in the Paper

This document describes how to generate evaluation figures (figure 6, 8 and 9) in our paper.

## Produce figures
```bash
git clone https://lmzheng@bitbucket.org/lmzheng/ansor-artifact.git
pip3 install matplotlib
cd ansor-artifact/paper-figures
python3 plot_all.py
```

The python script will output figures `figure_6.png`, `figure_8.png`, `figure_9_a.png`, `figure_9_b.png`, `figure_9_c.png` in the paper's
evaluation section to the same folder (`ansor-artifact/paper-figures`).

The script uses the benchmark data in the file `ansor-artifact/paper-figures/baseline/results.tsv`.
The sections below describe how the data is collected.

## Collect Benchmark Data for Baseline Systems
We use PyTorch, TensorFlow, Halide, FlexTensor, and AutoTVM as the baseline systems in the paper.
We provide the code and instructions on how to benchmark each of them.
The collected benchmark data is gathered to `ansor-artifact/paper-figures/baseline/results.tsv` for generating the figures above.

- PyTorch: see [paper-figures/baseline/pytorch/README.md](paper-figures/baseline/pytorch/README.md)
- TensorFlow: see [paper-figures/baseline/tensorflow/README.md](paper-figures/baseline/tensorflow/README.md)
- Halide: see [paper-figures/baseline/halide/README.md](paper-figures/baseline/halide/README.md)
- FlexTensor: see [paper-figures/baseline/FlexTensor/README.md](paper-figures/baseline/FlexTensor/README.md)
- AutoTVM: see [paper-figures/baseline/tvm/README.md](paper-figures/baseline/tvm/README.md)

## Collect Benchmark Data for Ansor
Follow the instructions in [README.md](README.md) to launch the docker container and build ansor for CPU or GPU.
Then run the scripts below.

```
cd scripts

# Intel CPU
python3 gather_ansor_data.py  --backend intel-cpu

# Nvidia GPU
python3 gather_ansor_data.py  --backend nvidia-gpu
```

The results will be saved to `scripts/results.tsv`.
Then you can append the content of `scripts/results.tsv` to `ansor-artifact/paper-figures/baseline/results.tsv`.

The above commands only run testing with pre-collected search results on AWS c5.9xlarge and p3.2xlarge instances.
The search results are stored in files `network-cpu-ansor-logs.json`, `network-gpu-ansor-logs.json`,
`op-subgraph-cpu-ansor-logs.json`, `op-subgraph-gpu-ansor-logs.json`.
If you want to do evaluation on new machines, you have to run the search by yourself following the
instructions provided by the README.md and description provided by the paper.

