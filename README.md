# Ansor : Generating High-Performance Tensor Programs for Deep Learning
-----------------------------------

This is the artifact for the paper "Ansor : Generating High-Performance Tensor Programs for Deep Learning".
Ansor is built on top of [TVM](https://github.com/apache/incubator-tvm), so this repo is built from a fork of TVM.

Ansor is a search-based compilation framework, it needs to run search and measurement on the specific hardware platforms to get the best perforamnce.
The search is time-consuming and computation-intensive.
It takes several weeks to run the search for all experiments and workloads in the paper on a single machine.

To make the artifact evaluation feasible, we provide the following two items:

1. The instructions to reproduce the evaluation results on specific machines.  
   We run the search in advance and provide the search results, so you can directly reuse the search results and skip the time-consuming search.
   Specifically, we use AWS c5.9xlarge instances for the CPU evaluation and AWS P3 series instances for the GPU evaluation. 
   The target cpu is a 18-core Intel Platinum 8124M. The target GPU is a NVIDIA V100.
2. The instructions to run the search for some example workloads.  
   You can test the functionality of the search and run the search if you have time.


# Instructions for reproducing the evaluation results on AWS instances
-----------------------------------
We use a AWS c5.9xlarge instance for the CPU evaluation and a AWS p3.2xlarge (or any other P3 series instance) for the GPU evaluation.
We will only use pytorch as the baseline in this artifact because of its easy usage. PyTorch is backed by vendor libraries such as Intel MKL-DNN and NVIDIA CuDNN.

## CPU Evaluation (AWS c5.9xlarge)
Launch a c5.9xlarge instance with deep learning AMI Deep Learning Base AMI (Ubuntu 18.04).

**Step 1: Build**  
We provide a docker image with all dependencies installed (The image is built from [this](docker/Dockerfile.ansor-artifact) Dockerfile).
The following commands clone this repo, launch a container and build Ansor inside the container.
Docker is required.

```bash
git clone https://lmzheng@bitbucket.org/lmzheng/ansor-artifact.git  
cd ansor-artifact  
docker/bash.sh cpu lmzheng/ansor-artifact  
cp cmake/config_cpu.cmake config.cmake  
make -j30
export PYTHONPATH=/workspace/python:/workspace/topi/python:/workspace/scripts
```

**Step 2: Get the PyTorch baseline**  

```bash
cd scripts
python3 pytorch/benchmark.py --backend cpu
```
  
**Step 3: Evaluate Ansor**  
AVX-512 is required, otherwise you will see fatal errors.

```bash
python3 evaluate_all_networks.py --backend intel-cpu 
```

**Step 4: Draw the table**  

```bash
python3 draw_table.py --backend intel-cpu
```

**Expeced Output**  
Here is the expected output, which also matches the Fig. 9 (a) in the paper.
```
-------------------------------------------------------------
       Inference Execution Time Evaluation (unit: ms)
-------------------------------------------------------------
   Network   | Batch size | PyTorch | Ansor (ous) | Speedup
-------------------------------------------------------------
resnet-50    |      1     |   35.54 |      5.81   |   6.12 X
resnet-50    |     16     |  434.28 |     65.62   |   6.62 X
mobilenet-v2 |      1     |   13.03 |      0.86   |  15.14 X
mobilenet-v2 |     16     |  184.21 |     11.67   |  15.78 X
resnet3d-18  |      1     |  615.13 |     33.91   |  18.14 X
resnet3d-18  |     16     | 8335.67 |    504.22   |  16.53 X
dcgan        |      1     |   16.69 |      1.15   |  14.47 X
dcgan        |     16     |  205.17 |      9.26   |  22.16 X
bert         |      1     |   29.85 |     20.24   |   1.48 X
bert         |     16     |  254.32 |    254.17   |   1.00 X
-------------------------------------------------------------
```


### GPU Evaluation (AWS P3 series)
Launch a p3.2xlarge instance (or any other P3 series instance) with deep learning AMI Deep Learning Base AMI (Ubuntu 18.04).

**Step 1: Build**  
We provide a docker image with all dependencies installed (The image is built from [this](docker/Dockerfile.ansor-artifact) Dockerfile).
The following commands clone this repo, launch a container and build Ansor inside the container.
Nvidia-docker is required.

```bash
git clone https://lmzheng@bitbucket.org/lmzheng/ansor-artifact.git  
cd ansor-artifact  
docker/bash.sh gpu lmzheng/ansor-artifact  
cp cmake/config_gpu.cmake config.cmake  
make -j10
export PYTHONPATH=/workspace/python:/workspace/topi/python:/workspace/scripts
```

**Step 2: Get the PyTorch baseline**  

```bash
cd scripts
python3 pytorch/benchmark.py --backend gpu
```
  
**Step 3: Evaluate Ansor**  

```bash
python3 evaluate_all_networks.py --backend nvidia-gpu
```

**Step 4: Draw the table**  

```bash
python3 draw_table.py --backend nvidia-gpu
```

**Expeced Output**  
Here is the expected output, which also matches the Fig. 9 (b) in the paper.
```
-------------------------------------------------------------
       Inference Execution Time Evaluation (unit: ms)
-------------------------------------------------------------
   Network   | Batch size | PyTorch | Ansor (ous) | Speedup
-------------------------------------------------------------
resnet-50    |      1     |    7.88 |      1.53   |   5.14 X
resnet-50    |     16     |   15.24 |     10.75   |   1.42 X
mobilenet-v2 |      1     |    6.83 |      0.46   |  14.89 X
mobilenet-v2 |     16     |    6.88 |      1.87   |   3.67 X
resnet3d-18  |      1     |    9.62 |      7.12   |   1.35 X
resnet3d-18  |     16     |  102.09 |     97.69   |   1.05 X
dcgan        |      1     |    0.76 |      0.31   |   2.43 X
dcgan        |     16     |    1.89 |      1.30   |   1.45 X
bert         |      1     |   13.44 |      4.25   |   3.16 X
bert         |     16     |   35.63 |     35.62   |   1.00 X
-------------------------------------------------------------
```

# Instructions for running the search
This section provides instructions to run the search by yourself on any machine.

### Build
We provide a docker image with all dependencies installed (The image is built from [this](docker/Dockerfile.ansor-artifact) Dockerfile).
The following commands clone this repo, launch a container and build Ansor inside the container.

**Build for CPU Only**  
Docker is required.

```bash
git clone https://lmzheng@bitbucket.org/lmzheng/ansor-artifact.git  
cd ansor-artifact  
docker/bash.sh cpu lmzheng/ansor-artifact  
cp cmake/config_cpu.cmake config.cmake  
make -j10
export PYTHONPATH=/workspace/python:/workspace/topi/python:/workspace/scripts
```

**Build for Both CPU and GPU**  
Nvidia-docker and CUDA are required.

```bash
git clone https://lmzheng@bitbucket.org/lmzheng/ansor-artifact.git  
cd ansor-artifact  
docker/bash.sh gpu lmzheng/ansor-artifact  
cp cmake/config_gpu.cmake config.cmake  
make -j10
export PYTHONPATH=/workspace/python:/workspace/topi/python:/workspace/scripts
```

### Warmup Example: Generating tensor programs for a matrix multiplication

To begin with, we do a quick test to test the functionality of Ansor.
We generate the code for a square matrix multiplication (N=M=K=512).

First, run the search and measure 100 candidate programs. The search can take several minutes.
```bash
cd scripts
python3 tune_test.py --wkl matmul-512 --n-trials 100
```

You can see Ansor prints the measurement results for these 100 programs.
When the search is done, Ansor prints the lowered IR for the best program we found as well as the evaluation result of this program.
One of the sample output could be

```
... (Omitted)
------------------------------------------------------------
-----------------------  [ Done ]
------------------------------------------------------------
==================== Lowered TIR ====================
... (Omitted)
==================== Evaluate ====================
Best program: 744.05 GFLOPS	cost: 0.361 ms
```

This means Ansor finds an implementation of matmul with 744.05 GFLOPs and its latency is 0.361ms .
The results will vary according to your test machine and the randomness of the search.
Note that this is only a warmup example, we use a relatively low `--n-trials` (100) in the example, which is not enough for the search to converge.
So the result may not be very good.
Typically, we use `--n-trials 1000` in the real evaluation, as mentioned in the Sec 7.1 of the paper.

### Network Evaluation : Generating the tensor programs for a neural network

We use mobilenet-v2 as an example. First, we do a quick warm-up test by generating the tensor programs with a small number of measurements.
Ansor will download the official model from tflite model zoo and optimize it. The search can take several minutes.

```bash
cd scripts
python3 tune_network.py --network mobilenet-v2 --n-trials 200
```
One sample output could be
```
... (Omitted)
=============== Compile ===============
DEBUG:ansor:Finish loading 204 records
=============== Compile Finish ===============
========== Evaluate ==========
Mean inference time (std dev): 2.59 ms (0.01 ms)
```

It means ansor measured about 200 programs and picked best of them to compile the whole neural network.
The inference execution time of the optimized neural network is 2.59ms.

As we mentioned in the Sec. 7.3 of the paper, we should use `n-trials = 1000 * number of subgraphs in the network` in the real evaluation.
For mobilenet-v2, there are around 30 unique subgraphs, so `n-trials` should be set to `30 * 1000 = 30000`.
In practice, the search usually converges after `n-trials = 20000`, so we can just use `n-trials = 20000` to save some time.
The commands to fully reproduce the results are listed below. The search can take several hours. A machine with many CPU cores (>16) is recommended.
```bash
# for cpus with AVX2
python3 tune_network.py --network mobilenet-v2 --n-trials 20000 --target "llvm -mcpu=core-avx2"

# for cpus with AVX-512
python3 tune_network.py --network mobilenet-v2 --n-trials 20000 --target "llvm -mcpu=skylake-avx512"
```


### More search examples
Here we provide some additional search examples for you to try. Please feel free to ask for more examples if necessary.
The search is computation-intensive, so a machine with many CPU cores (>16) is recommended.

Replace "llvm -mcpu=core-avx2" with "llvm -mcpu=skylake-avx512" in the commands below if your target CPU has avx-512 (e.g., AWS c5.9xlarge).  
Replace "llvm -mcpu=core-avx2" with "cuda" in the commands below if you target GPU.  

**Single Operator and Subgraph**
```bash
# Matrix multiplication
python3 tune_test.py --wkl matmul-1024 --n-trials 1000 --target "llvm -mcpu=core-avx2"

# The first convolution layer in resnet-50
python3 tune_test.py --wkl nhwc-resnet-50.C0 --n-trials 1000 --target "llvm -mcpu=core-avx2"

# The thrid convolution layer in resnet-50
python3 tune_test.py --wkl nhwc-resnet-50.C2 --n-trials 1000 --target "llvm -mcpu=core-avx2"
```

More examples can be found in `tune_test.py` and `tune_op_subgraph.py`.

**Network**
```bash
python3 tune_network.py --network resnet-50    --n-trials 25000 --batch-size 1 --target "llvm -mcpu=core-avx2"
python3 tune_network.py --network resnet3d-18  --n-trials 25000 --batch-size 1 --target "llvm -mcpu=core-avx2"
python3 tune_network.py --network mobilenet-v2 --n-trials 20000 --batch-size 1 --target "llvm -mcpu=core-avx2"
python3 tune_network.py --network bert         --n-trials 10000 --batch-size 1 --target "llvm -mcpu=core-avx2"
python3 tune_network.py --network dcgan        --n-trials 10000 --batch-size 1 --target "llvm -mcpu=core-avx2"
```

# Project Structure
-----------------------------------
```
- `scripts`: scripts to run the search
- `python/tvm/ansor`: Python frontend API
  - `auto_schedule.py`: User interface to use the auto-scheduler
  - `cost_model`: Python part code of the cost model. We use some python code 
                  because most machine learning frameworks are in python
  - `compute_dag.py`: Compute declaration graph and its related analysis tools
  - `dispatcher.py`: A global context to dispatch configurations. This is migrated from the old autotvm.
  - `env.py`: The scope to store global variables 
  - `feature.py`: Feature extraction for the cost model
  - `measure.py`: Python part code of measurement infrastructure. We use python's multiprocessing and exception handling.
  - `relay_integratoin.py`: Integrate ansor into Relay and TOPI
  - `serialization.py`: IO utilities for tuning records
  - `task_scheduler.py`: Task scheduler which tunes multiple workloads jointly. This is implemented in pure python.
  - `utils.py`: Other utilities
  - `workload_registry.py`: Workload registry
- `src/tvm/ansor`: C++ core
  - `cost_model`: Cost models
  - `search_policy`: Search policies
    - `sketch_search_policy.h`: The core search policy
	- `utils.h`: The common utilities for search policy
	- `search_policy.h`: The base class for search policy
  - `auto_cheduler.h`: User interface to use the auto-scheduler.
  - `compute_dag.h`: Compute declaration graph and its related analysis tools
  - `feature.h`: Feature extraction for the cost model
  - `loop_state.h`: An simplified loop structure IR for search. This defines the "state" for the search problem.
  - `measure.h`: Measurement infrastructure.
  - `search_task.h`: Meta information for a search task & Hardware parameters.
  - `serialization.h`: Json serialization format for dumping and loading tuning records
  - `transform_steps.h`: Schedule primitives (i.e. transform steps) for our IR. This defines the "action" for the search problem.
  - `utils.h`: Other common utilities
```
