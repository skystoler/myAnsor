# Help

**Use the official TVM repo instead of Ansor**.

### Build TVM
Follow this doc https://tvm.apache.org/docs/install/index.html

### Single-Op/Subgraph Evaluation
```
# cpu
python3 benchmark_op_subgraph.py --backend cpu
python3 benchmark_op_subgraph.py --eval --backend cpu

# gpu
python3 benchmark_op_subgraph.py --backend gpu
python3 benchmark_op_subgraph.py --eval --backend gpu
```
The results will be stored at `results.tsv`.
Then you can append it to our baseline collection file `paper-figures/baseline/results.tsv`


### Network Evaluation
Run the script
```
# intel cpu
python3 benchmark_network.py --backend intel-cpu

# arm cpu
python3 benchmark_network.py --backend arm-cpu

# nvidia gpu
python3 benchmark_network.py --backend nvidia-gpu
```

The results will be stored at `results.tsv`.
Then you can append it to our baseline collection file `paper-figures/baseline/results.tsv`

