# Help
1. Install [PyTroch](https://pytorch.org/)
2. Install transformers
```
pip3 install transformers --user
```

3. Run scripts
```
# CPU
MKL_ENABLE_INSTRUCTIONS=AVX2 python3 benchmark.py --backend cpu --wkl-type op
MKL_ENABLE_INSTRUCTIONS=AVX2 python3 benchmark.py --backend cpu --wkl-type subgraph
MKL_ENABLE_INSTRUCTIONS=AVX2 python3 benchmark.py --backend cpu --wkl-type subgraph --jit
python3 benchmark.py --backend cpu --wkl-type network
python3 benchmark.py --backend cpu --wkl-type network --jit

# GPU
python3 benchmark.py --backend gpu --wkl-type subgraph
python3 benchmark.py --backend gpu --wkl-type subgraph --jit
python3 benchmark.py --backend gpu --wkl-type network
python3 benchmark.py --backend gpu --wkl-type network --jit
```

The results will be stored at `results.tsv`.
Then you can append it to our baseline collection file `paper-figures/baseline/results.tsv`

