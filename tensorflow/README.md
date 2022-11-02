# Help

1. Install tensorflow >= 2.0
2. Install tf\_slim
```bash
pip3 install tf_slim --user
```
3. Run scripts
```
python3 benchmark.py --backend cpu --session True
python3 benchmark.py --backend gpu --session True
python3 benchmark.py --backend gpu --tensorrt True
```

The results will be stored at `results.tsv`.
Then you can append it to our baseline collection file `paper-figures/baseline/results.tsv`

