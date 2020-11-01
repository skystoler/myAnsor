# Help
1. Install our fork of [FlexTensor](https://github.com/jcf94/FlexTensor/tree/ansor_eval) at `ansor_eval` branch
```
git clone -b ansor_eval git@github.com:jcf94/FlexTensor.git 

# add the directory to python path
export PYTHONPATH=$PYTHONPATH:~/FlexTensor
```

2. Run scripts

```
# cpu
python3 benchmark_all.py --backend cpu

# gpu
python3 benchmark_all.py --backend gpu
```


The results will be stored at `results.tsv`.
Then you can append it to our baseline collection file `paper-figures/baseline/results.tsv`

