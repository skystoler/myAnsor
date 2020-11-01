# Help
1. Install Tensorflow (r2.0) and build TFLite (r2.0) runtime as this [link](https://github.com/apache/incubator-tvm/pull/4698) 
2. Run scripts
```
python3 benchmark.py --tflite-model-path mobilenet_v2_1.0_224.tflite --num-threads 4 --timeout 1000
```

The results will be stored at `results.tsv`.
Then you can append it to our baseline collection file `scripts/baseline/results.tsv`

