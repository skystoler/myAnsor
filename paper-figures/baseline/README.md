# Baseline

Put baseline results and instructions to run baseline here.
The instructions should contain build commands and a script to gather all results for this backend.

## Format
We save all baseline results into a single tsv file `results.tsv`, so we can share and query it easily.

The format for each line in this tsv file is
```
device   backend  workload_type  workload_name  library  algorithm  value  time_stamp
```

Example
```
1080ti    cuda  network  resnet-18.B1     cudnn           default  {"costs": [0.123, 0.234]}  99285.17
rk3399    cpu   network  resnet-18.B1     tflite          default  {"costs": [0.123, 0.234]}  93485.23
i7-9750h  cpu   op       resnet-18.C1.B1  pytorch-mkldnn  default  {"costs": [0.123, 0.234]}  93885.49
i7-9750h  cpu   op       resnet-18.C1.B1  halide          adam19   {"costs": [0.123, 0.234]}  93885.49
```

**Note**:  
`B1` in `resnet-18.B1` means batch size = 1  
`C1` in `resnet-18.C1.B1` means the 1st convolution layer
