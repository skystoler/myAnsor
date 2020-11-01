"""Test layout rewrite for a whole network"""

import argparse
import logging
import random
import os
import numpy as np

import tvm
from tvm import ansor, relay
import tvm.contrib.graph_runtime as runtime
from tvm.relay import testing

def get_np_array(var, dtype):
    return np.random.randn(*[int(x) for x in var.type_annotation.shape]).astype(dtype)

def get_relay_batchmm(batch=8, m=512, n=512, k=512):
    dtype = 'float32'
    d = relay.var('data', shape=(batch, m, k), dtype=dtype) 
    w = relay.var('weight', shape=(batch, n, k), dtype=dtype)
    y = relay.nn.batch_matmul(d, w)
    mod = tvm.IRModule()
    mod['main'] = relay.Function([d, w], y)
    data, weight = get_np_array(d, dtype), get_np_array(w, dtype)
    return mod, data, weight


def get_relay_dense(m=512, n=3072, k=768):
    dtype = 'float32'
    d = relay.var('data', shape=(m, k), dtype=dtype) 
    w = relay.var('weight', shape=(n, k), dtype=dtype)
    y = relay.nn.dense(d, w, units=n)
    mod = tvm.IRModule()
    mod['main'] = relay.Function([d, w], y)
    data, weight = get_np_array(d, dtype), get_np_array(w, dtype)
    return mod, data, weight


def get_relay_conv2d(outc=128, inc=64, height=14, width=14, kh=3, kw=3, batch=1, pad=0, stride=1, dilation=1, group=1, layout="NHWC"):
    dtype = 'float32'
    if layout == "NHWC":
        if group == 1:
            wlayout = "HWIO"
        else:
            wlayout = "HWOI"
        d = relay.var('data', shape=(batch, height, width, inc), dtype=dtype)
    elif layout == "NCHW":
        wlayout = "OIHW"
        d = relay.var('data', shape=(batch, inc, height, width), dtype=dtype)

    if group == 1:
        if wlayout == "HWIO":
            w = relay.var('weight', shape=(kh, kw, inc, outc), dtype=dtype)
        elif wlayout == "OIHW":
            w = relay.var('weight', shape=(outc, inc, kh, kw), dtype=dtype)
    elif group == inc:
        if wlayout == "OIHW":
            w = relay.var('weight', shape=(outc, 1, kh, kw), dtype=dtype)
        elif wlayout == "HWOI":
            w = relay.var('weight', shape=(kh, kw, outc, 1), dtype=dtype)

    y = relay.nn.conv2d(d, w, 
            padding=pad, 
            kernel_size=(kh, kw), 
            strides=(stride, stride),
            dilation=(dilation, dilation),
            channels=outc,
            groups=group,
            data_layout=layout,
            kernel_layout=wlayout)
    mod = tvm.IRModule()
    mod['main'] = relay.Function([d, w], y)
    data, weight = get_np_array(d, dtype), get_np_array(w, dtype)
    return mod, data, weight

def get_relay_conv3d(outc=128, inc=64, depth=4, height=14, width=14, kd=3, kh=3, kw=3, batch=1, pad=0, stride=1):
    dtype = 'float32'
    d = relay.var('data', shape=(batch, width, height, width, inc), dtype=dtype)
    w = relay.var('weight', shape=(kd, kh, kw, inc, outc), dtype=dtype)

    y = relay.nn.conv3d(d, w, 
            padding=pad, 
            kernel_size=(kd, kh, kw), 
            strides=(stride, stride, stride),
            channels=outc,
            data_layout="NDHWC",
            kernel_layout="DHWIO")
    mod = tvm.IRModule()
    mod['main'] = relay.Function([d, w], y)
    data, weight = get_np_array(d, dtype), get_np_array(w, dtype)
    return mod, data, weight


def tune_and_check(network, target):
    # Get network
    if network == "dense":
        mod, data, weight = get_relay_dense()
    elif network == "batchmm":
        mod, data, weight = get_relay_batchmm()
    elif network == "conv2d":
        mod, data, weight = get_relay_conv2d(kh=1, kw=1)
    elif network == "conv2d_winograd":
        mod, data, weight = get_relay_conv2d()
    elif network == "depthwise_conv2d":
        mod, data, weight = get_relay_conv2d(outc=64, inc=64, group=64)
    elif network == "conv3d":
        mod, data, weight = get_relay_conv3d()

    # Extract tasks from a relay program
    target = tvm.target.create(target)
    workloads, wkl_weights = ansor.extract_from_program(mod, target=target, params={})
    tasks = []
    for wkl_key in workloads:
        dag = ansor.workload_key_to_dag(wkl_key)
        tasks.append(ansor.SearchTask(dag, wkl_key, target))

    # Tune workloads
    log_file = 'test_layout_rewrite.json'
    measure_ctx = ansor.LocalRPCMeasureContext()
    runner = measure_ctx.runner
    tune_option = ansor.TuneOption(n_trials=1,
                                   num_measure_per_iter=1,
                                   measure_callbacks=[ansor.LogToFile(log_file)])

    tuner = ansor.SimpleTaskScheduler(tasks, strategy='round-robin')
    tuner.tune(tune_option, 'sketch.random')
    del measure_ctx

    # Compile and run
    def compile_and_run(disabled_pass={}):
        ctx = tvm.context(str(target))

        with ansor.apply_history_best(log_file):
            with tvm.transform.PassContext(opt_level=3, disabled_pass=disabled_pass):
                graph, lib, params = relay.build_module.build(
                    mod, target=target, params={'weight': weight})
                relay.backend.compile_engine.get().clear()

            module = runtime.create(graph, lib, ctx)
            module.set_input("data", data)
            for k, v in params.items():
                module.set_input(k, v)
            module.run()

        return module.get_output(0).asnumpy()

    # Check correctness 
    expected_output = compile_and_run(disabled_pass={"KernelLayoutTransform"})
    actual_output = compile_and_run()
    np.testing.assert_allclose(actual_output, expected_output, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    target = 'llvm'

    tune_and_check("dense", target)
    tune_and_check("batchmm", target)
    tune_and_check("conv2d", target)
    tune_and_check("conv2d_winograd", target)
    tune_and_check("depthwise_conv2d", target)
    tune_and_check("conv3d", target)

