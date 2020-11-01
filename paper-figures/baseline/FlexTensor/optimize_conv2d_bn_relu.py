import os
import sys
import argparse
import time
import json
import torch
import tvm 
import numpy as np
from tvm import rpc
from flextensor.utils import Config, RpcInfo
from flextensor.task import Task, TASK_TABLE
from flextensor.scheduler import schedule, schedule_with_config
from flextensor.measure import _evaluate
from flextensor.utils import to_tuple
from flextensor.configs.conv2d_bn_relu_config import conv2d_bn_relu_shapes

from utils import shape_dict, BenchmarkRecord, log_line

LOCAL_RPC = False
LIB_DIR = "."


def evaluate(name, s, bufs, target, dev_id, number, rpc_info):
    if rpc_info is not None:
        host = rpc_info.host
        port = rpc_info.port
    else:
        # local
        host = "0.0.0.0"
        port = 9090     # default port
    if host == "0.0.0.0":
        if LOCAL_RPC:
            use_rpc = True
        else:
            use_rpc = False
    else:
        use_rpc = True
    if use_rpc:
        remote = rpc.connect(host, port)
        ctx = remote.context(target, dev_id)
    else:
        ctx = tvm.context(target, dev_id)
    tvm_arys = []
    for buf in bufs:
        shape = to_tuple(buf.shape)
        tmp = np.random.uniform(-10, 10, size=shape).astype(buf.dtype)
        tmp = tvm.nd.array(tmp, ctx)
        tvm_arys.append(tmp)
    try:
        func_file = "{}.tar".format(name)
        if rpc_info is not None and rpc_info.target_host is not None:
            if 'llvm' in target:
                target = target_host = 'llvm -mcpu=core-avx2'
            else:
                target_host = rpc_info.target_host
            func = tvm.build(s, bufs, target=target, target_host=target_host)
        else:
            func = tvm.build(s, bufs, target=target)
        if use_rpc:
            func.export_library(os.path.join(LIB_DIR, func_file))
            remote.upload(os.path.join(LIB_DIR, func_file))
            func = remote.load_module(func_file)
        evaluator = func.time_evaluator(func.entry_name, ctx, number=number, min_repeat_ms=1000)
        time_cost = evaluator(*tvm_arys).mean * 1e3
    except Exception as e:
        print(e)
        time_cost = float("inf")
    finally:
        while len(tvm_arys) > 0:
            del tvm_arys[-1]
        if os.path.exists(os.path.join(LIB_DIR, func_file)):
            try:
                os.remove(os.path.join(LIB_DIR, func_file))
            except Exception as e:
                print(e)
        elif os.path.exists(os.path.join(LIB_DIR, func_file + ".so")):
            try:
                os.remove(os.path.join(LIB_DIR, func_file))
            except Exception as e:
                print(e)
    return time_cost


def optimize(shapes, slevel=4, rlevel=3, target="llvm", dev_id=0, timeout=4.0, trials=100, parallel=1, 
        method="searching", use_model=False, rpc_info=None, logfile=sys.stdout, force_inline=False):
    ret = dict()
    for i, shape in enumerate(shapes):
        print("Optimize conv2d_bn_relu shape %s [%.6f]" % (str(shape), time.time()), flush=True)
        N, H, W, CI, CO, kernel_size, strides, padding, dilation = shape
        # create an empty task but has the correct key we want
        task = Task(
            "conv2d_bn_relu",
            "conv2d_bn_relu", 
            None, 
            (N, H, W, CI, CO, kernel_size, strides, padding, dilation), 
            target, 
            dev_id
            )
        beg = time.time()
        s, bufs, configs = schedule(
            task.key, 
            slevel=slevel,
            rlevel=rlevel,
            op_trial=trials, 
            timeout=timeout, 
            op_stop=200, 
            method=method, 
            use_model=use_model,
            parallel=parallel,
            rpc_info=rpc_info,
            force_inline=force_inline
            )
        end = time.time()
        # print(tvm.lower(s, bufs, simple_mode=True))
        print("###################################### [%.6f]" % time.time())
        print("op schedules:")
        for config in configs.op_config_lst:
            print("----------------------------------")
            for name, value in config.items():
                if value:
                    print(name, value)
        print("graph schedules:")
        for name, value in configs.graph_config.items():
            if value:
                print(name, value)
        ret[task.key] = configs
        string = json.dumps(configs)
        line = task.key + ":" + string
        print(line, file=logfile, flush=True)
        s, bufs = schedule_with_config(task.key, configs)
        time_cost = _evaluate(s, bufs, target, task.dev_id, 10)
        print("Use", time_cost, "ms")
        print("Cost", end - beg, "s")
        print()
    return ret


def test(task_key, configs, dev_id=None, rpc_info=None):
    task = TASK_TABLE[task_key]
    s, bufs = schedule_with_config(task_key, configs)
    dev_id = dev_id if dev_id is not None else task.dev_id
    # print(tvm.lower(s, bufs, simple_mode=True))

    cost = evaluate(task_key, s, bufs, task.target, dev_id, 10, rpc_info)

    backend = "cpu" if "llvm" in task.target else "gpu"

    substrs = task_key.split('_')
    name = "_".join(substrs[:-2])
    workload_name = "%s%s" % (name[:len(name)//2], substrs[-2])
    print("%s\t%.3f ms" % (workload_name, cost))
    log_line(BenchmarkRecord('device', backend, 'subgraph', workload_name,
                 "FlexTensor", 'default', {"costs": [cost / 1e3]}, time.time()),
             'results.tsv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--from_", help="From which shape", type=int, default=0)
    parser.add_argument("-t", "--to", help="To which shape", type=int, default=-1)
    parser.add_argument("-l", "--log", help="Log file name", type=str, default="")
    parser.add_argument("--test", help="test file name", type=str, default="")
    parser.add_argument("--trials", help="number of trials for op", type=int, default=100)
    parser.add_argument("--target", help="target device type", type=str, default="llvm")
    parser.add_argument("--device", help="target device number", type=int, default=0)
    parser.add_argument("--timeout", help="timeout", type=float, default=4.0)
    parser.add_argument("--parallel", help="parallel", type=int, default=1)
    parser.add_argument("--use_model", help="use performance model", action="store_true")
    parser.add_argument("--method", help="how to schedule", type=str, default="searching")
    parser.add_argument("--slevel", type=int, default=4)
    parser.add_argument("--rlevel", type=int, default=3)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--target_host", type=str, default="llvm")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--force_inline", action="store_true")
    args = parser.parse_args()
    shapes = conv2d_bn_relu_shapes
    rpc_info = RpcInfo(args.host, args.port, args.target_host)
    if args.to < 0:
        end = len(shapes)
    else:
        end = args.to
    
    if args.test != "":
        with open(args.test, "r") as fin:
            for line in fin:
                name, string = line.split(":", 1)
                obj = json.loads(string)
                configs = Config(obj[0], obj[1])
                test(name, configs, args.device, rpc_info=rpc_info)

    elif args.log != "":
        with open(args.log, "a") as flog:
            ret = optimize(
                shapes[args.from_:end], 
                slevel=args.slevel,
                rlevel=args.rlevel,
                target=args.target, 
                dev_id=args.device, 
                timeout=args.timeout, 
                trials=args.trials, 
                parallel=args.parallel,
                use_model=args.use_model,
                method=args.method,
                force_inline=args.force_inline,
                logfile=flog,
                rpc_info=rpc_info
                )
    else:
        ret = optimize(
            shapes[args.from_:end], 
            slevel=args.slevel,
            rlevel=args.rlevel,
            target=args.target, 
            dev_id=args.device, 
            timeout=args.timeout, 
            trials=args.trials, 
            parallel=args.parallel,
            use_model=args.use_model,
            method=args.method,
            logfile=sys.stdout,
            rpc_info=rpc_info,
            force_inline=args.force_inline
            )

