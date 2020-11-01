import os
import sys
import argparse
import platform
import time
import json
import torch
import numpy as np

from tvm import rpc, te
import tvm 
import topi
import flextensor
from flextensor.utils import Config, RpcInfo
from flextensor.task import Task, TASK_TABLE
from flextensor.scheduler import schedule, schedule_with_config
from flextensor.measure import _evaluate
from flextensor.utils import to_tuple
from flextensor.configs.gemm_config import gemm_shapes
from flextensor.task import register_task, Task
from flextensor.model import WalkerGroup
from flextensor.scheduler import schedule

from utils import shape_dict, BenchmarkRecord, log_line

# override the multiprocessing context from 'spawn' to the default 'fork'
try:
    import torch.multiprocessing as _multi
except ImportError:
    import multiprocessing as _multi
flextensor.scheduler.multi = _multi

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

            # hack to improve the target
            if target == 'llvm':
                target = target_host = 'llvm -mcpu=core-avx2'
            else:
                target = task.target
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


def optimize(task, slevel=4, rlevel=3, target="llvm", dev_id=0, timeout=4.0, trials=100, parallel=1, 
        method="searching", use_model=False, rpc_info=None, force_inline=False, logfile=sys.stdout):
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
        force_inline=force_inline,
        rpc_info=rpc_info
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
    string = json.dumps(configs)
    line = task.key + ":" + string
    print(line, file=logfile, flush=True)
    s, bufs = schedule_with_config(task.key, configs)
    time_cost = _evaluate(s, bufs, target, task.dev_id, 10)
    print("Use", time_cost, "ms")
    print("Cost", end - beg, "s")
    print()

def test(task_key, configs, workload_type, dev_id=None, rpc_info=None, out_file="results.tsv"):
    task = TASK_TABLE[task_key]
    s, bufs = schedule_with_config(task_key, configs)
    dev_id = dev_id if dev_id is not None else task.dev_id
    #print(tvm.lower(s, bufs, simple_mode=True))
    #print(30 * "=")

    cost = evaluate(task_key, s, bufs, task.target, dev_id, 10, rpc_info) / 1e3

    if 'llvm' in task.target:
        device = platform.processor()
        backend = 'cpu'
    else:
        device = tvm.context(target).device_name
        backend = 'gpu'

    substrs = task_key.split('_')
    name = "_".join(substrs[:-2])
    workload_name = "%s%s" % (name[:len(name)//2], substrs[-2])
    print("%s\t%.3f ms" % (workload_name, cost * 1e3))
    log_line(BenchmarkRecord(device, backend, workload_type, workload_name,
                 "FlexTensor", 'default', {"costs": [cost]}, time.time()),
                 out_file)

########## TASK REGISTRATION ##########

def batch_matmul_nkkm(B, N, M, K):
    X = te.placeholder((B, N, K), name='A')
    Y = te.placeholder((B, K, M), name='B')
    k = te.reduce_axis((0, K), name='k')
    Z = te.compute((B, N, M), lambda b, i, j: te.sum(X[b][i][k] * Y[b][k][j], axis=[k]), name='C')
    return [Z.op], [X, Y, Z]

def conv1d_nlc(N, L, CI, CO, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    inputs = te.placeholder((N, L, CI), name='inputs')
    weight = te.placeholder((kernel_size, CI//groups, CO), name='weight')

    batch_size, in_len, in_channel = inputs.shape
    k_len, channel_per_group, out_channel = weight.shape
    out_channel_per_group = out_channel // groups
    out_len = (in_len + 2 * padding - dilation * (k_len - 1) - 1) // stride + 1
    rc = te.reduce_axis((0, channel_per_group), name='rc')
    rl = te.reduce_axis((0, k_len), name='rl')

    padded = topi.nn.pad(inputs, [0, padding, 0])
    output = te.compute(
        (batch_size, out_len, out_channel),
        lambda n, l, co: te.sum(
            (padded[n, l * stride + rl * dilation, co // out_channel_per_group * channel_per_group + rc] *
             weight[rl, rc, co]), axis=[rl, rc]),
        name='conv1d_nlc'
    )
    return [output.op], [inputs, weight, output]

def conv2d_nhwc(N, H, W, CI, CO, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    inputs = te.placeholder((N, H, W, CI), name='inputs')
    weight = te.placeholder((kernel_size, kernel_size, CI//groups, CO), name='weight')
    batch_size, in_h, in_w, in_channel = inputs.shape
    k_h, k_w, channel_per_group, out_channel = weight.shape
    out_channel_per_group = out_channel // groups

    out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
    rh = te.reduce_axis((0, k_h), name="rh")
    rw = te.reduce_axis((0, k_w), name="rw")
    rc = te.reduce_axis((0, channel_per_group), name="rc")

    padded = topi.nn.pad(inputs, [0, padding, padding, 0])
    output = te.compute(
        (batch_size, out_h, out_w, out_channel),
        lambda n, h, w, co: te.sum(
            (padded[n, h * stride + rh * dilation, w * stride + rw * dilation,
                    co // out_channel_per_group * channel_per_group + rc]
             * weight[rh, rw, rc, co]), axis=[rh, rw, rc]
        ),
        name='conv2d_nhwc'
    )
    return [output.op], [inputs, weight, output]

def conv2d_nchw(N, CI, H, W, CO, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    inputs = te.placeholder((N, CI, H, W), name='inputs')
    weight = te.placeholder((CO, CI//groups, kernel_size, kernel_size), name='weight')
    batch_size, in_channel, in_h, in_w = inputs.shape
    out_channel, channel_per_group, k_h, k_w, = weight.shape
    out_channel_per_group = out_channel // groups

    out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
    rc = te.reduce_axis((0, channel_per_group), name="rc")
    rh = te.reduce_axis((0, k_h), name="rh")
    rw = te.reduce_axis((0, k_w), name="rw")

    padded = topi.nn.pad(inputs, [0, 0, padding, padding])
    output = te.compute(
        (batch_size, out_channel, out_h, out_w),
        lambda n, co, h, w: te.sum(
            (padded[n, co // out_channel_per_group * channel_per_group + rc,
                    h * stride + rh * dilation, w * stride + rw * dilation]
             * weight[co, rc, rh, rw]), axis=[rc, rh, rw]
        ),
        name='conv2d_nchw'
    )
    return [output.op], [inputs, weight, output]

def conv3d_ndhwc(N, D, H, W, CI, CO, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    inputs = te.placeholder((N, D, H, W, CI))
    weight = te.placeholder((kernel_size, kernel_size, kernel_size, CI, CO))
    batch_size, in_d, in_h, in_w, in_channel = inputs.shape
    k_d, k_h, k_w, channel_per_group, out_channel = weight.shape
    out_channel_per_group = out_channel // groups

    out_d = (in_d + 2 * padding - dilation * (k_d - 1) - 1) // stride + 1
    out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
    rd = te.reduce_axis((0, k_d), name='rd')
    rh = te.reduce_axis((0, k_h), name='rh')
    rw = te.reduce_axis((0, k_w), name='rw')
    rc = te.reduce_axis((0, channel_per_group), name='rc')

    padded = topi.nn.pad(inputs, [0, padding, padding, padding, 0])
    output = te.compute(
        (batch_size, out_d, out_h, out_w, out_channel),
        lambda n, d, h, w, co: te.sum(
            (padded[n, d * stride + rd * dilation,
                    h * stride + rh * dilation, w * stride + rw * dilation,
                    co // out_channel_per_group * channel_per_group + rc]
             * weight[rd, rh, rw, rc, co]),
            axis=[rd, rh, rw, rc]
        ),
        name='conv3d_ndhwc'
    )
    return [output.op], [inputs, weight, output]

def depthwise_conv2d_nhwc(N, H, W, C, kernel_size, stride=1, padding=0, dilation=1, factor=1):
    inputs = te.placeholder((N, H, W, C))
    weight = te.placeholder((factor, kernel_size, kernel_size, C))

    batch_size, in_h, in_w, in_channel = inputs.shape
    factor, k_h, k_w, in_channel = weight.shape
    out_channel = in_channel * factor

    assert factor.value == 1, "Not optimized for factor != 1"

    out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
    rh = te.reduce_axis((0, k_h), name='rh')
    rw = te.reduce_axis((0, k_w), name='rw')

    padded = topi.nn.pad(inputs, [0, padding, padding, 0])
    output = te.compute(
        (batch_size, out_h, out_w, out_channel),
        lambda n, h, w, c: te.sum(
            (padded[n,  h * stride + rh * dilation, w * stride + rw * dilation, c // factor]
             * weight[c % factor, rh, rw, c // factor]),
            axis=[rh, rw]
        ),
        name="depth_conv2d_nhwc"
    )
    return [output.op], [inputs, weight, output]

def conv2d_transpose_nhwc(N, H, W, CI, CO, kernel_size, stride=1, padding=0):
    inputs = te.placeholder((N, H, W, CI), name='inputs')
    weight = te.placeholder((kernel_size, kernel_size, CI, CO), name='weight')

    batch, in_h, in_w, in_c = inputs.shape
    filter_h, filter_w, in_c, out_c = weight.shape
    stride_h, stride_w = (stride, stride)

    # compute padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = topi.nn.get_pad_tuple(padding, (filter_h, filter_w))
    bpad_top = filter_h - 1 - fpad_top
    bpad_bottom = filter_h - 1 - fpad_bottom
    bpad_left = filter_w - 1 - fpad_left
    bpad_right = filter_w - 1 - fpad_right

    # padding stage
    padded = topi.nn.pad(inputs,
                         [0, (bpad_top + stride_h - 1) // stride_h,
                          (bpad_left + stride_w - 1) // stride_w, 0],
                         [0, (bpad_bottom + stride_h - 1) // stride_h,
                          (bpad_right + stride_w - 1) // stride_w, 0])

    # remove extra padding introduced by dilatation
    idxdiv = tvm.te.indexdiv
    idxmod = tvm.te.indexmod
    border_h = idxmod(stride_h - idxmod(bpad_top, stride_h), stride_h)
    border_w = idxmod(stride_w - idxmod(bpad_left, stride_w), stride_w)

    # dilation stage
    strides = [1, stride_h, stride_w, 1]
    n = len(padded.shape)

    # We should embed this dilation directly into te.compute rather than creating a new te.compute.
    # Only in this way can we use unroll to eliminate the multiplication of zeros.
    def _dilate(*indices):
        not_zero = []
        index_tuple = []
        for i in range(n):
            if not strides[i] == 1:
                index_tuple.append(idxdiv(indices[i], strides[i]))
                not_zero.append(idxmod(indices[i], strides[i]).equal(0))
            else:
                index_tuple.append(indices[i])
        if not_zero:
            not_zero = te.all(*not_zero)
            return te.if_then_else(not_zero, padded(*index_tuple), tvm.tir.const(0.0, padded.dtype))
        return padded(*index_tuple)

    # convolution stage
    out_h = (in_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (in_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    rc = te.reduce_axis((0, in_c), name='rc')
    rh = te.reduce_axis((0, filter_h), name='rh')
    rw = te.reduce_axis((0, filter_w), name='rw')

    output = te.compute(
        (batch, out_h, out_w, out_c),
        lambda n, h, w, co: te.sum(
            _dilate(n, h + rh + border_h, w + rw + border_w, rc) *
            weight[filter_h - 1 - rh, filter_w - 1 - rw, rc, co],
            axis=[rh, rw, rc]),
        name="conv2d_transpose_nhwc",
        attrs={"auto_scheduler_always_unroll_inner": ["h", "w", "rh", "rw", "h_c", "w_c"]})
    # todo(lmzheng): add constraints on the tile size of h and w

    return [output.op], [inputs, weight, output]

def conv2d_capsule_nhwijc(N, H, W, CI, CO, kernel_size, stride=1, padding=0, capsule_size=4):
    inputs = te.placeholder((N, H, W, capsule_size, capsule_size, CI), name='inputs')
    weight = te.placeholder((kernel_size, kernel_size, capsule_size, capsule_size, CI, CO), name='weight')
    batch_size, in_h, in_w, _, _, in_channel = inputs.shape
    k_h, k_w, _, _, _, out_channel = weight.shape

    out_h = (in_h + 2 * padding - kernel_size) // stride + 1
    out_w = (in_w + 2 * padding - kernel_size) // stride + 1

    rh = te.reduce_axis((0, k_h), name="rh")
    rw = te.reduce_axis((0, k_w), name="rw")
    cap_k = te.reduce_axis((0, capsule_size), name='cap_k')
    rc = te.reduce_axis((0, in_channel), name="rc")

    padded = topi.nn.pad(inputs, [0, padding, padding, 0, 0, 0])
    output = te.compute(
        (batch_size, out_h, out_w, capsule_size, capsule_size, out_channel),
        lambda n, h, w, cap_i, cap_j, co: te.sum(
            (padded[n, h * stride + rh, w * stride + rw, cap_i, cap_k, rc]
             * weight[rh, rw, cap_k, cap_j, rc, co]), axis=[rh, rw, cap_k, rc]
        ),
        name='conv2d_capsule_nhwijc'
    )

    return [output.op], [inputs, weight, output]

def softmax_mn(M, N):
    A = te.placeholder((M, N), name='A')
    B = topi.nn.softmax(A, axis=1)

    return [B.op], [A, B]

def norm_bmn(B, M, N):
    A = te.placeholder((B, M, N), name='A')
    i = te.reduce_axis((0, M))
    j = te.reduce_axis((0, N))
    C = te.compute((B,), lambda b: te.sum(A[b][i][j] * A[b][i][j], axis=[i, j]), name='C')
    D = te.compute((B,), lambda b: tvm.sqrt(C[b]), name='D')

    return [D.op], [A, D]

def transpose_batch_matmul(batch, seq_len, n_head, n_dim):
    query = te.placeholder((batch, seq_len, n_head, n_dim), name='query')
    value = te.placeholder((batch, seq_len, n_head, n_dim), name='value')
    query_T = te.compute((batch, n_head, seq_len, n_dim),
                      lambda b, h, l, d: query[b, l, h, d], name="query_T")
    value_T = te.compute((batch, n_head, n_dim, seq_len),
                      lambda b, h, d, l: value[b, l, h, d], name="value_T")
    k = te.reduce_axis((0, n_dim), name='k')
    out = te.compute((batch, n_head, seq_len, seq_len), lambda b, h, i, j: te.sum(query_T[b][h][i][k] * value_T[b][h][k][j], axis=[k]), name='C')
    return [out.op], [query, value, out]

def flextensor_conv2d_nhwc_bn_relu(N, H, W, CI, CO, kernel_size, strides, padding, dilation=1):
    data, kernel, bias, bn_offset, bn_scale, out = conv2d_nhwc_bn_relu(N, H, W, CI, CO, kernel_size, strides, padding, dilation)
    return [out.op], [data, kernel, bias, bn_offset, bn_scale, out]

task_func_dict = {
    'GMM': batch_matmul_nkkm,
    'C1D': conv1d_nlc,
    'C2D': conv2d_nhwc,
    'C3D': conv3d_ndhwc,
    'GRP': conv2d_nhwc,
    'DIL': conv2d_nhwc,
    'DEP': depthwise_conv2d_nhwc,
    'T2D': conv2d_transpose_nhwc,
    'CAP': conv2d_capsule_nhwijc,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=int, default=-1)  # the no. of the task to tune, "-1" means all
    parser.add_argument('--wkl', type=str)
    parser.add_argument("--log", help="Log file name", type=str)
    parser.add_argument("--test", help="test file name", type=str, default="")
    parser.add_argument("--trials", help="number of trials for op", type=int, default=1000)
    parser.add_argument("--target", help="target device type", type=str, default="llvm")
    parser.add_argument("--device", help="target device number", type=int, default=0)
    parser.add_argument("--timeout", help="timeout", type=float, default=10.0)
    parser.add_argument("--parallel", help="parallel", type=int, default=1)
    parser.add_argument("--use_model", help="use performance model", action="store_true")
    parser.add_argument("--method", help="how to schedule", type=str, default="q")
    parser.add_argument("--slevel", type=int, default=4)
    parser.add_argument("--rlevel", type=int, default=3)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--target_host", type=str, default="llvm")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--no-force-inline", action='store_true')
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    log = args.log or 'flextensor_single_op.txt'

    rpc_info = RpcInfo(args.host, args.port, args.target_host)

    # register tasks
    all_task_list = []
    for wkl_meta_name, func in task_func_dict.items():
        if args.wkl is not None and wkl_meta_name != args.wkl:
            continue

        if args.test:
            batch_sizes = [1, 16]
        else:
            batch_sizes = [args.batch_size]

        for batch_size in batch_sizes:
            for shape in shape_dict[wkl_meta_name]:
                if shape[0] == 1:
                    shape = list(shape)
                    shape[0] = batch_size
                    shape = tuple(shape)

                task = Task(
                  wkl_meta_name,
                  wkl_meta_name,
                  func,
                  shape,
                  args.target,
                  args.device)
                register_task(task)
                all_task_list.append(task)

    if args.test != "":
        with open(args.test, "r") as fin:
            for line in fin:
                name, string = line.split(":", 1)
                obj = json.loads(string)
                configs = Config(obj[0], obj[1])
                test(name, configs, 'op', args.device, rpc_info=rpc_info)
    else:
        if args.i == -1:
            print("========== Get %d tasks to tune ==========" % len(all_task_list))
            for i in range(len(all_task_list)):
                cmd = "%s %s --trials %d --i %d --batch-size %d" % (sys.executable, sys.argv[0], args.trials, i, args.batch_size)
                if args.wkl is not None:
                    cmd += " --wkl %s" % args.wkl
                if args.no_force_inline:
                    cmd += " --no-force-inline"
                print(cmd)
                ret = os.system(cmd)
                if ret != 0:
                    exit(ret)
            exit()

        with open(log, "a") as flog:
            for i, task in enumerate(all_task_list):
                if args.i is not None and i != args.i:
                    continue

                optimize(
                    task,
                    slevel=args.slevel,
                    rlevel=args.rlevel,
                    target=args.target, 
                    dev_id=args.device, 
                    timeout=args.timeout, 
                    trials=args.trials, 
                    parallel=args.parallel,
                    use_model=args.use_model,
                    force_inline=not args.no_force_inline,
                    method=args.method,
                    logfile=flog,
                    rpc_info=rpc_info)

