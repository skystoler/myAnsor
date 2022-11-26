"""Tensorflow baseline"""
import argparse
import time
import timeit
from collections import namedtuple
import timeit
import os
import functools
import multiprocessing

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
#from tensorflow.python import _pywrap_util_port

from tf_models import resnet, resnet_3d, bert as bert_model
from tf_models.dcgan import dcgan as dcgan_model
from tf_models.mobilenet_v2 import mobilenet_v2 as mobilenet_v2_model

from utils import log_line, py_benchmark, BenchmarkRecord, shape_dict

USE_SESSION = False
USE_TENSOR_RT = False
USE_XLA = False

# ============ Op ============
def matmul(N, M, K):
   x = tf.random.uniform([N, K])
   y = tf.random.uniform([K, M])
   
   cost = py_benchmark("tf.matmul(x, y)", {**globals(), **locals()})
   flop = 2 * N * M * K
   return cost, flop

def batch_matmul(B, N, M, K):
    x = tf.random.uniform([B, N, K])
    y = tf.random.uniform([B, K, M])
    
    cost = py_benchmark("tf.matmul(x, y)", {**globals(), **locals()})
    flop = 2 * B * N * M * K
    return cost, flop

def conv1d(N, L, CI, CO, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    x = tf.random.uniform([N, CI, L])
    y = tf.random.uniform([kernel_size, CI // groups, CO])

    assert padding == kernel_size // 2
    assert groups == 1

    cost = py_benchmark("tf.nn.conv1d(x, y, stride, 'SAME', data_format='NCW')", {**globals(), **locals()})
    out_l = (L + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    flop = 2 * N * out_l * CO * CI * kernel_size
    return cost, flop

def conv2d(N, H, W, CI, CO, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    x = tf.random.uniform([N, CI, H, W])
    y = tf.random.uniform([kernel_size, kernel_size, CI // groups, CO])
    assert padding == kernel_size // 2
    assert groups == 1

    cost = py_benchmark("tf.nn.conv2d(x, y, stride, 'SAME', data_format='NCHW', dilations=%d)" % dilation,
                        {**globals(), **locals()})

    out_h = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    flop = 2 * N * out_h * out_w * CO * CI * kernel_size * kernel_size
    return cost, flop

def conv3d(N, D, H, W, CI, CO, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    x = tf.random.uniform([N, D, H, W, CI])
    y = tf.random.uniform([kernel_size, kernel_size, kernel_size, CI // groups, CO])

    assert padding == kernel_size // 2
    assert groups == 1

    strides = [1, stride, stride, stride, 1]
    dilations= [1, dilation, dilation, dilation, 1]

    cost = py_benchmark("tf.nn.conv3d(x, y, strides, 'SAME', data_format='NDHWC', dilations=%s)" % dilations,
                        {**globals(), **locals()})

    out_d = (D + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_h = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    flop = 2 * N * out_d * out_h * out_w * CO * CI * kernel_size * kernel_size * kernel_size
    return cost, flop

def group_conv2d(N, H, W, CI, CO, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    x = tf.random.uniform([N, CI, H, W])
    y = tf.random.uniform([kernel_size, kernel_size, CI // groups, CO])

    @tf.function(autograph=True, experimental_compile=False)
    def func(x, y):
        inputs = tf.split(axis=1, num_or_size_splits=groups, value=x)
        weights = tf.split(axis=3, num_or_size_splits=groups, value=y)
        return tf.concat(axis=1, values=[tf.nn.conv2d(a, b, stride, 'SAME', data_format='NCHW', dilations=dilation) for a, b in zip(inputs, weights)])

    assert padding == kernel_size // 2
    cost = py_benchmark("func(x, y)", {**globals(), **locals()})

    out_h = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    flop = 2 * N * out_h * out_w * CO * CI * kernel_size * kernel_size // groups
    return cost, flop

def norm(B, M, N):
    x = tf.random.uniform([B, M, N])

    if USE_SESSION:
        res = tf.norm(x, axis=[1, 2])
        sess.run(res)
        cost = py_benchmark("sess.run(res)", {**globals(), **locals()})
    else:
        cost = py_benchmark("tf.norm(x)", {**globals(), **locals()})
    flop = 2 * B * M * N
    return cost, flop

# ============ Subgraph ============
def conv2d_bn_relu(N, H, W, CI, CO, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    x = tf.random.uniform([N, CI, H, W])
    y = tf.random.uniform([kernel_size, kernel_size, CI // groups, CO])
    bias = tf.random.uniform([CO])
    bn_scale = tf.random.uniform([CO])
    bn_offset = tf.random.uniform([CO])
    assert padding == kernel_size // 2
    assert groups == 1

    if USE_SESSION:
        with tf.Graph().as_default():
            with tf.compat.v1.Session() as sess:
                res = tf.nn.relu(tf.add(
                            tf.multiply(
                                tf.add(
                                    tf.nn.conv2d(x, y, stride, 'SAME', data_format='NCHW', dilations=dilation),
                                    bias[None, :, None, None]),
                                bn_scale[None, :, None, None]),
                            bn_offset[None, :, None, None]))
                sess.run(res)
                cost = py_benchmark("sess.run(res)", {**globals(), **locals()})
    else:
        cost = py_benchmark("tf.nn.relu(tf.add(\
                                tf.multiply( \
                                    tf.add( \
                                        tf.nn.conv2d(x, y, stride, 'SAME', data_format='NCHW', dilations=%d), \
                                        bias[None, :, None, None]), \
                                    bn_scale[None, :, None, None]), \
                                bn_offset[None, :, None, None]))" % dilation,
                            {**globals(), **locals()})

    out_h = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    flop = 2 * N * out_h * out_w * CO * CI * kernel_size * kernel_size
    return cost, flop

def transpose_batch_matmul(B, N, M, K):
    x = tf.random.uniform([B, N, K])
    y = tf.random.uniform([B, K, M])

    if USE_SESSION:
        with tf.Graph().as_default():
            with tf.compat.v1.Session() as sess:
                res = tf.matmul(x, tf.transpose(y, [0, 2, 1]))
                sess.run(res)
                cost = py_benchmark("sess.run(res)", {**globals(), **locals()})
    else:
        cost = py_benchmark("tf.matmul(x, tf.transpose(y, [0, 2, 1]))", {**globals(), **locals()})
    flop = 2 * B * N * M * K
    return cost, flop

# ============ Network ============
def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func

def transfer_trt_graph(pb_graph_def, outputs, precision_mode, max_batch_size):
    trt_graph_def = trt.create_inference_graph(
        input_graph_def = pb_graph_def,
        outputs = outputs,
        max_batch_size = max_batch_size,
        max_workspace_size_bytes = 1 << 25,
        precision_mode = precision_mode,
        minimum_segment_size = 2,
        is_dynamic_op=True)
    return trt_graph_def

def evaluate_with_tensor_rt(network, N, net, output_node_name, input_shape, output_shape=None, input_dtype='float32'):
    trt_pb = './tf_models/%s-trt-B%d.pb' % (network, N)
    if os.path.exists(trt_pb):
        with tf.compat.v1.gfile.GFile(trt_pb, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
    else:
        with tf.Graph().as_default():
            with tf.compat.v1.Session() as sess:
                inputs = tf.compat.v1.placeholder(input_dtype, shape=input_shape, name="input")
                if network == 'bert':  # special branch for bert
                    model = net(input_ids=inputs, is_training=False)
                    output = model.get_pooled_output()
                else:
                    output = net(inputs, is_training=False)
                sess.run(tf.compat.v1.global_variables_initializer())
                graph_def = tf.compat.v1.get_default_graph().as_graph_def()
                graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, graph_def, [output_node_name])
                graph_def = transfer_trt_graph(graph_def, [output_node_name], 'FP32', N)
                with tf.compat.v1.gfile.GFile(trt_pb, 'wb') as f:
                    f.write(graph_def.SerializeToString())
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            if 'float' in input_dtype:
                input_data = tf.constant(np.random.randn(*input_shape).astype(input_dtype))
            else:
                input_data = tf.random.uniform(input_shape, dtype=tf.int32, maxval=30000)
            output_tensor, = tf.compat.v1.import_graph_def(graph_def,
                {"input": input_data,}, return_elements=[output_node_name + ":0"])
            res = sess.run(output_tensor)
            if output_shape:
                assert res.shape == output_shape
            cost = py_benchmark("sess.run(output_tensor)", {**globals(), **locals()})
    return cost

data_format = None

def dcgan(N):
    assert USE_SESSION
    image_shape = (N, 100)

    if USE_TENSOR_RT:
        net = wrapped_partial(dcgan_model, oshape=(64, 64, 3), batch_size=N, scope="dcgan")
        cost = evaluate_with_tensor_rt("dcgan", N, net, "dcgan/Tanh", image_shape, (N, 64, 64, 3))
    else:
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=args.intra_op_num_threads,
                                          inter_op_parallelism_threads=args.inter_op_num_threads)
        if USE_XLA:
            config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1

        with tf.Graph().as_default():
            with tf.compat.v1.Session(config=config) as sess:
                net = wrapped_partial(dcgan_model, oshape=(64, 64, 3), batch_size=N,
                                      scope="%d" % (int(np.random.randint(1 << 31))))
                inputs = tf.constant(np.random.randn(*image_shape).astype(np.float32))
                output = net(inputs, is_training=False)
                sess.run(tf.compat.v1.global_variables_initializer())
                cost = py_benchmark("sess.run(output)", {**globals(), **locals()})

    return cost

def resnet50(N):
    if USE_SESSION:
        image_shape = (N, 224, 224, 3)
        if USE_TENSOR_RT:
            net = resnet.imagenet_resnet_v2(resnet_size=50, num_classes=1000, data_format=data_format)
            cost = evaluate_with_tensor_rt("resnet50", N, net, "final_dense", image_shape, output_shape=(N, 1000))
        else:
            config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=args.intra_op_num_threads,
                                              inter_op_parallelism_threads=args.inter_op_num_threads)
            if USE_XLA:
                config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1

            with tf.Graph().as_default():
                with tf.compat.v1.Session(config=config) as sess:
                    net = resnet.imagenet_resnet_v2(resnet_size=50, num_classes=1000, data_format=data_format)
                    inputs = tf.constant(np.random.randn(*image_shape).astype(np.float32))
                    output = net(inputs, is_training=False)
                    sess.run(tf.compat.v1.global_variables_initializer())
                    cost = py_benchmark("sess.run(output)", {**globals(), **locals()})
    else:
        x = tf.random.uniform([N, 224, 224, 3])
        net = tf.keras.applications.ResNet50(weights=None)
        net.predict(x, steps=1)
        cost = py_benchmark("net.predict(x, steps=1)", {**globals(), **locals()})

    return cost

def resnet3d_18(N):
    assert USE_SESSION
    image_shape = (N, 16, 112, 112, 3)
    if USE_TENSOR_RT:
        net = resnet_3d.imagenet_resnet_v2(resnet_size=18, num_classes=1000, data_format=data_format)
        cost = evaluate_with_tensor_rt("resnet18", N, net, "final_dense", image_shape, output_shape=(N, 1000))
    else:
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=args.intra_op_num_threads,
                                          inter_op_parallelism_threads=args.inter_op_num_threads)

        if USE_XLA:
            config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1

        with tf.Graph().as_default():
            with tf.compat.v1.Session(config=config) as sess:
                net = resnet_3d.imagenet_resnet_v2(resnet_size=18, num_classes=1000, data_format=data_format)
                inputs = tf.constant(np.random.randn(*image_shape).astype(np.float32))
                output = net(inputs, is_training=False)
                sess.run(tf.compat.v1.global_variables_initializer())
                cost = py_benchmark("sess.run(output)", {**globals(), **locals()})

    return cost

def mobilenet_v2(N):
    if USE_SESSION:
        image_shape = (N, 224, 224, 3)
        if USE_TENSOR_RT:
            net = mobilenet_v2_model.mobilenet
            cost = evaluate_with_tensor_rt("mobilenet_v2", N, net, "MobilenetV2/Predictions/Reshape_1", image_shape,
                    output_shape=(N, 1001))
        else:
            config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=args.intra_op_num_threads,
                                              inter_op_parallelism_threads=args.inter_op_num_threads)
    
            if USE_XLA:
                config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
    
            with tf.Graph().as_default():
                with tf.compat.v1.Session(config=config) as sess:
                    net = mobilenet_v2_model.mobilenet
                    inputs = tf.constant(np.random.randn(*image_shape).astype(np.float32))
                    output = net(inputs)
                    sess.run(tf.compat.v1.global_variables_initializer())
                    cost = py_benchmark("sess.run(output)", {**globals(), **locals()})
    else:
        x = tf.random.uniform([N, 224, 224, 3])
        net = tf.keras.applications.MobileNetV2(weights=None)
        net.predict(x, steps=1)
        cost = py_benchmark("net.predict(x, steps=1)", {**globals(), **locals()})

    return cost

def bert(N):
    bert_config = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 2,
        "vocab_size": 30522
    }
    input_shape = (N, 128)

    if USE_SESSION:
        if USE_TENSOR_RT:
            net = wrapped_partial(bert_model.BertModel, bert_model.BertConfig(**bert_config))
            cost = evaluate_with_tensor_rt("bert", N, net, "bert/pooler/dense/Tanh", input_shape, output_shape=(N, 768),
                    input_dtype='int32')
        else:
            config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=args.intra_op_num_threads,
                                              inter_op_parallelism_threads=args.inter_op_num_threads)
    
            if USE_XLA:
                config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
    
            with tf.Graph().as_default():
                with tf.compat.v1.Session(config=config) as sess:
                    inputs = tf.constant(np.random.uniform(0, 30000, size=input_shape).astype(np.int32))
                    model = bert_model.BertModel(config=bert_model.BertConfig(**bert_config),
                                                 is_training=False, input_ids=inputs)
                    output = model.get_pooled_output()
                    sess.run(tf.compat.v1.global_variables_initializer())
                    cost = py_benchmark("sess.run(output)", {**globals(), **locals()})
    else:
        raise NotImplemented

    return cost

Workload = namedtuple("Workload", ['workload_type', 'workload_name', 'func'])

wkl_list = [
    Workload("op", "GMM", batch_matmul),
    Workload("op", "C1D", conv1d),
    Workload("op", "C2D", conv2d),
    Workload("op", "C3D", conv3d),
    #Workload("op", "GRP", group_conv2d),
    #Workload("op", "NRM", norm),
    #Workload("op", "SMX", softmax),
    Workload("subgraph", "conv2d_bn_relu", conv2d_bn_relu),
    #Workload("subgraph", "transpose_batch_matmul", transpose_batch_matmul),
    #Workload("network", "resnet_50", resnet50),
    #Workload("network", "resnet3d_18", resnet3d_18),
    #Workload("network", "dcgan", dcgan),
    #Workload("network", "mobilenet_v2", mobilenet_v2),
    #Workload("network", "bert", bert),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument("--device", type=str, default='device')
    parser.add_argument("--wkl", type=str)
    parser.add_argument("--out-file", type=str, default='results.tsv')
    parser.add_argument("--batch-size", type=int, default=-1)  # -1 means test both 1 and 16
    parser.add_argument("--session", type=str, default='False')
    parser.add_argument("--tensorrt", type=str, default='False')
    parser.add_argument("--xla", type=str, default='False')
    parser.add_argument("--inter-op-num-threads", type=int, default=1)
    parser.add_argument("--intra-op-num-threads", type=int, default=
            int(os.environ.get('TVM_NUM_THREADS', multiprocessing.cpu_count() // 2)))
    args = parser.parse_args()

    if args.tensorrt == 'True':
        assert args.backend == 'gpu'
        USE_TENSOR_RT = True
        try:
            from tensorflow.python.compiler.tensorrt import trt_convert as trt
        except:
            raise ImportError("Need to install TensorRT First.")
    elif args.xla == 'True':
        USE_XLA = True

    if args.session == 'True' or USE_TENSOR_RT or USE_XLA:
        USE_SESSION = True
        disable_eager_execution()

    if args.backend == 'cpu':
        device = '/CPU:0'
        data_format = 'channels_last'
        try:
            if _pywrap_util_port.IsMklEnabled():
                print("MKL-DNN is enabled in tensorflow")
                algorithm = 'mkldnn'
            else:
                print("WARNING: MKL-DNN is not enabled in tensorflow!!")
                algorithm = 'default'
        except:
            # Not able to check mkldnn
            algorithm = 'default'
    else:
        device = '/GPU:0'
        data_format = 'channels_first'
        algorithm = 'cudnn'

    if args.batch_size > 0:
        batch_size_list = [args.batch_size]
    else:
        batch_size_list = [1, 16]

    # Benchmark all workloads
    for wkl in wkl_list:
        for batch_size in batch_size_list:
            if args.wkl is not None and wkl.workload_name != args.wkl:
                continue

            if wkl.workload_type == 'op' or wkl.workload_type == 'subgraph':
                shape_configs = shape_dict[wkl.workload_name]

                for shape in shape_configs:
                    if shape[0] == 1:
                        shape = list(shape)
                        shape[0] = batch_size

                    with tf.device(device):
                       cost, flop = wkl.func(*shape)

                    workload_name = "%s%s" % (wkl.workload_name, tuple(shape))
                    print("%s\t%.3f ms\t%.2f GFLOPS" % (workload_name, cost * 1e3, (flop / 1e9) / cost))
                    log_line(BenchmarkRecord(args.device, args.backend, wkl.workload_type, workload_name,
                                             "tensorflow" if USE_SESSION else "tensorflow-eager", algorithm,
                                             {"costs": [cost]}, time.time()), args.out_file)
            elif wkl.workload_type == 'network':
                with tf.device(device):
                    cost = wkl.func(batch_size)
                workload_name = "%s.B%d" % (wkl.workload_name, batch_size)

                print("%s\t%.3f ms" % (workload_name, cost * 1e3))
                library = "tensorflow-tensorrt" if USE_TENSOR_RT else \
                          "tensorflow-xla" if USE_XLA else \
                          "tensorflow" if USE_SESSION else \
                          "tensorflow-eager"
                log_line(BenchmarkRecord(args.device, args.backend, wkl.workload_type, workload_name,
                                         library, algorithm,
                                         {"costs": [cost]}, time.time()), args.out_file)
            else:
                raise ValueError("Invalid worklaod type: " + wkl.workload_type)

