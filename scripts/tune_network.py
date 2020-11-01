""""Run auto-scheduler for a whole neural network"""
import argparse
import logging
import time
import random
import os

import numpy as np

import tvm
from tvm import ansor, relay
import tvm.contrib.graph_runtime as runtime
from tvm.contrib.debugger import debug_runtime
from tvm.contrib import util, ndk
from tvm.relay import testing
from tvm.ansor.utils import request_remote
from tvm.contrib.download import download_testdata

from common import str2bool, extract_tar, log_line, BenchmarkRecord, LogFileDatabase
from tune_test import create_tune_option
from bert_optimization import optimize_bert

def get_network(name, network_path, batch_size, layout):
    """Get the relay module and random weights for a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)
    input_name = 'data'
    input_dtype = dtype = 'float32'

    if name.startswith("resnet3d"):
        n_layer = int(name.split('-')[1])
        layout = "NDHWC"
        image_shape = (16, 112, 112, 3)
        input_shape = (batch_size, *image_shape)
        mod, params = relay.testing.resnet_3d.get_workload(num_layers=n_layer, batch_size=batch_size, image_shape=image_shape, dtype=dtype, layout=layout)
    elif name.startswith("resnet"):
        n_layer = int(name.split('-')[1])
        image_shape = (224, 224, 3) if layout == 'NHWC' else (3, 224, 224)
        input_shape = (batch_size, *image_shape)
        mod, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, layout=layout, image_shape=image_shape, dtype=dtype)
    elif "lstm" in name:
        mod, params = relay.testing.lstm.get_workload(iterations=10, num_hidden=512, batch_size=batch_size, dtype=dtype)
    elif "mlp" in name:
        input_shape = (batch_size, 1, 28, 28)
        mod, params = relay.testing.mlp.get_workload(batch_size=batch_size, dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif name == 'dcgan':
        input_shape = (batch_size, 100)
        mod, params = relay.testing.dcgan.get_workload(batch_size=batch_size, layout=layout)
    elif name == 'dqn':
        image_shape = (84, 84, 4) if layout == 'NHWC' else (4, 84, 84)
        input_shape = (batch_size, *image_shape)
        mod, params = relay.testing.dqn.get_workload(batch_size=batch_size, image_shape=image_shape, dtype=dtype, layout=layout)
    elif name == 'mobilenet':
        image_shape = (224, 224, 3) if layout == 'NHWC' else (3, 224, 224)
        input_shape = (batch_size, *image_shape)
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, layout=layout, image_shape=image_shape, dtype=dtype)
    elif name == 'squeezenet_v1.1':
        mod, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1', dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (batch_size, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'mxnet':
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model
        block = get_model('resnet18_v1', pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"input_name": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
        mod = relay.Module.from_expr(net)
    elif name == 'mobilenet-v2':
        import tflite.Model

        if network_path:
            tflite_model_file = network_path
        else:
            model_url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz"
            model_path = download_testdata(model_url, "mobilenet_v2_1.0_224.tgz", module=['tf', 'official'])
            model_dir = os.path.dirname(model_path)
            extract_tar(model_path)
            tflite_model_file = os.path.join(model_dir, "mobilenet_v2_1.0_224.tflite")

        tflite_model_buf = open(tflite_model_file, "rb").read()

        # Get TFLite model from buffer
        try:
            import tflite
            tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
        except AttributeError:
            import tflite.Model
            tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

        input_name = "input"
        input_shape = (batch_size, 224, 224, 3)
        output_shape = (batch_size, 1001)
        input_dtype = "float32"
        mod, params = relay.frontend.from_tflite(tflite_model,
                                                 shape_dict={input_name: input_shape},
                                                 dtype_dict={input_name: input_dtype})
    elif name == 'tflite-mobilenet-v2-int8':
        try:
            import tflite.Model
        except ImportError:
            raise ImportError("The tflite package must be installed")

        if network_path:
            tflite_model_file = network_path
        else:
            model_url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz"
            model_path = download_testdata(model_url, "mobilenet_v2_1.0_224_quant.tgz", module=['tf', 'official'])
            model_dir = os.path.dirname(model_path)
            extract_tar(model_path)
            tflite_model_file = os.path.join(model_dir, "mobilenet_v2_1.0_224_quant.tflite")

        input_name = "input"
        input_shape = (1, 224, 224, 3)
        output_shape = (1, 1001)
        input_dtype = "uint8"
        tflite_model_buf = open(tflite_model_file, "rb").read()
        
        # Get TFLite model from buffer
        try:
            import tflite
            tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
        except AttributeError:
            import tflite.Model
            tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

        mod, params = relay.frontend.from_tflite(tflite_model,
                                                 shape_dict={input_name: input_shape},
                                                 dtype_dict={input_name: input_dtype})
    elif name == 'tflite-resnet-101':
        import tflite.Model

        if network_path:
            tflite_model_file = network_path
        else:
            model_url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/resnet_v2_101.tgz"
            model_path = download_testdata(model_url, "resnet_v2_101.tgz", module=['tf', 'official'])
            model_dir = os.path.dirname(model_path)
            extract_tar(model_path)
            tflite_model_file = os.path.join(model_dir, "resnet_v2_101_299.tflite")

        tflite_model_buf = open(tflite_model_file, "rb").read()

        # Get TFLite model from buffer
        try:
            import tflite
            tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
        except AttributeError:
            import tflite.Model
            tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

        input_name = "input"
        input_shape = (batch_size, 299, 299, 3)
        output_shape = (batch_size, 1000)
        input_dtype = "float32"
        mod, params = relay.frontend.from_tflite(tflite_model,
                                                 shape_dict={input_name: input_shape},
                                                 dtype_dict={input_name: input_dtype})
    elif name == 'pytorch-mobilenet-v2':
        import torch

        model = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=False)
        model.eval()

        input_shape = [batch_size, 3, 224, 224]
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()

        input_name = 'input0'
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model,
                                                  shape_list)
    elif name == 'tf-bert':
        import tensorflow as tf

        bert_pb = '../../tvm-autoscheduler-clean2/scripts/baseline/tensorflow/tf_models/bert/bert-B%d.pb' % batch_size
        try:
            with tf.compat.v1.gfile.GFile(bert_pb, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
        except:
            raise ValueError("Need to run ./baseline/tensorflow/bert/generate_bert_pb.py to get model first")

        input_shape = (batch_size, 128)
        input_name = ['input']
        shape_dict = {
            'input': input_shape
        }
        out_names = [
            'bert/pooler/dense/Tanh'
        ]

        mod, params = relay.frontend.from_tensorflow(graph_def,
                                                    shape=shape_dict,
                                                    outputs=out_names)
    elif name == 'bert':
        import torch
        import transformers

        model_class = transformers.BertModel
        tokenizer_class = transformers.BertTokenizer

        # Better to download them manualy
        #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin
        #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
        #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json
        # Then rename to pytorch_model.bin, vocab.txt & config.json
        # weight = 'path to downloaded model dir'
        weight = 'bert-base-uncased'
        model = model_class.from_pretrained(weight)
        model.eval()

        # tokenizer = tokenizer_class.from_pretrained(weight)
        # A = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])
        # There is 30522 words in bert-base-uncased's vocabulary list
        input_shape = [batch_size, 128]
        input_name = 'input_ids'
        input_dtype = 'int64'
        A = torch.randint(30000, input_shape)
        scripted_model = torch.jit.trace(model, [A])
        shape_list = [('input_ids', input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        mod = optimize_bert(mod, params)
    elif name == 'pytorch-debug':
        import torch

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv2d = torch.nn.Conv2d(512, 512, 3, 1, 1)
            def forward(self, x):
                x = self.conv2d(x)
                return x

        net = Net()
        net.eval()

        input_shape = [batch_size, 512, 7, 7]
        data = torch.rand(*input_shape)
        scripted_model = torch.jit.trace(net, [data])
        shape_list = [('data', input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    elif name == 'debug':
        input_shape = [batch_size, 7, 7, 512]
        output_shape = input_shape

        data = relay.var("data", shape=input_shape, dtype=input_dtype)
        net = relay.testing.layers.conv2d(
            data=data,
            channels=512,
            kernel_size=3,
            strides=1,
            padding=1,
            data_layout="NHWC",
            kernel_layout="HWIO",
            name='')
        bias = relay.var("conv1_bias")
        net = relay.nn.bias_add(net, bias, 3)
        net = relay.nn.relu(net)
        mod, params = relay.testing.create_workload(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_name, input_shape, input_dtype, output_shape

def create_module(data_shape, data_dtype, graph, lib, target, input_name, params, debug_profile,
        local_measure, ndk_cc, rpc_device_key, rpc_host, rpc_port, rpc_num_threads, seed=43):
    if local_measure:
        ctx = tvm.context(str(target))
    else:
        print("=============== Request Remote ===============")
        remote = request_remote(rpc_device_key, rpc_host, rpc_port)

        print("=============== Export ===============")
        ctx = remote.cpu()
        temp = util.tempdir()
        if ndk_cc:
            os.environ['TVM_NDK_CC'] = ndk_cc
            filename = 'deploy_lib.so'
            path_lib = temp.relpath(filename)
            lib.export_library(path_lib, ndk.create_shared)
        else:
            filename = 'deploy_lib.tar'
            path_lib = temp.relpath(filename)
            lib.export_library(path_lib)

        print("=============== Upload ===============")
        remote.upload(path_lib)

        print("=============== Load ===============")
        lib = remote.load_module(filename)

        if rpc_num_threads:
            config_threadpool = remote.get_function('runtime.config_threadpool')
            config_threadpool(0, rpc_num_threads)

    np.random.seed(seed)
    np_data = 100 * (np.random.uniform(size=data_shape)).astype(data_dtype)
    data_tvm = tvm.nd.array(np_data, ctx=ctx)
    if debug_profile:
        module = debug_runtime.create(graph, lib, ctx)
    else:
        module = runtime.create(graph, lib, ctx)

    if type(input_name) == list:
        for name in input_name:
            module.set_input(name, data_tvm)
    else:
        module.set_input(input_name, data_tvm)
    for k, v in params.items():
        module.set_input(k, v)

    return module, ctx, np_data

def get_tflite_output(data, model_path):
    from tensorflow import lite as interpreter_wrapper
    interpreter = interpreter_wrapper.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # set input
    interpreter.set_tensor(input_details[0]['index'], data)

    # Run
    interpreter.invoke()

    # get output
    tflite_output = interpreter.get_tensor(output_details[0]['index'])

    return tflite_output

def estimate_by_log(log_file, log_n_lines, objective_func, tasks, target):
    database = LogFileDatabase(log_file, log_n_lines or -1)
    best_costs = []
    for i, task in enumerate(tasks):
        wkl_key = task.workload_key
        query_key = (target.keys[0], wkl_key)
        if query_key in database.best_by_targetkey:
            inp, res = database.best_by_targetkey[query_key]
            cost = np.mean([x.value for x in res.costs])
        else:
            print("Missing log for wkl_key: %s wkl_weight: %d" % (wkl_key, wkl_weights[i]))
            cost = 0
        best_costs.append(cost)
    print("Estimated end-to-end cost: %.2f ms" % (1000 * objective_func(best_costs)))

def tune_and_evaluate(network_arguments, target, target_host,
                      search_policy, task_scheduler_arguments, tune_option_arguments,
                      tune, estimate, debug_profile, check_correctness, log_n_lines, compare_with_tflite):
    # Extract tasks from relay program
    mod, params, input_name, data_shape, data_dtype, out_shape = get_network(**network_arguments)

    # Tune all
    if tune or estimate:
        print("=============== Extract Workloads ===============")
        workloads, wkl_weights = ansor.extract_from_program(mod, target=target, params=params)
        print("Extract %d workloads in total" % (len(workloads)))

        # create tasks
        tasks = []
        for i, wkl_key in enumerate(workloads):
            dag = ansor.workload_key_to_dag(wkl_key)
            tasks.append(ansor.SearchTask(dag, wkl_key, target, target_host))
            print("---------- Task %d ---------- (key: %s) \n" % (i, wkl_key), dag)

        # define objective function (end-to-end latency) for the task scheduler
        def objective_func(costs):
            return sum(c * w for c, w in zip(costs, wkl_weights))

        if estimate:  # estimate the objective function with history best records
            estimate_by_log(tune_option_arguments['log_file'], log_n_lines,
                            objective_func, tasks, target)
            exit()

        # Tune workloads with auto scheduler
        print("=============== Tune ===============")
        tuner = ansor.SimpleTaskScheduler(tasks, objective_func,
                                          **task_scheduler_arguments)
        tune_option, measure_ctx = create_tune_option(target, **tune_option_arguments)

        if tune_option_arguments['local_measure'] and 'cpu' in target.keys:
            os.environ['TVM_BIND_MASTER_CORE_0'] = "1"
        tuner.tune(tune_option, search_policy)

        if measure_ctx:
            del measure_ctx

    # Compile graph with best states found by auto-scheduler
    print("=============== Compile ===============")
    with ansor.apply_history_best(tune_option_arguments['log_file'], log_n_lines):
        os.environ['TVM_AUTO_CACHE_FLUSH'] = "0"

        with tvm.transform.PassContext(opt_level=3):
            graph, lib, opt_params = relay.build_module.build(
                mod, target=target, params=params)

        print("=============== Compile Finish ===============")

        module, ctx, data = create_module(data_shape, data_dtype, graph, lib, target, input_name,
                                          opt_params, debug_profile, **common_measure_parameters)

        # Evaluate
        print("========== Evaluate ==========")
        ftimer = module.module.time_evaluator("run", ctx, number=10, repeat=3, min_repeat_ms=1000)
        prof_res = np.array(ftimer().results)

        # display profile information
        if debug_profile or check_correctness or compare_with_tflite:
            module.run()
            if check_correctness or compare_with_tflite:
                actual_output = module.get_output(0).asnumpy()
            if compare_with_tflite:
                print("========== Compare with tflite ==========")
                assert network_arguments['network_path'], "tflite model path shouldn't be none"
                tflite_output = get_tflite_output(data, network_arguments['network_path'])
                np.testing.assert_allclose(actual_output, tflite_output, atol=1e-2, rtol=1e-2)

        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res) * 1000, np.std(prof_res) * 1000))
        log_line(BenchmarkRecord(target.target_name, 'gpu' if 'gpu' in target.keys else 'cpu', 'network',
                                 "%s.B%d" % (network_arguments['name'], network_arguments['batch_size']),
                                 'ours', network_arguments['layout'], {"costs": prof_res}, time.time()), 'results.tsv')

    if check_correctness:
        print("========== Check Correctness ==========")
        # clean relay cache
        relay.backend.compile_engine.get().clear()

        # disable layout rewrite
        target = tvm.target.create('llvm')
        with tvm.transform.PassContext(opt_level=3, disabled_pass={"KernelLayoutTransform"}):
            graph, lib, opt_params = relay.build_module.build(
                mod, target=target, params=params)

        module, _, _ = create_module(data_shape, data_dtype, graph, lib, target, input_name,
                                     opt_params, debug_profile, **common_measure_parameters)
        module.run()

        expected_output = module.get_output(0).asnumpy()
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Search task related arguments
    parser.add_argument("--network", type=str, required=True)
    parser.add_argument("--network-path", type=str, default=None, help="The path of tflite model")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--layout", type=str, default='NHWC')
    parser.add_argument("--target", type=str, default='llvm -mcpu=core-avx2')
    parser.add_argument("--target-host", type=str, default=None)
    parser.add_argument("--check-correctness", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--debug-profile", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--tune", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--estimate", type=str2bool, nargs='?', const=True, default=False,
            help="Estiamte the end-to-end execution time cost with the costs recroded in the log file")

    # Search strategy related arguments
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--policy", type=str, choices=['sketch', 'limit-space'], default='sketch')
    parser.add_argument("--cost-model", type=str, choices=['xgb', 'random', 'no-share'], default='xgb')
    parser.add_argument("--task-scheduler", type=str, default='gradient',
                        choices=['no', 'gradient', 'round-robin'],
                        help='The strategy of task scheduler')
    parser.add_argument("--seed", type=int, default=0, help='random seed')

    # Log file related arguments
    parser.add_argument("--log-file", type=str, help="Write measurement records to this log file")
    parser.add_argument("--load-log", type=str, help="Load history log to resume the status of search")
    parser.add_argument("--log-n-lines", type=int, help="Only load the first n lines for history log")
    parser.add_argument("--load-model", type=str, help="Load pre trained cost model file")

    # Measurement related and other arguments
    parser.add_argument("--num-measure-per-iter", type=int, default=64,
                        help="The number of programs to be measured at each iteration")
    parser.add_argument("--build-timeout", type=int, default=10)
    parser.add_argument("--run-timeout", type=int, default=25)
    parser.add_argument("--early-stopping", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--rpc-device-key", type=str, default=None)
    # Setting `--rpc-device-key` to None means using local devices for measurement
    parser.add_argument("--rpc-host", type=str, default='0.0.0.0')
    parser.add_argument("--rpc-port", type=int, default=9190)
    parser.add_argument("--rpc-num-threads", type=int, default=None)
    parser.add_argument("--n-parallel", type=int, default=1)
    parser.add_argument("--ndk-cc", type=str, default=None)
    parser.add_argument("--compare-with-tflite", type=str2bool, nargs='?', const=True, default=False)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    logging.basicConfig()
    logging.getLogger('ansor').setLevel(logging.DEBUG)
    os.environ["TOPHUB_LOCATION"] = "NONE"  # disable autotvm
    # TODO (FrozenGene):
    # For rasp, we don't have dot product instruction and want to
    # generate SMLAL instruction to get better performance, so we
    # want to use helper_no_fast_int8_hw_legalization in qnn legalization.
    # To avoid conflict with tensorize used in AutoTVM, here we use one
    # environment value to keep both working.
    # we should think of one better way to handle this.
    os.environ["TVM_ARM_NO_FAST_INT8"] = "True" # mark arm has no fast int8

    target = tvm.target.create(args.target)
    log_file = args.log_file or "%s-B%d-%s.json" % (args.network, args.batch_size,
                                                    target.target_name)
    load_log_file = args.load_log or log_file
    local_measure = args.rpc_device_key is None
    search_policy = "%s.%s" % (args.policy, args.cost_model)

    network_arguments = {
        'name': args.network,
        'network_path': args.network_path,
        'batch_size': args.batch_size,
        'layout': args.layout
    }

    task_scheduler_parameters = {
        'strategy': args.task_scheduler,
        'load_log_file': load_log_file,
        'load_model_file': args.load_model,
        'verbose': args.verbose,
    }

    common_measure_parameters = {
        'local_measure': local_measure,
        'rpc_device_key': args.rpc_device_key,
        'rpc_host': args.rpc_host,
        'rpc_port': args.rpc_port,
        'rpc_num_threads': args.rpc_num_threads,
        'ndk_cc': args.ndk_cc,
    }

    tune_option_arguments = {
        'log_file': log_file,
        'n_trials': args.n_trials,
        'num_measure_per_iter': args.num_measure_per_iter,
        'verbose': args.verbose,
        'n_parallel': args.n_parallel,
        'build_timeout': args.build_timeout,
        'run_timeout': args.run_timeout,
        'early_stopping': args.early_stopping,
        'pre_search_callbacks': [ansor.PreloadMeasuredStates(log_file)],
        **common_measure_parameters
    }

    tune_and_evaluate(network_arguments, target, args.target_host,
                      search_policy, task_scheduler_parameters, tune_option_arguments,
                      args.tune, args.estimate, args.debug_profile, args.check_correctness,
                      args.log_n_lines, args.compare_with_tflite)
