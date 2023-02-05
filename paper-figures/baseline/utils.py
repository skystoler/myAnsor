import sys
import os
from collections import namedtuple

import numpy as np


def get_network(name, batch_size):
    import tvm
    from tvm import relay
    from tvm.relay import testing

    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)
    input_dtype = dtype = 'float32'
    input_name = 'data'

    if "resnet3d" in name:
        from resnet3d import get_workload
        n_layer = int(name.split('-')[1])
        layout = "NCDHW"
        image_shape = (3, 16, 112, 112)
        input_shape = (batch_size, *image_shape)
        mod, params = get_workload(num_layers=n_layer, batch_size=batch_size, image_shape=image_shape, dtype=dtype, layout=layout)
    elif "resnet" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif name == 'dcgan':
        input_shape = (batch_size, 100)
        mod, params = relay.testing.dcgan.get_workload(batch_size=batch_size)
    elif name == 'mobilenet':
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'mobilenet_v2':
        import torch
        import torchvision

        model = getattr(torchvision.models, name)(pretrained=True)
        model = model.eval()

        # We grab the TorchScripted model via tracing
        input_shape = [batch_size, 3, 224, 224]
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()

        input_name = 'input0'  # only one input, set it to this name
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    elif name == 'squeezenet_v1.1':
        mod, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1', dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
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
    elif name == 'mxnet':
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model
        block = get_model('resnet18_v1', pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={input_name: input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
        mod = tvm.IRModule.from_expr(net)
    elif name == 'winograd_test':
        from tvm.relay.testing import layers, init
        N, CI, H, W, CO, kernel_size, stride, padding = 1, 256, 14, 14, 256, 3, 1, 1
        input_shape = [batch_size, CI, H, W]
        dtype = 'float32'

        def get_testnet():
            data = relay.var("data", shape=input_shape, dtype=dtype)
            conv1 = layers.conv2d(data, channels=CO, kernel_size=(kernel_size, kernel_size),
                                  strides=(stride, stride), padding=(padding, padding), name='')
            args = relay.analysis.free_vars(conv1)
            func = relay.Function(args, conv1)
            return init.create_workload(func)
        mod, params = get_testnet()
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_name, input_shape, input_dtype, output_shape

def optimize_bert(mod, params):
    import tvm

    class ShapeConstDedupMutator(tvm.relay.ExprMutator):
        def __init__(self):
            super().__init__()
            self.shape_consts = {}
    
        def visit_call(self, call):
            if (isinstance(call.op, tvm.ir.Op) and call.op.name == "reshape"
                and len(call.args) > 1 and isinstance(call.args[1], tvm.relay.Constant)):
                assert list(call.attrs.newshape) == list(call.args[1].data.asnumpy())
                new_fn = self.visit(call.op)
                new_args = [self.visit(arg) for arg in call.args]
                const = new_args[1]
                assert const.data.dtype.startswith('int') and len(const.data.shape)==1
                key = tuple(const.data.asnumpy())
                if key in self.shape_consts:
                    new_args[1] = self.shape_consts[key]
                else:
                    self.shape_consts[key] = new_args[1]
                return tvm.relay.Call(new_fn, new_args, call.attrs)
            return super().visit_call(call)
    
    @tvm.relay.transform.function_pass(opt_level=1)
    def ShapeConstDedup(fn, mod, ctx):
        return ShapeConstDedupMutator().visit(fn)

    new_mod = ShapeConstDedup(mod)
    new_mod = tvm.relay.transform.EliminateCommonSubexpr()(new_mod)
    BindPass = tvm.relay.transform.function_pass(lambda fn, new_mod, ctx:
            tvm.relay.build_module.bind_params_by_name(fn, params), opt_level=1)
    new_mod = BindPass(new_mod)
    new_mod = tvm.relay.transform.FoldConstant()(new_mod)
    new_mod = tvm.relay.transform.CombineParallelBatchMatmul()(new_mod)
    new_mod = tvm.relay.transform.FoldConstant()(new_mod)
    new_mod = tvm.relay.transform.SimplifyInference()(new_mod) # remove dropout
    return new_mod

# The format for a line in resulst file
BenchmarkRecord = namedtuple("BenchmarkRecord",
                             ['device', 'backend', 'workload_type', 'workload_name',
                              'library', 'algorithm', 'value', 'time_stamp'])

def run_cmd(cmd):
    print(cmd)
    ret_code = os.system(cmd)
    if ret_code != 0:
        exit(ret_code)

def to_str_round(x, decimal=6):
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)) or isinstance(x, np.ndarray):
        return "[" + ", ".join([to_str_round(y, decimal=decimal) for y in x]) + "]"
    if isinstance(x, dict):
        return str({k: eval(to_str_round(v)) for k, v in x.items()})
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        format_str = "%%.%df" % decimal
        return format_str % x
    raise ValueError("Invalid value: " + str(x))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def log_line(record, out_file):
    with open(out_file, 'a') as fout:
        fout.write("\t".join([to_str_round(x) for x in record]) + '\n')

def py_benchmark(stmt, context, min_repeat_second=1, setup='pass', finish='pass'):
    total_time = 0
    number = 10

    eval(stmt, context) # warmup
    total_time = timeit(stmt=stmt, setup=setup, finish=finish, number=number, globals=context)
    while total_time < min_repeat_second:
        number = int(number * (min_repeat_second / total_time)) + 1
        total_time = timeit(stmt=stmt, setup=setup, finish=finish, number=number, globals=context)

    return total_time / number

def measure_schedule(s,
                     bufs,
                     target,
                     target_host=None,
                     remote=None,
                     ndk_cc=None,
                     number=10,
                     repeat=3,
                     min_repeat_ms=500):
    """Measure the time cost of a schedule"""
    import tvm, topi
    func = tvm.build(s, bufs, target=target, target_host=target_host)
    if remote:
        ctx = remote.context(str(target), 0)
        temp = util.tempdir()
        if ndk_cc:
            os.environ['TVM_NDK_CC'] = ndk_cc
            filename = "tmp_deploy_lib.so"
            remote_path = temp.relpath(filename)
            func.export_library(remote_path, ndk.create_shared)
        else:
            filename = "tmp_deploy_lib.tar"
            remote_path = temp.relpath(filename)
            func.export_library(remote_path)
        remote.upload(remote_path)
        func = remote.load_module(filename)
    else:
        ctx = tvm.context(str(target), 0)

    if os.environ.get('TVM_AUTO_CACHE_FLUSH', '0') == '1':
        min_repeat_ms = 0
        number = 1

    time_f = func.time_evaluator(func.entry_name,
                                 ctx,
                                 number=number,
                                 repeat=repeat,
                                 min_repeat_ms=min_repeat_ms)

    np_args = [np.ones(topi.get_const_tuple(x.shape)).astype(x.dtype) for x in bufs]
    args = [tvm.nd.array(x, ctx=ctx) for x in np_args]
    ctx.sync()

    costs = time_f(*args).results

    return costs

# Import single operator evaluation workloads
sys.path.append('../..')
sys.path.append('..')
try:
    from timeit_v2 import timeit
except ModuleNotFoundError:
    from .timeit_v2 import timeit
from shape_configs import shape_dict
sys.path.pop()
sys.path.pop()

