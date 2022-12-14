import argparse
import time
from collections import namedtuple
import platform
import os

import numpy as np
import torch
import torchvision

import torchvision.models as models
from utils import log_line, py_benchmark, BenchmarkRecord, shape_dict

USE_TORCH_SCRIPT = False
USE_CUDA = False

# ============ Op ============
def matmul(N, M, K):
    class MyModule(torch.nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()

        def forward(self, A, B, C):
            torch.matmul(A, B, out=C)

    my_module = MyModule()
    my_module.eval()

    A = torch.rand(N, K)
    B = torch.rand(K, M)
    C = torch.rand(N, M)

    if USE_CUDA:
        my_module.cuda(); A = A.cuda(); B = B.cuda(); C = C.cuda()

    if USE_TORCH_SCRIPT:
        func = torch.jit.script(my_module)
        stmt = 'func(A, B, C)'
    else:
        stmt = 'my_module(A, B, C)'

    with torch.no_grad():
        t = py_benchmark(stmt, {**globals(), **locals()},
                         setup="torch.cuda.synchronize()" if USE_CUDA else "pass",
                         finish="torch.cuda.synchronize()" if USE_CUDA else "pass")

    return t, 2 * N * K * M

def batch_matmul(B, N, M, K):
    class MyModule(torch.nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()

        def forward(self, A, B, C):
            torch.matmul(A, B, out=C)

    my_module = MyModule()
    my_module.eval()

    x = torch.rand(B, N, K)
    y = torch.rand(B, K, M)
    z = torch.rand(B, N, M)

    if USE_CUDA:
        my_module.cuda(); x = x.cuda(); y = y.cuda(); z = z.cuda()

    if USE_TORCH_SCRIPT:
        func = torch.jit.script(my_module)
        stmt = 'func(x, y, z)'
    else:
        stmt = 'my_module(x, y, z)'

    with torch.no_grad():
        t = py_benchmark(stmt, {**globals(), **locals()},
                         setup="torch.cuda.synchronize()" if USE_CUDA else "pass",
                         finish="torch.cuda.synchronize()" if USE_CUDA else "pass")

    return t, 2 * B * N * K * M

def conv1d(N, H, CI, CO, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    class MyModule(torch.nn.Module):
        __constants__ = ['sh', 'ph', 'dh']

        def __init__(self):
            super(MyModule, self).__init__()
            self.sh = stride
            self.ph = padding
            self.dh = dilation

        def forward(self, A, B):
            return torch.nn.functional.conv1d(A, B,
                                             stride=self.sh,
                                             padding=self.ph,
                                             dilation=self.dh)

    OH = (H + 2 * padding - (kernel_size - 1) * dilation - 1) // stride + 1

    my_module = MyModule()
    my_module.eval()

    A = torch.rand(N, CI, H)
    B = torch.rand(CO, CI, kernel_size)

    if USE_CUDA:
        my_module.cuda(); A = A.cuda(); B = B.cuda();

    if USE_TORCH_SCRIPT:
        func = torch.jit.script(my_module)
        stmt = "func(A, B)"
    else:
        stmt = "my_module(A, B)"
    with torch.no_grad():
        t = py_benchmark(stmt, {**globals(), **locals()},
                         setup="torch.cuda.synchronize()" if USE_CUDA else "pass",
                         finish="torch.cuda.synchronize()" if USE_CUDA else "pass")

    return t, 2 * N * CO * CI * OH * kernel_size

def conv2d(N, H, W, CI, CO, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    class MyModule(torch.nn.Module):
        __constants__ = ['sh', 'sw', 'ph', 'pw', 'dh', 'dw', 'gs']

        def __init__(self):
            super(MyModule, self).__init__()
            self.sh = self.sw = stride
            self.ph = self.pw = padding
            self.dh = self.dw = dilation
            self.gs = groups

        def forward(self, A, B):
            return torch.nn.functional.conv2d(A, B,
                                              stride=(self.sh, self.sw),
                                              padding=(self.ph, self.pw),
                                              dilation=(self.dh, self.dw),
                                              groups=self.gs)

    my_module = MyModule()
    my_module.eval()

    A = torch.rand(N, CI, H, W)
    B = torch.rand(CO, CI // groups, kernel_size, kernel_size)

    if USE_CUDA:
        my_module.cuda(); A = A.cuda(); B = B.cuda()

    if USE_TORCH_SCRIPT:
        func = torch.jit.script(my_module)
        stmt = "func(A, B)"
    else:
        stmt = "my_module(A, B)"

    with torch.no_grad():
        t = py_benchmark(stmt, {**globals(), **locals()},
                         setup="torch.cuda.synchronize()" if USE_CUDA else "pass",
                         finish="torch.cuda.synchronize()" if USE_CUDA else "pass")

    OH = (H + 2 * padding - (kernel_size - 1) * dilation - 1) // stride + 1
    OW = (W + 2 * padding - (kernel_size - 1) * dilation - 1) // stride + 1

    return t, 2 * N * CO * CI * OH * OW * kernel_size * kernel_size / groups

def conv3d(N, D, H, W, CI, CO, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    class MyModule(torch.nn.Module):
        __constants__ = ['sd', 'sh', 'sw', 'pd', 'ph', 'pw', 'dd', 'dh', 'dw']

        def __init__(self):
            super(MyModule, self).__init__()
            self.sd = stride
            self.sh = stride
            self.sw = stride
            self.pd = padding
            self.ph = padding
            self.pw = padding
            self.dd = dilation
            self.dh = dilation
            self.dw = dilation

        def forward(self, A, B):
            return torch.nn.functional.conv3d(A, B,
                                              stride=(self.sd, self.sh, self.sw),
                                              padding=(self.pd, self.ph, self.pw),
                                              dilation=(self.dd, self.dh, self.dw))

    my_module = MyModule()
    my_module.eval()

    A = torch.rand(N, CI, D, H, W)
    B = torch.rand(CO, CI, kernel_size, kernel_size, kernel_size)

    if USE_CUDA:
        my_module.cuda(); A = A.cuda(); B = B.cuda()

    if USE_TORCH_SCRIPT:
        func = torch.jit.script(my_module)
        stmt = "func(A, B)"
    else:
        stmt = "my_module(A, B)"

    with torch.no_grad():
        t = py_benchmark(stmt, {**globals(), **locals()},
                         setup="torch.cuda.synchronize()" if USE_CUDA else "pass",
                         finish="torch.cuda.synchronize()" if USE_CUDA else "pass")

    OD = (D + 2 * padding - (kernel_size - 1) * dilation - 1) // stride + 1
    OH = (H + 2 * padding - (kernel_size - 1) * dilation - 1) // stride + 1
    OW = (W + 2 * padding - (kernel_size - 1) * dilation - 1) // stride + 1

    return t, 2 * N * CO * CI * OD * OH * OW * kernel_size * kernel_size * kernel_size

def depthwise_conv2d(N, H, W, C, kernel_size, stride=1, padding=0, dilation=1):
    return conv2d(N, H, W, C, C, kernel_size, stride, padding, dilation, C)

def conv2d_transpose(N, H, W, CI, CO, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    class MyModule(torch.nn.Module):
        __constants__ = ['sh', 'sw', 'ph', 'pw', 'dh', 'dw', 'gs']

        def __init__(self):
            super(MyModule, self).__init__()
            self.sh = stride
            self.sw = stride
            self.ph = padding
            self.pw = padding
            self.dh = dilation
            self.dw = dilation
            self.gs = groups

        def forward(self, A, B):
            return torch.nn.functional.conv_transpose2d(
                A, B,
                stride=(self.sh, self.sw),
                padding=(self.ph, self.pw),
                dilation=(self.dh, self.dw),
                groups=self.gs)

    my_module = MyModule()
    my_module.eval()

    A = torch.rand(N, CI, H, W)
    B = torch.rand(CI // groups, CO, kernel_size, kernel_size)
    if USE_CUDA:
        my_module.cuda(); A = A.cuda(); B = B.cuda()
    if USE_TORCH_SCRIPT:
        func = torch.jit.script(my_module)
        stmt = "func(A, B)"
    else:
        stmt = "my_module(A, B)"

    with torch.no_grad():
        t = py_benchmark(stmt, {**globals(), **locals()},
                         setup="torch.cuda.synchronize()" if USE_CUDA else "pass",
                         finish="torch.cuda.synchronize()" if USE_CUDA else "pass")

    OH = (H - 1) * stride - 2 * padding + kernel_size
    OW = (W - 1) * stride - 2 * padding + kernel_size

    return t, 2 * N * CO * CI * OH * OW * kernel_size * kernel_size / groups

def conv2d_capsule(N, H, W, CI, CO, kernel_size, stride=1, padding=0, dilation=1, groups=1, capsule_size=4):
    class MyModule(torch.nn.Module):
        __constants__ = ['sh', 'sw', 'ph', 'pw', 'dh', 'dw', 'gs', 'n', 'h', 'w', 'ci', 'co', 'k', 'p', 'oh', 'ow',
                         'USE_CUDA']

        def __init__(self):
            super(MyModule, self).__init__()
            self.sh = stride
            self.sw = stride
            self.ph = padding
            self.pw = padding
            self.dh = dilation
            self.dw = dilation
            self.gs = groups
            self.n  = N
            self.h  = H
            self.w  = W
            self.ci = CI
            self.co = CO
            self.k  = kernel_size
            self.p  = capsule_size
            self.oh = (H + 2 * padding - (kernel_size - 1) * dilation - 1) // stride + 1
            self.ow = (W + 2 * padding - (kernel_size - 1) * dilation - 1) // stride + 1
            self.USE_CUDA = USE_CUDA

        def forward(self, A, B):
            # Use im2col to implement this
            # lhs: (N x OH x OW x P_0,  P_r x KH x KW x CI)
            # rhs: (P_r x KH x KW x CI,  P_1 x CO)
            # out: (N x OH x OW x P_0,  P_1 x CO)
            X = torch.empty((self.n, self.oh, self.ow, self.p, self.p, self.k, self.k, self.ci))
            if self.USE_CUDA:
                X = X.cuda()
            for oh in range(self.oh):
                for ow in range(self.ow):
                    for kh in range(self.k):
                        for kw in range(self.k):
                            X[:, oh, ow, :, :, kh, kw, :] = A[:, oh * self.sh + kh, ow * self.sw + kw, :, :, :]
            X = X.view((self.n * self.oh * self.ow * self.p, self.k * self.k * self.ci * self.p))
            Y = B.view((self.p * self.k * self.k * self.ci, self.p * self.co))
            Z = torch.matmul(X, Y)
            return Z.view(self.n, self.oh, self.ow, self.p, self.p, self.co)

    assert groups == 1 and dilation == 1

    my_module = MyModule()
    my_module.eval()

    A = torch.rand(N, H + 2 * padding, W + 2 * padding, capsule_size, capsule_size, CI)
    B = torch.rand(capsule_size, kernel_size, kernel_size, CI, capsule_size, CO)
    if USE_CUDA:
        my_module.cuda(); A = A.cuda(); B = B.cuda()
    if USE_TORCH_SCRIPT:
        func = torch.jit.script(my_module)
        stmt = "func(A, B)"
    else:
        stmt = "my_module(A, B)"

    with torch.no_grad():
        t = py_benchmark(stmt, {**globals(), **locals()},
                         setup="torch.cuda.synchronize()" if USE_CUDA else "pass",
                         finish="torch.cuda.synchronize()" if USE_CUDA else "pass")

    OH = (H + 2 * padding - (kernel_size - 1) * dilation - 1) // stride + 1
    OW = (W + 2 * padding - (kernel_size - 1) * dilation - 1) // stride + 1

    return t, 2 * N * CO * CI * OH * OW * kernel_size * kernel_size * capsule_size * capsule_size * capsule_size / groups 

def norm(B, M, N):
    class MyModule(torch.nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()

        def forward(self, A):
            return torch.norm(A, dim=[1, 2])

    my_module = MyModule()
    my_module.eval()

    x = torch.rand(B, M, N)

    if USE_CUDA:
        my_module.cuda(); x = x.cuda()
    if USE_TORCH_SCRIPT:
        func = torch.jit.script(my_module)
        stmt = 'func(x)'
    else:
        stmt = 'my_module(x)'

    with torch.no_grad():
        t = py_benchmark(stmt, {**globals(), **locals()},
                         setup="torch.cuda.synchronize()" if USE_CUDA else "pass",
                         finish="torch.cuda.synchronize()" if USE_CUDA else "pass")

    return t, 2 * B * M * N

# ============ Subgraph ============
def conv2d_bn_relu(N, H, W, CI, CO, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    class MyModule(torch.nn.Module):
        __constants__ = ['sh', 'sw', 'ph', 'pw', 'dh', 'dw', 'gs']

        def __init__(self):
            super(MyModule, self).__init__()
            self.sh = self.sw = stride
            self.ph = self.pw = padding
            self.dh = self.dw = dilation
            self.gs = groups

        def forward(self, A, B, bias, bn_scale, bn_offset):
            return torch.nn.functional.relu(
                    torch.add(
                        torch.mul(
                            torch.add(
                                torch.nn.functional.conv2d(A, B,
                                                    stride=(self.sh, self.sw),
                                                    padding=(self.ph, self.pw),
                                                    dilation=(self.dh, self.dw),
                                                    groups=self.gs),
                                bias[None, :, None, None]),
                            bn_scale[None, :, None, None]),
                        bn_offset[None, :, None, None]), inplace=True)

    my_module = MyModule()
    my_module.eval()

    A = torch.rand(N, CI, H, W)
    B = torch.rand(CO, CI // groups, kernel_size, kernel_size)
    bias = torch.rand(CO)
    bn_scale = torch.rand(CO)
    bn_offset = torch.rand(CO)
    if USE_CUDA:
        my_module.cuda(); A = A.cuda(); B = B.cuda(); bias = bias.cuda(); 
        bn_scale = bn_scale.cuda(); bn_offset = bn_offset.cuda()
    if USE_TORCH_SCRIPT:
        func = torch.jit.script(my_module)
        stmt = "func(A, B, bias, bn_scale, bn_offset)"
    else:
        stmt = "my_module(A, B, bias, bn_scale, bn_offset)"

    with torch.no_grad():
        with torch.jit.optimized_execution(True):
            t = py_benchmark(stmt, {**globals(), **locals()},
                             setup="torch.cuda.synchronize()" if USE_CUDA else "pass",
                             finish="torch.cuda.synchronize()" if USE_CUDA else "pass")

    OH = (H + 2 * padding - (kernel_size - 1) * dilation - 1) // stride + 1
    OW = (W + 2 * padding - (kernel_size - 1) * dilation - 1) // stride + 1

    return t, 2 * N * CO * CI * OH * OW * kernel_size * kernel_size / groups

def transpose_batch_matmul(batch, seq_len, n_head, n_dim):
    class MyModule(torch.nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()

        def forward(self, A, B):
            A = A.permute(0, 2, 1, 3)
            B = B.permute(0, 2, 3, 1)
            return torch.matmul(A, B)

    my_module = MyModule()
    my_module.eval()

    x = torch.rand(batch, seq_len, n_head, n_dim)
    y = torch.rand(batch, seq_len, n_head, n_dim)

    if USE_CUDA:
        my_module.cuda(); x = x.cuda(); y = y.cuda()
    if USE_TORCH_SCRIPT:
        func = torch.jit.script(my_module)
        stmt = "func(x, y)"
    else:
        stmt = "my_module(x, y)"

    with torch.no_grad():
        with torch.jit.optimized_execution(True):
            t = py_benchmark(stmt, {**globals(), **locals()},
                             setup="torch.cuda.synchronize()" if USE_CUDA else "pass",
                             finish="torch.cuda.synchronize()" if USE_CUDA else "pass")

    return t, 2 * batch * n_head * seq_len * seq_len * n_dim

def transpose_batch_matmul_softmax(batch, seq_len, n_head, n_dim):
    class MyModule(torch.nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()

        def forward(self, A, B):
            A = A.permute(0, 2, 1, 3)
            B = B.permute(0, 2, 3, 1)
            C = torch.matmul(A, B)
            return torch.softmax(C, dim=-1)

    my_module = MyModule()
    my_module.eval()

    x = torch.rand(batch, seq_len, n_head, n_dim)
    y = torch.rand(batch, seq_len, n_head, n_dim)

    if USE_CUDA:
        my_module.cuda(); x = x.cuda(); y = y.cuda()
    if USE_TORCH_SCRIPT:
        func = torch.jit.script(my_module)
        stmt = "func(x, y)"
    else:
        stmt = "my_module(x, y)"

    with torch.no_grad():
        with torch.jit.optimized_execution(True):
            t = py_benchmark(stmt, {**globals(), **locals()},
                             setup="torch.cuda.synchronize()" if USE_CUDA else "pass",
                             finish="torch.cuda.synchronize()" if USE_CUDA else "pass")

    return t, 2 * batch * n_head * seq_len * seq_len * n_dim


def softmax(M, N):
    class MyModule(torch.nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()

        def forward(self, A):
            return torch.nn.functional.softmax(A, dim=-1)

    my_module = MyModule()
    my_module.eval()

    x = torch.rand(M, N)

    if USE_CUDA:
        my_module.cuda(); x = x.cuda()
    if USE_TORCH_SCRIPT:
        func = torch.jit.script(my_module)
        stmt = 'func(x)'
    else:
        stmt = 'my_module(x)'

    with torch.no_grad():
        with torch.jit.optimized_execution(True):
            t = py_benchmark(stmt, {**globals(), **locals()},
                             setup="torch.cuda.synchronize()" if USE_CUDA else "pass",
                             finish="torch.cuda.synchronize()" if USE_CUDA else "pass")

    return t, 5 * M * N

def batch_norm(M, N):
    class MyModule(torch.nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()

        def forward(self, A):
            return torch.nn.functional.batch_norm(A, torch.mean(A, dim=0), A.var(dim=0))

    my_module = MyModule()
    my_module.eval()

    x = torch.rand(M, N)

    if USE_CUDA:
        my_module.cuda(); x = x.cuda()

    if USE_TORCH_SCRIPT:
        func = torch.jit.script(my_module)
        stmt = 'func(x)'
    else:
        stmt = 'my_module(x)'

    with torch.no_grad():
        with torch.jit.optimized_execution(True):
            t = py_benchmark(stmt, {**globals(), **locals()},
                             setup="torch.cuda.synchronize()" if USE_CUDA else "pass",
                             finish="torch.cuda.synchronize()" if USE_CUDA else "pass")

    return t, 11 * M * N

# ============ Network ============
def resnet50(N):
    
    my_module = models.resnet50(pretrained=False)
    #my_module = torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=False)
    my_module.eval()

    A = torch.rand(N, 3, 224, 224)
    if USE_CUDA:
        my_module.cuda(); A = A.cuda()

    if USE_TORCH_SCRIPT:
        func = torch.jit.script(my_module)
        stmt = "func(A)"
    else:
        stmt = "my_module(A)"

    with torch.no_grad():
        with torch.jit.optimized_execution(True):
            t = py_benchmark(stmt, {**globals(), **locals()},
                             setup="torch.cuda.synchronize()" if USE_CUDA else "pass",
                             finish="torch.cuda.synchronize()" if USE_CUDA else "pass")

    return t

def mobilenet_v2(N):
    my_module = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=False)
    my_module.eval()

    A = torch.rand(N, 3, 224, 224)
    if USE_CUDA:
        my_module.cuda(); A = A.cuda()

    if USE_TORCH_SCRIPT:
        func = torch.jit.script(my_module)
        stmt = "func(A)"
    else:
        stmt = "my_module(A)"

    with torch.no_grad():
        with torch.jit.optimized_execution(True):
            t = py_benchmark(stmt, {**globals(), **locals()},
                             setup="torch.cuda.synchronize()" if USE_CUDA else "pass",
                             finish="torch.cuda.synchronize()" if USE_CUDA else "pass")

    return t

def dcgan(N):
    class Reshape(torch.nn.Module):
        def __init__(self, *args):
            super(Reshape, self).__init__()
            self.shape = args
    
        def forward(self, x):
            return x.view(self.shape)

    class MyModule(torch.nn.Module):
        random_len : torch.jit.Final[int] = 100
        ngf : torch.jit.Final[int] = 128

        def __init__(self):
            super(MyModule, self).__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(self.random_len, 4 * 4 * self.ngf * 8),
                torch.nn.ReLU(inplace=True),
                Reshape(-1, self.ngf * 8, 4, 4),
                torch.nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1),
                torch.nn.BatchNorm2d(self.ngf * 4),
                torch.nn.ReLU(inplace=True),
                torch.nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1),
                torch.nn.BatchNorm2d(self.ngf * 2),
                torch.nn.ReLU(inplace=True),
                torch.nn.ConvTranspose2d(self.ngf * 2, self.ngf * 1, 4, 2, 1),
                torch.nn.BatchNorm2d(self.ngf * 1),
                torch.nn.ReLU(inplace=True),
                torch.nn.ConvTranspose2d(self.ngf * 1, 3, 4, 2, 1),
                torch.nn.Tanh(),
           )

        def forward(self, z):
            return self.net(z)

    my_module = MyModule()
    my_module.eval()
    z = torch.rand(N, 100)

    if USE_CUDA:
        my_module.cuda(); z = z.cuda()

    if USE_TORCH_SCRIPT:
        func = torch.jit.script(my_module)
        stmt = "func(z)"
    else:
        stmt = "my_module(z)"

    with torch.no_grad():
        with torch.jit.optimized_execution(True):
            t = py_benchmark(stmt, {**globals(), **locals()},
                             setup="torch.cuda.synchronize()" if USE_CUDA else "pass",
                             finish="torch.cuda.synchronize()" if USE_CUDA else "pass")

    return t

def dqn(N):
    class MyModule(torch.nn.Module):
        num_action : torch.jit.Final[int] = 18

        def __init__(self):
            super(MyModule, self).__init__()

            def conv2d_size_out(size, kernel_size, stride):
                return (size - (kernel_size - 1) - 1) // stride  + 1
            conv_out = \
                conv2d_size_out(
                    conv2d_size_out(
                        conv2d_size_out(84, 8, 4),
                        4, 2),
                    3, 1)
            linear_input_size = conv_out * conv_out * 64

            self.conv = torch.nn.Sequential(
                # input (batch_size, 4, 84, 84)
                torch.nn.Conv2d(4, 32, 8, 4, bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(32, 64, 4, 2, bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(64, 64, 3, 1, bias=True),
                torch.nn.ReLU(inplace=True))

            self.linear = torch.nn.Sequential(
                torch.nn.Linear(linear_input_size, 512, bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(512, self.num_action, bias=True)
           )

        def forward(self, inp):
            x = self.conv(inp)
            return self.linear(x.view(x.size(0), -1))

    my_module = MyModule()
    my_module.eval()
    z = torch.rand(N, 4, 84, 84)

    if USE_CUDA:
        my_module.cuda(); z = z.cuda()

    if USE_TORCH_SCRIPT:
        func = torch.jit.script(my_module)
        stmt = "func(z)"
    else:
        stmt = "my_module(z)"

    with torch.no_grad():
        with torch.jit.optimized_execution(True):
            t = py_benchmark(stmt, {**globals(), **locals()},
                             setup="torch.cuda.synchronize()" if USE_CUDA else "pass",
                             finish="torch.cuda.synchronize()" if USE_CUDA else "pass")

    return t

def bert(N):
    # need to `pip install transformers` first
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
    A = torch.randint(30000, [N, 128])
    if USE_CUDA:
        model.cuda(); A = A.cuda()

    if USE_TORCH_SCRIPT:
        func = torch.jit.trace(model, [A])
        stmt = "func(A)"
    else:
        stmt = "model(A)"

    with torch.no_grad():
        with torch.jit.optimized_execution(True):
            t = py_benchmark(stmt, {**globals(), **locals()},
                             setup="torch.cuda.synchronize()" if USE_CUDA else "pass",
                             finish="torch.cuda.synchronize()" if USE_CUDA else "pass")

    return t

def resnet3d_18(N):
    model = getattr(torchvision.models.video, 'r3d_18')(pretrained=False)
    model = model.eval()

    input_shape = [N, 3, 16, 112, 112]
    A = torch.randn(input_shape)

    if USE_CUDA:
        model.cuda(); A = A.cuda()

    if USE_TORCH_SCRIPT:
        func = torch.jit.script(model)
        stmt = "func(A)"
    else:
        stmt = "model(A)"

    with torch.no_grad():
        with torch.jit.optimized_execution(True):
            t = py_benchmark(stmt, {**globals(), **locals()},
                             setup="torch.cuda.synchronize()" if USE_CUDA else "pass",
                             finish="torch.cuda.synchronize()" if USE_CUDA else "pass")

    return t

Workload = namedtuple("Workload", ['workload_type', 'workload_name', 'func'])

wkl_list = [
    # Workload("op", "GMM", batch_matmul),
    # Workload("op", "C1D", conv1d),
    # Workload("op", "C2D", conv2d),
    # Workload("op", "C3D", conv3d),
    # Workload("op", "GRP", conv2d),
    # Workload("op", "DIL", conv2d),
    # Workload("op", "DEP", depthwise_conv2d),
    # Workload("op", "T2D", conv2d_transpose),
    # Workload("op", "CAP", conv2d_capsule),
    # Workload("op", "NRM", norm),
    # Workload("op", "SMX", softmax),
    Workload("subgraph", "conv2d_bn_relu", conv2d_bn_relu),
    # Workload("subgraph", "transpose_batch_matmul", transpose_batch_matmul),
    # Workload("subgraph", "transpose_batch_matmul_softmax", transpose_batch_matmul_softmax),
    # Workload("network", "resnet_50", resnet50),
    # Workload("network", "mobilenet_v2", mobilenet_v2),
    # Workload("network", "resnet3d_18", resnet3d_18),
    # Workload("network", "dcgan", dcgan),
    # Workload("network", "dqn", dqn),
    # Workload("network", "bert", bert),
]

import torch
import torch.nn as nn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument("--device", type=str, default=platform.processor())
    parser.add_argument("--wkl", type=str)
    parser.add_argument("--wkl-type", type=str)
    parser.add_argument("--out-file", type=str, default='results.tsv')
    parser.add_argument("--batch-size", type=int, default=-1)  # -1 means test both 1 and 16
    parser.add_argument("--jit", action='store_true')
    args = parser.parse_args()

    if 'TVM_NUM_THREADS' in os.environ:
        torch.set_num_threads(int(os.environ['TVM_NUM_THREADS']))
        print("Set the number of threads to TVM_NUM_THREADS (%s)" % os.environ['TVM_NUM_THREADS'])

    if args.backend == 'cpu':
        # Check mkl-dnn
        if "MKL-DNN" in torch.__config__.show():
            print("MKL-DNN is enabled in pytorch")
            algorithm = 'mkldnn'
        else:
            print("WARNING: MKL-DNN is not enabled in pytorch!!")
            algorithm = 'default'
    elif args.backend == 'gpu':
        USE_CUDA = True
        args.device = torch.cuda.get_device_name()
        torch.backends.cudnn.benchmark = True
        # Check CuDNN
        if "CuDNN" in torch.__config__.show():
            print("CuDNN is enabled in pytorch")
            algorithm = 'cudnn'
        else:
            print("WARNING: CuDNN is not enabled in pytorch!!")
            algorithm = 'default'

    if args.jit:
        USE_TORCH_SCRIPT = True
        algorithm += "-jit"

    if args.batch_size > 0:
        batch_size_list = [args.batch_size]
    else:
        batch_size_list = [1, 16]

    # Benchmark all workloads
    for wkl in wkl_list:
        for batch_size in batch_size_list:
            if args.wkl is not None and wkl.workload_name != args.wkl:
                continue

            if args.wkl_type is not None and wkl.workload_type != args.wkl_type:
                continue
    
            if wkl.workload_type == 'op' or wkl.workload_type == 'subgraph':
                shape_configs = shape_dict[wkl.workload_name]
    
                for shape in shape_configs:
                    if shape[0] == 1:
                        shape = list(shape)
                        shape[0] = batch_size

                    cost, flop = wkl.func(*shape)
                    workload_name = "%s%s" % (wkl.workload_name, tuple(shape))
                    print("%s\t%.3f ms\t%.2f GFLOPS" % (workload_name, cost * 1e3, (flop / 1e9) / cost))
                    log_line(BenchmarkRecord(args.device, args.backend, wkl.workload_type, workload_name,
                                             "pytorch", algorithm, {"costs": [cost]}, time.time()),
                                             args.out_file)
            elif wkl.workload_type == 'network':
                cost = wkl.func(batch_size)
                workload_name = "%s.B%d" % (wkl.workload_name, batch_size)
    
                print("%s\t%.3f ms" % (workload_name, cost * 1e3))
                log_line(BenchmarkRecord(args.device, args.backend, wkl.workload_type, workload_name,
                                         "pytorch", algorithm, {"costs": [cost]}, time.time()),
                                         args.out_file)
            else:
                raise ValueError("Invalid worklaod type: " + wkl.workload_type)

