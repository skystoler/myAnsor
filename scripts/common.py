# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Common utility for scripts"""
import argparse
import math
import os
import re
import time
import itertools
from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import topi
import tvm
from tvm import te, ansor
from tvm.ansor import (LogReader, make_workload_key_func,
                       register_workload_func,
                       write_measure_records_to_file, measure)
from tvm.contrib import ndk, util
from topi.util import get_const_tuple
from topi.nn.util import get_pad_tuple

############################################################
######################  Test Workloads  ####################
############################################################

########## Reduction workloads ##########
@register_workload_func
def min_mn(M, N):
    A = te.placeholder((M, N), name='A')
    B = topi.min(A, axis=1)

    return [A, B]

@register_workload_func
def argmin_mn(M, N):
    A = te.placeholder((M, N), name='A')
    B = topi.argmin(A, axis=1)

    return [A, B]

@register_workload_func
def softmax_mn(M, N):
    A = te.placeholder((M, N), name='A')
    B = topi.nn.softmax(A, axis=1)

    return [A, B]

@register_workload_func
def softmax_abcd(a, b, c, d):
    A = te.placeholder((a, b, c, d), name='A')
    #B = topi.nn.softmax(A, axis=-1)
    B = topi.nn.fast_softmax(A, axis=-1)

    return [A, B]

@register_workload_func
def norm_bmn(B, M, N):
    A = te.placeholder((B, M, N), name='A')
    i = te.reduce_axis((0, M))
    j = te.reduce_axis((0, N))
    C = te.compute((B,), lambda b: te.sum(A[b][i][j] * A[b][i][j], axis=[i, j]), name='C')
    D = te.compute((B,), lambda b: te.sqrt(C[b]), name='D')

    return [A, D]

@register_workload_func
def max_pool_2d_nchw(N, C, H, W):
    data = te.placeholder((N, C, H, W), name='data')
    out = topi.nn.pool(data, (2, 2), (1, 1), (0, 0, 0, 0), pool_type='max', ceil_mode=True,
                       layout="NCHW", count_include_pad=True)

    return [data, out]

@register_workload_func
def add_min_relu(M, N):
    A = te.placeholder((M, N), name='A')
    B = te.placeholder((M, N), name='B')
    C = topi.add(A, B)
    D = topi.min(C, axis=1)
    out = topi.nn.relu(D)
    return [A, B, out]

@register_workload_func
def mean_nhwc(N, H, W, C):
    A = te.placeholder((N, H, W, C), name='A')
    B = topi.sum(A, axis=[1, 2])
    C = B / (H * W)

    return [A, C]

########## Matmul workload ##########
@register_workload_func
def add_mn(M, N):
    A = te.placeholder((M, N), name='A')
    B = te.placeholder((M, N), name='B')
    C = te.compute((M, N), lambda i, j: A[i][j] + B[i][j], name='C')

    return [A, B, C]

@register_workload_func
def matmul_nkkm(N, M, K, in_type='float32', out_type='float32',
                tensor_core_support=False):
    if tensor_core_support:
        A = te.placeholder((N // 16, K // 16, 16, 16), name='A', dtype=in_type)
        B = te.placeholder((K // 16, M // 16, 16, 16), name='B', dtype=in_type)
        k = te.reduce_axis((0, K // 16), name='k')
        kk = te.reduce_axis((0, 16), name='kk')
        if not ((in_type == 'float16' and out_type == 'float32') or \
            (in_type == 'int8' and out_type == 'int32')):
            raise ValueError
        C = te.compute((N // 16, M // 16, 16, 16),
            lambda i, j, ii, jj: te.sum(A[i][k][ii][kk].astype(out_type) * B[k][j][kk][jj].astype(out_type),
                                    axis=[k, kk]),
            name='C')
    else:
        A = te.placeholder((N, K), name='A', dtype=in_type)
        B = te.placeholder((K, M), name='B', dtype=in_type)
        k = te.reduce_axis((0, K), name='k')
        C = te.compute((N, M),
                       lambda i, j: te.sum(A[i][k] * B[k][j], axis=[k]),
                       name='C')
        #C = topi.nn.relu(C)

    return [A, B, C]

@register_workload_func
def double_matmul(N):
    A = te.placeholder((N, N), name='A', dtype='float32')
    B = te.placeholder((N, N), name='B', dtype='float32')
    C = te.placeholder((N, N), name='C', dtype='float32')
    k = te.reduce_axis((0, N), name='k')
    D = te.compute((N, N),
                   lambda i, j: te.sum(A[i][k] * B[k][j], axis=[k]),
                   name='D')
    k = te.reduce_axis((0, N), name='k')
    E = te.compute((N, N),
                   lambda i, j: te.sum(D[i][k] * C[k][j], axis=[k]),
                   name='E')

    return [A, B, C, E]

@register_workload_func
def dense_layer(batch, in_dim, out_dim):
    A = te.placeholder((batch, in_dim), name='A')
    B = te.placeholder((out_dim, in_dim), name='B')
    k = te.reduce_axis((0, in_dim), name='k')
    C = te.compute((batch, out_dim), lambda i, j: te.sum(A[i][k] * B[j][k], axis=[k]), name='C')

    return [A, B, C]

########## Sparse workloads ##########
def random_bsr_matrix(M, N, BS_R, BS_C, density, dtype):
    import itertools
    import scipy.sparse as sp
    np.random.seed(42)
    Y = np.zeros((M, N), dtype=dtype)
    assert M % BS_R == 0
    assert N % BS_C == 0
    nnz = int(density * M * N)
    num_blocks = int(nnz / (BS_R * BS_C)) + 1
    candidate_blocks = np.asarray(list(itertools.product(range(0, M, BS_R), range(0, N, BS_C))))
    assert candidate_blocks.shape[0] == M // BS_R * N // BS_C
    chosen_blocks = candidate_blocks[np.random.choice(candidate_blocks.shape[0], size=num_blocks, replace=False)]
    for i in range(len(chosen_blocks)):
        r, c = chosen_blocks[i]
        Y[r:r + BS_R, c:c + BS_C] = np.random.randn(BS_R, BS_C)
    s = sp.bsr_matrix(Y, blocksize=(BS_R, BS_C))
    assert s.data.shape == (num_blocks, BS_R, BS_C)
    assert s.indices.shape == (num_blocks, )
    assert s.indptr.shape == (M // BS_R + 1, )
    return s

def sparse_dense_bsr_compute(data, weight_data, weight_indices, weight_indptr):
    (m, k) = get_const_tuple(data.shape)
    (_, bs_r, bs_c) = get_const_tuple(weight_data.shape)
    (num_block_row_plus_1, ) = get_const_tuple(weight_indptr.shape)
    num_block_row = num_block_row_plus_1 - 1

    def body_func(i, j):
        block_j = j // bs_r
        jj = j % bs_r

        row_start = weight_indptr[block_j]
        row_end = weight_indptr[block_j + 1]
        row_offset = te.reduce_axis((0, row_end - row_start), name='row_offset')
        kk = te.reduce_axis((0, bs_c), name='kk')
        block_idx = row_start + row_offset

        return te.sum(data[i, weight_indices[block_idx] * bs_c + kk] * \
                      weight_data[block_idx, jj, kk], axis=[row_offset, kk],)

    out = te.compute((m, num_block_row * bs_r), lambda i, j : body_func(i, j),
                     attrs={"ansor_no_split_at_inner": ["row_offset", "kk"],
                            "FLOP": 2 * m * num_block_row * bs_r * k},
                     name='sparse_dense')

    return out

@register_workload_func
def sparse_dense_bsr(M, N, K, BS_R, BS_C, density, use_relu):
    dtype = "float32"
    W_sp_np = random_bsr_matrix(N, K, BS_R, BS_C, density=density, dtype=dtype)

    # register these special buffers for measurement
    prefix = "sparse_dense_bsr_%d_%d_%d_%d_%d_%.2f_" % (M, N, K, BS_R, BS_C, density)
    measure.register_special_buffer(prefix + "W_data", W_sp_np.data)
    measure.register_special_buffer(prefix + "W_indices", W_sp_np.indices)
    measure.register_special_buffer(prefix + "W_indptr", W_sp_np.indptr)

    W_data = te.placeholder(shape=W_sp_np.data.shape, dtype=str(W_sp_np.data.dtype),
                            name=prefix + "W_data")
    W_indices = te.placeholder(shape=W_sp_np.indices.shape, dtype=str(W_sp_np.indices.dtype),
                               name=prefix + "W_indices")
    W_indptr = te.placeholder(shape=W_sp_np.indptr.shape, dtype=str(W_sp_np.indptr.dtype),
                              name=prefix + "W_indptr")
    X = te.placeholder(shape=(M, K), dtype=dtype, name='X')

    Y = sparse_dense_bsr_compute(X, W_data, W_indices, W_indptr)

    if use_relu:
        Y = topi.nn.relu(Y)

    return [X, W_data, W_indices, W_indptr, Y]

def random_csr_matrix(M, N, density, dtype):
    import scipy.sparse as sp
    np.random.seed(42)
    Y = np.zeros((M, N), dtype=dtype)
    nnz = int(density * M * N)
    chosen_indices = np.random.choice(M * N, size=nnz, replace=False)
    for idx in chosen_indices:
        i, j = idx // N, idx % N
        Y[i, j] = np.random.randn()
    s = sp.csr_matrix(Y)
    assert s.data.shape == (nnz,)
    assert s.indices.shape == (nnz,)
    assert s.indptr.shape == (M+1,)
    return s

def sparse_conv2d_csr_compute(data, weight_data, weight_indices, weight_indptr, kernel_sizes, strides, padding, dilation):
    assert isinstance(strides, int) or len(strides) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(strides, int):
        stride_h = stride_w = strides
    else:
        stride_h, stride_w = strides

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    KH, KW = kernel_sizes

    assert dilation_h == dilation_w == 1

    batch, in_channel, in_height, in_width = get_const_tuple(data.shape)
    out_channel_plus_1, = get_const_tuple(weight_indptr.shape)
    out_channel = out_channel_plus_1 - 1

    # compute the output shape
    dilated_kernel_h = (KH - 1) * dilation_h + 1
    dilated_kernel_w = (KW - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_height = (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1
    out_width = (in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    padded = topi.nn.pad(data, pad_before, pad_after, name="pad_temp")

    def body(n, co, oh, ow):
        row_start = weight_indptr[co]
        row_end = weight_indptr[co + 1]

        row_offset = te.reduce_axis((0, row_end - row_start), name='row_offset')
        idx = row_start + row_offset

        col = weight_indices[idx]
        kw = col % KW
        kh = col // KW % KH
        ci = col // KH // KW

        return te.sum(
            padded[n, ci, oh * stride_h + kh * dilation_h,
                 ow * stride_w + kw * dilation_w] *
            weight_data[idx],
            axis=[row_offset])

    return te.compute(
        (batch, out_channel, out_height, out_width),
        lambda n, co, oh, ow: body(n, co, oh, ow),
        attrs={"ansor_no_split_at_inner": ["row_offset"],
               "FLOP": 2 * batch * out_channel * out_height * out_width * in_channel * KH * KW},
        name='sparse_conv2d_csr',
    )

@register_workload_func
def sparse_conv2d_csr(N, H, W, CI, CO, KH, KW, strides, padding, dilation, density, use_relu):
    dtype = "float32"
    W_sp_np = random_csr_matrix(CO, CI * KH * KW, density=density, dtype=dtype)

    # register these special buffers for measurement
    prefix = "sparse_conv2d_csr_" + \
             "_".join(["%s" % x for x in (N, H, W, CI, CO, KH, KW, strides, padding, dilation)]) + \
             "%.2f" % density

    measure.register_special_buffer(prefix + "W_data", W_sp_np.data)
    measure.register_special_buffer(prefix + "W_indices", W_sp_np.indices)
    measure.register_special_buffer(prefix + "W_indptr", W_sp_np.indptr)

    W_data = te.placeholder(shape=W_sp_np.data.shape, dtype=str(W_sp_np.data.dtype),
                            name=prefix + "W_data")
    W_indices = te.placeholder(shape=W_sp_np.indices.shape, dtype=str(W_sp_np.indices.dtype),
                               name=prefix + "W_indices")
    W_indptr = te.placeholder(shape=W_sp_np.indptr.shape, dtype=str(W_sp_np.indptr.dtype),
                              name=prefix + "W_indptr")
    X = te.placeholder(shape=(N, CI, H, W), dtype=dtype, name='X')

    Y = sparse_conv2d_csr_compute(X, W_data, W_indices, W_indptr, (KH, KW), strides, padding, dilation)

    if use_relu:
        Y = topi.nn.relu(Y)

    return [X, W_data, W_indices, W_indptr, Y]

########## Conv2d NCHW ##########
@register_workload_func
def conv2d_relu_softmax_min(N, H, W, CI, CO, KH, KW, strides, padding, dilation):
    data = te.placeholder((N, CI, H, W), name='data')
    kernel = te.placeholder((CO, CI, KH, KW), name='kernel')
    conv = topi.nn.conv2d_nchw(data, kernel, strides, padding, dilation)
    relu = topi.nn.relu(conv)
    softmax = topi.nn.softmax(relu, axis=1)
    out = topi.min(softmax, axis=1)

    return [data, kernel, out]

@register_workload_func
def conv2d_nchw_bias(N, H, W, CI, CO, KH, KW, strides, padding, dilation):
    data = te.placeholder((N, CI, H, W), name='data')
    kernel = te.placeholder((CO, CI, KH, KW), name='kernel')
    bias = te.placeholder((CO, 1, 1), name='bias')
    conv = topi.nn.conv2d_nchw(data, kernel, strides, padding, dilation)
    #out = topi.nn.relu(conv)
    out = topi.add(conv, bias)
    return [data, kernel, bias, out]

@register_workload_func
def conv2d_nchw_bn_relu(N, H, W, CI, CO, kernel_size, strides, padding, dilation=1):
    data = te.placeholder((N, CI, H, W), name='data')
    kernel = te.placeholder((CO, CI, kernel_size, kernel_size), name='kernel')
    bias = te.placeholder((CO, 1, 1), name='bias')
    bn_scale = te.placeholder((CO, 1, 1), name='bn_scale')
    bn_offset = te.placeholder((CO, 1, 1), name='bn_offset')

    OH = (H + 2 * padding - (kernel_size - 1) * dilation - 1) // strides + 1
    OW = (W + 2 * padding - (kernel_size - 1) * dilation - 1) // strides + 1

    conv = topi.nn.conv2d_nchw(data, kernel, strides, padding, dilation)
    conv = te.compute((N, CO, OH, OW),
                      lambda i, j, k, l: conv[i, j, k, l] + bias[j, 0, 0],
                      name='bias_add')
    conv = te.compute((N, CO, OH, OW),
                      lambda i, j, k, l: conv[i, j, k, l] * bn_scale[j, 0, 0],
                      name='bn_mul')
    conv = te.compute((N, CO, OH, OW),
                      lambda i, j, k, l: conv[i, j, k, l] + bn_offset[j, 0, 0],
                      name='bn_add')
    out = topi.nn.relu(conv)

    return [data, kernel, bias, bn_offset, bn_scale, out]

########## Conv2d NHWC ##########
def conv2d_nhwc_without_layout_rewrite(Input, Filter, stride, padding, dilation, out_dtype='float32'):
    """A copy of `topi.nn.conv2d_nhwc` but without the 'layout_free` attribute.
    We use this in single op and subgraph evaluation because we don't want to introduce graph level optimization.
    """
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_height, in_width, in_channel = Input.shape
    kernel_h, kernel_w, channel, num_filter = Filter.shape

    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = topi.nn.get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_height = topi.util.simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = topi.util.simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    PaddedInput = topi.nn.pad(Input, pad_before, pad_after, name="PaddedInput")
    rc = te.reduce_axis((0, in_channel), name='rc')
    ry = te.reduce_axis((0, kernel_h), name='ry')
    rx = te.reduce_axis((0, kernel_w), name='rx')
    Output = te.compute(
        (batch, out_height, out_width, out_channel),
        lambda nn, yy, xx, ff: te.sum(
            PaddedInput[nn, yy * stride_h + ry * dilation_h,
                        xx * stride_w + rx * dilation_w, rc].astype(out_dtype) *
            Filter[ry, rx, rc, ff].astype(out_dtype)
            , axis=[ry, rx, rc]),
        name="Conv2dOutput", tag="conv2d_nhwc")
    return Output

@register_workload_func
def dense_conv2d_nhwc(N, H, W, CI, CO, KH, KW, strides, padding, dilation):
    data = te.placeholder((N, H, W, CI), name='data')
    kernel = te.placeholder((KH, KW, CI, CO), name='kernel')
    conv = conv2d_nhwc_without_layout_rewrite(data, kernel, strides, padding, dilation)
    return [data, kernel, conv]

@register_workload_func
def conv2d_nhwc_bias(N, H, W, CI, CO, KH, KW, strides, padding, dilation):
    data = te.placeholder((N, H, W, CI), name='data')
    kernel = te.placeholder((KH, KW, CI, CO), name='kernel')
    bias = te.placeholder((CO, ), name='bias')
    conv = conv2d_nhwc_without_layout_rewrite(data, kernel, strides, padding, dilation)
    out = topi.add(conv, bias)
    return [data, kernel, bias, out]

@register_workload_func
def conv2d_nhwc_bn_relu(N, H, W, CI, CO, kernel_size, strides, padding, dilation=1):
    data = te.placeholder((N, H, W, CI), name='data')
    kernel = te.placeholder((kernel_size, kernel_size, CI, CO), name='kernel')
    bias = te.placeholder((CO,), name='bias')
    bn_scale = te.placeholder((CO,), name='bn_scale')
    bn_offset = te.placeholder((CO,), name='bn_offset')

    OH = (H + 2 * padding - (kernel_size - 1) * dilation - 1) // strides + 1
    OW = (W + 2 * padding - (kernel_size - 1) * dilation - 1) // strides + 1

    conv = conv2d_nhwc_without_layout_rewrite(data, kernel, strides, padding, dilation)
    conv = te.compute((N, OH, OW, CO),
                       lambda i, j, k, l: conv[i, j, k, l] + bias[l],
                       name='bias_add')
    conv = te.compute((N, OH, OW, CO),
                       lambda i, j, k, l: conv[i, j, k, l] * bn_scale[l],
                       name='bn_mul')
    conv = te.compute((N, OH, OW, CO),
                       lambda i, j, k, l: conv[i, j, k, l] + bn_offset[l],
                       name='bn_add')
    out = topi.nn.relu(conv)

    return [data, kernel, bias, bn_offset, bn_scale, out]

@register_workload_func
def conv2d_nhwc_bias_with_rewrite(N, H, W, CI, CO, KH, KW, strides, padding, dilation):
    data = te.placeholder((N, H, W, CI), name='data')
    kernel = te.placeholder((KH, KW, CI, CO), name='kernel')
    bias = te.placeholder((CO, ), name='bias')
    conv = topi.nn.conv2d_nhwc(data, kernel, strides, padding, dilation)
    out = topi.add(conv, bias)
    return [data, kernel, bias, out]

@register_workload_func
def depthwise_conv2d_nhwc_bias_with_rewrite(N, H, W, CI, CO, KH, KW, strides, padding, dilation):
    data = te.placeholder((N, H, W, CI), name='data')
    kernel = te.placeholder((KH, KW, CI, 1), name='kernel')
    bias = te.placeholder((CO, ), name='bias')
    conv = topi.nn.depthwise_conv2d_nhwc(data, kernel, strides, padding, dilation)
    out = topi.add(conv, bias)
    return [data, kernel, bias, out]

resnet_conv2d_configs = {
    # format : N, H, W, CI, CO, KH, KW, strides, padding, dilation
    '18': [
        (1, 224, 224, 3, 64, 7, 7, (2, 2), (3, 3), (1, 1)),
        (1, 56, 56, 64, 128, 3, 3, (2, 2), (1, 1), (1, 1)),
        (1, 56, 56, 64, 128, 1, 1, (2, 2), (0, 0), (1, 1)),
        (1, 56, 56, 64, 64, 3, 3, (1, 1), (1, 1), (1, 1)),
        (1, 56, 56, 64, 64, 1, 1, (1, 1), (0, 0), (1, 1)),
        (1, 28, 28, 128, 256, 3, 3, (2, 2), (1, 1), (1, 1)),
        (1, 28, 28, 128, 256, 1, 1, (2, 2), (0, 0), (1, 1)),
        (1, 28, 28, 128, 128, 3, 3, (1, 1), (1, 1), (1, 1)),
        (1, 14, 14, 256, 512, 3, 3, (2, 2), (1, 1), (1, 1)),
        (1, 14, 14, 256, 512, 1, 1, (2, 2), (0, 0), (1, 1)),
        (1, 14, 14, 256, 256, 3, 3, (1, 1), (1, 1), (1, 1)),
        (1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1), (1, 1)),
    ],
    '50': [
        (1, 224, 224, 3, 64, 7, 7, (2, 2), (3, 3), (1, 1)),
        (1, 56, 56, 256, 512, 1, 1, (2, 2), (0, 0), (1, 1)),
        (1, 56, 56, 256, 128, 1, 1, (2, 2), (0, 0), (1, 1)),
        (1, 56, 56, 256, 64, 1, 1, (1, 1), (0, 0), (1, 1)),
        (1, 56, 56, 64, 256, 1, 1, (1, 1), (0, 0), (1, 1)),
        (1, 56, 56, 64, 64, 3, 3, (1, 1), (1, 1), (1, 1)),
        (1, 56, 56, 64, 64, 1, 1, (1, 1), (0, 0), (1, 1)),
        (1, 28, 28, 512, 1024, 1, 1, (2, 2), (0, 0), (1, 1)),
        (1, 28, 28, 512, 256, 1, 1, (2, 2), (0, 0), (1, 1)),
        (1, 28, 28, 512, 128, 1, 1, (1, 1), (0, 0), (1, 1)),
        (1, 28, 28, 128, 512, 1, 1, (1, 1), (0, 0), (1, 1)),
        (1, 28, 28, 128, 128, 3, 3, (1, 1), (1, 1), (1, 1)),
        (1, 14, 14, 1024, 2048, 1, 1, (2, 2), (0, 0), (1, 1)),
        (1, 14, 14, 1024, 512, 1, 1, (2, 2), (0, 0), (1, 1)),
        (1, 14, 14, 1024, 256, 1, 1, (1, 1), (0, 0), (1, 1)),
        (1, 14, 14, 256, 1024, 1, 1, (1, 1), (0, 0), (1, 1)),
        (1, 14, 14, 256, 256, 3, 3, (1, 1), (1, 1), (1, 1)),
        (1, 7, 7, 2048, 512, 1, 1, (1, 1), (0, 0), (1, 1)),
        (1, 7, 7, 512, 2048, 1, 1, (1, 1), (0, 0), (1, 1)),
        (1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1), (1, 1)),
    ],
}

# number of appearance for all conv2ds in resnet
resnet_conv2d_weights = {
    '18': [1, 1, 1, 4, 1, 1, 1, 3, 1, 1, 3, 3],
    '50': [1, 1, 1, 2, 4, 3, 1, 1, 1, 3, 4, 4, 1, 1, 5, 6, 6, 2, 3, 3],
}

########## Workload name parser ##########
def parse_workload_name(name: str) -> List[str]:
    """Parse workload name with wildcard character and abbreviation to standard names"""
    if name.startswith('matmul-'):  # e.g. matmul-512, matmul-1024, matmul-+
        N = name.split('-', maxsplit=1)[1]
        if N == '+':
            cfg_list = [256, 512, 1024]
        else:
            cfg_list = [N]
        return ["matmul-%s" % x for x in cfg_list]
    #elif name.startswith('dense-'):  # e.g. dense-1-512-1024, dense-16-512-512
    #    N = name.split('-', maxsplit=1)[1]
    #    if N == '+':
    #        cfg_list = ["1-512-512", "16-512-512"]
    #    else:
    #        cfg_list = [N]
    #    return ["dense-%s" % x for x in cfg_list]
    elif name.startswith('min-'):  # e.g. min-4096
        N = name.split('-', maxsplit=1)[1]
        if N == '+':
            cfg_list = [4096, 8192, 16384]
        else:
            cfg_list = [N]
        return ["min-%s" % x for x in cfg_list]
    elif name.startswith('argmin-'):  # e.g. argmin-4096
        N = name.split('-', maxsplit=1)[1]
        if N == '+':
            cfg_list = [4096, 8192, 16384]
        else:
            cfg_list = [N]
        return ["argmin-%s" % x for x in cfg_list]
    elif name.startswith('softmax-'):  # e.g. softmax-4096
        N = name.split('-', maxsplit=1)[1]
        if N == '+':
            cfg_list = [4096, 8192, 16384]
        else:
            cfg_list = [N]
        return ["softmax-%s" % x for x in cfg_list]
    elif name.startswith('add-'):  # e.g. add-4096
        N = name.split('-', maxsplit=1)[1]
        if N == '+':
            cfg_list = [4096, 8192, 16384]
        else:
            cfg_list = [N]
        return ["add-%s" % x for x in cfg_list]
    elif name.startswith('norm-'):  # e.g. norm-1024
        N = name.split('-', maxsplit=1)[1]
        if N == '+':
            cfg_list = [4096, 8192, 16384]
        else:
            cfg_list = [N]
        return ["norm-%s" % x for x in cfg_list]
    elif name.startswith('add-min-relu'):  # e.g. add-min-relu-4096
        N = name.split('-', maxsplit=3)[3]
        if N == '+':
            cfg_list = [4096, 8192, 16384]
        else:
            cfg_list = [N]
        return ["add-min-relu-%s" % x for x in cfg_list]
    elif name.startswith('nhwc-resnet-'):  # e.g.  nhwc-resnet-50.C1
        res = re.match(r'nhwc-resnet-(\d+).C([\d\+]+)(.B(\d+))?', name)
        n_layers = res.group(1)
        if res.group(2) == '+':
            idx_list = range(len(resnet_conv2d_configs[n_layers]))
        else:
            idx_list = [int(res.group(2))]

        batch_size = 1 if res.group(4) is None else int(res.group(4))
        return ['nhwc-resnet-%s.C%d.B%d' % (n_layers, i, batch_size) for i in idx_list]
    elif name.startswith('resnet-'):  # e.g.  resnet-50.C1, resnet-50.C1.B2, resnet-50.C+.B2
        res = re.match(r'resnet-(\d+).C([\d\+]+)(.B(\d+))?', name)
        n_layers = res.group(1)
        if res.group(2) == '+':
            idx_list = range(len(resnet_conv2d_configs[n_layers]))
        else:
            idx_list = [int(res.group(2))]

        batch_size = 1 if res.group(4) is None else int(res.group(4))
        return ['resnet-%s.C%d.B%d' % (n_layers, i, batch_size) for i in idx_list]
    elif name in ['conv2d-bn-relu', 'conv2d-relu-softmax-min', 'max-pool-2d', 'conv2d-rewrite',
                  'depthwise-conv2d-rewrite', 'bert-softmax', 'sparse-dense-bsr', 'sparse-conv2d-csr',
                  'dense-conv2d', 'mean-nhwc', 'double-matmul']:
        return [name]
    else:
        raise ValueError("Invalid workload " + name)


def get_workload_keys(name: str) -> List[str]:
    """Parse workload name and return the workload keys"""
    normalized_names = parse_workload_name(name)

    ret = []
    for name in normalized_names:
        if name.startswith('matmul-'):
            name_split = name.split('-')
            in_type = out_type = 'float32'
            tensor_core_support = False
            if len(name_split) == 2:    # e.g. matmul-512
                N = K = M = int(name_split[1])
            elif len(name_split) == 4:  # e.g. matmul-32-256-512
                N = int(name_split[1])
                K = int(name_split[2])
                M = int(name_split[3])
            elif len(name_split) == 6:  # e.g. matmul-32-512-512-float16-float32
                N = int(name_split[1])
                K = int(name_split[2])
                M = int(name_split[3])
                in_type = name_split[4]
                out_type = name_split[5]
            elif len(name_split) == 7:  # e.g. matmul-32-512-512-float16-float32-tc
                N = int(name_split[1])
                K = int(name_split[2])
                M = int(name_split[3])
                in_type = name_split[4]
                out_type = name_split[5]
                tensor_core_support = name_split[6] == "tc"
            else:
                raise ValueError("Invalid matmul workload")
            ret.append(make_workload_key_func(matmul_nkkm,
                                              (N, M, K, in_type, out_type, tensor_core_support)))
        #elif name.startswith('dense-'):  # e.g. dense-1-512-1024, dense-16-512-512
        #    name_split = name.split('-')
        #    assert len(name_split) == 4
        #    batch = int(name_split[1])
        #    in_dim = int(name_split[2])
        #    out_dim = int(name_split[3])
        #    ret.append(make_workload_key_func(dense_layer, (batch, in_dim, out_dim)))
        elif name.startswith('min-'):  # e.g. min-4096
            name_split = name.split('-')
            if len(name_split) == 2:
                M = 64
                N = int(name_split[1])
            elif len(name_split) == 3:
                M = int(name_split[1])
                N = int(name_split[2])
            else:
                raise ValueError("Invalid min workload")
            ret.append(make_workload_key_func(min_mn, (M, N)))
        elif name.startswith('argmin-'):  # e.g. argmin-4096
            name_split = name.split('-')
            if len(name_split) == 2:
                M = 64
                N = int(name_split[1])
            elif len(name_split) == 3:
                M = int(name_split[1])
                N = int(name_split[2])
            else:
                raise ValueError("Invalid argmin workload")
            ret.append(make_workload_key_func(argmin_mn, (M, N)))
        elif name.startswith('softmax-'):  # e.g. softmax-4096
            name_split = name.split('-')
            if len(name_split) == 2:
                M = 64
                N = int(name_split[1])
            elif len(name_split) == 3:
                M = int(name_split[1])
                N = int(name_split[2])
            else:
                raise ValueError("Invalid softmax workload")
            ret.append(make_workload_key_func(softmax_mn, (M, N)))
        elif name.startswith('add-min-relu'):  # e.g. add-min-relu-4096
            name_split = name.split('-')
            if len(name_split) == 4:
                M = 64
                N = int(name_split[3])
            elif len(name_split) == 5:
                M = int(name_split[3])
                N = int(name_split[4])
            else:
                raise ValueError("Invalid workload")
            ret.append(make_workload_key_func(add_min_relu, (M, N)))
        elif name.startswith('add-'):  # e.g. add-4096
            name_split = name.split('-')
            if len(name_split) == 2:
                N = M = int(name_split[1])
            elif len(name_split) == 3:
                M = int(name_split[1])
                N = int(name_split[2])
            else:
                raise ValueError("Invalid add workload")
            ret.append(make_workload_key_func(add_mn, (M, N)))
        elif name.startswith('norm-'):  # e.g. norm-4096
            name_split = name.split('-')
            B = 2
            if len(name_split) == 2:
                N = M = int(name_split[1])
            elif len(name_split) == 3:
                M = int(name_split[1])
                N = int(name_split[2])
            else:
                raise ValueError("Invalid norm workload")
            ret.append(make_workload_key_func(norm_bmn, (B, M, N)))
        elif name.startswith('nhwc-resnet-'):  # e.g.  nhwc-resnet-50.C1.B2
            res = re.match(r'nhwc-resnet-(\d+).C(\d+).B(\d+)', name)
            n_layers = res.group(1)
            idx = int(res.group(2))
            batch_size = 1 if res.group(3) is None else int(res.group(3))
            args = list(resnet_conv2d_configs[n_layers][idx])
            args[0] = batch_size
            ret.append(make_workload_key_func(conv2d_nhwc_bias, args))
        elif name.startswith('resnet-'):  # e.g.  resnet-50.C1.B2
            res = re.match(r'resnet-(\d+).C(\d+).B(\d+)', name)
            n_layers = res.group(1)
            idx = int(res.group(2))
            batch_size = 1 if res.group(3) is None else int(res.group(3))
            args = list(resnet_conv2d_configs[n_layers][idx])
            args[0] = batch_size
            ret.append(make_workload_key_func(conv2d_nchw_bias, args))
        elif name == 'max-pool-2d':
            return [make_workload_key_func(max_pool_2d_nchw, (2, 512, 7, 7))]
        elif name == 'conv2d-bn-relu':
            return [make_workload_key_func(conv2d_nhwc_bn_relu,
                                           (1, 7, 7, 512, 512, 3, 1, 1, 1)) ]
        elif name == 'conv2d-rewrite':
            return [ make_workload_key_func(conv2d_nhwc_bias_with_rewrite,
                                            (1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1), (1, 1)))]
        elif name == 'depthwise-conv2d-rewrite':
            return [ make_workload_key_func(depthwise_conv2d_nhwc_bias_with_rewrite,
                                            (1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1), (1, 1)))]
        elif name == 'conv2d-relu-softmax-min':
            return [make_workload_key_func(conv2d_relu_softmax_min,
                                           (1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1), (1, 1)))]
        elif name == 'bert-softmax':
            return [make_workload_key_func(softmax_abcd, (16, 12, 128, 128))]
        elif name == 'sparse-dense-bsr':
            return [make_workload_key_func(sparse_dense_bsr, (128, 3072, 768, 16, 1, 0.15, False))]
        elif name == 'sparse-conv2d-csr':
            return [make_workload_key_func(sparse_conv2d_csr, (1, 7, 7, 512, 512, 3, 3, 1, 1, 1, 0.15, False))]
        elif name == 'dense-conv2d':
            return [make_workload_key_func(dense_conv2d_nhwc, (1, 7, 7, 512, 512, 3, 3, 1, 1, 1))]
        elif name == 'mean-nhwc':
            return [make_workload_key_func(mean_nhwc, (1, 7, 7, 128))]
        elif name == 'double-matmul':
            return [make_workload_key_func(double_matmul, (512,))]
        else:
            raise ValueError("Invalid workload " + name)

    return ret


def get_workload_weights(name: str) -> List[float]:
    """Return weights for workload name"""
    if name.startswith('resnet-'):
        res = re.match(r'resnet-(\d+).C+', name)
        n_layers = res.group(1)
        return np.array(resnet_conv2d_weights[n_layers])
    else:
        return np.ones(len(get_workload_keys(name)))

########## Load workloads from a network ##########
def load_network(network_name, network_path, batch_size, layout):
    from tune_network import get_network
    # Extract tasks from relay program
    print("Load tasks from network %s.B%d (%s)..." % (network_name, batch_size, layout))
    mod, params, input_name, data_shape, data_dtype, out_shape = get_network(
            network_name, network_path, batch_size, layout)
    workloads, wkl_weights = ansor.extract_from_program(mod, target='llvm', params=params)

############################################################
######################  Measure Tools   ####################
############################################################

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
    for i, arg in enumerate(bufs):
        if measure.get_special_buffer(arg.name) is not None:
            np_args[i] = measure.get_special_buffer(arg.name)
    args = [tvm.nd.array(x, ctx=ctx) for x in np_args]
    ctx.sync()

    costs = time_f(*args).results

    return costs

def check_correctness(s, bufs, s_ref, buf_ref, target, target_host=None, remote=None, ndk_cc=None):
    """Check the correctness of a schedule against a reference schedule"""
    func = tvm.build(s, bufs, target=target, target_host=target_host)
    func_ref = tvm.build(s_ref, buf_ref, target='llvm')

    if remote:
        raise NotImplemented
    else:
        ctx = tvm.context(str(target), 0)
        ctx_ref = tvm.cpu()

    np_args = [np.random.randn(*topi.get_const_tuple(x.shape)).astype(x.dtype) for x in bufs]
    for i, arg in enumerate(bufs):
        if measure.get_special_buffer(arg.name) is not None:
            np_args[i] = measure.get_special_buffer(arg.name)
    args = [tvm.nd.array(x, ctx=ctx) for x in np_args]
    args_ref = [tvm.nd.array(x, ctx=ctx_ref) for x in np_args]
    ctx.sync()

    func(*args)
    func_ref(*args_ref)

    for arr, arr_ref in zip(args, args_ref):
        np.testing.assert_allclose(arr.asnumpy(), arr_ref.asnumpy(), rtol=1e-3, atol=1e-3)


############################################################
#####################  Other Utilities  ####################
############################################################

def verify_gpu_code(stmt):
    check_gpu = {
	#"max_shared_memory_per_block": ctx.max_shared_memory_per_block,
	#"max_threads_per_block": ctx.max_threads_per_block,
	#"max_thread_x": max_dims[0],
	#"max_thread_y": max_dims[1],
	#"max_thread_z": max_dims[2]
        'max_vector_bytes': 16,
    }
    valid = tvm.tir.analysis.verify_gpu_code(stmt['main'], check_gpu)
    print("Valid : %s" % valid)
    exit()


def geomean(xs):
    """Compute geometric mean"""
    return math.exp(math.fsum(math.log(x) for x in xs) / len(xs))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def run_cmd(cmd):
    print(cmd)
    ret = os.system(cmd)
    if ret != 0:
        exit(ret)


global last_tic
last_tic = None


def PRINT_TIME(msg):
    """Print time interval between differnt calls. This is for debug so we make the name letters capital"""
    global last_tic
    now = time.time()

    if last_tic is None:
        last_tic = now

    print(msg, now - last_tic)
    last_tic = now


############################################################
######################  I/O Utilities  #####################
############################################################

# The format for a line in resulst file
BenchmarkRecord = namedtuple("BenchmarkRecord",
                             ['device', 'backend', 'workload_type', 'workload_name',
                              'library', 'algorithm', 'value', 'time_stamp'])

def log_line(record, out_file):
    with open(out_file, 'a') as fout:
        fout.write("\t".join([to_str_round(x) for x in record]) + '\n')

def extract_tar(path):
    import tarfile
    if path.endswith("tgz") or path.endswith("gz"):
        dir_path = os.path.dirname(path)
        tar = tarfile.open(path)
        tar.extractall(path=dir_path)
        tar.close()
    else:
        raise RuntimeError('Could not decompress the file: ' + path)

class BaselineDatabase:
    """A class for query records in baseline database"""
    def __init__(self, filename):
        self.filename = filename

        self.lines = []
        for line in open(filename):
            if line.startswith('#') or line.isspace():
                continue
            self.lines.append(line.split('\t'))

    def filter_records(self, devices=None, backends=None, wkl_names=None, libraries=None):
        ret = []
        for line in self.lines:
            line = BenchmarkRecord(*line)

            if devices is not None and line.device not in devices:
                continue
            if backends is not None and line.backend not in backends:
                continue
            if wkl_names is not None and line.workload_name not in wkl_names:
                continue
            if libraries is not None and line.library not in libraries:
                continue

            ret.append(line)
        return ret

    def get_data_dict(self, device, target, wkl_names) -> Tuple[Dict, List]:
        """Return a data dict s.t.  data[wkl][library] = cost"""
        data = defaultdict(lambda: defaultdict(lambda: 1e10))

        all_libraries = set()

        if "cpu" in target.keys:
            backends = ['cpu']
        elif "gpu" in target.keys:
            backends = ['gpu']
        else:
            raise ValueError("Invalid target: " + target)

        # Read costs for baselines
        records = self.filter_records(devices=[device], backends=backends, wkl_names=wkl_names)
        for record in records:
            # use min over (possible) multiple algorithms
            all_libraries.add(record.library)
            data[record.workload_name][record.library] = \
                min(data[record.workload_name][record.library],
                    np.mean(eval(record.value)['costs']))

        return data, list(all_libraries)


class LogFileDatabase:
    """A class for indexing best records in a log file"""
    def __init__(self, filename: str, n_lines: int = -1):
        inputs, results = LogReader(filename).read_lines(n_lines)

        # best records, search by (target_key, workload_key).  e.g. ('gpu', 'conv2d...')
        self.best_by_targetkey = {}

        # best according to (model, workload_key).  e.g. ('1080ti', 'conv2d...'))
        self.best_by_model = {}

        # find best records and build the index
        for inp, res in zip(inputs, results):
            if res.error_no != 0:
                continue

            # use target keys in tvm target system as key to build best map
            for target_key in inp.task.target.keys:
                key = (target_key, inp.task.workload_key)
                if key not in self.best_by_targetkey:
                    self.best_by_targetkey[key] = (inp, res)
                else:
                    _, other_res = self.best_by_targetkey[key]
                    if np.mean([x.value for x in other_res.costs]) > \
                            np.mean([x.value for x in res.costs]):
                        self.best_by_targetkey[key] = (inp, res)

            # use model as key to build best map
            key = (inp.task.target.model, inp.task.workload_key)
            if key not in self.best_by_model:
                if inp.task.target.model != 'unknown':
                    self.best_by_model[key] = (inp, res)
            else:
                _, other_res = self.best_by_model[key]
                if np.mean([x.value for x in other_res.costs]) > \
                        np.mean([x.value for x in res.costs]):
                    self.best_by_model[key] = (inp, res)

    def write_best(self, filename: str):
        visited = set()
        inputs, results = [], []
        for inp, res in self.best_by_targetkey.values():
            if inp in visited:
                continue
            visited.add(inp)
            inputs.append(inp)
            results.append(res)
        write_measure_records_to_file(filename, inputs, results)


############################################################
######################  Plot Utilities  ####################
############################################################

def to_str_round(x, decimal=6):
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)) or isinstance(x, np.ndarray):
        return "[" + ", ".join([to_str_round(y, decimal=decimal)
                                for y in x]) + "]"
    if isinstance(x, dict):
        return str({k: eval(to_str_round(v)) for k, v in x.items()})
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        format_str = "%%.%df" % decimal
        return format_str % x
    raise ValueError("Invalid value: " + str(x))

############################## Curve
def max_curve(raw_curve):
    """Return b[i] = max(a[:i]) """
    ret = []
    cur_max = -np.inf
    for x in raw_curve:
        cur_max = max(cur_max, x)
        ret.append(cur_max)
    return ret

def min_curve(raw_curve):
    """Return b[i] = min(a[:i]) """
    ret = []
    cur_min = np.inf
    for x in raw_curve:
        cur_min = min(cur_min, x)
        ret.append(cur_min)
    return ret

def mean_curve(raw_curve, window_size=None):
    """Return b[i] = mean(a[:i]) """
    ret = []
    mean = 0
    if window_size is None:
        for i, x in enumerate(raw_curve):
            mean = (mean * i + x) / (i + 1)
            ret.append(mean)
    else:
        for i, x in enumerate(raw_curve):
            if i >= window_size:
                mean = (mean * window_size + x - raw_curve[i - window_size]) / window_size
            else:
                mean = (mean * i + x) / (i + 1)
            ret.append(mean)
    return ret

def plot_max_curve(xs_list, ys_list, legends, out_file, x_label, y_label="GFLOPS",
                   x_min=0, x_max=None, curve_type='max',
                   fmts=None, colors=None, title='Best Performance', figure_size=None,
                   mean_window_size=None):
    colors = colors or ['C%d' % i for i in range(10)]
    fig, ax = plt.subplots()
    fontsize = 19
    if curve_type == 'max':
        curve_func = max_curve
    elif curve_type == 'min':
        curve_func = min_curve
    elif curve_type == 'mean':
        curve_func = lambda curve : mean_curve(curve, mean_window_size)
    else:
        raise ValueError("Invalid curve type: %s" % curve_type)

    for i, (xs, ys) in enumerate(zip(xs_list, ys_list)):
        if fmts is None:
            plt.plot(xs, curve_func(ys), color=colors[i])
        else:
            plt.plot(xs, curve_func(ys), fmts[i], color=colors[i])

    plt.legend([x for x in legends if x], fontsize=fontsize)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    if title:
        plt.title(title, fontsize=fontsize)
    plt.xlim(left=x_min)
    if x_max is not None:
        plt.xlim(right=x_max)
    plt.ylim(bottom=0)
    if figure_size:
        fig.set_size_inches(figure_size)

    fig.savefig(out_file, bbox_inches='tight')
    print("Output to file %s" % out_file)

############################################################
##################### Cost Model Metric ####################
############################################################

def top_k(arr, k):
    """Return top k elements in an array"""
    return arr[np.argpartition(arr, -k)[-k:]]

def compute_r_squared(preds, labels):
    """Compute R squared value"""
    s_tot = np.sum(np.square(labels - np.mean(labels)))
    s_res = np.sum(np.square(labels - preds))
    return 1 - s_res / s_tot

def compute_rmse(preds, labels):
    """Compute RMSE (Rooted mean square error)"""
    return np.sqrt(np.mean(np.square(preds - labels)))

def compute_pairwise_comp_accuracy(preds, labels, group_sizes):
    """Compute the accuracy of pairwise comparision.

    The test set can contain samples from different tasks.
    We compute pairwise accuracy inside each task and then compute a weighted average over all tasks.
    The weight of a task is equal to the number of samples belonging to the task.
    """
    def compute_one_group(preds, labels):
        correct_ct = wrong_ct = 0
        predicted_order = np.argsort(preds)

        for i in range(len(preds)):
            for j in range(i+1, len(preds)):
                idx1 = predicted_order[i]
                idx2 = predicted_order[j]

                if labels[idx1] <= labels[idx2]:
                    correct_ct += 1
                else:
                    wrong_ct += 1
            accuracy = correct_ct * 1.0 / (correct_ct + wrong_ct)

        return accuracy

    accuracy_list = []
    begin = 0
    for i in range(len(group_sizes)):
        accuracy_list.append(compute_one_group(
            preds[begin:begin + group_sizes[i]], labels[begin:begin + group_sizes[i]]))
        begin += group_sizes[i]

    weights = group_sizes / np.sum(group_sizes)
    return np.sum(w * a for w, a in zip(weights, accuracy_list))

def compute_top_k_recall(preds, labels, group_sizes, top_k):
    """Compute recall of top-k@k = |(top-k according to prediction) intersect (top-k according to ground truth)| / k.

    The test set can contain samples from different tasks.
    We compute top-k recall inside each task and then compute a weighted average over all tasks.
    The weight of a task is equal to the number of samples belonging to the task.
    """
    def compute_one_group(preds, labels):
        real_top_k = set(np.argsort(labels)[:top_k])
        predicted_top_k = set(np.argsort(preds)[:top_k])
        recalled = real_top_k.intersection(predicted_top_k)
        return 1.0 * len(recalled) / top_k

    recall_list = []
    begin = 0
    for i in range(len(group_sizes)):
        recall_list.append(compute_one_group(
            preds[begin:begin + group_sizes[i]], labels[begin:begin + group_sizes[i]]))
        begin += group_sizes[i]

    weights = group_sizes / np.sum(group_sizes)
    return np.sum(w * a for w, a in zip(weights, recall_list))

