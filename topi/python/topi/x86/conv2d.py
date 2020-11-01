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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
# pylint: disable=no-value-for-parameter,import-outside-toplevel
"""Conv2D schedule on x86"""

import logging

import tvm
from tvm import te
from tvm import autotvm
from tvm import ansor
from .. import nn
from ..nn.conv2d import conv2d_infer_layout, _get_workload as _get_conv2d_workload
from ..nn.conv2d import unpack_NCHWc_to_nchw
from ..nn.depthwise_conv2d import _get_workload as _get_depthwise_conv2d_workload
from ..nn.util import get_pad_tuple, get_const_int
from ..nn.winograd_util import winograd_transform_matrices
from ..util import get_const_tuple, traverse_inline
from . import conv2d_avx_1x1, conv2d_avx_common

logger = logging.getLogger('topi')

def _get_default_config(cfg, data, kernel, strides, padding, out_dtype, is_depthwise=False,
                        layout='NCHW'):
    """
    Get default schedule config for the workload
    """
    static_data_shape = []
    for dim in get_const_tuple(data.shape):
        if isinstance(dim, tvm.tir.Var):
            static_data_shape.append(1)
        else:
            static_data_shape.append(dim)
    data = te.placeholder(static_data_shape, dtype=data.dtype)
    if is_depthwise:
        wkl = _get_depthwise_conv2d_workload(data, kernel, strides, padding, out_dtype)
        from .depthwise_conv2d import _fallback_schedule
        _fallback_schedule(cfg, wkl)
    else:
        wkl = _get_conv2d_workload(data, kernel, strides, padding, out_dtype, layout)
        is_kernel_1x1 = wkl.hkernel == 1 and wkl.wkernel == 1
        if is_kernel_1x1:
            conv2d_avx_1x1._fallback_schedule(cfg, wkl)
        else:
            conv2d_avx_common._fallback_schedule(cfg, wkl)

@conv2d_infer_layout.register("cpu")
def _conv2d_infer_layout(workload, cfg):
    _, data, kernel, strides, padding, dilation, layout, _, dtype = workload
    batch_size, in_channel, in_height, in_width = data[1]
    out_channel, _, k_height, k_width = kernel[1]
    idxdiv = tvm.tir.indexdiv

    pt, pl, pb, pr = get_pad_tuple(padding, (k_height, k_width))
    out_height = idxdiv(in_height + pt + pb - k_height, strides[0]) + 1
    out_width = idxdiv(in_width + pl + pr - k_width, strides[1]) + 1
    tile_ic, tile_oc = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
    in_shape = (batch_size, idxdiv(in_channel, tile_ic), in_height, in_width, tile_ic)
    in_layout = "NCHW%dc" % tile_ic
    out_shape = (batch_size, idxdiv(out_channel, tile_oc), out_height, out_width, tile_oc)
    out_layout = "NCHW%dc" % tile_oc
    return ((in_shape, in_layout),), ((out_shape, out_layout),)

def schedule_conv2d_nhwc(outs):
    """Create schedule for conv2d_nhwc"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    output_op = outs[0].op

    def _callback(op):
        if 'conv2d_nhwc' in op.tag:
            conv = op.output(0)
            kernel = op.input_tensors[1]
            if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()

            data = op.input_tensors[0]
            data_pad = None
            if isinstance(data.op, tvm.te.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            n_pad, h_pad, w_pad, c_pad = data_pad.op.axis
            pad_fused = s[data_pad].fuse(n_pad, h_pad)
            s[data_pad].parallel(pad_fused)
            C = conv
            n, h, w, c = C.op.axis
            s[C].vectorize(c)

            O = output_op.output(0)
            if len(O.op.axis) == 4: # schedule bias + bn + relu
                n, h, w, c = O.op.axis
                fused = s[O].fuse(n, h, w)
                s[O].parallel(fused)
                channels = int(O.shape[-1])
                if channels % 64 == 0:
                    c, ci = s[O].split(c, 64)
                    s[O].vectorize(ci)
                if C != O:
                    s[C].compute_at(s[O], c)

    traverse_inline(s, output_op, _callback)
    return s

def conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype):
    layout = "NCHW"
    packed_out = conv2d_NCHWc(data, kernel, strides, padding, dilation,
                              layout, layout, out_dtype)
    return unpack_NCHWc_to_nchw(packed_out, out_dtype)

def schedule_conv2d_nchw(outs):
    """Create schedule for tensors"""
    return schedule_conv2d_NCHWc(outs)

def _pack_data(cfg, data, kernel):
    n, _, ih, iw = get_const_tuple(data.shape)
    oc, ic, kh, kw = get_const_tuple(kernel.shape)
    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]

    ic_chunk = ic // ic_bn
    oc_chunk = oc // oc_bn

    data = te.compute((n, ic_chunk, ih, iw, ic_bn),
                      lambda bs, c, h, w, vc: data[bs, c*ic_bn + vc, h, w],
                      name="data_vec")

    kernel = te.compute(
        (oc_chunk, ic_chunk, kh, kw, ic_bn, oc_bn),
        lambda occ, icc, k_h, k_w, icb, ocb:
        kernel[occ * oc_bn + ocb, icc * ic_bn + icb, k_h, k_w],
        name="kernel_vec")

    return data, kernel

@autotvm.register_topi_compute("conv2d_NCHWc.x86")
def conv2d_NCHWc(cfg, data, kernel, strides, padding, dilation, layout, out_layout, out_dtype):
    """Compute conv2d with NCHWc layout."""
    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    if len(data.shape) == 5:
        n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
        oc_chunk, ic_chunk_group, kernel_height, kernel_width, _, oc_bn = \
            get_const_tuple(kernel.shape)
        in_channel = ic_chunk * ic_bn
        num_filter = oc_chunk * oc_bn
    else:
        n, in_channel, ih, iw = get_const_tuple(data.shape)
        num_filter, _, kernel_height, kernel_width = get_const_tuple(kernel.shape)

    # Define autotvm tuning space
    is_kernel_1x1 = kernel_height == 1 and kernel_width == 1
    pt, pl, pb, pr = get_pad_tuple(padding, (kernel_height, kernel_width))
    sh, sw = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    oh = (ih - kernel_height + pt + pb) // sh + 1
    ow = (iw - kernel_width + pl + pr) // sw + 1

    cfg.define_split("tile_ic", in_channel, num_outputs=2)
    cfg.define_split("tile_oc", num_filter, num_outputs=2)
    cfg.define_split("tile_ow", ow, num_outputs=2, filter=lambda y: y.size[-1] <= 64,
                     policy="verbose")
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if oh > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])

    # If no config was set, we can fallback to default config.
    if cfg.is_fallback:
        _get_default_config(cfg, te.placeholder((n, in_channel, ih, iw), dtype=data.dtype),
                            te.placeholder((num_filter, in_channel, kernel_height, kernel_width),
                                           dtype=kernel.dtype),
                            strides, padding, out_dtype)

    # Pack data if raw 4-D data is provided.
    # This can only happen when autotuning.
    if len(data.shape) == 4:
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # Directly use modified data layout placeholder.
            dshape = (n, in_channel // cfg["tile_ic"].size[-1],
                      ih, iw, cfg["tile_ic"].size[-1])
            data = tvm.te.placeholder(dshape, data.dtype, name="data")
            kshape = (num_filter // cfg["tile_oc"].size[-1],
                      in_channel // cfg["tile_ic"].size[-1],
                      kernel_height, kernel_width,
                      cfg["tile_ic"].size[-1],
                      cfg["tile_oc"].size[-1])
            kernel = tvm.te.placeholder(kshape, kernel.dtype, name="kernel")
        else:
            data, kernel = _pack_data(cfg, data, kernel)

    return nn.conv2d_NCHWc(data,
                           kernel,
                           strides,
                           padding,
                           dilation,
                           layout,
                           out_layout,
                           out_dtype)

@autotvm.register_topi_schedule("conv2d_NCHWc.x86")
def schedule_conv2d_NCHWc(cfg, outs):
    """Create schedule for tensors"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if 'conv2d_NCHWc' in op.tag:
            conv_out = op.output(0)
            kernel_vec = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]

            args = [s, cfg, data_vec, kernel_vec, conv_out, outs[0]]
            _, _, kh, kw, _, _, = get_const_tuple(kernel_vec.shape)
            if kh == 1 and kw == 1:
                conv2d_avx_1x1._schedule_conv_NCHWc(*args)
            else:
                conv2d_avx_common._schedule_conv_NCHWc(*args)

    traverse_inline(s, outs[0].op, _callback)
    return s


# FIXME - https://github.com/apache/incubator-tvm/issues/4122
# _declaration_conv_nhwc_pack expects kernel layout to be HWOI. However, the tests use HWIO
# layout. Commenting until we have clarity about the nhwc_pack implementation from the author.
# elif layout == 'NHWC' and kh == 1 and kw == 1 and kernel.dtype == "int8":
#     if cfg.is_fallback:
#         _get_default_config(cfg, data, kernel, strides, padding, out_dtype, False, layout)
#     # specialize for INT8 1X1 conv on X86
#     return conv2d_avx_1x1._declaration_conv_nhwc_pack(cfg, data, kernel, strides,
#                                                       padding, dilation, out_dtype)

def _conv2d_nhwc_winograd_impl(input, weight, strides, padding, dilation, out_dtype, tile_size, pre_computed=False, ansor_kernel_layout=""):
    """Conv2D NHWC Winograd implementation.

    Parameters
    ----------
    input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    filter : tvm.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]

    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    out_dtype : str, optional
        Specifies the output data type.

    tile_size : int
        The size of the tile to use for the Winograd filter

    pre_computed: bool
        Whether the kernel is precomputed

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    N, H, W, CI = get_const_tuple(input.shape)
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    assert (dilation_h, dilation_w) == (1, 1), "Does not support dilation"
    if not pre_computed:
        KH, KW, CI, CO = get_const_tuple(weight.shape)
    else:
        if ansor_kernel_layout:
            if len(weight.shape) >= 14:
                # For cpu tile structure SSRSRS
                base = len(weight.shape) - 14
                H_CAT = get_const_int(weight.shape[0 + base] * weight.shape[3 + base] *
                                        weight.shape[7 + base] * weight.shape[11 + base])
                W_CAT = get_const_int(weight.shape[1 + base] * weight.shape[4 + base] *
                                        weight.shape[8 + base] * weight.shape[12 + base])
                CO = get_const_int(weight.shape[2 + base] * weight.shape[5 + base] *
                                     weight.shape[9 + base] * weight.shape[13 + base])
                CI = get_const_int(weight.shape[6 + base] * weight.shape[10 + base])
                assert base % 3 == 0
                for i in range(base // 3):
                    H_CAT *= get_const_int(weight.shape[i * 3])
                    W_CAT *= get_const_int(weight.shape[i * 3 + 1])
                    CO *= get_const_int(weight.shape[i * 3 + 2])
            elif len(weight.shape) == 10:
                # For cpu tile structure SRS
                H_CAT = get_const_int(weight.shape[0] * weight.shape[3] * weight.shape[7])
                W_CAT = get_const_int(weight.shape[1] * weight.shape[4] * weight.shape[8])
                CO = get_const_int(weight.shape[2] * weight.shape[5] * weight.shape[9])
                CI = get_const_int(weight.shape[6])
            elif len(weight.shape) == 7:
                # For cpu tile structure SRS
                H_CAT = get_const_int(weight.shape[0] * weight.shape[4])
                W_CAT = get_const_int(weight.shape[1] * weight.shape[5])
                CO = get_const_int(weight.shape[2] * weight.shape[6])
                CI = get_const_int(weight.shape[3])
            elif len(weight.shape) == 4:
                H_CAT, W_CAT, CO, CI = get_const_tuple(weight.shape)
            else:
                raise ValueError("Unhandlede case for weight shape: " + str(weight))
        else:
            assert len(weight.shape) == 4, len(weight.shape)
            H_CAT, W_CAT, CO, CI = get_const_tuple(weight.shape)
        KH, KW = H_CAT - tile_size + 1, W_CAT - tile_size + 1
    pad_t, pad_l, pad_d, pad_r = get_pad_tuple(padding, weight)
    HPAD = pad_t + pad_d
    WPAD = pad_l + pad_r
    HSTR, WSTR = (strides, strides) if isinstance(strides, int) else strides
    assert HSTR == 1 and WSTR == 1 and KH == 3 and KW == 3

    data_pad = nn.pad(input, (0, pad_t, pad_l, 0), (0, pad_d, pad_r, 0), name="data_pad")

    r = KW
    m = tile_size
    alpha = m + r - 1
    A, B, G = winograd_transform_matrices(m, r, out_dtype)

    H = (H + HPAD - KH) // HSTR + 1
    W = (W + WPAD - KW) // WSTR + 1
    nH, nW = (H + m - 1) // m, (W + m - 1) // m
    P = N * nH * nW
    r_kh = te.reduce_axis((0, KH), name='r_kh')
    r_kw = te.reduce_axis((0, KW), name='r_kw')
    if not pre_computed:
        kernel_pack = te.compute((alpha, alpha, CO, CI), lambda eps, nu, co, ci:
                                  te.sum(weight[r_kh][r_kw][ci][co] *
                                         G[eps][r_kh] * G[nu][r_kw],
                                         axis=[r_kh, r_kw]), name='kernel_pack')
    else:
        kernel_pack = weight

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod
    # pack input tile
    input_tile = te.compute((alpha, alpha, P, CI), lambda eps, nu, p, ci:
                             data_pad[idxdiv(p, (nH * nW))][idxmod(idxdiv(p, nW), nH) * m + eps]
                                     [idxmod(p, nW) * m + nu][ci], name='input_tile')

    # transform data
    r_a = te.reduce_axis((0, alpha), 'r_a')
    r_b = te.reduce_axis((0, alpha), 'r_b')
    data_pack = te.compute((alpha, alpha, P, CI), lambda eps, nu, p, ci:
                            te.sum(input_tile[r_a][r_b][p][ci] * B[r_a][eps] * B[r_b][nu],
                                    axis=[r_a, r_b]), name='data_pack',
                            attrs={"ansor_simplify_const_tensor_indices": ["eps", "nu", "r_a", "r_b"]})

    # do batch gemm
    ci = te.reduce_axis((0, CI), name='ci')
    bgemm = te.compute((alpha, alpha, P, CO), lambda eps, nu, p, co:
                        te.sum(data_pack[eps][nu][p][ci] *
                               kernel_pack[eps][nu][co][ci],
                               axis=[ci]), name='bgemm',
                               attrs={"layout_free_placeholders": [kernel_pack],
                                      "ansor_task_scheduler_tag": "conv2d_winograd_%d_%d" % (r, m)})
    if ansor_kernel_layout != "":
        bgemm = ansor.rewrite_compute_body(bgemm, kernel_pack, ansor_kernel_layout)

    # inverse transform
    r_a = te.reduce_axis((0, alpha), 'r_a')
    r_b = te.reduce_axis((0, alpha), 'r_b')
    inverse = te.compute((m, m, P, CO), lambda vh, vw, p, co:
                          te.sum(bgemm[r_a][r_b][p][co] * A[r_a][vh] * A[r_b][vw],
                                  axis=[r_a, r_b]), name='inverse',
                          attrs={"ansor_simplify_const_tensor_indices": ["vh", "vw", "r_a", "r_b"]})

    # output
    output = te.compute((N, H, W, CO), lambda n, h, w, co:
                         inverse[idxmod(h, m),
                                 idxmod(w, m),
                                 n * nH * nW + idxdiv(h, m) * nW + idxdiv(w, m),
                                 co],
                         name='conv2d_winograd')

    return output

def conv2d_nhwc_winograd(input, weight, strides, padding, dilation, out_dtype, pre_computed=False, ansor_kernel_layout=""):
    """Conv2D NHWC Winograd implementation.

    Parameters
    ----------
    input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    filter : tvm.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]

    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    out_dtype : str, optional
        Specifies the output data type.

    pre_computed: bool
        Whether the kernel is precomputed

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    tile_size = 4
    return _conv2d_nhwc_winograd_impl(input, weight, strides, padding, dilation, out_dtype, tile_size, pre_computed, ansor_kernel_layout)

def conv2d_nhwc_winograd_without_weight_transform(input, weight, strides, padding,
                                                  dilation, out_dtype, ansor_kernel_layout=""):
    return conv2d_nhwc_winograd(input, weight, strides, padding,
                                dilation, out_dtype, pre_computed=True,
                                ansor_kernel_layout=ansor_kernel_layout)
