import argparse
import time
import sys
import platform
import logging

import numpy as np

import tvm
from tvm import autotvm, te
import topi
import topi.math
from topi.util import traverse_inline, get_const_tuple
from utils import log_line, py_benchmark, BenchmarkRecord, measure_schedule
from utils import shape_dict

from topi.x86.batch_matmul import batch_matmul as decl_batch_matmul, schedule_batch_matmul
@autotvm.template('benchmark_batch_matmul_nkkm')
def batch_matmul_nkkm(B, N, M, K):
    X = te.placeholder((B, N, K), name='A')
    Y = te.placeholder((B, M, K), name='B')

    Z = decl_batch_matmul(X, Y)
    s = schedule_batch_matmul([Z])

    return s, [X, Y, Z]

@autotvm.template('benchmark_conv1d_nlc')
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

    # manual schedule
    O = output
    cfg = autotvm.get_config()
    cfg.define_split("tile_l", L, num_outputs=2)
    cfg.define_split("tile_co", CO, num_outputs=2)

    s = te.create_schedule([O.op])
    fused = s[padded].fuse(*s[padded].op.axis)
    s[padded].parallel(fused)
    b, l, co = s[O].op.axis
    rl, rc = s[O].op.reduce_axis
    lo, li = cfg["tile_l"].apply(s, O, l)
    coo, coi = cfg["tile_co"].apply(s, O, co)
    s[O].reorder(b, lo, coo, rl, rc, li, coi)
    outer = s[O].fuse(b, lo, coo)
    s[O].parallel(outer)
    s[O].unroll(li)
    s[O].vectorize(coi)

    return s, [inputs, weight, output]

from topi.x86.conv2d import conv2d_nchw as decl_conv2d_nchw, schedule_conv2d_nchw
@autotvm.template("benchmark_conv2d_nchw")
def conv2d_nchw(N, CI, H, W, CO, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    inputs = te.placeholder((N, CI, H, W), name='inputs')
    weight = te.placeholder((CO, CI//groups, kernel_size, kernel_size), name='weight')

    assert groups == 1
    output = decl_conv2d_nchw(inputs, weight, stride, padding, dilation, 'float32')
    s = schedule_conv2d_nchw([output])

    if autotvm.GLOBAL_SCOPE.in_tuning:
        queue = [output]
        while len(queue) > 0:
            now = queue.pop()
            if now.op.name == 'data':
                inputs = now
            elif now.op.name == 'kernel':
                weight = now
            for x in now.op.input_tensors:
                queue.append(x)

    return s, [inputs, weight, output]

@autotvm.template("benchmark_conv2d_nhwc")
def conv2d_nhwc(N, H, W, CI, CO, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    inputs = te.placeholder((N, H, W, CI), name='inputs')
    weight = te.placeholder((kernel_size, kernel_size, CI//groups, CO), name='weight')
    batch_size, in_h, in_w, in_channel = get_const_tuple(inputs.shape)
    k_h, k_w, channel_per_group, out_channel = get_const_tuple(weight.shape)
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

    # manual schedule
    O = output
    cfg = autotvm.get_config()
    cfg.define_split("tile_h", out_h, num_outputs=2)
    cfg.define_split("tile_w", out_w, num_outputs=2)
    cfg.define_split("tile_co", out_channel, num_outputs=2)

    s = te.create_schedule([O.op])
    fused = s[padded].fuse(*s[padded].op.axis)
    s[padded].parallel(fused)
    n, h, w, co = s[O].op.axis
    rh, rw, rc = s[O].op.reduce_axis
    ho, hi = cfg["tile_h"].apply(s, O, h)
    wo, wi = cfg["tile_w"].apply(s, O, w)
    coo, coi = cfg["tile_co"].apply(s, O, co)

    s[O].reorder(n, ho, wo, coo, rh, rw, rc, hi, wi, coi)
    outer = s[O].fuse(n, ho, wo, coo)
    s[O].parallel(outer)
    s[O].unroll(wi)
    s[O].vectorize(coi)

    return s, [inputs, weight, output]

from topi.x86.conv3d import conv3d_ncdhw as decl_conv3d_ncdhw, schedule_conv3d_ncdhw
@autotvm.template("benchmark_conv3d_ncdhw")
def conv3d_ncdhw(N, D, H, W, CI, CO, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    inputs = te.placeholder((N, CI, D, H, W))
    weight = te.placeholder((CO, CI//groups, kernel_size, kernel_size, kernel_size))

    assert groups == 1

    output = decl_conv3d_ncdhw(inputs, weight, (stride, stride, stride),
            (padding, padding, padding), dilation, 'float32')
    s = schedule_conv3d_ncdhw([output])

    return s, [inputs, weight, output]

from topi.x86.depthwise_conv2d import depthwise_conv2d_nchw as decl_depthwise_conv2d_nchw, schedule_depthwise_conv2d_nchw
@autotvm.template("benchmark_depthwise_conv2d_nchw")
def depthwise_conv2d_nchw(N, H, W, C, kernel_size, stride=1, padding=0, dilation=1, factor=1):
    inputs = te.placeholder((N, C, H, W))
    weight = te.placeholder((C, factor, kernel_size, kernel_size))

    output = decl_depthwise_conv2d_nchw(inputs, weight, strides=(stride, stride), padding=(padding, padding), dilation=dilation, out_dtype='float32')
    s = schedule_depthwise_conv2d_nchw([output])

    if autotvm.GLOBAL_SCOPE.in_tuning:
        queue = [output]
        while len(queue) > 0:
            now = queue.pop()
            if now.op.name == 'data':
                inputs = now
            elif now.op.name == 'kernel':
                weight = now
            for x in now.op.input_tensors:
                queue.append(x)

    return s, [inputs, weight, output]

from topi.x86.conv2d_transpose import conv2d_transpose_nchw as decl_conv2d_transpose_nchw, schedule_conv2d_transpose_nchw
@autotvm.template("benchmark_conv2d_transpose_nchw")
def conv2d_transpose_nchw(N, H, W, CI, CO, kernel_size, stride=1, padding=0):
    inputs = te.placeholder((N, CI, H, W), name='inputs')
    weight = te.placeholder((CI, CO, kernel_size, kernel_size), name='weight')

    output = decl_conv2d_transpose_nchw(inputs, weight, strides=(stride, stride), padding=(padding, padding), out_dtype='float32')
    s = schedule_conv2d_transpose_nchw([output])

    if autotvm.GLOBAL_SCOPE.in_tuning:
        queue = [output]
        while len(queue) > 0:
            now = queue.pop()
            if now.op.name == 'data':
                inputs = now
            elif now.op.name == 'kernel':
                weight = now
            for x in now.op.input_tensors:
                queue.append(x)

    return s, [inputs, weight, output]

@autotvm.template("benchmark_conv2d_capsule_nhwijc")
def conv2d_capsule_nhwijc(N, H, W, CI, CO, kernel_size, stride=1, padding=0, capsule_size=4):
    inputs = te.placeholder((N, H, W, capsule_size, capsule_size, CI), name='inputs')
    weight = te.placeholder((kernel_size, kernel_size, capsule_size, capsule_size, CI, CO), name='weight')
    batch_size, in_h, in_w, _, _, in_channel = get_const_tuple(inputs.shape)
    k_h, k_w, _, _, _, out_channel = get_const_tuple(weight.shape)

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

    # manual schedule
    O = output
    cfg = autotvm.get_config()
    cfg.define_split("tile_h", out_h, num_outputs=2)
    cfg.define_split("tile_w", out_w, num_outputs=2)
    cfg.define_split("tile_cap_i", capsule_size, num_outputs=2)
    cfg.define_split("tile_cap_j", capsule_size, num_outputs=2)
    cfg.define_split("tile_co", out_channel, num_outputs=2)

    s = te.create_schedule([O.op])
    fused = s[padded].fuse(*s[padded].op.axis)
    s[padded].parallel(fused)
    n, h, w, cap_i, cap_j, co = s[O].op.axis
    rh, rw, cap_k, rc = s[O].op.reduce_axis
    ho, hi = cfg["tile_h"].apply(s, O, h)
    wo, wi = cfg["tile_w"].apply(s, O, w)
    cap_io, capii = cfg["tile_cap_i"].apply(s, O, cap_i)
    cap_jo, capji = cfg["tile_cap_j"].apply(s, O, cap_j)
    coo, coi = cfg["tile_co"].apply(s, O, co)

    s[O].reorder(n, ho, wo, cap_io, cap_jo, coo, rh, rw, cap_k, rc, hi, wi, capii, capji, coi)
    outer = s[O].fuse(n, ho, wo, cap_io, cap_jo, coo)
    s[O].parallel(outer)
    s[O].unroll(capji)
    s[O].vectorize(coi)

    return s, [inputs, weight, output]


def _schedule_reduce(sch, op, is_idx_reduce=False):
    if is_idx_reduce:
        real_out = op.output(0)
        fused = sch[real_out].fuse(*sch[real_out].op.axis)
        out = op.input_tensors[0]
    else:
        out = op.output(0)

    const_shape = True
    out_shape = get_const_tuple(out.shape)
    for d in out_shape:
        if not isinstance(d, int):
            const_shape = False
            break

    if const_shape:
        naxes = len(sch[out].op.axis)
        parallelism = 1
        fuse_axes = []
        # We choose a heuristic number 128 to limit the maximum parallelism
        while len(fuse_axes) < naxes and parallelism < 128:
            ivar = sch[out].op.axis[len(fuse_axes)]
            parallelism *= int(ivar.dom.extent)
            fuse_axes.append(ivar)
        fused = sch[out].fuse(*fuse_axes)
        sch[out].parallel(fused)
    else:
        if len(sch[out].op.axis) >= 5:
            # avoid too many parallelism
            fused = sch[out].fuse(sch[out].op.axis[0], sch[out].op.axis[1], sch[out].op.axis[2])
            sch[out].parallel(fused)
        else:
            fused = sch[out].fuse(*sch[out].op.axis)
            sch[out].parallel(fused)

from topi.x86.reduction import schedule_reduce as x86_schedule_reduce
@autotvm.template("benchmark_norm_bmn")
def norm_bmn(B, M, N):
    A = te.placeholder((B, M, N), name='A')
    i = te.reduce_axis((0, M))
    j = te.reduce_axis((0, N))
    C = te.compute((B,), lambda b: te.sum(A[b][i][j] * A[b][i][j], axis=[i, j]), name='C', tag='comm_reduce')
    #D = te.compute((B,), lambda b: tvm.sqrt(C[b]), name='D')

    s = x86_schedule_reduce([C])
    return s, [A, C]

from topi.cuda.conv2d import conv2d_nchw as decl_conv2d_nchw_cuda, schedule_conv2d_nchw as schedule_conv2d_nchw_cuda
@autotvm.template("benchmark_conv2d_nchw_bn_relu")
def conv2d_nchw_bn_relu(N, H, W, CI, CO, kernel_size, strides, padding, dilation=1):
    data = te.placeholder((N, CI, H, W), name='data')
    kernel = te.placeholder((CO, CI, kernel_size, kernel_size), name='kernel')
    bias = te.placeholder((CO, 1, 1), name='bias')
    bn_scale = te.placeholder((CO, 1, 1), name='bn_scale')
    bn_offset = te.placeholder((CO, 1, 1), name='bn_offset')

    OH = (H + 2 * padding - (kernel_size - 1) * dilation - 1) // strides + 1
    OW = (W + 2 * padding - (kernel_size - 1) * dilation - 1) // strides + 1

    target = tvm.target.Target.current()
    cfg = autotvm.get_config()
    if 'cpu' in target.keys: 
        conv = decl_conv2d_nchw(data, kernel, strides, padding, dilation, 'float32')
        conv = te.compute((N, CO, OH, OW),
                           lambda i, j, k, l: conv[i, j, k, l] + bias[j, 0, 0],
                           name='bias_add', tag='elemwise')
        conv = te.compute((N, CO, OH, OW),
                           lambda i, j, k, l: conv[i, j, k, l] * bn_scale[j, 0, 0],
                           name='bn_add', tag='elemwise')
        conv = te.compute((N, CO, OH, OW),
                           lambda i, j, k, l: conv[i, j, k, l] + bn_offset[j, 0, 0],
                           name='bn_mul', tag='elemwise')
        out = topi.nn.relu(conv)
        s = schedule_conv2d_nchw([out])

        if autotvm.GLOBAL_SCOPE.in_tuning:
            queue = [out]
            while len(queue) > 0:
                now = queue.pop()
                if now.op.name == 'data':
                    data = now
                elif now.op.name == 'kernel':
                    kernel = now
                for x in now.op.input_tensors:
                    queue.append(x)
    else:
        conv = decl_conv2d_nchw_cuda(data, kernel, (strides, strides), (padding, padding),
                dilation, 'float32')
        conv = te.compute((N, CO, OH, OW),
                           lambda i, j, k, l: conv[i, j, k, l] + bias[j, 0, 0],
                           name='bias_add', tag='elemwise')
        conv = te.compute((N, CO, OH, OW),
                           lambda i, j, k, l: conv[i, j, k, l] * bn_scale[j, 0, 0],
                           name='bn_add', tag='elemwise')
        conv = te.compute((N, CO, OH, OW),
                           lambda i, j, k, l: conv[i, j, k, l] + bn_offset[j, 0, 0],
                           name='bn_mul', tag='elemwise')
        out = topi.nn.relu(conv)

        s = schedule_conv2d_nchw_cuda([out])

    return s, [data, kernel, bias, bn_offset, bn_scale, out]

from topi.cuda.batch_matmul import batch_matmul as decl_batch_matmul_cuda, schedule_batch_matmul as schedule_batch_matmul_cuda
@autotvm.template("benchmark_transpose_batch_matmul")
def transpose_batch_matmul(batch, seq_len, n_head, n_dim):
    target = tvm.target.Target.current()
    if 'cpu' in target.keys: 
        query = te.placeholder((batch, seq_len, n_head, n_dim), name='query')
        value = te.placeholder((batch, seq_len, n_head, n_dim), name='value')
        query_T = te.compute((batch, n_head, seq_len, n_dim),
                          lambda b, h, l, d: query[b, l, h, d], name="query_T")
        value_T = te.compute((batch, n_head, n_dim, seq_len),
                          lambda b, h, d, l: value[b, l, h, d], name="value_T")
        k = te.reduce_axis((0, n_dim), name='k')
        out = te.compute((batch, n_head, seq_len, seq_len), lambda b, h, i, j: te.sum(query_T[b][h][i][k] * value_T[b][h][k][j], axis=[k]), name='C')

        cfg = autotvm.get_config()
        s = te.create_schedule([out.op])
        cfg.define_split("tile_h", seq_len, num_outputs=2)
        cfg.define_split("tile_i", n_head, num_outputs=2)
        cfg.define_split("tile_j", n_dim, num_outputs=2)
        b, h, i, j = s[out].op.axis
        k, = s[out].op.reduce_axis
        ho, hi = cfg["tile_h"].apply(s, out, h)
        io, ii = cfg["tile_i"].apply(s, out, i)
        jo, ji = cfg["tile_j"].apply(s, out, j)
        s[out].reorder(b, ho, io, jo, k, hi, ii, ji)
        s[out].parallel(s[out].fuse(b, ho, io, jo))
        s[out].unroll(ii)
        s[out].vectorize(ji)
        def parallel_injective(X):
            s[X].parallel(s[X].fuse(s[X].op.axis[0], s[X].op.axis[1]))
        parallel_injective(query_T)
        parallel_injective(value_T)
    else:
        query = te.placeholder((batch, seq_len, n_head, n_dim), name='query')
        value = te.placeholder((batch, seq_len, n_head, n_dim), name='value')
        query_T = te.compute((batch * n_head, seq_len, n_dim),
                          lambda b, l, d: query[b // n_head, l, b % n_head, d], name="query_T")
        value_T = te.compute((batch * n_head, seq_len, n_dim),
                          lambda b, l, d: value[b // n_head, l, b % n_head, d], name="value_T")
        k = te.reduce_axis((0, n_dim), name='k')
        out = decl_batch_matmul_cuda(query_T, value_T)
        s = schedule_batch_matmul_cuda([out])
        s[query_T].compute_inline()
        s[value_T].compute_inline()

    return s, [query, value, out]


def softmax_common(x, axis, use_fast_exp):
    shape = x.shape
    if axis < 0:
        axis = len(shape) + axis
    if axis >= len(shape):
        ValueError("axis parameter should be less than input dim")

    k1 = te.reduce_axis((0, shape[axis]), name='k')
    k2 = te.reduce_axis((0, shape[axis]), name='k')

    def insert_reduce_index(indices, reduce_index):
        return indices[:axis] + (reduce_index,) + indices[axis:]

    def get_non_reduce_indices(indices):
        return tuple([var for (i, var) in enumerate(indices) if i != axis])

    def _compute_max(*indices):
        eval_range = insert_reduce_index(indices, k1)
        return tvm.te.max(x[eval_range], axis=k1)

    def _compute_delta(max_elem, *indices):
        non_reduce_indices = get_non_reduce_indices(indices)
        return x[indices] - max_elem[non_reduce_indices]

    def _compute_exp(max_elem, *indices):
        non_reduce_indices = get_non_reduce_indices(indices)
        return te.exp(x[indices] - max_elem[non_reduce_indices])

    def _compute_expsum(exp, *indices):
        eval_range = insert_reduce_index(indices, k2)
        return te.sum(exp[eval_range], axis=k2)

    def _normalize(exp, expsum, *indices):
        non_reduce_indices = get_non_reduce_indices(indices)
        return exp[indices] / expsum[non_reduce_indices]

    reduced_shape = tuple([dim for (i, dim) in enumerate(shape) if i != axis])
    max_elem = te.compute(reduced_shape, _compute_max, name='T_softmax_maxelem')

    if use_fast_exp:
        delta = te.compute(shape, lambda *indices: _compute_delta(max_elem, *indices),
                           name='T_softmax_delta')
        exp = topi.math.fast_exp(delta)
    else:
        exp = te.compute(shape, lambda *indices: _compute_exp(max_elem, *indices),
                         name='T_softmax_exp')
    expsum = te.compute(reduced_shape, lambda *indices: _compute_expsum(exp, *indices),
                        name='T_softmax_expsum')
    return te.compute(shape, lambda *indices: _normalize(exp, expsum, *indices),
                      name='T_softmax_norm', attrs={"axis" : axis})

@tvm.te.tag_scope(tag='softmax_output')
def softmax(x, axis=-1):
    return softmax_common(x, axis, False)

@tvm.te.tag_scope(tag='fast_softmax_output')
def fast_softmax(x, axis=-1):
    return softmax_common(x, axis, True)

""" Use this patch to fix the codegen for fast_softmax on cuda backend

diff --git a/src/target/source/codegen_c.cc b/src/target/source/codegen_c.cc
index 7c3c8309e..bfdf93284 100644
--- a/src/target/source/codegen_c.cc
+++ b/src/target/source/codegen_c.cc
@@ -630,11 +630,18 @@ void CodeGenC::VisitExpr_(const CallNode* op, std::ostream& os) {  // NOLINT(*)
       os << " == NULL)";
     } else if (op->op.same_as(builtin::reinterpret())) {
       // generate (*( TYPE *)(&(ARG)))
+      //os << "(*(";
+      //this->PrintType(op->dtype, os);
+      //os << " *)(&(";
+      //this->PrintExpr(op->args[0], os);
+      //os << ")))";
+      //
+      int ssa_scope = BeginScope();
+      std::string rhs = SSAGetID(PrintExpr(op->args[0]), op->args[0]->dtype);
       os << "(*(";
       this->PrintType(op->dtype, os);
-      os << " *)(&(";
-      this->PrintExpr(op->args[0], os);
-      os << ")))";
+      os << " *)(&(" << rhs << ")))";
+      EndScope(ssa_scope);
     } else if (op->op.same_as(builtin::isnan())) {
       os << "(";
       this->PrintExpr(op->args[0], os);
"""

from topi.cuda.injective import schedule_injective_from_existing as schedule_injective_from_existing_cuda
@autotvm.template("benchmark_transpose_batch_matmul_softmax")
def transpose_batch_matmul_softmax(batch, seq_len, n_head, n_dim):
    target = tvm.target.Target.current()
    if 'cpu' in target.keys: 
        query = te.placeholder((batch, seq_len, n_head, n_dim), name='query')
        value = te.placeholder((batch, seq_len, n_head, n_dim), name='value')
        query_T = te.compute((batch, n_head, seq_len, n_dim),
                          lambda b, h, l, d: query[b, l, h, d], name="query_T")
        value_T = te.compute((batch, n_head, n_dim, seq_len),
                          lambda b, h, d, l: value[b, l, h, d], name="value_T")
        k = te.reduce_axis((0, n_dim), name='k')
        out = te.compute((batch, n_head, seq_len, seq_len), lambda b, h, i, j: te.sum(query_T[b][h][i][k] * value_T[b][h][k][j], axis=[k]), name='C')
        sm = fast_softmax(out)

        cfg = autotvm.get_config()
        s = te.create_schedule([sm.op])
        cfg.define_split("tile_h", seq_len, num_outputs=2)
        cfg.define_split("tile_i", n_head, num_outputs=2)
        cfg.define_split("tile_j", n_dim, num_outputs=2)
        b, h, i, j = s[out].op.axis
        k, = s[out].op.reduce_axis
        ho, hi = cfg["tile_h"].apply(s, out, h)
        io, ii = cfg["tile_i"].apply(s, out, i)
        jo, ji = cfg["tile_j"].apply(s, out, j)
        s[out].reorder(b, ho, io, jo, k, hi, ii, ji)
        s[out].parallel(s[out].fuse(b, ho, io, jo))
        s[out].unroll(ii)
        s[out].vectorize(ji)
        def parallel_injective(X):
            s[X].parallel(s[X].fuse(s[X].op.axis[0], s[X].op.axis[1]))
        parallel_injective(query_T)
        parallel_injective(value_T)

        exp = sm.op.input_tensors[0]
        delta = exp.op.input_tensors[0]
        expsum = sm.op.input_tensors[1]
        max_elem = s[delta].op.input_tensors[1]
        axis = int(sm.op.attrs['axis'])

        # only parallelize outer dimensions up to axis
        outer_axes = [s[sm].op.axis[i] for i in range(0, axis)]
        fused_outer_axes = s[sm].fuse(*outer_axes)
        s[sm].parallel(fused_outer_axes)

        # move computations with the same outer dimensions under the same root
        s[max_elem].compute_at(s[sm], fused_outer_axes)
        s[expsum].compute_at(s[sm], fused_outer_axes)
        s[delta].compute_at(s[sm], fused_outer_axes)
        s[exp].compute_at(s[sm], fused_outer_axes)
    else:
        query = te.placeholder((batch, seq_len, n_head, n_dim), name='query')
        value = te.placeholder((batch, seq_len, n_head, n_dim), name='value')
        query_T = te.compute((batch * n_head, seq_len, n_dim),
                          lambda b, l, d: query[b // n_head, l, b % n_head, d], name="query_T")
        value_T = te.compute((batch * n_head, seq_len, n_dim),
                          lambda b, l, d: value[b // n_head, l, b % n_head, d], name="value_T")
        k = te.reduce_axis((0, n_dim), name='k')
        out = decl_batch_matmul_cuda(query_T, value_T)
        sm = fast_softmax(out)
        s = te.create_schedule([sm.op])

        def _schedule_batch_matmul(cfg, op):
            C = op.output(0)
            A, B = s[C].op.input_tensors
            _, M, N = get_const_tuple(C.shape)
            AA = s.cache_read(A, "shared", [C])
            AL = s.cache_read(AA, "local", [C])
            BB = s.cache_read(B, "shared", [C])
            BL = s.cache_read(BB, "local", [C])
            CC = s.cache_write(C, "local")

            b, y, x = s[C].op.axis
            k, = s[CC].op.reduce_axis
    
            cfg.define_split("tile_y", y, num_outputs=3)
            cfg.define_split("tile_x", x, num_outputs=3)
            cfg.define_split("tile_k", k, num_outputs=2)
            cfg.define_knob("auto_unroll_max_step", [8, 16, 32, 64])
            target = tvm.target.Target.current()
            cfg.define_knob("unroll_explicit", [0, 1])
    
            if cfg.is_fallback:
                y_bn = get_max_power2_factor(M, 64)
                x_bn = get_max_power2_factor(N, 64)
                y_nthreads = min(y_bn, 8)
                x_nthreads = min(x_bn, 8)
                cfg['tile_x'] = SplitEntity([-1, x_nthreads, x_bn // x_nthreads])
                cfg['tile_y'] = SplitEntity([-1, y_nthreads, y_bn // y_nthreads])
                cfg['tile_k'] = SplitEntity([-1, 8])
                cfg['auto_unroll_max_step'] = OtherOptionEntity(16)
    
            by, ty, yi = cfg["tile_y"].apply(s, C, y)
            bx, tx, xi = cfg["tile_x"].apply(s, C, x)
    
            thread_x = te.thread_axis("threadIdx.x")
            thread_y = te.thread_axis("threadIdx.y")
    
            s[C].reorder(b, by, bx, ty, tx, yi, xi)
            s[C].bind(b, te.thread_axis("blockIdx.z"))
            s[C].bind(by, te.thread_axis("blockIdx.y"))
            s[C].bind(bx, te.thread_axis("blockIdx.x"))
            s[C].bind(ty, thread_y)
            s[C].bind(tx, thread_x)
            s[C].pragma(yi, "auto_unroll_max_step", cfg['auto_unroll_max_step'].val)
            s[C].pragma(yi, 'unroll_explicit', cfg['unroll_explicit'].val)
    
            s[CC].compute_at(s[C], tx)
            _, yi, xi = s[CC].op.axis
            ko, ki = cfg["tile_k"].apply(s, CC, k)
            s[CC].reorder(ko, ki, yi, xi)
            s[CC].pragma(ki, "auto_unroll_max_step", cfg['auto_unroll_max_step'].val)
            s[CC].pragma(ki, 'unroll_explicit', cfg['unroll_explicit'].val)
    
            s[AA].compute_at(s[CC], ko)
            s[AL].compute_at(s[CC], ki)
            s[BB].compute_at(s[CC], ko)
            s[BL].compute_at(s[CC], ki)
            _, y, k = s[AA].op.axis
            ty, yi = s[AA].split(y, nparts=cfg["tile_y"].size[1])
            tx, ki = s[AA].split(k, nparts=cfg["tile_x"].size[1])
            s[AA].reorder(ty, tx, yi, ki)
            s[AA].bind(ty, thread_y)
            s[AA].bind(tx, thread_x)
            s[AA].pragma(yi, "auto_unroll_max_step", cfg['auto_unroll_max_step'].val)
            s[AA].pragma(yi, 'unroll_explicit', cfg['unroll_explicit'].val)
    
            _, x, k = s[BB].op.axis
            ty, xi = s[BB].split(x, nparts=cfg["tile_y"].size[1])
            tx, ki = s[BB].split(k, nparts=cfg["tile_x"].size[1])
            s[BB].bind(ty, thread_y)
            s[BB].bind(tx, thread_x)
            s[BB].reorder(ty, tx, xi, ki)
            s[BB].pragma(xi, "auto_unroll_max_step", cfg['auto_unroll_max_step'].val)
            s[BB].pragma(xi, 'unroll_explicit', cfg['unroll_explicit'].val)

        _schedule_batch_matmul(autotvm.get_config(), out.op)

        s[query_T].compute_inline()
        s[value_T].compute_inline()

        exp = sm.op.input_tensors[0]
        delta = exp.op.input_tensors[0]
        expsum = sm.op.input_tensors[1]
        max_elem = delta.op.input_tensors[1]

        s[exp].compute_inline()
        s[delta].compute_inline()

        for ten in [max_elem, expsum, sm]:
            s = schedule_injective_from_existing_cuda(s, ten)
        # print(tvm.lower(s, [query, value, out], simple_mode=True))
        # exit()

    return s, [query, value, out]

task_func_dict = {
    'GMM': (batch_matmul_nkkm, 'batch_matmul_nkkm'),
    'C1D': (conv1d_nlc, 'conv1d_nlc'),
    'C2D': (conv2d_nchw, 'conv2d_nchw'),
    'C3D': (conv3d_ncdhw, 'conv3d_ncdhw'),
    'GRP': (conv2d_nhwc, "conv2d_nhwc"),
    'DIL': (conv2d_nhwc, "conv2d_nhwc"),
    'DEP': (depthwise_conv2d_nchw, "depthwise_conv2d_nchw"),
    'T2D': (conv2d_transpose_nchw, "conv2d_transpose_nchw"),
    'CAP': (conv2d_capsule_nhwijc, "conv2d_capsule_nhwijc"),
    'NRM': (norm_bmn, "norm_bmn"),

    'conv2d_bn_relu': (conv2d_nchw_bn_relu, "conv2d_nchw_bn_relu"),
    'transpose_batch_matmul': (transpose_batch_matmul, "transpose_batch_matmul"),
    'transpose_batch_matmul_softmax': (transpose_batch_matmul_softmax, "transpose_batch_matmul_softmax"),
}

single_op_eval_wkls = ['C1D', 'C2D', 'C3D']
subgraph_eval_wkls = ['conv2d_bn_relu']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wkl", type=str)
    parser.add_argument("--batch-size", type=int, default=-1)
    parser.add_argument("--backend", type=str, choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument("--out-file", type=str, default='results.tsv')
    parser.add_argument("--n-trials", type=int, default=1200)
    parser.add_argument("--eval", action='store_true')
    args = parser.parse_args()

    # logging config (for printing tuning log to the screen)
    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

    measure_option = autotvm.measure_option(
            builder='local', runner=autotvm.LocalRunner(number=10, min_repeat_ms=300, timeout=30))

    backend = args.backend
    if backend == 'cpu':
        target = tvm.target.create('llvm -mcpu=core-avx2')
    elif backend == 'gpu':
        target = tvm.target.create('cuda')
    else:
        raise ValueError("Invalid target: " + backend)
    device = 'device_name'

    if args.wkl == 'op':
        wkl_meta_names = single_op_eval_wkls
    elif args.wkl == 'subgraph':
        wkl_meta_names = subgraph_eval_wkls
    elif args.wkl == 'all':
        wkl_meta_names = single_op_eval_wkls + subgraph_eval_wkls
    elif args.wkl is None:
        if backend == 'cpu':
            wkl_meta_names = single_op_eval_wkls + subgraph_eval_wkls
        else:
            wkl_meta_names = subgraph_eval_wkls
    else:
        wkl_meta_names = [args.wkl]

    if args.batch_size < 0:
        batch_sizes = [1, 16]
    else:
        batch_sizes = [args.batch_size]

    print("Workloads: %s" % wkl_meta_names)

    for wkl_meta_name in wkl_meta_names:
        func, func_name = task_func_dict[wkl_meta_name]
        log_file = '%s.log' % wkl_meta_name

        for batch_size in batch_sizes:
            for shape in shape_dict[wkl_meta_name]:
                if shape[0] == 1:
                    shape = list(shape)
                    shape[0] = batch_size
                    shape = tuple(shape)
    
                if args.eval:
                    with autotvm.apply_history_best(log_file):
                        with tvm.target.create(target):
                            s, bufs = func(*shape)
                            #print(tvm.lower(s, bufs, simple_mode=True))
                            cost = np.mean(measure_schedule(s, bufs, target))
                            workload_name = "%s%s" % (wkl_meta_name, shape)
                            print("%s\t%.3fms" % (workload_name, cost * 1e3))
                        log_line(BenchmarkRecord(device, backend, 'op', workload_name,
                                                 "AutoTVM", 'default', {"costs": [cost]}, time.time()),
                                                 args.out_file)
                else:
                    task_name = "benchmark_" + func_name
                    task = autotvm.task.create(task_name, args=shape, target=target)
                    tuner = autotvm.tuner.XGBTuner(task, feature_type='knob')
                    #tuner = autotvm.tuner.GATuner(task)
                    tuner.tune(n_trial=args.n_trials,
                               measure_option=measure_option,
                               callbacks=[autotvm.callback.log_to_file(log_file)])

