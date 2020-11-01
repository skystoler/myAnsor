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
"""TVM operator fully connected compute."""
from tvm import te, ansor
from .. import tag

def dense(data, weight, bias=None, out_dtype=None, ansor_kernel_layout=""):
    """The default implementation of dense in topi.

    Parameters
    ----------
    data : tvm.te.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.te.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.te.Tensor, optional
        1-D with shape [out_dim]

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [batch, out_dim]
    """
    assert len(data.shape) == 2, \
        "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    if out_dtype is None:
        out_dtype = data.dtype
    batch, in_dim = data.shape

    if ansor_kernel_layout:
        # infer shape for the rewritten layout
        if len(weight.shape) >= 6:
            # For cpu tile structure SSRSRS
            base = len(weight.shape) - 6
            red_dim = weight.shape[2 + base] * weight.shape[4 + base]
            out_dim = weight.shape[3 + base] * weight.shape[5 + base]
            for i in range(base + 2):
                out_dim *= weight.shape[i]
        else:
            out_dim, red_dim = weight.shape
    else:
        out_dim, red_dim = weight.shape
    assert in_dim == red_dim

    k = te.reduce_axis((0, in_dim), name='k')
    matmul = te.compute((batch, out_dim), \
                        lambda i, j: te.sum(data[i, k].astype(out_dtype) * \
                                            weight[j, k].astype(out_dtype), axis=k), \
                        name='T_dense', tag='dense', attrs={"layout_free_placeholders": [weight]})

    if ansor_kernel_layout:
        matmul = ansor.rewrite_compute_body(matmul, weight, ansor_kernel_layout)

    if bias is not None:
        matmul = te.compute((batch, out_dim), \
                            lambda i, j: matmul[i, j] + bias[j].astype(out_dtype), \
                            tag=tag.BROADCAST)
    return matmul
