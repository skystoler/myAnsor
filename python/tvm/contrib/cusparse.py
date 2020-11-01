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
"""External function interface to cuBLAS libraries."""
import tvm
from tvm import te


def sparse_dense(data, weight_data, weight_indices, weight_indptr):
    """
    Computes sparse-dense matrix multiplication of `data` and
    `(weight_data, weight_indices, weight_indptr).T`

    Parameters
    ----------
    data : tvm.te.Tensor
        2-D with shape [M, K], float32

    weight_data : tvm.te.Tensor
        1-D with shape [nnz] (CSR) or
        3-D with shape [num_blocks, bs_r, bs_c] (BSR)

    weight_indices : tvm.te.Tensor
        1-D with shape [nnz] (CSR) or
        1-D with shape [num_blocks] (BSR)

    weight_indptr : tvm.te.Tensor
        1-D with shape [N + 1] (CSR) or
        1-D with shape [(N + 1) // bs_r] (BSR)

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [M, N]
    """
    m = data.shape[0]
    n = (weight_indptr.shape[0] - 1) * weight_data.shape[1]
    return te.extern(
        (m, n), [data, weight_data, weight_indices, weight_indptr],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.cusparse.sparse_dense",
            ins[0], ins[1], ins[2], ins[3], outs[0]), dtype=data.dtype, name="C")
