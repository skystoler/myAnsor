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

import itertools

import numpy as np
import scipy.sparse as sp


import tvm
from tvm.ir import IRModule
from tvm.contrib import cusparse
import tvm.runtime.ndarray
from tvm import te


def random_bsr_matrix(M, N, BS_R, BS_C, density, dtype="float32"):
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
        Y[r:r+BS_R,c:c+BS_C] = np.random.randn(BS_R, BS_C)
    s = sp.bsr_matrix(Y, blocksize=(BS_R, BS_C))
    assert s.data.shape == (num_blocks, BS_R, BS_C)
    assert s.data.size >= nnz
    assert s.indices.shape == (num_blocks, )
    assert s.indptr.shape == (M // BS_R + 1, )
    return s

def test_cusparse_bsr_sparse_dense():
    m = 312
    n = 768
    k = 132
    bs = 3
    dtype = "float32"
    itype = "int32"

    weights = random_bsr_matrix(n, k, bs, bs, 0.1)
    data = np.random.uniform(size=(m, k)).astype(np.float32)

    A = te.placeholder((m, k), name='A', dtype=dtype)
    B_data = te.placeholder(weights.data.shape, name='B_data', dtype=dtype)
    B_indices = te.placeholder(weights.indices.shape, name='B_indices', dtype=itype)
    B_indptr = te.placeholder(weights.indptr.shape, name='B_indptr', dtype=itype)
    C = cusparse.sparse_dense(A, B_data, B_indices, B_indptr)
    s = te.create_schedule(C.op)

    if not tvm.runtime.enabled("cuda"):
        print("skip because cuda is not enabled...")
        return
    if not tvm.get_global_func("tvm.contrib.cusparse.sparse_dense", True):
        print("skip because extern function is not available")
        return
    ctx = tvm.gpu()
    f = tvm.build(s, [A, B_data, B_indices, B_indptr, C], "cuda")
    a = tvm.nd.array(data.astype(dtype), ctx)
    b_data = tvm.nd.array(weights.data.astype(dtype), ctx)
    b_indices = tvm.nd.array(weights.indices.astype(itype), ctx)
    b_indptr = tvm.nd.array(weights.indptr.astype(itype), ctx)
    c = tvm.nd.array(np.zeros((m, n), dtype=dtype), ctx)
    f(a, b_data, b_indices, b_indptr, c)

    verified = data @ weights.transpose()
    tvm.testing.assert_allclose(verified, c.asnumpy(), rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_cusparse_bsr_sparse_dense()
