#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <dlpack/dlpack.h>
#include <dmlc/thread_local.h>

#include <iostream>

#include "../../cuda/cuda_common.h"
#include "../cblas/gemm_common.h"

namespace tvm {
namespace contrib {
void CHECK_CUSPARSE_ERROR(cusparseStatus_t status) {
  if (status != CUSPARSE_STATUS_SUCCESS) {
    std::cerr << "ERROR" << std::endl;
    exit(1);
  }
}

// Cache cusparse handle between invocations
struct CuSparseThreadEntry {
  CuSparseThreadEntry() { CHECK_CUSPARSE_ERROR(cusparseCreate(&handle)); }

  ~CuSparseThreadEntry() {
    if (handle) {
      cusparseDestroy(handle);
      handle = 0;
    }
  }
  typedef dmlc::ThreadLocalStore<CuSparseThreadEntry> CuSparseThreadStore;

  static CuSparseThreadEntry* ThreadLocal() {
    auto stream = runtime::CUDAThreadEntry::ThreadLocal()->stream;
    CuSparseThreadEntry* retval = CuSparseThreadStore::Get();
    CHECK_CUSPARSE_ERROR(cusparseSetStream(retval->handle, static_cast<cudaStream_t>(stream)));
    return retval;
  }

  cusparseHandle_t handle{nullptr};
};

template <class T>
inline T* Data(DLTensor* tensor) {
  return reinterpret_cast<T*>(static_cast<char*>(tensor->data) + tensor->byte_offset);
};

void cusparsebsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA,
                   cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const float* alpha,
                   const cusparseMatDescr_t descrA, const float* bsrValA, const int* bsrRowPtrA,
                   const int* bsrColIndA, int blockDim, const float* B, int ldb, const float* beta,
                   float* C, int ldc) {
  CHECK_CUSPARSE_ERROR(cusparseSbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA,
                                      bsrValA, bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C,
                                      ldc));
}

void cusparsebsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA,
                   cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const double* alpha,
                   const cusparseMatDescr_t descrA, const double* bsrValA, const int* bsrRowPtrA,
                   const int* bsrColIndA, int blockDim, const double* B, int ldb,
                   const double* beta, double* C, int ldc) {
  CHECK_CUSPARSE_ERROR(cusparseDbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA,
                                      bsrValA, bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C,
                                      ldc));
}

template <class T>
void sparse_dense(TVMArgs args) {
  // Compute B * A^T with dense B and sparse A
  DLTensor* B = args[0];
  DLTensor* A_data = args[1];
  DLTensor* A_indices = args[2];
  DLTensor* A_indptr = args[3];
  DLTensor* C = args[4];

  CHECK_EQ(A_data->ndim, 3);
  CHECK_EQ(A_indices->ndim, 1);
  CHECK_EQ(A_indptr->ndim, 1);
  CHECK_EQ(B->ndim, 2);
  CHECK_EQ(C->ndim, 2);

  CHECK_EQ(ElementStride(A_data), 1);
  CHECK_EQ(ElementStride(A_indices), 1);
  CHECK_EQ(ElementStride(A_indptr), 1);
  CHECK_EQ(ElementStride(B), 1);
  CHECK_EQ(ElementStride(C), 1);

  // C can never be transposed.
  CHECK(!IsInPlaceTransposed(C));

  // Reversed strides indicates an in-place transpose operation.
  auto transb = IsInPlaceTransposed(B) ? false : true;

  CHECK(TypeMatch(A_indices->dtype, kDLInt, sizeof(int) * 8))
      << "Sparse matrix indices must be int, but are " << A_indices->dtype;
  CHECK(TypeMatch(A_indptr->dtype, kDLInt, sizeof(int) * 8))
      << "Sparse matrix column pointers must be int, but are " << A_indptr->dtype;

  CuSparseThreadEntry* entry_ptr = CuSparseThreadEntry::ThreadLocal();

  cusparseMatDescr_t descrA = nullptr;
  CHECK_CUSPARSE_ERROR(cusparseCreateMatDescr(&descrA));

  int bs_r = A_data->shape[1];
  int bs_c = A_data->shape[2];
  CHECK_EQ(bs_r, bs_c) << "cuSPARSE only supports square blocks";
  CHECK_GT(bs_r, 1) << "cuSPARSE only supports block size > 1";
  // A: (mb * b_r) x (kb * b_c)
  // B : (kb * b_c) x n
  int mb = A_indptr->shape[0] - 1;
  int m = mb * bs_r;
  int k = RowCount(B, transb);
  CHECK(k % bs_r == 0) << "Number of rows in B is not a multiple of the block size of A";
  int kb = k / bs_r;
  int nnzb = A_data->shape[0];
  T alpha = 1;
  T beta = 0;
  cusparsebsrmm(entry_ptr->handle,
                CUSPARSE_DIRECTION_ROW,  // Are blocks row major?
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                // bsrmm is A * B, but we want A * B^T
                transb ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE, mb,
                ColumnCount(B, transb), kb, nnzb, &alpha, descrA, Data<T>(A_data),
                Data<int>(A_indptr), Data<int>(A_indices), bs_r, Data<T>(B), k, &beta, Data<T>(C),
                m);
  // We've computed A * B^T, so we need to transpose
  std::swap(C->shape[0], C->shape[1]);
  // Assume C has no strides
  C->strides = new int64_t[2];
  C->strides[0] = m;
  C->strides[1] = 1;
}

// matrix multiplication for row major
TVM_REGISTER_GLOBAL("tvm.contrib.cusparse.sparse_dense")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      auto dtype = static_cast<DLTensor*>(args[0])->dtype;
      if (TypeMatch(dtype, kDLFloat, sizeof(float) * 8)) {
        sparse_dense<float>(args);
      } else if (TypeMatch(dtype, kDLFloat, sizeof(double) * 8)) {
        sparse_dense<double>(args);
      } else {
        CHECK(false) << "cuSPARSE only supports float or double matrices, but you provided "
                     << dtype;
      }
    });
}  // namespace contrib
}  // namespace tvm
