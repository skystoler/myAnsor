/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file batch_matmul_weight_transpose.cc
 * \brief Transpose the layout of weight tensor of batch_matmul from BMK to BKM
 */

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/attrs/nn.h>
#include <functional>

namespace tvm {
namespace relay {

// Defined in src/relay/op/tensor/transform.cc
extern Expr MakeTranspose(Expr data, Array<Integer> axes);

class BatchMatmulWeightTransposeMutator : public ExprMutator {
 public:
  BatchMatmulWeightTransposeMutator(): ExprMutator() {}

  Expr VisitExpr_(const CallNode* n) {
    auto new_n = ExprMutator::VisitExpr_(n);

    const auto* call = new_n.as<CallNode>();
    if (call && call->op.as<OpNode>() &&
        call->op.as<OpNode>()->name == "nn.batch_matmul") {
      const auto* ref_attrs = call->attrs.as<BatchMatmulAttrs>();

      if (ref_attrs->weight_transposed) {
        auto new_attrs = make_object<BatchMatmulAttrs>();
        new_attrs->weight_transposed = false;

        const auto* trans = call->args[1].as<CallNode>();
        if (trans && trans->op.as<OpNode>() &&
            trans->op.as<OpNode>()->name == "transpose") {
          // Case 1: if the rhs is a transpose, then we can remove this transpose
          const auto* transpose_attrs = trans->attrs.as<TransposeAttrs>();
          if (transpose_attrs->axes.size() == 3 && transpose_attrs->axes[0] == 0 &&
              transpose_attrs->axes[1] == 2 && transpose_attrs->axes[2] == 1) {
            Array<Expr> new_args = {call->args[0], trans->args[0]};
            new_n = Call(call->op, new_args, Attrs(new_attrs));
          }
        } else if (call->args[1]->IsInstance<ConstantNode>()) {
          // Case 2: if the rhs is a constant, then we can add a transpose and
          // rely on FoldConstant to transform the weight
          Expr transposed_arg = MakeTranspose(call->args[1], Array<Integer>{0, 2, 1});
          Array<Expr> new_args = {call->args[0], transposed_arg};
          new_n = Call(call->op, new_args, Attrs(new_attrs));
        }
      }
    }
    return new_n;
  }
};

Expr BatchMatmulWeightTranspose(const Expr& expr) {
  return BatchMatmulWeightTransposeMutator().Mutate(expr);
}

namespace transform {

Pass BatchMatmulWeightTranspose() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
      return Downcast<Function>(relay::BatchMatmulWeightTranspose(f));
  };
  return CreateFunctionPass(pass_func, 3, "BatchMatmulWeightTranspose", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.BatchMatmulWeightTranspose")
.set_body_typed(BatchMatmulWeightTranspose);

}  // namespace transform

}  // namespace relay
}  // namespace tvm

