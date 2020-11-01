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

#include "kernel_layout_transform.h"

#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/te/operation.h>

#include <tuple>
#include <unordered_map>
#include <deque>
#include <functional>
#include <string>
#include <vector>

#include "pattern_util.h"
#include "../backend/compile_engine.h"


namespace tvm {
namespace relay {

// Two global variables for receiving layout information from python
std::deque<std::string> KernelLayoutTransformer::global_ori_layouts_queue;
std::deque<std::string> KernelLayoutTransformer::global_new_layouts_queue;


// Copy an Attrs but with a new ansor_kernel_layout filed.
template <typename T>
Attrs CopyAttrsWithNewLayout(const T* ptr, const std::string& layout) {
  auto n = make_object<T>(*ptr);
  n->ansor_kernel_layout = layout;
  return Attrs(n);
}

// Mutate ops in a function
class FuncMutator : public ExprMutator {
 public:
  FuncMutator(const std::deque<std::string>& ori_layouts_queue,
              const std::deque<std::string>& new_layouts_queue) :
      ExprMutator(), ori_layouts_queue_(ori_layouts_queue), new_layouts_queue_(new_layouts_queue) {}

  Expr VisitExpr_(const CallNode* n) {
    auto new_n = ExprMutator::VisitExpr_(n);

    const auto* call = new_n.as<CallNode>();
    if (call && call->op.as<OpNode>() &&
         (std::find(target_ops_.begin(), target_ops_.end(), n->op.as<OpNode>()->name) !=
             target_ops_.end()) &&
         !ori_layouts_queue_.empty() && !new_layouts_queue_.empty()) {

      // Pop a new layout from the queue
      const std::string ori_layout = ori_layouts_queue_.front();
      const std::string new_layout = new_layouts_queue_.front();
      ori_layouts_queue_.pop_front();
      new_layouts_queue_.pop_front();

      // Insert a new op to do layout transform. (This will be simplified by FoldConstant later).
      Expr updated_kernel = MakeKernelLayoutTransform(call->args[1], ori_layout, new_layout);
      Array<Expr> updated_args = {call->args[0], updated_kernel};

      // Update the attrs
      Attrs updated_attrs;
      if (auto pattr = call->attrs.as<Conv2DAttrs>()) {
        updated_attrs = CopyAttrsWithNewLayout(pattr, new_layout);
      } else if (auto pattr = call->attrs.as<Conv2DWinogradAttrs>()) {
        updated_attrs = CopyAttrsWithNewLayout(pattr, new_layout);
      } else if (auto pattr = call->attrs.as<Conv3DAttrs>()) {
        updated_attrs = CopyAttrsWithNewLayout(pattr, new_layout);
      } else if (auto pattr = call->attrs.as<DenseAttrs>()) {
        updated_attrs = CopyAttrsWithNewLayout(pattr, new_layout);
      } else if (auto pattr = call->attrs.as<BatchMatmulAttrs>()) {
        updated_attrs = CopyAttrsWithNewLayout(pattr, new_layout);
      }
      new_n = Call(call->op, updated_args, updated_attrs);
    }
    return new_n;
  }

 private:
  std::deque<std::string> ori_layouts_queue_;
  std::deque<std::string> new_layouts_queue_;

  std::vector<std::string> target_ops_{"nn.contrib_conv2d_winograd_without_weight_transform",
                                       "nn.conv2d", "nn.conv3d", "nn.dense", "nn.batch_matmul"};
};


Expr KernelLayoutTransformer::VisitExpr_(const CallNode* n) {
  auto new_n = ExprMutator::VisitExpr_(n);
  const auto* call = new_n.as<CallNode>();
  if (call) {
    const auto* func = call->op.as<FunctionNode>();
    if (func) {
      global_ori_layouts_queue.clear();
      global_new_layouts_queue.clear();

      // Use ScheduleGetter to call python lower functions
      // This is used to get the layout transform information
      // The layout transformation will be recorded to global_ori_layout_queue
      // in ComputeDAG::RewriteLayout
      auto f = runtime::Registry::Get("ansor.enter_layout_rewrite");
      CHECK(f) << "Could not find ansor.enter_layout_rewrite function.";
      (*f)();

      ScheduleGetter(Target::Current()).Create(GetRef<Function>(func));

      f = runtime::Registry::Get("ansor.exit_layout_rewrite");
      CHECK(f) << "Could not find ansor.exit_layout_rewrite function.";
      (*f)();

      // Mutate the called function
      if (!global_ori_layouts_queue.empty() && !global_new_layouts_queue.empty()) {
        auto ret = FuncMutator(global_ori_layouts_queue, global_new_layouts_queue).VisitExpr(new_n);
        return ret;
      }
    }
  }

  return new_n;
}


Expr KernelLayoutTransform(const Expr& expr) {
  // Do a post-order DSF to mutate the layout of
  // all "layout free" placeholders for the auto-scheduler.
  return KernelLayoutTransformer().Mutate(expr);
}


namespace transform {

Pass KernelLayoutTransform() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(relay::KernelLayoutTransform(f));
      };
  return CreateFunctionPass(pass_func, 3, "KernelLayoutTransform", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.KernelLayoutTransform").set_body_typed(KernelLayoutTransform);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
