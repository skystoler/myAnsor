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
 * \file relay/backend/compile_engine.h
 * \brief Internal compialtion engine handle function cache.
 *  and interface to low level code generation.
 */
#ifndef TVM_RELAY_BACKEND_COMPILE_ENGINE_H_
#define TVM_RELAY_BACKEND_COMPILE_ENGINE_H_

#include <topi/tags.h>
#include <tvm/driver/driver_api.h>
#include <tvm/ir/type_functor.h>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/op_strategy.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/module.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/te/schedule_pass.h>

#include <functional>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../transforms/pass_util.h"
#include "utils.h"

namespace tvm {
namespace relay {

/*! \brief Indicate whether the data or shape or both of a parameter is used in the shape func. */
enum ShapeFuncParamState {
  kNoNeed = 0,
  kNeedInputData = 1,
  kNeedInputShape = 2,
  kNeedBoth = 3,
};

struct LoweredOutputNode : public Object {
  /*! \brief The outputs to the function */
  tvm::Array<te::Tensor> outputs;
  /*! \brief The implementation used to compute the output */
  OpImplementation implementation;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("outputs", &outputs);
    v->Visit("implementation", &implementation);
  }

  static constexpr const char* _type_key = "relay.LoweredOutput";
  TVM_DECLARE_FINAL_OBJECT_INFO(LoweredOutputNode, Object);
};

class LoweredOutput : public ObjectRef {
 public:
  TVM_DLL LoweredOutput(tvm::Array<te::Tensor> outputs, OpImplementation impl);

  TVM_DEFINE_OBJECT_REF_METHODS(LoweredOutput, ObjectRef, LoweredOutputNode);
};

/*! \brief Node container to represent a cached function. */
struct CachedFuncNode : public Object {
  /* \brief compiled target */
  tvm::Target target;
  /*! \brief Function name */
  std::string func_name;
  /* \brief The inputs to the function */
  tvm::Array<te::Tensor> inputs;
  /* \brief The outputs to the function */
  tvm::Array<te::Tensor> outputs;
  /*! \brief The schedule to the function */
  te::Schedule schedule;
  /*! \brief The lowered functions to support the function. */
  IRModule funcs = IRModule();

  /*! \brief Parameter usage states in the shape function. */
  tvm::Array<Integer> shape_func_param_states;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("target", &target);
    v->Visit("func_name", &func_name);
    v->Visit("inputs", &inputs);
    v->Visit("outputs", &outputs);
    v->Visit("schedule", &schedule);
    v->Visit("funcs", &funcs);
    v->Visit("shape_func_param_states", &shape_func_param_states);
  }

  static constexpr const char* _type_key = "relay.CachedFunc";
  TVM_DECLARE_FINAL_OBJECT_INFO(CachedFuncNode, Object);
};

class CachedFunc : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(CachedFunc, ObjectRef, CachedFuncNode);
};

class CCacheKey;
/*! \brief Compile cache key */
class CCacheKeyNode : public Object {
 public:
  /*! \brief The source function to be lowered. */
  Function source_func;
  /*! \brief The hardware target.*/
  Target target;

  bool disabled;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("source_func", &source_func);
    v->Visit("target", &target);
  }
  /*! \return The hash value of CCacheKey. */
  inline size_t Hash() const;
  /*!
   * \brief check content equality
   * \param other The other value.
   * \return The result of equality check.
   */
  inline bool Equal(const CCacheKeyNode* other) const;

  static constexpr const char* _type_key = "relay.CCacheKey";
  TVM_DECLARE_FINAL_OBJECT_INFO(CCacheKeyNode, tvm::Object);

 private:
  /*!
   * \brief internal cached hash value.
   */
  mutable size_t hash_{0};
};

/*! \brief cache entry used in compile engine */
class CCacheKey : public ObjectRef {
 public:
  CCacheKey() {}
  explicit CCacheKey(ObjectPtr<Object> n) : ObjectRef(n) {}

  /*!
   * \brief The constructor
   * \param source_func The source function.
   * \param target The target device.
   */
  TVM_DLL CCacheKey(Function source_func, Target target);

  const CCacheKeyNode* operator->() const { return static_cast<const CCacheKeyNode*>(get()); }
  // comparator
  inline bool operator==(const CCacheKey& other) const {
    CHECK(defined() && other.defined());
    return (*this)->Equal(other.operator->());
  }
  using ContainerType = CCacheKeyNode;
};

/*! \brief Node container for compile cache. */
class CCacheValueNode : public Object {
 public:
  /*! \brief The corresponding function */
  CachedFunc cached_func;
  /*! \brief Result of Packed function generated by JIT */
  PackedFunc packed_func;
  /*! \brief usage statistics */
  int use_count{0};

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("cached_func", &cached_func);
    v->Visit("use_count", &use_count);
  }
  static constexpr const char* _type_key = "relay.CCacheValue";
  TVM_DECLARE_FINAL_OBJECT_INFO(CCacheValueNode, tvm::Object);
};

/*! \brief cache entry used in compile engine */
class CCacheValue : public ObjectRef {
 public:
  CCacheValue() {}
  explicit CCacheValue(ObjectPtr<Object> n) : ObjectRef(n) {}
  CCacheValueNode* operator->() { return static_cast<CCacheValueNode*>(get_mutable()); }
  const CCacheValueNode* operator->() const { return static_cast<const CCacheValueNode*>(get()); }
  using ContainerType = CCacheValueNode;
};

/*!
 * \brief Backend compilation engine for
 *        low level code generation.
 */
class CompileEngineNode : public Object {
 public:
  /*! \brief destructor */
  virtual ~CompileEngineNode() {}
  /*!
   * \brief Get lowered result.
   * \param key The key to the cached function.
   * \return The result.
   */
  virtual CachedFunc Lower(const CCacheKey& key) = 0;
  /*!
   * \brief Just in time compile to get a PackedFunc.
   * \param key The key to the cached function.
   * \return The result.
   */
  virtual PackedFunc JIT(const CCacheKey& key) = 0;
  /*!
   * \brief Lower the shape function.
   * \param key The key to the cached function.
   * \return The result.
   */
  virtual CachedFunc LowerShapeFunc(const CCacheKey& key) = 0;
  /*!
   * \brief Lower the external function using external codegen tools.
   * \return The runtime moduels for each needed external codegen tool.
   */
  virtual tvm::Array<tvm::runtime::Module> LowerExternalFunctions() = 0;

  /*! \brief clear the cache. */
  virtual void Clear() = 0;

  // VisitAttrs
  void VisitAttrs(AttrVisitor*) {}

  static constexpr const char* _type_key = "relay.CompileEngine";
  TVM_DECLARE_FINAL_OBJECT_INFO(CompileEngineNode, Object);
};

/*! \brief cache entry used in compile engine */
class CompileEngine : public ObjectRef {
 public:
  CompileEngine() {}
  explicit CompileEngine(ObjectPtr<Object> n) : ObjectRef(n) {}
  CompileEngineNode* operator->() { return static_cast<CompileEngineNode*>(get_mutable()); }
  using ContainerType = CompileEngineNode;
  /*! \brief The global compile engine. */
  TVM_DLL static const CompileEngine& Global();
};

/*!
 * \brief Check if the type is dynamic.
 * \param ty The type to be checked.
 * \return The result.
 */
bool IsDynamic(const Type& ty);

// implementations
inline size_t CCacheKeyNode::Hash() const {
  if (hash_ != 0) return hash_;
  // do structral hash, avoid 0.
  hash_ = tvm::StructuralHash()(this->source_func);
  hash_ = dmlc::HashCombine(hash_, std::hash<std::string>()(target->str()));
  if (hash_ == 0) hash_ = 1;
  return hash_;
}

inline bool CCacheKeyNode::Equal(const CCacheKeyNode* other) const {
  if (disabled) return false;
  if (Hash() != other->Hash()) return false;
  return this->target->str() == other->target->str() &&
         tvm::StructuralEqual()(this->source_func, other->source_func);
}

Array<IndexExpr> GetShape(const Array<IndexExpr>& shape);

// The getter to get schedule from compile engine.
// Get schedule from functor.
class ScheduleGetter: public backend::MemoizedExprTranslator<Array<te::Tensor>> {
 public:
  explicit ScheduleGetter(Target target)
      : target_(target), device_copy_op_(Op::Get("device_copy")) {}

  CachedFunc Create(const Function& prim_func) {
    auto cache_node = make_object<CachedFuncNode>();
    cache_node->target = target_;
    for (Var param : prim_func->params) {
      Array<tvm::te::Tensor> inputs;
      if (const auto* ttype = param->checked_type().as<TensorTypeNode>()) {
        tvm::te::Tensor tensor = tvm::te::placeholder(GetShape(ttype->shape), ttype->dtype);
        cache_node->inputs.push_back(tensor);
        inputs.push_back(tensor);
      } else {
        // flatten tuple of tensor type.
        const auto* tuple_type = param->type_as<TupleTypeNode>();
        for (Type field : tuple_type->fields) {
          const auto* ttype = field.as<TensorTypeNode>();
          // TODO(@icemelon): Allow recursive tuple
          CHECK(ttype != nullptr);
          tvm::te::Tensor tensor = tvm::te::placeholder(GetShape(ttype->shape), ttype->dtype);
          cache_node->inputs.push_back(tensor);
          inputs.push_back(tensor);
        }
      }
      memo_[param] = inputs;
    }
    readable_name_stream_ << "fused";
    cache_node->outputs = this->VisitExpr(prim_func->body);
    auto candidate_name = readable_name_stream_.str();
    constexpr static size_t kMaxFuncNameLength = 80;
    if (candidate_name.size() > kMaxFuncNameLength) {
      std::stringstream truncated_name;
      truncated_name << candidate_name.substr(0, kMaxFuncNameLength);
      truncated_name << "_" << std::hash<std::string>{}(candidate_name) << "_";
      candidate_name = truncated_name.str();
    }
    cache_node->func_name = candidate_name;

    CHECK(master_op_.defined());
    // Fusion over tupled results may leave identity relationships
    // between inputs and outputs, and those should not be scheduled.
    // Hence schedule only non PlaceholderOp outputs.
    tvm::Array<te::Tensor> tensor_outs;
    for (const auto& tensor : cache_node->outputs) {
      if (!tensor->op.as<te::PlaceholderOpNode>()) {
        tensor_outs.push_back(tensor);
      }
    }
    te::Schedule schedule;
    // No need to register schedule for device copy op.
    if (master_attrs_.as<DeviceCopyAttrs>() == nullptr) {
      CHECK(master_implementation_.defined());
      schedule = master_implementation_.Schedule(master_attrs_, tensor_outs, target_);
      for (const auto& scalar : scalars_) {
        if (schedule->Contain(scalar)) {
          schedule[scalar].compute_inline();
        }
      }
    }
    cache_node->schedule = std::move(schedule);
    return CachedFunc(cache_node);
  }

  Array<te::Tensor> VisitExpr_(const VarNode* op) final {
    LOG(FATAL) << "Free variable " << op->name_hint();
    return {};
  }

  Array<te::Tensor> VisitExpr_(const ConstantNode* op) final {
    using tir::make_const;
    CHECK(op->is_scalar());
    void* data = op->data->data;
    DataType dtype = DataType(op->data->dtype);
    auto value = te::compute(
        {},
        [&](const Array<tvm::tir::Var>&) {
          if (dtype == DataType::Int(32)) {
            return make_const(dtype, static_cast<const int32_t*>(data)[0]);
          } else if (dtype == DataType::Int(64)) {
            return make_const(dtype, static_cast<const int64_t*>(data)[0]);
          } else if (dtype == DataType::Float(32)) {
            return make_const(dtype, static_cast<const float*>(data)[0]);
          } else if (dtype == DataType::Float(64)) {
            return make_const(dtype, static_cast<const double*>(data)[0]);
          } else if (dtype == DataType::Bool()) {
            return make_const(dtype, static_cast<const uint8_t*>(data)[0]);
          } else {
            LOG(FATAL) << "not handled";
            return tvm::PrimExpr();
          }
        },
        "compile_engine_const", topi::kBroadcast);
    scalars_.push_back(value->op);
    return {value};
  }

  Array<te::Tensor> VisitExpr_(const CallNode* call_node) final {
    static auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");
    static auto flower_call = tvm::runtime::Registry::Get("relay.backend.lower_call");
    CHECK(flower_call) << "relay.backend.lower_call is not registered.";

    Array<te::Tensor> inputs;
    int count_tuple = 0;
    for (Expr arg : call_node->args) {
      if (arg->checked_type().as<TupleTypeNode>()) {
        ++count_tuple;
      }
      for (te::Tensor tensor : VisitExpr(arg)) {
        inputs.push_back(tensor);
      }
    }
    if (count_tuple) {
      CHECK_EQ(call_node->args.size(), 1U) << "Only allow function with a single tuple input";
    }

    CHECK(call_node->op.as<OpNode>()) << "Primitive function only allows call into primitive ops";
    Op op = Downcast<Op>(call_node->op);

    Array<te::Tensor> outputs;
    OpImplementation impl;
    // Skip fcompute for device copy operators as it is not registered.
    if (op == device_copy_op_) {
      const auto* copy_input = inputs[0].operator->();
      outputs.push_back(te::Tensor(copy_input->shape, copy_input->dtype, te::Operation(), 0));
    } else {
      LoweredOutput lowered_out = (*flower_call)(GetRef<Call>(call_node), inputs, target_);
      outputs = lowered_out->outputs;
      impl = lowered_out->implementation;
    }

    int op_pattern = fpattern[op];
    if (op_pattern >= kCommReduce) {
      CHECK(!master_op_.defined() || master_op_pattern_ < kCommReduce)
          << "Two complicated op in a primitive function "
          << " master=" << master_op_ << " current=" << op;
    }
    if (op_pattern >= master_op_pattern_) {
      master_op_ = op;
      master_attrs_ = call_node->attrs;
      master_op_pattern_ = op_pattern;
      master_implementation_ = impl;
    }
    if (outputs.size() != 1) {
      const auto* tuple_type = call_node->checked_type().as<TupleTypeNode>();
      CHECK(tuple_type) << "Expect output to be a tuple type";
      CHECK_EQ(tuple_type->fields.size(), outputs.size());
    }
    // Set the name to `__copy`. It will be detected in graph runtime to perform
    // data copy across devices.
    if (op == device_copy_op_) {
      readable_name_stream_.str(std::string());
      readable_name_stream_ << "__copy";
    } else {
      readable_name_stream_ << '_' << op->name;
    }
    return outputs;
  }

  Array<te::Tensor> VisitExpr_(const FunctionNode* op) final {
    LOG(FATAL) << "Do not support sub function";
    return Array<te::Tensor>();
  }

  Array<te::Tensor> VisitExpr_(const LetNode* op) final {
    Array<te::Tensor> val = VisitExpr(op->value);
    CHECK(!memo_.count(op->var));
    memo_[op->var] = val;
    return VisitExpr(op->body);
  }

  Array<te::Tensor> VisitExpr_(const TupleNode* op) final {
    Array<te::Tensor> fields;
    for (Expr field : op->fields) {
      CHECK(field->checked_type().as<TensorTypeNode>()) << "Only allow Tuple of Tensor";
      Array<te::Tensor> res = VisitExpr(field);
      CHECK_EQ(res.size(), 1);
      fields.push_back(res[0]);
    }
    return fields;
  }

  Array<te::Tensor> VisitExpr_(const TupleGetItemNode* op) final {
    const auto* tuple_type = op->tuple->type_as<TupleTypeNode>();
    Array<te::Tensor> tuple = VisitExpr(op->tuple);
    CHECK_EQ(tuple_type->fields.size(), tuple.size());
    CHECK_GE(op->index, 0);
    CHECK_LT(static_cast<size_t>(op->index), tuple.size());
    return {tuple[op->index]};
  }

 private:
  tvm::Target target_;
  Op master_op_;
  Attrs master_attrs_;
  int master_op_pattern_{0};
  OpImplementation master_implementation_;
  std::ostringstream readable_name_stream_;
  Array<te::Operation> scalars_;
  // Cache device copy op for equivalence checking to reduce registry lookup
  // overhead for each invocation of call node when retrieving schedules.
  const Op& device_copy_op_;
};
}  // namespace relay
}  // namespace tvm

namespace std {
// overload hash
template <>
struct hash<::tvm::relay::CCacheKey> {
  size_t operator()(const ::tvm::relay::CCacheKey& key) const {
    CHECK(key.defined());
    return key->Hash();
  }
};
}  // namespace std
#endif  // TVM_RELAY_BACKEND_COMPILE_ENGINE_H_
