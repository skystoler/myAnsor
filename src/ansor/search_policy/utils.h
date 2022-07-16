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
 * \file ansor/search_policy/utils.cc
 * \brief Common utilities for search policies
 */

#ifndef TVM_ANSOR_SEARCH_POLICY_UTILS_H_
#define TVM_ANSOR_SEARCH_POLICY_UTILS_H_

#include <tvm/te/operation.h>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "../cost_model/cost_model.h"
#include "../utils.h"
#include "../loop_state.h"
#include "../transform_step.h"
#include "search_policy.h"

namespace tvm {
namespace ansor {

// Get an integer from a tvm str Map
inline int GetIntParam(const Map<String, ObjectRef>& attr_dict,
                       const std::string& key) {
  CHECK_GT(attr_dict.count(key), 0) << "Cannot find key: \"" << key << "\" in " << attr_dict;
  auto pint = attr_dict[key].as<IntImmNode>();
  CHECK(pint != nullptr);
  return pint->value;
}

// Get a double from a tvm str Map
inline double GetDoubleParam(const Map<String, ObjectRef>& attr_dict,
                             const std::string& key) {
  CHECK_GT(attr_dict.count(key), 0) << "Cannot find key: \"" << key << "\" in " << attr_dict;
  auto pdouble = attr_dict[key].as<FloatImmNode>();
  CHECK(pdouble != nullptr);
  return pdouble->value;
}

// Get a string from a tvm str Map
inline std::string GetStringParam(const Map<String, ObjectRef>& attr_dict,
                                  const std::string& key) {
  CHECK_GT(attr_dict.count(key), 0)
      << "Cannot find key: \"" << key << "\" in " << attr_dict;
  const auto& target = attr_dict[key];
  if (auto pstr = target.as<StringImmNode>()) {
    return pstr->value;
  }
  auto pstr = target.as<StringObj>();
  CHECK(pstr != nullptr);
  return pstr->data;
}

// Get a iterator name set from a tvm str Map
inline std::set<std::string> GetIterNameSetParam(const Map<String, ObjectRef>& attr_dict,
                                                 const std::string& key) {
  std::set<std::string> ret;
  CHECK_GT(attr_dict.count(key), 0) << "Cannot find key: \"" << key << "\" in " << attr_dict;
  auto names = attr_dict[key].as<ArrayNode>();
  CHECK(names != nullptr);
  for (const auto & name : *names) {
    ret.insert(name.as<StringObj>()->data);
  }
  return ret;
}

// Get axes that should not be splitted according to the attribute from tvm.compute
inline std::pair<std::set<std::string>, std::set<std::string> > GetNoSplitAxisAttr(
    const Stage& stage) {
  std::pair<std::set<std::string>, std::set<std::string> > ret;
  if (stage->op->attrs.count(SearchPolicyNode::no_split_at_inner_key)) {
    ret.first = GetIterNameSetParam(stage->op->attrs, SearchPolicyNode::no_split_at_inner_key);
  }
  if (stage->op->attrs.count(SearchPolicyNode::no_split_at_outer_key)) {
    ret.second = GetIterNameSetParam(stage->op->attrs, SearchPolicyNode::no_split_at_outer_key);
  }
  return ret;
}

// Get the the setting of maximum innermost split factor
inline int GetMaxInnermostSplitFactor(const SearchTask& task,
                                      const Map<String, ObjectRef>& search_policy_params) {
  int limit_inner_most_tile_size = GetIntParam(search_policy_params, "limit_inner_most_tile_size");
  return limit_inner_most_tile_size > 0 ?
      limit_inner_most_tile_size : task->hardware_params->max_innermost_split_factor;
}

// Convert operation to stage id
inline int OperationToStage(const te::Operation& op, const State& state) {
  for (size_t i = 0; i < state->stages.size(); ++i) {
    if (op == state->stages[i]->op) {
      return i;
    }
  }
  LOG(FATAL) << "Cannot find op: " << op;
  return -1;
}

// Return whether an op is strict-inlineable
inline bool IsStrictInlineable(const SearchTask& task, const State& state, int stage_id) {
  if (state->task_dag.defined()) {
    return state->task_dag->access_analyzer.IsStrictInlineable(state->stages[stage_id]->op);
  } else {
    return task->compute_dag->access_analyzer.IsStrictInlineable(state->stages[stage_id]->op);
  }
}

// Return whether an op is an output op
inline bool IsOutputOp(const SearchTask& task, const State& state, int stage_id) {
  if (state->task_dag.defined()) {
    return state->task_dag->access_analyzer.IsOutput(state->stages[stage_id]->op);
  } else {
    return task->compute_dag->access_analyzer.IsOutput(state->stages[stage_id]->op);
  }
}

// Return whether an op needs multi level tiling
inline bool NeedsMultilevelTiling(const SearchTask& task, const State& state, int stage_id) {
  if (state->task_dag.defined()) {
    return state->task_dag->access_analyzer.NeedsMultiLevelTiling(state->stages[stage_id]->op);
  } else {
    return task->compute_dag->access_analyzer.NeedsMultiLevelTiling(state->stages[stage_id]->op);
  }
}

// Get all consumers for a stage. This function propagates the relation for inlined ops.
inline std::set<int> GetConsumers(const SearchTask& task, const State& state, int stage_id) {
  std::unordered_set<te::Operation, ObjectHash, ObjectEqual> consumers;
  std::set<int> ret;

  if (state->task_dag.defined()) {
    state->task_dag->access_analyzer.GetConsumers(state, state->stages[stage_id]->op, &consumers);
  } else {
    task->compute_dag->access_analyzer.GetConsumers(state, state->stages[stage_id]->op, &consumers);
  }

  for (const auto& op : consumers) {
    ret.insert(OperationToStage(op, state));
  }
  return ret;
}

// Get all producers for a stage. This function propagates the relation for inlined ops.
inline std::set<int> GetProducers(const SearchTask& task, const State& state, int stage_id) {
  std::unordered_set<te::Operation, ObjectHash, ObjectEqual> producers;
  std::set<int> ret;

  if (state->task_dag.defined()) {
    state->task_dag->access_analyzer.GetProducers(state, state->stages[stage_id]->op, &producers);
  } else {
    task->compute_dag->access_analyzer.GetProducers(state, state->stages[stage_id]->op, &producers);
  }

  for (const auto& op : producers) {
    ret.insert(OperationToStage(op, state));
  }
  return ret;
}

// Get all producers for a stage. This function DOES NOT propagates the relation for inlined ops.
inline std::set<int> GetDirectProducers(const SearchTask& task, const State& state, int stage_id) {
  std::unordered_set<te::Operation, ObjectHash, ObjectEqual> producers;
  std::set<int> ret;

  if (state->task_dag.defined()) {
    state->task_dag->access_analyzer.GetDirectProducers(state,
            state->stages[stage_id]->op, &producers);
  } else {
    task->compute_dag->access_analyzer.GetDirectProducers(state,
            state->stages[stage_id]->op, &producers);
  }

  for (const auto& op : producers) {
    ret.insert(OperationToStage(op, state));
  }
  return ret;
}

// Get the number of common outer iterators
// This function propagates the relation for chains with multiple ops.
inline int GetNumCommonOuterIterator(const SearchTask& task, const State& state,
        int stage_id, int target_stage_id) {
  if (state->task_dag.defined()) {
    return state->task_dag->access_analyzer.GetNumCommonOuterIterator(state,
            state->stages[stage_id]->op, state->stages[target_stage_id]->op);
  } else {
    return task->compute_dag->access_analyzer.GetNumCommonOuterIterator(state,
            state->stages[stage_id]->op, state->stages[target_stage_id]->op);
  }
}

// Return whether two ops are elementwise-matched
inline bool ElementwiseMatch(const SearchTask& task, const State& state, int stage_id,
                             int target_stage_id) {
  const auto& op = state->stages[stage_id]->op;
  const auto& target_op = state->stages[target_stage_id]->op;
  if (state->task_dag.defined()) {
    return state->task_dag->access_analyzer.ElementWiseMatch(op, target_op);
  } else {
    return task->compute_dag->access_analyzer.ElementWiseMatch(op, target_op);
  }
}

// Return whether the search task is targeting a GPU
inline bool IsGPUTask(const SearchTask& task) {
  return IsGPUDevice(task->target->device_type);
}

// Return the extent of an iterator
inline int64_t GetExtent(const Iterator& it) {
  if (it->range.defined()) {
    if (auto pint = it->range->extent.as<IntImmNode>()) {
      return pint->value;
    }
  }
  return -1;
}

// Compute the product of lengths of all space iters and all reduce iters, respectively
inline std::pair<int64_t, int64_t> GetCumulativeSpaceAndReductionLengh(const Stage& stage) {
  int64_t cum_space_len = 1, cum_reduce_len = 1;
  for (const auto& iter : stage->iters) {
    if (iter->iter_type == kSpace) {
      cum_space_len *= GetExtent(iter);
    } else if (iter->iter_type == kReduce) {
      cum_reduce_len *= GetExtent(iter);
    }
  }
  return std::make_pair(cum_space_len, cum_reduce_len);
}

// Return whether this stage needs rfactor
inline bool NeedsRfactor(const SearchTask& task, const State& state, int stage_id) {
  const auto& op = state->stages[stage_id]->op;
  if (op->IsInstance<te::ComputeOpNode>()) {
    // Compute the product of lengths of all space iters and all reduce iters
    int cum_space_len, cum_reduce_len;
    std::tie(cum_space_len, cum_reduce_len) = 
        GetCumulativeSpaceAndReductionLengh(state->stages[stage_id]);

    if (NeedsMultilevelTiling(task, state, stage_id)) {
      // Do not use rfactor if we have enough parallelism on space iters
      if (cum_space_len > cum_reduce_len ||
          cum_space_len > task->hardware_params->num_cores * 16) {
        return false;
      } else {
        return true;
      }
    } else if (cum_reduce_len > 1) {
      // Always try rfactor for reduction ops
      if (IsGPUTask(task)) {
        return cum_reduce_len > task->hardware_params->warp_size;
      } else {
        return cum_reduce_len > task->hardware_params->num_cores;
      }
    }
  }

  return false;
}

// Return whether this stage needs rfactor
inline bool NeedsCrossThreadReduction(const SearchTask& task, const State& state, int stage_id) {
  const auto& op = state->stages[stage_id]->op;
  if (op->IsInstance<te::ComputeOpNode>()) {
    // Compute the product of lengths of all space iters and all reduce iters
    int cum_space_len, cum_reduce_len;
    std::tie(cum_space_len, cum_reduce_len) = 
        GetCumulativeSpaceAndReductionLengh(state->stages[stage_id]);

    if (NeedsMultilevelTiling(task, state, stage_id)) {
      // Do rfactor if we do not have enough parallelism on space iters
      return cum_space_len < cum_reduce_len;
    } else if (cum_reduce_len > 1) {
      // Try rfactor for other reduction operators
      return cum_reduce_len > task->hardware_params->warp_size;
    }
  }

  return false;
}

// Return whether the stage has reduce iterators
inline bool HasReduceIter(const Stage& stage) {
  for (const auto& iter : stage->iters) {
    if (iter->iter_type != kSpace) {
      return true;
    }
  }
  return false;
}

// Return whether the stage has specific annotated iterators
inline bool HasAnnotatedIter(const Stage& stage, IteratorAnnotation type) {
  for (const auto& iter : stage->iters) {
    if (iter->annotation == type) {
      return true;
    }
  }
  return false;
}

// Return whether the stage does cross thread reduction
inline bool HasCrossThreadReduction(const State& state, int stage_id) {
  std::function<bool(const Stage&)> check_stage = [](const Stage& in_stage) {
    for (const auto& iter : in_stage->iters) {
      if (iter->annotation == kThreadX && iter->iter_type == kReduce) {
        return true;
      }
    }
    return false;
  };

  // Check the stage itself
  if (check_stage(state->stages[stage_id])) {
    return true;
  }

  // Check the attached stages
  for (size_t iter_id = 0; iter_id < state->stages[stage_id]->iters.size(); iter_id++) {
    const auto& res = state->attach_map->iter_to_attached_stages.find(
          std::make_pair(stage_id, iter_id));
    if (res != state->attach_map->iter_to_attached_stages.end()) {
      for (int attached_stage_id : res->second) {
        if (check_stage(state->stages[attached_stage_id])) {
          return true;
        }
      }
    }
  }

  return false;
}

// Return whether the stage has only one consumer and they are elementwise-matched
inline bool HasSingleElementwiseMatchedConsumer(const SearchTask& task,
    const State& state, int stage_id, int* target_stage_id) {
  const std::set<int>& consumers = GetConsumers(task, state, stage_id);
  if (consumers.size() == 1) {
    *target_stage_id = *consumers.begin();
    if (ElementwiseMatch(task, state, stage_id, *target_stage_id) &&
        (!(HasReduceIter(state->stages[stage_id]) &&
         HasReduceIter(state->stages[*target_stage_id]))) &&
        (!StrEndsWith(state->stages[*target_stage_id]->op->name, ".shared"))) {
      return true;
    }
  }
  return false;
}

// Return whether the step is a stage number changing step
inline bool IsStageNumberChangingStep(const Step& step) {
  return step->IsInstance<CacheWriteStepNode>() ||
         step->IsInstance<CacheReadStepNode>() ||
         step->IsInstance<RfactorStepNode>();
}

// Return whether the state does cache_write for stage_id
inline bool HasCacheWriteStage(const State& s, int stage_id) {
  for (int i = static_cast<int>(s->transform_steps.size()) - 1; i >= 0; --i) {
    if (auto ps = s->transform_steps[i].as<CacheWriteStepNode>()) {
      if (stage_id == ps->stage_id) {
        return true;
      }
    }

    if (IsStageNumberChangingStep(s->transform_steps[i])) {
      if (stage_id > s->transform_steps[i]->stage_id) {
        stage_id--;
      }
    }
  }
  return false;
}

// Return whether the state does cache_read for stage_id
inline bool HasCacheReadStage(const State& s, int stage_id) {
  for (int i = static_cast<int>(s->transform_steps.size()) - 1; i >= 0; --i) {
    if (auto ps = s->transform_steps[i].as<CacheReadStepNode>()) {
      if (stage_id == ps->stage_id) {
        return true;
      }
    }

    if (IsStageNumberChangingStep(s->transform_steps[i])) {
      if (stage_id > s->transform_steps[i]->stage_id) {
        stage_id--;
      }
    }
  }
  return false;
}

// Return whether the state does rfactor for stage_id
inline bool HasRfactorStage(const State& s, int stage_id) {
  for (int i = static_cast<int>(s->transform_steps.size()) - 1; i >= 0; --i) {
    if (auto ps = s->transform_steps[i].as<RfactorStepNode>()) {
      if (stage_id == ps->stage_id) {
        return true;
      }
    }

    if (IsStageNumberChangingStep(s->transform_steps[i])) {
      if (stage_id > s->transform_steps[i]->stage_id) {
        stage_id--;
      }
    }
  }
  return false;
}

// 

// Return whether the stage has been tiled already
inline bool IsTiled(const Stage& stage) {
  auto op = stage->op.as<te::ComputeOpNode>();
  CHECK(op != nullptr);
  return stage->iters.size() != op->axis.size() + op->reduce_axis.size();
}

// Extract primitive iterators from a nested fused or splitted iterator's name
inline void ExtractOriginalIterators(const std::string& name, std::set<std::string>* rets) {
  size_t last_pos = 0;
  for (size_t i = 0; i < name.size(); ++i) {
    if (name[i] == '@' || name[i] == '.') {  // '@' for fuse and '.' for split
      if (!isdigit(name[last_pos]) && name[last_pos] != '@' && name[last_pos] != '.') {
        rets->insert(name.substr(last_pos, i - last_pos));
      }
      last_pos = i + 1;
    }
  }

  if (last_pos < name.size() && !isdigit(name[last_pos]) &&
      name[last_pos] != '@' && name[last_pos] != '.') {
    rets->insert(name.substr(last_pos, name.size() - last_pos));
  }
}

// Get the last space iterator in the outer most tile
inline const Iterator& GetLastSpaceIteratorInOutermostTile(const Stage& stage) {
  auto pop = stage->op.as<te::ComputeOpNode>();
  CHECK(pop != nullptr);
  std::set<std::string> original_names;

  for (const auto& iter : stage->iters) {
    ExtractOriginalIterators(iter->name, &original_names);
    if (original_names.size() == pop->axis.size()) {
      return iter;
    }
  }

  LOG(FATAL) << "Cannot find the iterator.";
  return stage->iters[0];
}

// Get the last reduce iterator in the outermost reduce tile
inline const Iterator& GetLastReduceIteratorInOutermostReduceTile(const Stage& stage) {
  auto pop = stage->op.as<te::ComputeOpNode>();
  CHECK(pop != nullptr);
  std::set<std::string> original_names;

  auto no_split_name_pair = GetNoSplitAxisAttr(stage);
  std::set<std::string> no_split_at_inner_name_set = no_split_name_pair.first;
  size_t reduce_axis_size = 0;
  for (const auto axis : pop->reduce_axis) {
    if (!no_split_at_inner_name_set.count(axis->var->name_hint)) {
      reduce_axis_size++;
    }
  }
  if (reduce_axis_size) {
    for (const auto& iter : stage->iters) {
      if (iter->iter_type == kReduce) {
        ExtractOriginalIterators(iter->name, &original_names);
        if (original_names.size() == reduce_axis_size) {
          return iter;
        }
      }
    }
  } else {
    for (size_t i = stage->iters.size() - 1; i >= 0; i--) {
      if (stage->iters[i]->iter_type == kReduce) {
        return stage->iters[i];
      }
    }
  }

  LOG(FATAL) << "Cannot find the iterator.";
  return stage->iters[0];
}

// Get the last reduce iterator in the outermost reduce tile
inline const Iterator& GetLastReduceIteratorInSecondOutermostReduceTile(const Stage& stage) {
  auto pop = stage->op.as<te::ComputeOpNode>();
  CHECK(pop != nullptr);
  std::set<std::string> original_names;

  auto no_split_name_pair = GetNoSplitAxisAttr(stage);
  std::set<std::string> no_split_at_inner_name_set = no_split_name_pair.first;
  size_t reduce_axis_size = 0;
  for (const auto axis : pop->reduce_axis) {
    if (!no_split_at_inner_name_set.count(axis->var->name_hint)) {
      reduce_axis_size++;
    }
  }

  if (reduce_axis_size) {
    size_t i = 0;
    for (; i < stage->iters.size(); ++i) {
      if (stage->iters[i]->iter_type == kReduce) {
        ExtractOriginalIterators(stage->iters[i]->name, &original_names);
        if (original_names.size() == reduce_axis_size) {
          ++i;
          break;
        }
      }
    }
    original_names.clear();
    for (; i < stage->iters.size(); ++i) {
      if (stage->iters[i]->iter_type == kReduce) {
        ExtractOriginalIterators(stage->iters[i]->name, &original_names);
        if (original_names.size() == reduce_axis_size) {
          return stage->iters[i];
        }
      }
    }
  }

  LOG(FATAL) << "Cannot find the iterator.";
  return stage->iters[0];
}

inline int GetSingleConsumerId(const SearchTask& task, const State& state, int stage_id) {
  const std::set<int>& consumers = GetConsumers(task, state, stage_id);
  if (consumers.empty()) {
    return -1;
  }

  if (consumers.size() == 1) {
    return *consumers.begin();
  } else {
    // check all consumers share a common root
    int common_root_id = -1;
    bool mismatch = false;
    for (const auto& consumer_stage_id : consumers) {
      int root_id = -1;
      if (state->stages[consumer_stage_id]->compute_at == kRoot) {
        root_id = consumer_stage_id;
      } else if (state->stages[consumer_stage_id]->compute_at == kIter) {
        root_id = state->attach_map->stage_to_attach_iter.at(consumer_stage_id).first;
      } else {
        LOG(FATAL) << "Invalid case";
      }

      if (common_root_id == -1) {
        common_root_id = root_id;
      } else {
        if (common_root_id != root_id) {
          mismatch = true;
          break;
        }
      }
    }

    return mismatch ? -1 : common_root_id;
  }
}

// Fuse all reduction iterators
inline State FuseAllReductionIterators(State state, int stage_id, Iterator* fused_iter,
        std::vector<Iterator>* space_iters, std::vector<Iterator>* reduce_iters) {
  space_iters->clear();
  reduce_iters->clear();

  for (const auto &iter : state->stages[stage_id]->iters) {
    if (iter->iter_type == kSpace) {
      space_iters->push_back(iter);
    } else if (iter->iter_type == kReduce) {
      reduce_iters->push_back(iter);
    }
  }

  CHECK(!reduce_iters->empty());
  if (reduce_iters->size() > 1) {
    *fused_iter = state.fuse(stage_id, *reduce_iters);
  } else {
    *fused_iter = (*reduce_iters)[0];
  }
  return state;
}

// Fuse all outer level space iterators
inline State FuseAllOuterSpaceIterators(State state, int stage_id, Iterator* fused_iter) {
  std::vector<Iterator> to_fuse;
  for (size_t iter_id = 0; iter_id < state->stages[stage_id]->iters.size(); ++iter_id) {
    const auto& it = state->stages[stage_id]->iters[iter_id];
    if (it->iter_type == kReduce || it->annotation != kNone) {
      break;
    }

    if (state->attach_map->iter_to_attached_stages.count(std::make_pair(stage_id, iter_id - 1))) {
      break;
    }
    to_fuse.push_back(it);
  }

  CHECK(!to_fuse.empty());
  if (to_fuse.size() > 1) {
    *fused_iter = state.fuse(stage_id, to_fuse);
  } else {
    *fused_iter = to_fuse[0];
  }
  return state;
}

// Compute prefix-sum probabiilty based on the given weights
inline void ComputePrefixSumProb(const std::vector<float>& weights,
                                 std::vector<double>* prefix_sum_probs) {
  // Compute selection probabilities.
  float sum = 0.0;
  prefix_sum_probs->resize(weights.size());
  for (size_t i = 0; i < weights.size(); ++i) {
    sum += std::max(weights[i], 0.0f);
    (*prefix_sum_probs)[i] = sum;
  }
  for (size_t i = 0; i < weights.size(); ++i) {
    (*prefix_sum_probs)[i] /= sum;
  }
}

// Random sample states
inline void RandomSampleStates(const std::vector<State>& in_states, std::mt19937* random_gen,
        size_t out_size, std::vector<State>* out_states) {
  out_states->clear();
  for (size_t i = 0; i < out_size; i++) {
    out_states->push_back(in_states[(*random_gen)() % in_states.size()]);
  }
}

// Random choose an index according to a prefix sum probability
inline int RandomChoose(const std::vector<double>& prefix_sum_probs, std::mt19937* random_gen) {
  std::uniform_real_distribution<> dis(0.0, 1.0);
  double x = dis(*random_gen);

  CHECK(!prefix_sum_probs.empty());

  return std::lower_bound(prefix_sum_probs.begin(), prefix_sum_probs.end(), x) -
      prefix_sum_probs.begin();
}

// Print all states
inline void PrintAllStates(const std::vector<State>& states) {
  for (size_t i = 0; i < states.size(); ++i) {
    std::cerr << i << std::endl;
    std::cerr << states[i];
    std::cerr << "==============================================" << std::endl;
  }
}

// Get all split steps for one stage
inline void GetSplitStepIds(const State& s, int stage_id, std::vector<int>* split_step_ids) {
  for (int i = static_cast<int>(s->transform_steps.size()) - 1; i >= 0; --i) {
    if (auto ps = s->transform_steps[i].as<SplitStepNode>()) {
      if (stage_id == ps->stage_id) {
        split_step_ids->push_back(i);
      }
    }

    if (IsStageNumberChangingStep(s->transform_steps[i])) {
      if (stage_id > s->transform_steps[i]->stage_id) {
        stage_id--;
      }
    }
  }
}

// Get the target stage id of a history step in the new state.
// We need this because the stage_id in the history may be stale due to later steps
inline int GetTargetStageIDInState(const State&s, int step_id) {
  int stage_inc = 0;  

  for (size_t i = step_id + 1; i < s->transform_steps.size(); ++i) {
    if (IsStageNumberChangingStep(s->transform_steps[i])) {
      if (s->transform_steps[i]->stage_id <= s->transform_steps[step_id]->stage_id + stage_inc)
        stage_inc++;
    }
  }
  return s->transform_steps[step_id]->stage_id + stage_inc;
}

// Get all split steps on spatial iterators for one multi-level tiled stage
void GetSpaceSplitStepIds(const State& s, int stage_id, std::vector<int>* spatial_split_step_ids);

// Get the possible compute locations for a stage
std::vector<std::pair<int, int>> GetComputeLocationCandidates(const SearchTask& task,
                                                              const State& state, int stage_id);

// Apply multi-level tiling structure according to a string format,
// where "S" stands a space level, "R" stands for a reudciton level.
// For example, if the format is "SSRSRS", the we will
// use tiling structure:  space_L0, space_L1, reduce_L0, space_L2, reduce_L1, space_L3
// For example, if apply "SSRSRS" to matrix multiplication,
// we have space iterators i and j, reduce iterator k.
// Then the tiling structure is : i0, j0, i1, j1, k0, i2, j2, k1, i3, j3
State DoMultiLevelTiling(const State& state, int stage_id, const std::string& format,
                         std::vector<int>* spatial_split_step_ids);

// Apply tiling structure: space, space, space, ..., with tile sizes from other SplitStep
State FollowTiling(const State& state, int stage_id,
                   const std::vector<int>& split_step_ids, int n_split);

// Randomly mutate the tile size of one SplitStep
State RandomMutateTileSize(const State& old_state, SplitFactorizationMemo* split_memo,
                           std::mt19937* random_gen, int max_innermost_split_factor);

// Randomly mutate the value of one auto_unroll_max_step PragmaStep
State RandomMutateMaxUnrollStep(const State& old_state, std::mt19937* random_gen,
                                const std::vector<int>& auto_unroll_configs);

// Randomly mutate the parallel degree of one stage.
State RandomMutateParallel(const State& old_state, std::mt19937* random_gen,
                           const SearchTask& task, int verbose = 0);

// Randomly mutate the computation location of one stage.
State RandomMutateComputeLocation(const State& old_state, std::mt19937* random_gen,
                                  const SearchTask& task);

// GA: Crossover two states
State CrossOverState(const SearchTask& task, std::mt19937* random_gen, const State& p1
                    const State& p2, std::vector<int>* fail_counters,float proportion);
                      
// Prune invalid states and return the results in-place.
void PruneInvalidState(const SearchTask& task, std::vector<State>* states);

}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_SEARCH_POLICY_UTILS_H_
