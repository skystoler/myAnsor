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
 * \file ansor/search_policy/sketch_search_policy.h
 * \brief The search policy that searches in a hierarchical search space defined by sketches.
 * The policy randomly samples programs from the space defined by sketches
 * and use evolutionary search to fine-tune them.
 */

#include "sketch_search_policy.h"
#include <tvm/runtime/registry.h>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include "utils.h"

namespace tvm {
namespace ansor {

TVM_REGISTER_NODE_TYPE(SketchSearchPolicyNode);
TVM_REGISTER_OBJECT_TYPE(PreloadCustomSketchRuleNode);

SketchSearchPolicy::SketchSearchPolicy(CostModel program_cost_model,
                                       Map<String, ObjectRef> params,
                                       int seed) {
  auto node = make_object<SketchSearchPolicyNode>();
  node->program_cost_model = std::move(program_cost_model);
  node->rand_gen_ = std::mt19937(seed);
  node->params = std::move(params);
  data_ = std::move(node);
}

State SketchSearchPolicyNode::Search(SearchTask task, int n_trials,
    int early_stopping, int num_measure_per_iter, int verbose,
    ProgramMeasurer measurer, Array<SearchCallback> pre_search_callbacks) {
  std::vector<State> best_states, random_states;
  this->cur_task = task;
  this->verbose = verbose;
  num_measure_per_iter_ = num_measure_per_iter;

  PrintTitle("Call pre-search callbacks", verbose);
  RunCallbacks(pre_search_callbacks);

  if (n_trials <= 1) {  // no measurement is allowed
    SearchOneRound(&best_states, 0, &random_states);
    CHECK_GT(best_states.size(), 0);
    return best_states[0];
  } else {
    std::vector<MeasureInput> inputs;
    std::vector<MeasureResult> results;
    int num_random = static_cast<int>(GetDoubleParam(params, "eps_greedy") * num_measure_per_iter);

    measurer->Reset();

    early_stopping = early_stopping < 0 ? std::numeric_limits<int>::max() >> 1 : early_stopping;

    int ct = 0;
    while (ct < n_trials) {
      if (!inputs.empty()) {
        // retrain cost models
        PrintTitle("Train cost model", verbose);
        program_cost_model->Update(inputs, results);
      }

      // Search one round to get promising states
      PrintTitle("Search", verbose);
      SearchOneRound(&best_states, num_random, &random_states);

      // Infer bound. This is necessary for computing the correct ToStr() for redundancy check
      cur_task->compute_dag.InferBound(&best_states);
      cur_task->compute_dag.InferBound(&random_states);

      // Pick `num_measure_per_iter` states to measure, check hash to remove already measured state
      // Also pick some random states to do eps-greedy
      PickStatesWithEpsGreedy(&inputs, best_states, random_states, n_trials - ct);

      // Have traversed all of the search space
      if (inputs.empty()) {
        StdCout(verbose) << "All candidates in the search space have been measured." << std::endl;
        break;
      }

      // Measure candidate states
      PrintTitle("Measure", verbose);
      measurer->Measure(cur_task, GetRef<SearchPolicy>(this), inputs, &results);
      ct += inputs.size();

      if (ct - measurer->best_ct[cur_task->workload_key] > early_stopping) {
        StdCout(verbose) << "Meet the early stopping condition." << std::endl;
        break;
      }

      // Update measured states. These states will join the LocalMutation in later rounds
      for (const auto& res : results) {
        measured_states_throughputs_.push_back(1.0 / FloatArrayMean(res->costs));
      }
    }
    PrintTitle("Done", verbose);

    return measurer->best_state[cur_task->workload_key];
  }
}

std::pair<Array<MeasureInput>, Array<MeasureResult> >
    SketchSearchPolicyNode::ContinueSearchOneRound(
    SearchTask task, int num_measure, int verbose, ProgramMeasurer measurer) {
  if (cur_task.defined()) {
    CHECK_EQ(cur_task, task);
  } else {
    cur_task = task;
  }
  this->verbose = verbose;
  num_measure_per_iter_ = num_measure;

  std::vector<State> best_states, random_states;
  std::vector<MeasureInput> inputs;
  std::vector<MeasureResult> results;
  int num_random = static_cast<int>(GetDoubleParam(params, "eps_greedy") * num_measure);

  // Search one round to get promising states
  PrintTitle("Search", verbose);
  SearchOneRound(&best_states, num_random * 2, &random_states);

  // Infer bound. This is necessary for computing the correct ToStr() for redundancy check
  cur_task->compute_dag.InferBound(&best_states);
  cur_task->compute_dag.InferBound(&random_states);

  // Pick `num_measure` states to measure, check hash to remove already measured state
  // Also pick some random states to do eps-greedy
  PickStatesWithEpsGreedy(&inputs, best_states, random_states, num_measure);

  // Measure candidate states
  PrintTitle("Measure", verbose);
  measurer->Measure(cur_task, GetRef<SearchPolicy>(this), inputs, &results);

  // Update throughputs of measured states. These states will join the LocalMutation in later rounds
  for (const auto& res : results) {
    measured_states_throughputs_.push_back(1.0 / FloatArrayMean(res->costs));
  }

  // Update the cost model
  Array<MeasureInput> inputs_arr(std::make_move_iterator(inputs.begin()),
                                 std::make_move_iterator(inputs.end()));
  Array<MeasureResult> results_arr(std::make_move_iterator(results.begin()),
                                   std::make_move_iterator(results.end()));

  PrintTitle("Train cost model", verbose);
  program_cost_model->Update(inputs_arr, results_arr);
  return std::make_pair(std::move(inputs_arr), std::move(results_arr));
}

void SketchSearchPolicyNode::PickStatesWithEpsGreedy(
    std::vector<MeasureInput>* inputs,
    const std::vector<State>& best_states,
    const std::vector<State>& random_states,
    int remaining_n_trials) {
  int num_random = static_cast<int>(GetDoubleParam(params, "eps_greedy") * num_measure_per_iter_);
  int num_good = num_measure_per_iter_ - num_random;

  inputs->clear();
  size_t offset_best = 0, offset_random = 0;

  while (static_cast<int>(inputs->size()) < std::min(num_measure_per_iter_, remaining_n_trials)) {
    const State* pstate;

    bool has_best = offset_best < best_states.size();
    bool has_random = offset_random < random_states.size();

    if (static_cast<int>(inputs->size()) < num_good) {
      // prefer best states
      if (has_best) {
        pstate = &best_states[offset_best++];
      } else if (has_random) {
        pstate = &random_states[offset_random++];
      } else {
        break;
      }
    } else {
      // prefer random states
      if (has_random) {
        pstate = &random_states[offset_random++];
      } else if (has_best) {
        pstate = &best_states[offset_best++];
      } else {
        break;
      }
    }

    // Check if it has already been measured
    std::string state_str = pstate->ToStr();

    if (measured_states_set_.count(state_str)) { continue; }
    measured_states_set_.insert(std::move(state_str));

    inputs->push_back(MeasureInput(cur_task, *pstate));
    measured_states_vector_.push_back(*pstate);
  }
}

void SketchSearchPolicyNode::SearchOneRound(std::vector<State>* best_states,
    int num_random_states, std::vector<State>* random_states) {
  best_states->clear();
  random_states->clear();

  // Get parameters
  int population = GetIntParam(params, "evolutionary_search_population");
  int num_use_measured = std::min(static_cast<int>(measured_states_vector_.size()),
      static_cast<int>(
          GetDoubleParam(params, "evolutionary_search_use_measured_ratio") * population));
  bool have_cost_model = !program_cost_model->IsInstance<RandomModelNode>();
  if (IsGPUTask(cur_task)) {
    auto_unroll_configs_ = {0, 16, 64, 512, 1024};
  } else {
    auto_unroll_configs_ = {0, 16, 64, 512};
  }

  if (!have_cost_model) {
    num_use_measured = 0;
  }

  // Generate sketches
  if (sketch_cache_.empty()) {
    sketch_cache_ = GenerateSketches();
  }

  if (GetBoolEnv("ANSOR_DEBUG_SKETCH_GENERATION")) {
    PrintAllStates(sketch_cache_);
    exit(0);
  }

  // Sample the init population
  std::vector<State> init_population;
  SampleInitPopulation(sketch_cache_, population - num_use_measured, &init_population);

  // PrintAllStates(init_population);
  // exit(0);

  if (have_cost_model) {
    // Also insert already measured good states to the initial population
    std::vector<int> indices;
    Argsort(measured_states_throughputs_, &indices);
    for (int i = 0; i < num_use_measured; i++) {
      init_population.push_back(measured_states_vector_[indices[i]]);
    }

    // Perform evolutionary search
    EvolutionarySearch(init_population, num_measure_per_iter_ * 2, best_states);
  } else {
    // If the cost model is useless (i.e. RandomCostModel), skip evolutionary search
    RandomSampleStates(init_population, &rand_gen_, num_measure_per_iter_ * 3, best_states);
  }

  // Sample some random states for eps-greedy
  RandomSampleStates(init_population, &rand_gen_, num_random_states * 10, random_states);
}

static inline bool ShouldBeCacheRead(
    const SketchSearchPolicyNode* policy, const State& state, int stage_id) {
  const SearchTask& task = policy->cur_task;

  // Don't cache_read a stage if it has multiple consumers
  const std::set<int>& consumers = GetConsumers(task, state, stage_id);
  if (consumers.size() != 1) {
    return false;
  }

  // Don't cache_read a stage if its consumer does not need multi-level tiling
  int target_stage_id = *consumers.begin();
  if (!NeedsMultilevelTiling(task, state, target_stage_id)) {
    return false;
  }

  // Don't cache_read a stage if its consumer does cross-thread reduction
  if (HasCrossThreadReduction(state, target_stage_id)) {
    return false;
  }

  // Only direct producers can be cache read
  const std::set<int>& producers = GetDirectProducers(task, state, target_stage_id);
  if (producers.find(stage_id) == producers.end()) {
    return false;
  }

  return true;
}

static inline bool ShouldAlwaysBeInlined(
    const SketchSearchPolicyNode* policy, const State& state, int stage_id) {
  const SearchTask& task = policy->cur_task;
  const Stage& stage = state->stages[stage_id];

  // check the inline condition of TE
  if (stage->op_type == kPlaceholder || IsOutputOp(task, state, stage_id) || HasReduceIter(stage)) {
    return false;
  }

  if (IsGPUTask(policy->cur_task)) {  // Greedily inline all inlinable ops on gpu
    return true;
  } else {
     // Only always-inline strict-inlinable ops on cpu.
     // The computation location of other ops will be tuned in InitPopulationChangeComputeLocation 
     // and EvolutionarySearch.
    return IsStrictInlineable(task, state, stage_id);
  }
}

// Handle special cases in Winograd transformation for GPU.
// We need to change the compute location of the producers of 
// compute ops that perform "fake reduction" with const tensors.
class RuleSpecialComputeLocationGPU : public SketchGenerationRule {
 public:
  ConditionEnum MeetCondition(const SketchSearchPolicyNode* policy,
      const State& state, int stage_id) final {
    const SearchTask& task = policy->cur_task;

    if (!ShouldAlwaysBeInlined(policy, state, stage_id)) {
      return kPass;
    }

    const std::set<int>& producers = GetProducers(task, state, stage_id);
    if (producers.empty()) {
      return kPass;
    }

    const std::set<int>& consumers = GetConsumers(task, state, stage_id);
    if (consumers.size() == 1 && state->stages[*consumers.begin()]->op->attrs.count(
                SearchPolicyNode::simplify_const_tensor_indices_key)) {
      return kApplyAndSkipRest;
    }

    return kPass;
  }

  std::vector<std::pair<State, int> > Apply(const SketchSearchPolicyNode* policy,
      const State& state, int stage_id) final {
    const SearchTask& task = policy->cur_task;

    State tmp_s = state;
    const std::set<int>& consumers = GetConsumers(task, state, stage_id);
    CHECK_EQ(consumers.size(), 1);

    // Get the last outer space iterator that is not unrolled.
    const Stage& target_stage = state->stages[*consumers.begin()];
    for (size_t i = 0; i < target_stage->iters.size(); ++i) {
      if (target_stage->iters[i]->annotation == kUnroll) {
        CHECK_GT(i, 0);

        tmp_s.compute_at(stage_id, *consumers.begin(),
                target_stage->iters[i-1]);
        break;
      }
    }

    return {std::make_pair(std::move(tmp_s), stage_id - 1)};
  }
};

// The rule that inlines simple elementwise ops
class RuleAlwaysInline : public SketchGenerationRule {
 public:
  ConditionEnum MeetCondition(const SketchSearchPolicyNode* policy,
      const State& state, int stage_id) final {
    return ShouldAlwaysBeInlined(policy, state, stage_id) ? kApplyAndSkipRest : kPass;
  }

  std::vector<std::pair<State, int> > Apply(const SketchSearchPolicyNode* policy,
      const State& state, int stage_id) final {
    State tmp_s = state;
    tmp_s.compute_inline(stage_id);
    return {std::make_pair(std::move(tmp_s), stage_id - 1)};
  }
};

// The rule that simply skips the current stage
class RuleSkipStage : public SketchGenerationRule {
 public:
  ConditionEnum MeetCondition(const SketchSearchPolicyNode* policy,
                              const State& state, int stage_id) final {
    return kApply;
  }

  std::vector<std::pair<State, int> > Apply(const SketchSearchPolicyNode* policy,
                                            const State& state, int stage_id) final {
    return {std::make_pair(state, stage_id - 1)};
  }
};

// The rule that performs multi-level tiling
class RuleMultiLevelTiling : public SketchGenerationRule {
 public:
  ConditionEnum MeetCondition(const SketchSearchPolicyNode* policy,
                              const State& state, int stage_id) final {
    const SearchTask& task = policy->cur_task;

    return NeedsMultilevelTiling(task, state, stage_id) ? kApplyAndSkipRest : kPass;
  }

  std::vector<std::pair<State, int> > Apply(const SketchSearchPolicyNode* policy,
                                            const State& state, int stage_id) final {
    std::string multi_level_tiling_structure = IsGPUTask(policy->cur_task) ?
        GetStringParam(policy->params, "gpu_multi_level_tiling_structure") :
        GetStringParam(policy->params, "cpu_multi_level_tiling_structure");

    std::vector<int> spatial_split_step_ids;
    State tmp_s = state;
    tmp_s = DoMultiLevelTiling(tmp_s, stage_id, multi_level_tiling_structure,
        &spatial_split_step_ids);
    return {std::make_pair(std::move(tmp_s), stage_id-1)};
  }
};

// The rule that performs multi-level tiling and fuses later consumers
class RuleMultiLevelTilingWithFusion : public SketchGenerationRule {
 public:
  ConditionEnum MeetCondition(const SketchSearchPolicyNode* policy,
                              const State& state, int stage_id) final {
    const SearchTask& task = policy->cur_task;
    int target_stage_id;

    if (NeedsMultilevelTiling(task, state, stage_id) &&
        HasSingleElementwiseMatchedConsumer(task, state, stage_id, &target_stage_id)) {

      // Always do fusion for stage with cache_write or GPU
      return HasCacheWriteStage(state, stage_id) || IsGPUTask(task) ?
          kApplyAndSkipRest : kApply;
    }

    return kPass;
  }

  std::vector<std::pair<State, int> > Apply(const SketchSearchPolicyNode* policy,
                                            const State& state, int stage_id) final {
    const SearchTask& task = policy->cur_task;

    std::string multi_level_tiling_structure = IsGPUTask(policy->cur_task) ?
        GetStringParam(policy->params, "gpu_multi_level_tiling_structure") :
        GetStringParam(policy->params, "cpu_multi_level_tiling_structure");

    std::vector<int> spatial_split_step_ids;
    int target_stage_id;

    CHECK(HasSingleElementwiseMatchedConsumer(task, state, stage_id, &target_stage_id));

    State base_state = state;
    base_state = DoMultiLevelTiling(base_state, stage_id,
        multi_level_tiling_structure, &spatial_split_step_ids);
    std::vector<int> follow_tiling_levels;
    if (IsGPUTask(task)) {
      follow_tiling_levels.push_back(3);
    } else {
      follow_tiling_levels.push_back(1);
      follow_tiling_levels.push_back(2);
    }

    std::vector<std::pair<State, int> > ret;
    for (int level : follow_tiling_levels) {
      if (tolower(multi_level_tiling_structure[level-1]) != 's') {
        continue;
      }
      State tmp_s = base_state;
      tmp_s = FollowTiling(tmp_s, target_stage_id, spatial_split_step_ids, level);
      const Iterator &target_iter = tmp_s->stages[target_stage_id]->iters[
          level * spatial_split_step_ids.size() - 1];
      tmp_s.compute_at(stage_id, target_stage_id, target_iter);

      ret.emplace_back(std::move(tmp_s), stage_id - 1);
    }

    return ret;
  }
};

// The rule that adds a cache write stage
class RuleAddCacheWrite : public SketchGenerationRule {
 public:
  ConditionEnum MeetCondition(const SketchSearchPolicyNode* policy,
                              const State& state, int stage_id) final {
    const SearchTask& task = policy->cur_task;
    int target_stage_id;

    // Don't cache_write a stage if it does cross-thread reduction
    if (HasCrossThreadReduction(state, stage_id)) {
      return kPass;
    }

    // Add cache write if a stage needs multi-level tiling,
    // but does not have a element-wise matched consumer
    if (NeedsMultilevelTiling(task, state, stage_id) &&
       !HasSingleElementwiseMatchedConsumer(task, state, stage_id, &target_stage_id)) {
      // Always do cache_write on GPU
      return IsGPUTask(task) ? kApplyAndSkipRest : kApply;
    }

    return kPass;
  }

  std::vector<std::pair<State, int> > Apply(const SketchSearchPolicyNode* policy,
                                            const State& state, int stage_id) final {
    const SearchTask& task = policy->cur_task;

    State tmp_s = state;
    tmp_s.cache_write(stage_id, "local", task->compute_dag);
    return {std::make_pair(std::move(tmp_s), stage_id)};
  }
};

// The rule that adds a cache read stage
// Mainly used for GPU cooperative fetching
// Currently only support 1 to 1 match cache read
class RuleAddCacheRead : public SketchGenerationRule {
 public:
  ConditionEnum MeetCondition(const SketchSearchPolicyNode* policy,
                              const State& state, int stage_id) final {
    return ShouldBeCacheRead(policy, state, stage_id) ? kApplyAndSkipRest : kPass;
  }

  std::vector<std::pair<State, int> > Apply(const SketchSearchPolicyNode* policy,
                                            const State& state, int stage_id) final {
    const SearchTask& task = policy->cur_task;
    const std::set<int>& consumers = GetConsumers(task, state, stage_id);
    CHECK_EQ(consumers.size(), 1);
    int target_stage_id = *consumers.begin();
    State tmp_s = state;

    // Cache read add shared memory
    int added_stage_id = tmp_s.cache_read(stage_id, "shared",
                                          {target_stage_id},
                                          task->compute_dag);
    target_stage_id++;
    const auto& share_read_pos = GetLastReduceIteratorInOutermostReduceTile(
        tmp_s->stages[target_stage_id]);
    tmp_s.compute_at(added_stage_id, target_stage_id, share_read_pos);
    return {std::make_pair(tmp_s, stage_id)};
  }
};

// The rule that adds rfactor stage
class RuleAddRfactor : public SketchGenerationRule {
 public:
  ConditionEnum MeetCondition(const SketchSearchPolicyNode* policy,
                              const State& state, int stage_id) final {
    const SearchTask& task = policy->cur_task;

    return NeedsRfactor(task, state, stage_id) && !HasCacheWriteStage(state, stage_id) ?
      kApply : kPass;
  }

  std::vector<std::pair<State, int> > Apply(const SketchSearchPolicyNode* policy,
                                            const State& state, int stage_id) final {
    const SearchTask& task = policy->cur_task;

    std::vector<std::pair<State, int> > ret;

    State tmp_s = state;

    // fuse all reduction iters
    std::vector<Iterator> space_iters, reduce_iters;
    Iterator fused_reduce_iter;
    tmp_s = FuseAllReductionIterators(tmp_s, stage_id,
            &fused_reduce_iter, &space_iters, &reduce_iters);

    // todo(lmzheng): We can do more analysis here to generate less and more efficient sketches.
    // In some cases, we only need rfactor for more parallel
    // In some cases, we only need rfactor for vectorization.
    // Now we will generate two versions and let the search figure out the bette one.

    // split reduction iters
    const auto &split_res = tmp_s.split(stage_id, fused_reduce_iter, {1});
    int factor_axis_id = static_cast<int>(space_iters.size());
    State base_state = tmp_s;
    for (const auto &split_iter : split_res) {
      tmp_s = base_state;
      int rstage_id = tmp_s.rfactor(stage_id, split_iter, factor_axis_id, task->compute_dag);

      // reorder the space iterator to innermost for vectorization
      if (split_iter == split_res[1]) {
        std::vector<Iterator> new_order;
        for (size_t i = 0; i < tmp_s->stages[rstage_id]->iters.size(); ++i) {
          if (i != space_iters.size()) {
            new_order.push_back(tmp_s->stages[rstage_id]->iters[i]);
          }
        }
        new_order.push_back(tmp_s->stages[rstage_id]->iters[space_iters.size()]);
        tmp_s.reorder(rstage_id, new_order);
      }

      ret.emplace_back(std::move(tmp_s), rstage_id - 1);
    }

    return ret;
  }
};

// The rule that adds a cache write stage
class RuleCrossThreadReduction : public SketchGenerationRule {
 public:
  ConditionEnum MeetCondition(const SketchSearchPolicyNode* policy,
                              const State& state, int stage_id) final {
    const SearchTask& task = policy->cur_task;

    CHECK(IsGPUTask(task));

    // If it is an intermidiate state created by RuleAddCacheWrite,
    // we just skip it.
    if (HasCacheWriteStage(state, stage_id)) {
      return kPass;
    }

    return NeedsCrossThreadReduction(task, state, stage_id) ? kApply : kPass;
  }

  std::vector<std::pair<State, int> > Apply(const SketchSearchPolicyNode* policy,
                                            const State& state, int stage_id) final {
    const SearchTask& task = policy->cur_task;
    State tmp_s = state;

    // fuse all reduction iters
    std::vector<Iterator> space_iters, reduce_iters;
    Iterator fused_reduce_iter;
    tmp_s = FuseAllReductionIterators(tmp_s, stage_id,
            &fused_reduce_iter, &space_iters, &reduce_iters);

    // Check the opportunity for kernel fusion
    bool fusible = false;
    int target_stage_id = GetSingleConsumerId(policy->cur_task, tmp_s, stage_id);
    int num_common_outer = -1;
    if (target_stage_id >= 0) {
      num_common_outer = GetNumCommonOuterIterator(policy->cur_task, tmp_s,
              stage_id, target_stage_id);
      if (num_common_outer > 0 &&
          !NeedsMultilevelTiling(policy->cur_task, state, target_stage_id)) {
          fusible = true;
      }
    }

    if (fusible) {
      const Stage& target_stage = state->stages[target_stage_id];
      std::vector<int> split_step_ids;

      GetSplitStepIds(tmp_s, target_stage_id, &split_step_ids);

      if (split_step_ids.size() == 0) {
        // If the target stage does not have split step,
        // it must be a simple stage without reduce iters.
        // We then should do a split for it.
        CHECK(!HasReduceIter(target_stage));
        const auto &split_res = tmp_s.split(target_stage_id, target_stage->iters.back(),
                                            {task->hardware_params->warp_size});
        tmp_s.bind_thread(target_stage_id, split_res[1], kThreadX);
        split_step_ids.push_back(tmp_s->transform_steps.size() - 2);
      }

      CHECK_EQ(split_step_ids.size(), 1);

      const Iterator& target_iter = tmp_s->stages[target_stage_id]->iters[num_common_outer - 1];
      const auto &split_res = tmp_s.follow_split(stage_id, fused_reduce_iter,
              split_step_ids[0], 1);
      tmp_s.bind_thread(stage_id, split_res[1], kThreadX);
      tmp_s.compute_at(stage_id, target_stage_id, target_iter);
    } else {
      const auto &split_res = tmp_s.split(stage_id, fused_reduce_iter,
                                          {task->hardware_params->warp_size});
      tmp_s.bind_thread(stage_id, split_res[1], kThreadX);
    }

    return {std::make_pair(std::move(tmp_s), stage_id - 1)};
  }
};

// The rule that deals with compute ops that perform "fake reduction" with const tensors.
// This kind of op comes from winograd transformation.
class RuleSimplifyComputeWithConstTensor : public SketchGenerationRule {
 public:
  ConditionEnum MeetCondition(const SketchSearchPolicyNode* policy,
                              const State& state, int stage_id) final {

    if (state->stages[stage_id]->op->attrs.count(
                SearchPolicyNode::simplify_const_tensor_indices_key)) { 
      return kApplyAndSkipRest;
    }

    return kPass;
  }

  std::vector<std::pair<State, int> > Apply(const SketchSearchPolicyNode* policy,
                                            const State& state, int stage_id) final {
    std::set<std::string> const_tensor_indices = GetIterNameSetParam(
            state->stages[stage_id]->op->attrs,
            SearchPolicyNode::simplify_const_tensor_indices_key);

    State tmp_s = state;
    std::vector<std::vector<Iterator>> tiled_outer_iters;
    std::vector<Iterator> unrolled_inner_iters;

    size_t tile_level = 2;
 
    for (const auto& iter : state->stages[stage_id]->iters) {
      if (const_tensor_indices.count(iter->name)) {
        // unroll indices of const tensors
        unrolled_inner_iters.push_back(tmp_s.unroll(stage_id, iter));
      } else {
        // tile other space indices
        CHECK_EQ(iter->iter_type, kSpace);

        tiled_outer_iters.push_back(tmp_s.split(stage_id, iter,
                    std::vector<PrimExpr>(tile_level - 1)));
      }
    }

    // reorder them
    std::vector<Iterator> new_order;
    for (size_t i = 0; i < tile_level; ++i) {
      for (size_t j = 0; j < tiled_outer_iters.size(); ++j) {
        new_order.push_back(tiled_outer_iters[j][i]);
      }
    }
    new_order.insert(new_order.end(), unrolled_inner_iters.begin(),
            unrolled_inner_iters.end());
    tmp_s.reorder(stage_id, new_order);

    return {std::make_pair(tmp_s, stage_id - 1)};
  }
};


std::vector<State> SketchSearchPolicyNode::GenerateSketches() {
  State init_state = cur_task->compute_dag.GetInitState();

  // two ping pong buffers to avoid copy
  std::vector<State> states_buf1, states_buf2;
  std::vector<State> *pnow, *pnext;
  pnow = &states_buf1;
  pnext = &states_buf2;
  pnow->push_back(init_state);

  // A map that maps state to its current working position (stage_id)
  std::unordered_map<State, int, ObjectHash, ObjectEqual> cur_stage_id_map;
  cur_stage_id_map[init_state] = static_cast<int>(init_state->stages.size() - 1);

  static RuleSkipStage rule_skip_stage;
  static RuleAlwaysInline rule_always_inline;
  static RuleMultiLevelTiling rule_multi_level_tiling;
  static RuleMultiLevelTilingWithFusion rule_multi_level_tiling_with_fusion;
  static RuleAddCacheWrite rule_add_cache_write_stage;
  static RuleAddCacheRead rule_add_cache_read_stage;
  static RuleAddRfactor rule_add_rfactor;
  static RuleCrossThreadReduction rule_cross_thread_reduction;
  static RuleSimplifyComputeWithConstTensor rule_simplify_compute_with_const_tensor;
  static RuleSpecialComputeLocationGPU rule_special_compute_location_gpu;;

  if (sketch_rules.empty()) {
    // Some rules require us to skip all the rest rules after they be applied.
    // So the rules below should be ordered carefully.

    if (IsGPUTask(cur_task)) {
      sketch_rules.push_back(&rule_add_cache_read_stage);
      sketch_rules.push_back(&rule_special_compute_location_gpu);
      sketch_rules.push_back(&rule_always_inline);
      sketch_rules.push_back(&rule_simplify_compute_with_const_tensor);
      sketch_rules.push_back(&rule_cross_thread_reduction);
      sketch_rules.push_back(&rule_add_cache_write_stage);
      sketch_rules.push_back(&rule_multi_level_tiling_with_fusion);
      sketch_rules.push_back(&rule_multi_level_tiling);
      sketch_rules.push_back(&rule_skip_stage);
    } else {
      sketch_rules.push_back(&rule_always_inline);
      sketch_rules.push_back(&rule_simplify_compute_with_const_tensor);
      sketch_rules.push_back(&rule_add_rfactor);
      sketch_rules.push_back(&rule_add_cache_write_stage);
      sketch_rules.push_back(&rule_multi_level_tiling_with_fusion);
      sketch_rules.push_back(&rule_multi_level_tiling);
      sketch_rules.push_back(&rule_skip_stage);
    }
  }

  // Derivation rule based enumeration
  std::vector<State> out_states;
  while (!pnow->empty()) {
    pnext->clear();

    for (const State& state : *pnow) {
      int stage_id = cur_stage_id_map[state];

      // Reaches to the terminal stage
      if (stage_id < 0) {
        out_states.push_back(state);
        continue;
      }

      // Try all derivation rules
      for (const auto& rule : sketch_rules) {
        auto cond = rule->MeetCondition(this, state, stage_id);
        if (cond == SketchGenerationRule::ConditionEnum::kApply ||
            cond == SketchGenerationRule::ConditionEnum::kApplyAndSkipRest) {
          for (const auto& pair : rule->Apply(this, state, stage_id)) {
            cur_stage_id_map[pair.first] = pair.second;
            pnext->push_back(pair.first);
          }

          // Skip the reset rules
          if (cond == SketchGenerationRule::ConditionEnum::kApplyAndSkipRest) {
            break;
          }
        }
      }
    }

    std::swap(pnow, pnext);
  }

  // Hack for rfactor: Replace the split factor for rfactor to the undefined Expr(),
  // so later we can sample random value for the split factor.
  // Why don't we use Expr() when doing the split for rfactor at the first time?
  // Because during ApplySteps, a rfactor with undefined Expr() will crash TVM.
  // So rfactor with undefined Expr() will conflict with cache_write, cache_read, rfactor
  // in other stages
  for (size_t i = 0; i < out_states.size(); ++i) {
    auto pstate = out_states[i].CopyOnWrite();
    for (size_t step_id = 0; step_id < pstate->transform_steps.size(); ++step_id) {
      if (pstate->transform_steps[step_id]->IsInstance<RfactorStepNode>()) {
        CHECK_GE(step_id, 1);
        int split_step_id = static_cast<int>(step_id - 1);
        auto step = pstate->transform_steps[split_step_id].as<SplitStepNode>();
        CHECK(step != nullptr);
        pstate->transform_steps[split_step_id]
            = SplitStep(step->stage_id, step->iter_id, step->extent, {PrimExpr()},
                        step->inner_to_outer);
      }
    }
  }

  StdCout(verbose) << "Generate Sketches\t\t#s: " << out_states.size() << std::endl;
  return out_states;
}

int InitPopulationFillTileSize(const SketchSearchPolicyNode& policy,
                               State* state, std::mt19937* rand_gen,
                               SplitFactorizationMemo* split_memo) {
  int max_innermost_split_factor = GetMaxInnermostSplitFactor(policy.cur_task, policy.params);

  // Scan the transformation history and randomly fill tiles size for all SplitStep
  for (size_t step_id = 0; step_id < (*state)->transform_steps.size(); ++step_id) {
    if (auto ps = (*state)->transform_steps[step_id].as<SplitStepNode>()) {
      bool defined = true;
      for (const PrimExpr& len : ps->lengths) {
        if (!len.defined()) {
          defined = false;
        }
      }

      if (defined) {
        continue;
      }

      int extent = GetIntImm(ps->extent);
      const std::vector<std::vector<PrimExpr> >& candidate_lens =
          split_memo->GetFactorizationSchemes(extent, ps->lengths.size(),
                                              max_innermost_split_factor);

      StateNode* pstate = state->CopyOnWrite();
      pstate->transform_steps[step_id] = SplitStep(
          ps->stage_id, ps->iter_id, ps->extent,
          candidate_lens[(*rand_gen)() % candidate_lens.size()],
          ps->inner_to_outer);
    }
  }

  return 0;
}

int InitPopulationThreadBind(const SketchSearchPolicyNode* policy, State* state) {
  std::set<int> multi_level_tiling_root_set;

  for (size_t stage_id = 0; stage_id < (*state)->stages.size(); ++stage_id) {
    if (NeedsMultilevelTiling(policy->cur_task, *state, stage_id)) {
      const Stage& stage = (*state)->stages[stage_id];
      if (stage->compute_at != kIter) {
        // This stage is not multi-level tiled,
        // so it must be produced by RuleCrossThreadReduction.
        CHECK(HasCrossThreadReduction(*state, stage_id));
        continue;
      }
      CHECK_EQ(stage->compute_at, kIter);
      const auto res = (*state)->attach_map->stage_to_attach_iter.find(stage_id);
      CHECK(res != (*state)->attach_map->stage_to_attach_iter.end());

      multi_level_tiling_root_set.insert(res->second.first);
    }
  }

  for (size_t stage_id = 0; stage_id < (*state)->stages.size(); ++stage_id) {
    const Stage& stage = (*state)->stages[stage_id];

    if (stage->compute_at == kInlined || stage->op_type == kPlaceholder) {
      continue;
    }

    // Deal with the cross-thread reduction generated by RuleCrossThreadReduction
    if (HasCrossThreadReduction(*state, stage_id)) {
      if (stage->compute_at != kRoot) {
        continue;
      }

      Iterator fused_it;
      *state = FuseAllOuterSpaceIterators(*state, stage_id, &fused_it);
      state->bind_thread(stage_id, fused_it, kBlockX);
      continue;
    }

    // Skip if this stage has already been annotaed with threadIdx.x or has been tensorized
    if (HasAnnotatedIter(stage, IteratorAnnotation::kThreadX) ||
        HasAnnotatedIter(stage, IteratorAnnotation::kTensorized)) {
      continue;
    }

    if (stage->compute_at == kRoot) {
      // This stage has not been tiled, but in GPU schedule, we must tile the root stage
      // to do thread binding

      if (!multi_level_tiling_root_set.count(stage_id)) {
        Iterator fused_it;
        *state = FuseAllOuterSpaceIterators(*state, stage_id, &fused_it);

        if (GetExtent(fused_it) <= policy->cur_task->hardware_params->warp_size) {
          state->bind_thread(stage_id, fused_it, kThreadX);
        } else {
          // Set threadIdx.x = default_warp_size by default.
          // The later EvolutionarySearch will try more possiblity
          const auto& split_its = state->split(stage_id, fused_it,
              {policy->cur_task->hardware_params->warp_size});
          state->bind_thread(stage_id, split_its[0], kBlockX);
          state->bind_thread(stage_id, split_its[1], kThreadX);
        }
        continue;
      }

      auto pop = stage->op.as<te::ComputeOpNode>();
      std::vector<Iterator> to_fuse;

      // The remaining part deals with the thread binding for multi-level tiled stages
      int total_space_extent = 1;
      for (const auto& i : pop->root_iter_vars()) {
        CHECK(i->dom.defined());
        const auto& pint = i->dom->extent.as<IntImmNode>();
        CHECK(pint);
        total_space_extent *= pint->value;
      }

      // Check if the total space extent is too small for multi-level thread binding
      if (total_space_extent <= policy->cur_task->hardware_params->warp_size) {
        Iterator fused_it;
        *state = FuseAllOuterSpaceIterators(*state, stage_id, &fused_it);
        state->bind_thread(stage_id, fused_it, kThreadX);
        continue;
      }

      // Fuse the outermost space tile as blockIdx
      for (size_t i = 0; i < pop->axis.size(); i++) {
        const auto& it = (*state)->stages[stage_id]->iters[i];
        if (!StrEndsWith(it->name, ".0")) {
          break;
        }
        to_fuse.push_back(it);
      }
      const auto& blockidx_it = state->fuse(stage_id, to_fuse);
      state->bind_thread(stage_id, blockidx_it, kBlockX);

      // Fuse the second outermost space tile as vthread
      to_fuse.clear();
      for (size_t i = 1; i < pop->axis.size() + 1; i++) {
        const auto& it = (*state)->stages[stage_id]->iters[i];
        if (!StrEndsWith(it->name, ".1")) {
          break;
        }
        to_fuse.push_back((*state)->stages[stage_id]->iters[i]);
      }
      const auto& vthread_it = state->fuse(stage_id, to_fuse);
      if (GetExtent(vthread_it) > policy->cur_task->hardware_params->max_vthread_extent) {
        return -1;
      }
      state->bind_thread(stage_id, vthread_it, kVThread);

      // Fuse the third outermost space tile as threadIdx
      to_fuse.clear();
      for (size_t i = 2; i < pop->axis.size() + 2; i++) {
        const auto& it = (*state)->stages[stage_id]->iters[i];
        if (!StrEndsWith(it->name, ".2")) {
          break;
        }
        to_fuse.push_back((*state)->stages[stage_id]->iters[i]);
      }
      const auto& threadidx_it = state->fuse(stage_id, to_fuse);
      if (GetExtent(threadidx_it) < policy->cur_task->hardware_params->warp_size) {
        return -1;
      }
      state->bind_thread(stage_id, threadidx_it, kThreadX);
    } else if (stage->compute_at == kIter && StrEndsWith(stage->op->name, ".shared")) {
      // Do cooperative fetching for the cache read stage.
      // Get spatial_split_step_ids from the root stage
      CHECK(stage->compute_at == kIter);
      const auto& it = (*state)->attach_map->stage_to_attach_iter.find(stage_id);
      CHECK(it != (*state)->attach_map->stage_to_attach_iter.end());
      std::vector<int> spatial_split_step_ids;
      GetSpaceSplitStepIds(*state, it->second.first, &spatial_split_step_ids);

      // Fuse all iterators to do cooperative fetching
      Iterator fused = state->fuse(stage_id, (*state)->stages[stage_id]->iters);
      // Split out an extra iterator for vectorization
      // The later EvolutionarySearch will try more possiblity
      const auto& iters0 = state->split(stage_id, fused, {1});
      state->vectorize(stage_id, iters0[1]);
      // Follow split to keep a same thread extent with the root stage
      const auto& iters1 = state->follow_fused_split(stage_id, iters0[0],
                                                    spatial_split_step_ids,
                                                    1, true);
      state->bind_thread(stage_id, iters1[1], kThreadX);
    }
  }

  return 0;
}

int InitPopulationChangeComputeLocation(const SketchSearchPolicyNode* policy,
                                        State* state, std::mt19937* rand_gen) {
  if (GetIntParam(policy->params, "disable_change_compute_location")) {
    return 0;
  }

  // Randomly change the computation location for some stages
  for (int stage_id = static_cast<int>((*state)->stages.size()) - 1; stage_id >= 0; stage_id--) {
    const Stage& stage = (*state)->stages[stage_id];

    // Skip the inlined stages and placeholders
    if (stage->op_type == kPlaceholder || stage->compute_at == kInlined) {
      continue;
    }
    // Skip the tiled stages
    if (IsTiled(stage) || NeedsMultilevelTiling(policy->cur_task, *state, stage_id)) {
      continue;
    }

    int target_stage_id = GetSingleConsumerId(policy->cur_task, *state, stage_id);
    if (target_stage_id < 0) {
      continue;
    }

    std::vector<std::pair<int, int>> candidates =
        GetComputeLocationCandidates(policy->cur_task, *state, stage_id);

    int choice = (*rand_gen)() % (candidates.size() + 2);

    if (choice == 0) {
      if (!HasReduceIter(stage)) {
        const auto& stage_to_attach_iter = (*state)->attach_map->stage_to_attach_iter;
        if (stage_to_attach_iter.find(stage_id) != stage_to_attach_iter.end()) {
          state->compute_inline(stage_id);
        }
      }
    } else if (choice == 1) {
      state->compute_root(stage_id);
    } else {
      choice = choice - 2;
      const Stage& stage = (*state)->stages[candidates[choice].first];
      state->compute_at(stage_id, candidates[choice].first,
                        stage->iters[candidates[choice].second]);
    }
  }

  return 0;
}

int InitPopulationParallel(const SketchSearchPolicyNode* policy,
                           State* state) {
  // Annotate parallel for CPU
  std::function<void(const SketchSearchPolicyNode*, State*, int stage_id, int iter_offset)>
      annotate_parallel;

  annotate_parallel = [&annotate_parallel](
          const SketchSearchPolicyNode* policy, State* state, int stage_id, int iter_offset) {
    const Stage& stage = (*state)->stages[stage_id];

    std::vector<Iterator> to_fuse;
    int64_t parallel_degree = 1;

    // strategy: try to fuse and parallel the outermost n iterators
    // Stop if we meet reduce iterator or we have enough parallel degree
    size_t iter_id = iter_offset;
    for (; iter_id < stage->iters.size(); ++iter_id) {
      const Iterator& it = stage->iters[iter_id];
      if (it->iter_type == kReduce || it->annotation != kNone) {
        break;
      }

      to_fuse.push_back(it);
      parallel_degree *= GetExtent(it);

      if (parallel_degree > policy->cur_task->hardware_params->num_cores * 16) {
        break;
      }

      if ((*state)->attach_map->iter_to_attached_stages.count(
          std::make_pair(stage_id, iter_id))) {
        break;
      }
    }

    if (parallel_degree == 1) {
      auto res =
          (*state)->attach_map->iter_to_attached_stages.find(std::make_pair(stage_id, iter_id));
      if (res != (*state)->attach_map->iter_to_attached_stages.end()) {
        for (int attached_stage_id : res->second) {
          annotate_parallel(policy, state, attached_stage_id, 0);
        }
        annotate_parallel(policy, state, stage_id, iter_id + 1);
      }
    }

    if (!to_fuse.empty()) {
      if (to_fuse.size() == 1) {
        state->parallel(stage_id, to_fuse[0]);
      } else {
        Iterator fused_iter = state->fuse(stage_id, to_fuse);
        state->parallel(stage_id, fused_iter);
      }
    }
  };

  for (size_t stage_id = 0; stage_id < (*state)->stages.size(); ++stage_id) {
    const Stage& stage = (*state)->stages[stage_id];
    if (stage->compute_at != kRoot || stage->op_type == kPlaceholder) {
      continue;
    }

    annotate_parallel(policy, state, stage_id, 0);
  }

  return 0;
}

int InitPopulationVectorization(const SketchSearchPolicyNode* policy,
                                State* state, std::mt19937* rand_gen) {
  for (size_t stage_id = 0; stage_id < (*state)->stages.size(); ++stage_id) {
    const Stage& stage = (*state)->stages[stage_id];

    if (stage->compute_at == kInlined || stage->op_type == kPlaceholder) {
      continue;
    }

    // Skip cooperative fetching stage
    if (IsGPUTask(policy->cur_task) && HasCacheReadStage(*state, stage_id - 1)) {
      continue;
    }

    if (HasAnnotatedIter(stage, IteratorAnnotation::kTensorized)) {
      // Skip if this stage has been tensorized
      continue;
    }

    // try to fuse and vectorize the space iterators in the inner most tile
    int cum_length_prod = 1;

    int num_fusible = 0;
    while (num_fusible < static_cast<int>(stage->iters.size())) {
      int iter_id = static_cast<int>(stage->iters.size()) - 1 - num_fusible;
      if ((*state)->attach_map->iter_to_attached_stages.count(
          std::make_pair(stage_id, iter_id))) {
        break;
      }

      const Iterator& it = stage->iters[iter_id];

      // Stop if we meet a reduce iterator
      if (it->iter_type == kReduce || it->annotation != kNone) {
        break;
      }

      // Stop if the memory access is not continuous (vectorizable)
      // Note: The check is too hard, so we use heuristic here
      if (IsTiled(stage) && num_fusible != 0) {
        // If the stage is tiled, then the memory access must not be continuous
        // for the innermost two iterators
        break;
      }

      cum_length_prod *= GetExtent(it);
      if (cum_length_prod > policy->cur_task->hardware_params->max_unroll_vec) {
        break;
      }

      num_fusible++;
    }

    if (num_fusible > 1) {
      num_fusible = 1 + (*rand_gen)() % (num_fusible - 1);  // Select a random range to fuse
    }

    if (num_fusible == 1) {
      state->vectorize(stage_id, stage->iters.back());
    } else if (num_fusible > 1) {
      std::vector<Iterator> to_fuse(stage->iters.end() - num_fusible,
                                    stage->iters.end());
      state->vectorize(stage_id, state->fuse(stage_id, to_fuse));
    }
  }

  return 0;
}

int InitPopulationUnroll(const SketchSearchPolicyNode* policy,
                         State* state, std::mt19937* rand_gen,
                         const std::vector<int>& auto_unroll_configs) {
  // Add pragma auto_unroll_max_step for some stages
  for (size_t stage_id = 0; stage_id < (*state)->stages.size(); ++stage_id) {
    const Stage& stage = (*state)->stages[stage_id];

    if (stage->compute_at == kInlined || stage->op_type == kPlaceholder) {
      continue;
    }

    if (stage->op->attrs.count(SearchPolicyNode::always_unroll_inner_key)) {
      // Special unroll policy
      const auto& to_unroll_name_set = GetIterNameSetParam(stage->op->attrs,
              SearchPolicyNode::always_unroll_inner_key);
      std::set<std::string> visited_names;

      // Unroll the space iterators and reduce iterators listed in the attrs
      // in the innermost tile
      int n = static_cast<int>(stage->iters.size()) - 1;
      visited_names.clear();
      while (n >= 0) {
        const Iterator& it = stage->iters[n];

        // If we meet two iterators that come from a same original iterator,
        // then we are out of the innermost tile
        size_t size_before = visited_names.size();
        ExtractOriginalIterators(it->name, &visited_names);
        if (size_before == visited_names.size()) {
          break;
        }

        std::set<std::string> name;
        ExtractOriginalIterators(it->name, &name);

        if (name.size() == 1 && to_unroll_name_set.count(*name.begin())) {
          if (it->annotation == kNone) {
            state->unroll(stage_id, it);
          }
        }

        n--;
      }
    }

    bool annotate_auto_unroll = HasReduceIter(stage);
    if (IsGPUTask(policy->cur_task)) {
      if (!NeedsMultilevelTiling(policy->cur_task, *state, stage_id)
          || HasRfactorStage(*state, stage_id)) {
        annotate_auto_unroll = false;
      }
    }

    if (annotate_auto_unroll) {
      // use auto unroll for multi level tiled stage
      int value = auto_unroll_configs[(*rand_gen)() % auto_unroll_configs.size()];
      state->pragma(stage_id, (*state)->stages[stage_id]->iters[0],
                    std::string("auto_unroll_max_step") + "$" + std::to_string(value));
    }
  }

  return 0;
}

void SketchSearchPolicyNode::SampleInitPopulation(const std::vector<State>& sketches,
    int out_size, std::vector<State>* out_states) {
  auto tic_begin = std::chrono::high_resolution_clock::now();

  std::uniform_real_distribution<> dis(0.0, 1.0);
  int fail_ct = 0;

  // TODO(lmzheng, jcf94): Try to parallel this while loop
  while (static_cast<int>(out_states->size()) < out_size
          && fail_ct < static_cast<int>(out_size)) {
    State tmp_s = sketches[rand_gen_() % sketches.size()];

    InitPopulationFillTileSize(*this, &tmp_s, &rand_gen_, &split_memo_);

    if (IsGPUTask(cur_task)) {
      tmp_s = cur_task->compute_dag.InferBound(tmp_s);

      if (InitPopulationThreadBind(this, &tmp_s)) {
        fail_ct++;
        continue;
      }
    } else {
      InitPopulationChangeComputeLocation(this, &tmp_s, &rand_gen_);

      tmp_s = cur_task->compute_dag.InferBound(tmp_s);

      InitPopulationParallel(this, &tmp_s);
    }

    if (cur_task->target->target_name != "cuda") {  // don't explicitly do vectorization for CUDA
      InitPopulationVectorization(this, &tmp_s, &rand_gen_);
    }

    InitPopulationUnroll(this, &tmp_s, &rand_gen_, this->auto_unroll_configs_);

    out_states->push_back(std::move(tmp_s));
  }

  double duration = std::chrono::duration_cast<std::chrono::duration<double> >(
      std::chrono::high_resolution_clock::now()-  tic_begin).count();
  StdCout(verbose) << "Sample Initial Population\t#s: " << out_states->size()
                   << "\tfail_ct: " << fail_ct << "\tTime elapsed: "
                   << std::fixed << std::setprecision(2) << duration << std::endl;
}

void SketchSearchPolicyNode::EvolutionarySearch(
    const std::vector<State>& init_population,
    int num_best_states, std::vector<State>* best_states) {
  auto tic_begin = std::chrono::high_resolution_clock::now();

  int max_innermost_split_factor = GetMaxInnermostSplitFactor(cur_task, params);

  // Set parameters for genetic algorithm
  size_t population = GetIntParam(params, "evolutionary_search_population");
  int num_iters =  GetIntParam(params, "evolutionary_search_num_iters");
  double mutation_prob = GetDoubleParam(params, "evolutionary_search_mutation_prob");
  double crossover_ratio = GetDoubleParam(params, "evolutionary_search_crossover_ratio");
  int num_cross_over = static_cast<int>(population * crossover_ratio);
  int num_cross_over_trial_upper_bound = num_cross_over;

  // Two ping pong buffers to avoid copy
  std::vector<State> states_buf1, states_buf2;
  std::vector<State> *pnow = &states_buf1, *pnext = &states_buf2;
  states_buf1.reserve(population);
  states_buf2.reserve(population);
  states_buf1.insert(states_buf1.begin(), init_population.begin(), init_population.end());

  // A heap to keep the best states during evolution
  using StateHeapItem = std::pair<State, float>;
  auto cmp = [](const StateHeapItem& left, const StateHeapItem& right) {
    return left.second > right.second;
  };
  std::vector<StateHeapItem> heap;
  std::unordered_set<std::string> in_heap(measured_states_set_);
  heap.reserve(num_best_states);

  // auxiliary global variables
  std::vector<float> pop_scores;
  std::vector<double> pop_selection_probs;
  double max_score = 0.0;
  pop_scores.reserve(population);
  pop_selection_probs.reserve(population);
  std::uniform_real_distribution<> dis(0.0, 1.0);

  // Mutation and crossover counters
  int mutation_success_ct, mutation_fail_ct;
  int crossover_success_ct, crossover_fail_ct;
  std::vector<int> crossover_fail_counters = {0, 0, 0, 0, 0};
  mutation_success_ct = mutation_fail_ct = crossover_success_ct = crossover_fail_ct = 0;

  // Mutation rule selection probability
  std::vector<float> rule_weights;
  std::vector<double> rule_selection_probs;
  if (IsGPUTask(cur_task)) {
    rule_weights = {0.90, 0.10, 0.00, 0.00};
  } else {
    rule_weights = {0.90, 0.03, 0.05, 0.02};
  }
  if (GetIntParam(params, "disable_change_compute_location")) {
    rule_weights[2] = 0.0;
  }
  ComputePrefixSumProb(rule_weights, &rule_selection_probs);
  
  // Genetic Algorithm
  for (int k = 0; k < num_iters + 1; ++k) {
    // Maintain the heap
    cur_task->compute_dag.InferBound(pnow);
    PruneInvalidState(cur_task, pnow);
    program_cost_model->Predict(cur_task, *pnow, &pop_scores);

    for (size_t i = 0; i < pnow->size(); ++i) {
      const State& state = (*pnow)[i];
      std::string state_str = state.ToStr();

      if (in_heap.count(state_str) == 0) {
        if (static_cast<int>(heap.size()) < num_best_states) {
          heap.emplace_back((*pnow)[i], pop_scores[i]);
          std::push_heap(heap.begin(), heap.end(), cmp);
          in_heap.insert(state_str);
        } else if (pop_scores[i] > heap.front().second) {
          std::string old_state_str = heap.front().first.ToStr();
          in_heap.erase(old_state_str);
          in_heap.insert(state_str);

          std::pop_heap(heap.begin(), heap.end(), cmp);
          heap.back() = StateHeapItem(state, pop_scores[i]);
          std::push_heap(heap.begin(), heap.end(), cmp);
        }
        if (pop_scores[i] > max_score) {
          max_score = pop_scores[i];
        }
      }
    }

    if (k % 5 == 0 || k == num_iters) {
      StdCout(verbose) << "GA Iter: " << k << std::fixed << std::setprecision(4)
                       << "\tMax score: " << max_score
                       << "\tMin score: " << heap.front().second
                       << "\t#Pop: " << pnow->size()
                       << "\t#C+: " << crossover_success_ct / (k+1)
                       << "\t#C-: " << crossover_fail_ct / (k+1)
                       << "\t#M+: " << mutation_success_ct / (k+1)
                       << "\t#M-: " << mutation_fail_ct / (k+1)
                       << std::endl;
      //std::cerr << "Crossover fail counters : ";
      //for (int x : crossover_fail_counters) {
      //    std::cerr << x / (k+1) << " ";
      //}
      //std::cerr << "\n";
    }
    if (k == num_iters) {
      break;
    }

    // Compute selection probability
    ComputePrefixSumProb(pop_scores, &pop_selection_probs);

    // Do cross over
    int ct = 0;
    while (cross_over_enabled_ &&
           static_cast<int>(pnext->size()) < num_cross_over &&
           ct < num_cross_over_trial_upper_bound) {
      int p1 = RandomChoose(pop_selection_probs, &rand_gen_);
      int p2 = RandomChoose(pop_selection_probs, &rand_gen_);

      if (p1 == p2 || (*pnow)[p1].ToStr() == (*pnow)[p2].ToStr()) {
        pnext->push_back((*pnow)[p1]);
      } else {
        State tmp_s = CrossOverState(cur_task, &rand_gen_, (*pnow)[p1], (*pnow)[p2], &crossover_fail_counters);
        if (tmp_s.defined()) {
          //std::cerr << (*pnow)[p1] << std::endl;
          //std::cerr << "========================================" << std::endl;
          //std::cerr << (*pnow)[p2] << std::endl;
          //std::cerr << "========================================" << std::endl;
          //tmp_s = cur_task->compute_dag.InferBound(tmp_s);
          //std::cerr << tmp_s << std::endl;
          //std::cerr << "========================================" << std::endl;
          ////std::cerr << cur_task->compute_dag.PrintStepsAsPython(tmp_s->transform_steps);
          //exit(0);
          pnext->push_back(std::move(tmp_s));
          crossover_success_ct++;
        } else {
          crossover_fail_ct++;
        }
      }
      ct++;
    }

    // Turn off crossover forever if we cannot perform it successfully
    if (crossover_success_ct == 0) {
      cross_over_enabled_ = false;
      crossover_success_ct = crossover_fail_ct = -1;
    }

    // Do mutation
    while (pnext->size() < population) {
      int id = RandomChoose(pop_selection_probs, &rand_gen_);

      if (dis(rand_gen_) < mutation_prob) {
        int rule_id = RandomChoose(rule_selection_probs, &rand_gen_);

        State tmp_s;
        switch (rule_id) {
          case 0:
            tmp_s = RandomMutateTileSize((*pnow)[id], &split_memo_, &rand_gen_,max_innermost_split_factor);
            break;
          case 1:
            tmp_s = RandomMutateMaxUnrollStep((*pnow)[id], &rand_gen_, auto_unroll_configs_);
            break;
          case 2:
            tmp_s = RandomMutateComputeLocation((*pnow)[id], &rand_gen_, cur_task);
            break;
          case 3:
            tmp_s = RandomMutateParallel((*pnow)[id], &rand_gen_, cur_task);
            break;
	        case 4:
            tmp_s = RandomReorder((*pnow)[id], &rand_gen_, cur_task);
	          break;
          default:
            LOG(FATAL) << "Invalid rule id: " << rule_id;
        }

        if (tmp_s.defined()) {
          mutation_success_ct++;
          pnext->push_back(std::move(tmp_s));
        } else {
          mutation_fail_ct++;
        }
      } else {
        pnext->push_back((*pnow)[id]);
      }
    }

    std::swap(pnext, pnow); pnext->clear();
  }

  // Copy best states in the heap to out_states
  std::sort(heap.begin(), heap.end(), cmp);
  best_states->clear();
  for (auto& item : heap) {
    best_states->push_back(std::move(item.first));
  }

  double duration = std::chrono::duration_cast<std::chrono::duration<double> >(
      std::chrono::high_resolution_clock::now() - tic_begin).count();
  StdCout(verbose) << "EvolutionarySearch\t\t#s: " << best_states->size()
                   << "\tTime elapsed: "
                   << std::fixed << std::setprecision(2) << duration << std::endl;
}

/*!
 * \brief Base class for custom sketch generation rules
 */
class RuleCustomSketch : public SketchGenerationRule {
 public:
  RuleCustomSketch(PackedFunc meet_condition_func, PackedFunc apply_func) :
      meet_condition_func_(std::move(meet_condition_func)),
      apply_func_(std::move(apply_func)) {}

  inline ConditionEnum MeetCondition(const SketchSearchPolicyNode* policy,
                                     const State& state, int stage_id) final {
    auto ret = meet_condition_func_(
        tvm::runtime::GetRef<SketchSearchPolicy>(policy), state, stage_id);
    if (ret.type_code() == 0) {
      return ConditionEnum(static_cast<int>(ret));
    } else {
      return kApplyAndSkipRest;
    }
  }

  inline std::vector<std::pair<State, int> > Apply(
      const SketchSearchPolicyNode* policy,
      const State& state, int stage_id) final {
    std::vector<std::pair<State, int> > ret;

    Array<Array<ObjectRef>> apply_ret = apply_func_(
        tvm::runtime::GetRef<SketchSearchPolicy>(policy), state, stage_id);

    for (const auto& item : apply_ret) {
      CHECK_EQ(item.size(), 2);
      auto next = item[1].as<IntImmNode>();
      ret.emplace_back(Downcast<State>(item[0]), next->value);
    }
    return ret;
  }

 private:
  PackedFunc meet_condition_func_;
  PackedFunc apply_func_;
};

PreloadCustomSketchRule::PreloadCustomSketchRule(PackedFunc meet_condition_func,
                                                 PackedFunc apply_func) {
  auto node = make_object<PreloadCustomSketchRuleNode>();
  node->meet_condition_func = std::move(meet_condition_func);
  node->apply_func = std::move(apply_func);
  data_ = std::move(node);
}

void PreloadCustomSketchRuleNode::callback(SearchPolicyNode* policy) {
  CHECK(policy->IsInstance<SketchSearchPolicyNode>());
  auto sketch_policy = dynamic_cast<SketchSearchPolicyNode*>(policy);
  sketch_policy->sketch_rules.emplace_back(
      new RuleCustomSketch(meet_condition_func, apply_func));
  StdCout(policy->verbose) << "Custom sketch rule added." << std::endl;
}

TVM_REGISTER_GLOBAL("ansor.SketchSearchPolicy")
.set_body_typed([](CostModel program_cost_model, Map<String, ObjectRef> params, int seed) {
  return SketchSearchPolicy(program_cost_model, params, seed);
});

TVM_REGISTER_GLOBAL("ansor.SketchSearchPolicyGenerateSketches")
.set_body_typed([](SketchSearchPolicy policy, SearchTask task){
  policy->cur_task = std::move(task);
  return Array<State>(policy->GenerateSketches());
});

TVM_REGISTER_GLOBAL("ansor.PreloadCustomSketchRule")
.set_body_typed([](PackedFunc meet_condition_func, PackedFunc apply_func) {
  return PreloadCustomSketchRule(meet_condition_func, apply_func);
});

}  // namespace ansor
}  // namespace tvm
