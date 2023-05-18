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
 * and use evolutionary search to  fine-tune them.
 */

#ifndef TVM_ANSOR_SEARCH_POLICY_SKETCH_SEARCH_POLICY_H_
#define TVM_ANSOR_SEARCH_POLICY_SKETCH_SEARCH_POLICY_H_

#include <vector>
#include <string>
#include <utility>
#include <unordered_set>
#include <set>
#include "search_policy.h"
#include "../cost_model/cost_model.h"
#include "../utils.h"


namespace tvm {
namespace ansor {

// LRU
class LRUList{
public:
      LRUList(std::string x, LRUList *y, LRUList *z){
          val = x, pre = y, next = z;
      }
      std::string val;
      LRUList *pre, *next;
};

class SketchSearchPolicyNode;

/*!
 * \brief The base class for derivation rules used in the sketch generation
 */
class SketchGenerationRule {
 public:
  enum ConditionEnum {
    kPass, kApply, kApplyAndSkipRest
  };

  virtual ConditionEnum MeetCondition(const SketchSearchPolicyNode* policy,
                                      const State& state, int stage_id) = 0;
  virtual std::vector<std::pair<State, int> > Apply(const SketchSearchPolicyNode* policy,
                                                    const State& state, int stage_id) = 0;
};

/*!
 * \brief The search policy that searches in a hierarchical search space defined by sketches.
 * The policy randomly samples programs from the space defined by sketches
 * and use evolutionary search to  fine-tune them.
 */
class SketchSearchPolicyNode: public SearchPolicyNode {
 public:
  /*! \brief The cost model for complete programs */
  CostModel program_cost_model;
  /*! \brief Random generator */
  std::mt19937 rand_gen_;
  /*! \brief The parameters for search. It stores the following parameters:
   * int evolutionary_search_population    // The population size for evolutionary search
   * int evolutionary_search_num_iters;    // The number of iterations for evolutionary search
   * int evolutionary_search_mutation_prob // The probability of mutation for evolutionary search
   * double evolutionary_search_crossover_ratio;  // The ratio of states created by crossover in evolutionary search
   * double evolutionary_search_use_measured_ratio;  // The maximum percentage of measured states in the initial
   *                                             // population for evolutionary search
   * double eps_greedy;          // Always allocate this percentage of measurements to random sampled states
   * str cpu_multi_level_tiling_structure;  // The structure of multi-level tiling for CPU
   * str gpu_multi_level_tiling_structure;  // The structure of multi-level tiling for GPU
   *
   * int disable_change_compute_location;  // Disable changing compute location.
   * int limit_inner_most_tile_size;       // Limit the value of inner most tile size.
   *                                       // The above two params are used to limit the search space.
   */
  Map<String, ObjectRef> params;
  /*! \brief The rules to generate sketches */
  std::vector<SketchGenerationRule*> sketch_rules;

  /*! \brief Search and make n_trails measurements.
   *  \returns the best state */
  State Search(SearchTask task, int n_trials,
               int early_stopping, int num_measure_per_iter,
               int verbose, ProgramMeasurer measurer,
               Array<SearchCallback> pre_search_callbacks) final;

  /*! \brief Continue search for one round. This is used by JointTuner
   * \returns the measurement pairs */
  std::pair<Array<MeasureInput>, Array<MeasureResult> > ContinueSearchOneRound(
      SearchTask task, int num_measure, int verbose, ProgramMeasurer measurer) final;

  /*! \brief Generate sketches
   *  \returns The list of generated sketches */
  std::vector<State> GenerateSketches();

  static constexpr const char *_type_key = "ansor.SketchSearchPolicy";

  TVM_DECLARE_FINAL_OBJECT_INFO(SketchSearchPolicyNode, SearchPolicyNode);

 protected:
  /*! \brief Pick states from best states and random states with eps-greedy policy */
  void PickStatesWithEpsGreedy(std::vector<MeasureInput>* inputs,
                               const std::vector<State>& best_states,
                               const std::vector<State>& random_states,
                               int remaining_n_trials);

 private:
  // Run one round of the search pipeline
  void SearchOneRound(std::vector<State>* best_states,
                      int num_random_states, std::vector<State>* random_states);

  State LNS(State now_s);

  State DestroyAndRepair(State old_s);

  // Sample init population
  void SampleInitPopulation(const std::vector<State>& sketches,
      int out_size, std::vector<State>* out_states);

  // Perform evolutionary search
  void EvolutionarySearch(const std::vector<State>& init_population,
      int num_best_states, std::vector<State>* best_states);

  SplitFactorizationMemo split_memo_;  // Memorize split space for Split
  int num_measure_per_iter_;   // The number of states to measure per iteration
  std::vector<int> auto_unroll_configs_;  // All possible candidates for auto_unroll
  std::vector<State> sketch_cache_;  // cached sketches
  bool cross_over_enabled_{true};   // Whether the crossover in evolutionary search is
                                    // enabled for the current task
};

/*!
 * \brief Managed reference to SketchSearchPolicyNode.
 * \sa SketchSearchPolicyNode
 */
class SketchSearchPolicy : public SearchPolicy {
 public:
  SketchSearchPolicy(CostModel program_cost_model,
                     Map<String, ObjectRef> params,
                     int seed);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SketchSearchPolicy, SearchPolicy,
                                        SketchSearchPolicyNode);
};

/*! \brief Pre-search callback function to load custom rules for sketch generation */
class PreloadCustomSketchRuleNode : public SearchCallbackNode {
 public:
  // TODO(jcf94): Use tvm::runtime::TypedPackedFunc?
  PackedFunc meet_condition_func;
  PackedFunc apply_func;

  void callback(SearchPolicyNode* policy) final;

  static constexpr const char *_type_key = "ansor.PreloadCustomSketchRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(PreloadCustomSketchRuleNode, SearchCallbackNode);
};

/*!
 * \brief Managed reference to PreloadCustomSketchRuleNode.
 * \sa PreloadCustomSketchRuleNode
 */
class PreloadCustomSketchRule : public SearchCallback {
 public:
  PreloadCustomSketchRule(PackedFunc meet_condition_func,
                          PackedFunc apply_func);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(PreloadCustomSketchRule, SearchCallback,
                                        PreloadCustomSketchRuleNode);
};

}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_SEARCH_POLICY_SKETCH_SEARCH_POLICY_H_
