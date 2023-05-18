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

#include "utils.h"
#include "search_policy.h"

namespace tvm {
namespace ansor {

void GetSpaceSplitStepIds(const State& s, int stage_id, std::vector<int>* spatial_split_step_ids) {
  auto pop = s->stages[stage_id]->op.as<te::ComputeOpNode>();
  CHECK(pop != nullptr);

  const auto& no_split_name_pair = GetNoSplitAxisAttr(s->stages[stage_id]);
  const std::set<std::string>& no_split_at_inner_name_set = no_split_name_pair.first;
  const std::set<std::string>& no_split_at_outer_name_set = no_split_name_pair.second;

  size_t reduce_count = 0;
  for (const auto axis : pop->reduce_axis) {
    if (!no_split_at_inner_name_set.count(axis->var->name_hint) &&
        !no_split_at_outer_name_set.count(axis->var->name_hint)) {
      reduce_count++;
    }
  }

  for (int i = static_cast<int>(s->transform_steps.size()) - 1; i >= 0; --i) {
    if (IsStageNumberChangingStep(s->transform_steps[i])) {
      if (stage_id > s->transform_steps[i]->stage_id) {
        stage_id--;
      }
    } else if (auto ps = s->transform_steps[i].as<SplitStepNode>()) {
      if (stage_id == ps->stage_id) {
        // Assume SplitStep on reduction axes are always after SplitStep on spatial axes.
        // TODO(jcf94): do not rely on this assumption
        if (reduce_count) {
          reduce_count--;
        } else {
          spatial_split_step_ids->push_back(i);
        }
      }
    }
  }
}

std::vector<std::pair<int, int>> GetComputeLocationCandidates(const SearchTask& task,
                                                              const State& state, int stage_id) {
  int target_stage_id = GetSingleConsumerId(task, state, stage_id);
  if (target_stage_id < 0) {
    return {};
  }
  const Stage& target_stage = state->stages[target_stage_id];

  std::vector<std::pair<int, int>> candidates;
  bool target_compute_at_other = target_stage->compute_at == kIter;
  bool target_is_tiled = IsTiled(target_stage);

  bool visited_reduce = false;
  // Enumerate compute_at location at target_stage
  // TODO(merrymercy): More analysis here to make smarter choices
  for (size_t i = 0; i < target_stage->iters.size(); ++i) {
    const Iterator& target_iter = target_stage->iters[i];
    if (target_iter->iter_type == kReduce) {
      visited_reduce = true;
      if (!target_is_tiled) {  // Do not go into reduce iter
        break;
      }
    } else if (target_iter->iter_type == kSpace) {
      if (visited_reduce) {  // Do not go into inner tile
        break;
      }
    }

    if (target_iter->annotation == kUnroll) {
      // Do not go into the unroll region of const tensor indices
      break;
    }

    if (GetExtent(target_iter) == 1) {
      // Skip iterators with length of 1
      continue;
    }
    if (target_compute_at_other && target_iter->iter_type == kSpace &&
        StrEndsWith(target_iter->name, ".0")) {
      // Skip the first level iterators if target stage compute_at another stage
      // In this case, the lengths of first level iterators are always one
      continue;
    }
    candidates.emplace_back(target_stage_id, i);

    if (state->attach_map->iter_to_attached_stages.count(std::make_pair(target_stage_id, i))) {
      break;
    }
  }

  // if the target_stage is already compute_at another stage X, try also compute_at X
  // We call stage X as `target_target_stage`
  if (target_compute_at_other) {
    int target_target_stage_id;
    target_target_stage_id = state->attach_map->stage_to_attach_iter.at(target_stage_id).first;
    const Stage& target_target_stage = state->stages[target_target_stage_id];

    for (size_t i = 0; i < target_target_stage->iters.size(); ++i) {
      const Iterator& target_target_iter = target_target_stage->iters[i];
      if (target_target_iter->iter_type == kReduce ||
          state->attach_map->iter_to_attached_stages.count(
              std::make_pair(target_target_stage_id, i))) {
        break;
      }

      if (target_target_iter->annotation == kUnroll) {
        // Do not go into the unroll region of const tensor indices
        break;
      }

      if (GetExtent(target_target_iter) == 1) {  // skip iterators with length of 1
        continue;
      }

      candidates.emplace_back(target_target_stage_id, i);
    }
  }

  return candidates;
}

State DoMultiLevelTiling(const State& state, int stage_id, const std::string& format,
                         std::vector<int>* spatial_split_step_ids) {
  std::vector<std::vector<Iterator> > space_levels;
  std::vector<std::vector<Iterator> > reduce_levels;
  std::vector<Iterator> space_outer, space_inner, reduce_outer, reduce_inner;
  std::vector<Iterator> split_res;

  for (const auto c : format) {
    if (tolower(c) == 's') {
      space_levels.emplace_back();
    } else if (tolower(c) == 'r') {
      reduce_levels.emplace_back();
    } else {
      LOG(FATAL) << "Invalid multi-level tiling format: " << format;
    }
  }
  size_t n_space = space_levels.size();
  size_t n_reduce = reduce_levels.size();

  spatial_split_step_ids->clear();

  State tmp_s = state;
  const Stage& stage = state->stages[stage_id];
  const auto& no_split_name_pair = GetNoSplitAxisAttr(stage);  // handle special split strategy
  const std::set<std::string>& no_split_at_inner_name_set = no_split_name_pair.first;
  const std::set<std::string>& no_split_at_outer_name_set = no_split_name_pair.second;

  for (const auto& iter : state->stages[stage_id]->iters) {
    if (iter->iter_type == kSpace) {
      if (!no_split_at_inner_name_set.count(iter->name) &&
          !no_split_at_outer_name_set.count(iter->name)) {
        CHECK_GE(n_space, 1);

        if (n_space == 1) {
          space_levels[0].push_back(iter);
        } else {
          split_res = tmp_s.split(stage_id, iter, std::vector<PrimExpr>(n_space - 1));
          for (int i = 0; i < static_cast<int>(n_space); i++) {
            space_levels[i].push_back(std::move(split_res[i]));
          }
          spatial_split_step_ids->push_back(tmp_s->transform_steps.size() - 1);
        }
      } else {
        if (no_split_at_inner_name_set.count(iter->name)) {
          space_inner.push_back(iter);
        }
        if (no_split_at_outer_name_set.count(iter->name)) {
          space_outer.push_back(iter);
        }
      }
    } else if (iter->iter_type == kReduce) {
      if (!no_split_at_inner_name_set.count(iter->name) &&
          !no_split_at_outer_name_set.count(iter->name)) {
        CHECK_GE(n_reduce, 1);

        if (n_reduce == 1) {
          reduce_levels[0].push_back(iter);
        } else {
          split_res = tmp_s.split(stage_id, iter, std::vector<PrimExpr>(n_reduce - 1));
          for (size_t i = 0; i < n_reduce; i++) {
            reduce_levels[i].push_back(std::move(split_res[i]));
          }
        }
      } else {
        if (no_split_at_inner_name_set.count(iter->name)) {
          reduce_inner.push_back(iter);
        }
        if (no_split_at_outer_name_set.count(iter->name)) {
          reduce_outer.push_back(iter);
        }
      }
    } else {
      LOG(FATAL) << "Invalid iter type: " << iter->iter_type;
    }
  }

  if (!space_outer.empty()) {
    CHECK(!space_levels.empty());
    space_levels.front().insert(space_levels.front().begin(),
            std::make_move_iterator(space_outer.begin()),
            std::make_move_iterator(space_outer.end()));
  }
  if (!space_inner.empty()) {
    CHECK(!space_levels.empty());
    space_levels.back().insert(space_levels.back().begin(),
            std::make_move_iterator(space_inner.begin()),
            std::make_move_iterator(space_inner.end()));
  }

  if (!reduce_outer.empty()) {
    CHECK(!reduce_levels.empty());
    reduce_levels.front().insert(reduce_levels.front().begin(),
        std::make_move_iterator(reduce_outer.begin()),
        std::make_move_iterator(reduce_outer.end()));
  }
  if (!reduce_inner.empty()) {
    CHECK(!reduce_levels.empty());
    reduce_levels.back().insert(reduce_levels.back().begin(),
        std::make_move_iterator(reduce_inner.begin()),
        std::make_move_iterator(reduce_inner.end()));
  }

  std::vector<Iterator> order;
  int space_ct = 0, reduce_ct = 0;
  for (const auto c : format) {
    if (tolower(c) == 's') {
      order.insert(order.end(), std::make_move_iterator(space_levels[space_ct].begin()),
              std::make_move_iterator(space_levels[space_ct].end()));
      space_ct++;
    } else if (tolower(c) == 'r') {
      order.insert(order.end(), std::make_move_iterator(reduce_levels[reduce_ct].begin()),
              std::make_move_iterator(reduce_levels[reduce_ct].end()));
      reduce_ct++;
    } else {
      LOG(FATAL) << "Invalid multi level tiling format: " << format;
    }
  }

  tmp_s.reorder(stage_id, order);
  return tmp_s;
}

State FollowTiling(const State& state, int stage_id,
                   const std::vector<int>& split_step_ids, int n_split) {
  if (n_split < 1 || n_split > 3) {
    LOG(FATAL) << "Invalid split parts, currently only support 1, 2 and 3";
  }
  // Apply up to three-level tiling structure:  space_L0, space_L1, space_L2
  std::vector<Iterator> space_0, space_1, space_2, space_3;
  std::vector<Iterator> split_res, tmp_order;

  auto pop = state->stages[stage_id]->op.as<te::ComputeOpNode>();
  CHECK(pop != nullptr);
  const Stage& stage = state->stages[stage_id];
  const auto& no_split_name_pair = GetNoSplitAxisAttr(stage);  // handle special split strategy
  const std::set<std::string>& no_split_at_inner_name_set = no_split_name_pair.first;
  const std::set<std::string>& no_split_at_outer_name_set = no_split_name_pair.second;
  int no_split_at_inner_name_in_stage_cnt = 0;
  int no_split_at_outer_name_in_stage_cnt = 0;
  for (const auto& iter : state->stages[stage_id]->iters) {
    no_split_at_inner_name_in_stage_cnt += no_split_at_inner_name_set.count(iter->name);
    no_split_at_outer_name_in_stage_cnt += no_split_at_outer_name_set.count(iter->name);
  }

  CHECK_EQ(state->stages[stage_id]->iters.size()
               - no_split_at_inner_name_in_stage_cnt
               - no_split_at_outer_name_in_stage_cnt,
           split_step_ids.size());

  State tmp_s = state;
  int ct = 0;
  for (const auto& iter : state->stages[stage_id]->iters) {
    if (iter->iter_type == kSpace) {
      // For spatial iterator, split it into multi iterators
      if (!no_split_at_inner_name_set.count(iter->name) &&
          !no_split_at_outer_name_set.count(iter->name)) {
        IteratorAnnotation ann_type = iter->annotation;
        split_res = tmp_s.follow_split(stage_id, iter, split_step_ids[ct],
                                       n_split);
        // Restore annotation. Move unroll and vectorize to inner, move parallel
        // to outer
        switch (ann_type) {
          case kUnroll:
            split_res[n_split] = tmp_s.unroll(stage_id, split_res[n_split]);
            break;
          case kVectorize:
            split_res[n_split] = tmp_s.vectorize(stage_id, split_res[n_split]);
            break;
          case kParallel:
            split_res[0] = tmp_s.parallel(stage_id, split_res[0]); break;
          default:
            break;
        }

        space_0.push_back(std::move(split_res[0]));
        space_1.push_back(std::move(split_res[1]));
        if (n_split >= 2) {
          space_2.push_back(std::move(split_res[2]));
          if (n_split == 3) {
            space_3.push_back(std::move(split_res[3]));
          }
        }
        ct++;
      } else {
        if (no_split_at_outer_name_set.count(iter->name)) {
          space_0.push_back(iter);
        }
        if (no_split_at_inner_name_set.count(iter->name)) {
          if (n_split == 1) {
            space_1.push_back(iter);
          } else if (n_split == 2) {
            space_2.push_back(iter);
          } else {
            CHECK_EQ(n_split, 3);
            space_3.push_back(iter);
          }
        }
      }
    } else {
      LOG(FATAL) << "Invalid iter type: " << iter->iter_type;
    }
  }

  if (n_split == 3) {
    ConcatenateMove(&tmp_order, &space_0, &space_1, &space_2, &space_3);
  } else if (n_split == 2) {
    ConcatenateMove(&tmp_order, &space_0, &space_1, &space_2);
  } else {
    ConcatenateMove(&tmp_order, &space_0, &space_1);
  }
  tmp_s.reorder(stage_id, tmp_order);
  return tmp_s;
}

State RandomMutateTileSize(const State& old_state, SplitFactorizationMemo* split_memo,
                           std::mt19937* random_gen, int max_innermost_split_factor) {
  State tmp_s = old_state;

  // Extract all SplitStep
  std::vector<size_t> split_step_ids;
  for (size_t i = 0; i < tmp_s->transform_steps.size(); ++i) {
    if (auto ps = tmp_s->transform_steps[i].as<SplitStepNode>()) {
      if (ps->extent.defined() && ps->extent->IsInstance<IntImmNode>() &&
          GetIntImm(ps->lengths.back()) <= max_innermost_split_factor) {
        split_step_ids.push_back(i);
      }
    }
  }
  if (split_step_ids.empty()) {
    return State();
  }

  // Find a SplitStep with extent != 1
  int retry_ct = 0;
  int64_t extent = 1;
  int step_id;
  const SplitStepNode* ps;

  do {
    step_id = split_step_ids[(*random_gen)() % split_step_ids.size()];
    ps = tmp_s->transform_steps[step_id].as<SplitStepNode>();
    CHECK(ps != nullptr);
    extent = GetIntImm(ps->extent);
    retry_ct += 1;
  } while (retry_ct < static_cast<int>(split_step_ids.size()) << 2 &&
           (extent == 1 || extent == 0));

  if (extent == 0 || extent == 1) {
    return State();
  }

  // Mutate tile size
  std::vector<int> lengths(ps->lengths.size() + 1, 1);
  for (int i = 0; i < static_cast<int>(ps->lengths.size()); ++i) {
    lengths[i + 1] = GetIntImm(ps->lengths[i]);
  }
  lengths[0] = extent / ElementProduct(lengths);

  std::vector<int> random_perm;
  RandomPermutation(lengths.size(), &random_perm, random_gen);

  for (size_t i = 0; i < random_perm.size(); ++i) {
    size_t src_idx = random_perm[i];
    int length = lengths[src_idx];

    if (length <= 1) {
      continue;
    }

    // Divide one factor from lengths[src_idx] and multiply it to lengths[dst_idx]
    size_t dst_idx = random_perm[(i + 1) % random_perm.size()];

    const std::vector<int>& factors = split_memo->GetFactors(length);
    CHECK_GE(factors.size(), 1);

    int divide_factor;
    if (dst_idx == lengths.size() - 1) {
      // Maintain the restriction of hardware_params.max_innermost_split_factor
      int max_factor_index = static_cast<int>(factors.size()) - 1;
      for (; max_factor_index >= 1; max_factor_index--) {
        if (factors[max_factor_index] * lengths[dst_idx] <= max_innermost_split_factor) {
          break;
        }
      }
      if (max_factor_index == 0) {
        // failed on this dst_idx, try next one
        continue;
      }
      divide_factor = factors[1 + (*random_gen)() % (max_factor_index)];
    } else {
      divide_factor = factors[1 + (*random_gen)() % (factors.size() - 1)];
    }

    std::vector<PrimExpr> new_lengths;
    for (size_t j = 1; j < lengths.size(); ++j) {
      if (j == src_idx) {
        new_lengths.emplace_back(lengths[j] / divide_factor);
      } else if (j == dst_idx) {
        new_lengths.emplace_back(lengths[j] * divide_factor);
      } else {
        new_lengths.emplace_back(lengths[j]);
      }
    }

    CHECK_LE(GetIntImm(new_lengths.back()), max_innermost_split_factor);

    auto pstate = tmp_s.CopyOnWrite();
    pstate->transform_steps[step_id] =
        SplitStep(ps->stage_id, ps->iter_id, ps->extent, new_lengths, ps->inner_to_outer);
    return tmp_s;
  }

  return State();
}

State RandomMutateMaxUnrollStep(const State& old_state, std::mt19937* random_gen,
    const std::vector<int>& auto_unroll_configs) {
  State tmp_s = old_state;

  // Extract all auto_unroll_max_step pragma steps.
  std::vector<int> pragma_steps;
  for (size_t i = 0; i < old_state->transform_steps.size(); ++i) {
    if (auto ps = tmp_s->transform_steps[i].as<PragmaStepNode>()) {
      if (ps->pragma_type.find("auto_unroll_max_step") != std::string::npos) {
        pragma_steps.push_back(i);
      }
    }
  }
  if (pragma_steps.empty()) {
    return State();
  }

  // Randomly pick one step.
  auto step_id = pragma_steps[(*random_gen)() % pragma_steps.size()];
  auto ps = tmp_s->transform_steps[step_id].as<PragmaStepNode>();
  int val = auto_unroll_configs[(*random_gen)() % auto_unroll_configs.size()];

  auto pstate = tmp_s.CopyOnWrite();
  pstate->transform_steps[step_id] = PragmaStep(
      ps->stage_id, ps->iter_id,
      std::string("auto_unroll_max_step") + "$" + std::to_string(val));
  pstate->stages[ps->stage_id].CopyOnWrite()->attrs.auto_unroll_max_step = val;
  return tmp_s;
}

State RandomMutateParallel(const State& old_state, std::mt19937* random_gen,
                           const SearchTask& task, int verbose) {
  // This mutation rule only focuses on a case that parallel was added to
  // the outermost loop and the loop is generated by fusing other loops.
  // In short, we mutate the fusion step before the parallel step.
  verbose = 1;

  // Extract all parallel steps.
  std::vector<int> parallel_steps;
  for (size_t s = 0; s < old_state->transform_steps.size(); ++s) {
    auto ps = old_state->transform_steps[s].as<AnnotationStepNode>();
    if (!ps || ps->annotation != kParallel) {
      continue;
    }

    // Skip non-outermost loop or the parallel step without fusion beforehand.
    if (ps->iter_id != 0 || s == 0 || !old_state->transform_steps[s - 1].as<FuseStepNode>()) {
      continue;
    }
    auto fuse_step = old_state->transform_steps[s - 1].as<FuseStepNode>();
    CHECK(!fuse_step->fused_ids.empty());
    if (fuse_step->fused_ids[0] != 0) {
      continue;
    }

    parallel_steps.push_back(s);
  }
  if (parallel_steps.empty()) {
    return State();
  }

  // Randomly pick one parallel step.
  size_t step_id = parallel_steps[(*random_gen)() % parallel_steps.size()];

  // Replay a new state until the picked fuse step.
  State tmp_s = task->compute_dag.GetInitState();
  for (size_t s = 0; s < step_id - 1; ++s) {
    const auto& step = old_state->transform_steps[s];
    tmp_s.CopyOnWrite()->transform_steps.push_back(step);
    tmp_s.DoStep(step, task->compute_dag);
  }

  // Compute all possible fusion granularities
  auto fuse_step = old_state->transform_steps[step_id - 1].as<FuseStepNode>();
  CHECK(fuse_step != nullptr);
  int stage_id = fuse_step->stage_id;
  const Stage& stage = tmp_s->stages[stage_id];
  size_t max_fusible_id;
  for (max_fusible_id = 0; max_fusible_id < stage->iters.size(); ++max_fusible_id) {
    const Iterator& it = stage->iters[max_fusible_id];
    if (it->iter_type == kReduce || it->annotation != kNone) {
      break;
    }

    if (tmp_s->attach_map->iter_to_attached_stages.count(
                std::make_pair(stage_id, max_fusible_id))) {
      break;
    }
  }

  if (max_fusible_id == 0) {
    return State();
  }

  // Randomly pick one granularity
  int fuse_to_iter_id = (*random_gen)() % max_fusible_id + 1;
  std::vector<int> fused_ids;
  for (int i = 0; i < fuse_to_iter_id; ++i) {
    fused_ids.push_back(i);
  }
  int iter_offset = fuse_step->fused_ids.back() - fused_ids.back();
  if (iter_offset == 0) {
    return State();
  }

  // Replay the mutated fused and annotation step.
  auto new_fuse_step = FuseStep(stage_id, fused_ids);
  tmp_s.CopyOnWrite()->transform_steps.push_back(new_fuse_step);
  tmp_s.DoStep(new_fuse_step, task->compute_dag);
  tmp_s.CopyOnWrite()->transform_steps.push_back(old_state->transform_steps[step_id]);
  tmp_s.DoStep(old_state->transform_steps[step_id], task->compute_dag);

  // Replay the rest steps.
  for (size_t s = step_id + 1; s < old_state->transform_steps.size(); ++s) {
    auto step = old_state->transform_steps[s];
    if (step->stage_id == stage_id) {
      // Since we change the loop structure, iter ID in later steps to the same stage
      // has to be adjusted.
      if (auto ps = step.as<AnnotationStepNode>()) {
        if (ps->iter_id == 0) {
          step = AnnotationStep(ps->stage_id, 0, ps->annotation);
        } else {
          CHECK_LE(ps->iter_id + iter_offset, tmp_s->stages[stage_id]->iters.size());
          step = AnnotationStep(ps->stage_id, ps->iter_id + iter_offset, ps->annotation);
        }
      } else if (auto ps = step.as<PragmaStepNode>()) {
        if (ps->iter_id == 0) {
          step = PragmaStep(ps->stage_id, 0, ps->pragma_type);
        } else {
          CHECK_LE(ps->iter_id + iter_offset, tmp_s->stages[stage_id]->iters.size());
          step = PragmaStep(ps->stage_id, ps->iter_id + iter_offset, ps->pragma_type);
        }
      } else {
        StdCout(verbose) << "Parallel mutation: Cannot apply " << step
                         << " after fuse"  << std::endl;
        return State();
      }
    }
    if (IsStageNumberChangingStep(step)) {
      // For these steps, we have to update stage_id because these steps will make stage_id out-dated.
      // But here we just simply give up this mutation for simplicity.
      // This is not an issue because this will never happend in normal cases where all these steps
      // are before parallel steps.
      return State();
    }
    tmp_s.CopyOnWrite()->transform_steps.push_back(step);
    try {
      tmp_s.DoStep(tmp_s->transform_steps.back(), task->compute_dag);
    } catch (dmlc::Error &e) {
      return State();
    }
  }

  return tmp_s;
}


State RandomMutateComputeLocation(const State& old_state, std::mt19937* random_gen,
                                  const SearchTask& task) {
  // Extract all compute_at steps.
  std::vector<int> compute_at_steps;
  for (size_t s = 0; s < old_state->transform_steps.size(); ++s) {
    if (auto ps = old_state->transform_steps[s].as<ComputeAtStepNode>()) {
      int stage_inc = GetTargetStageIDInState(old_state, s) - ps->stage_id;

      if (IsTiled(old_state->stages[ps->stage_id + stage_inc])) {
        continue;
      }

      if (NeedsMultilevelTiling(task, old_state, ps->stage_id + stage_inc)) {
        continue;
      }
      compute_at_steps.push_back(s);
    }
  }
  if (compute_at_steps.empty()) {
    return State();
  }

  // Randomly pick one step
  size_t step_id = compute_at_steps[(*random_gen)() % compute_at_steps.size()];
  auto ps = old_state->transform_steps[step_id].as<ComputeAtStepNode>();
  int stage_inc = GetTargetStageIDInState(old_state, step_id) - ps->stage_id;
  CHECK(ps != nullptr);


  // Randomly pick a new computation location
  int new_compute_at_stage_id;
  int new_compute_at_iter_id;
  std::vector<std::pair<int, int>> candidates =
      GetComputeLocationCandidates(task, old_state, ps->stage_id + stage_inc);
  if (candidates.empty()) {
    return State();
  }
  int choice = (*random_gen)() % (candidates.size());
  new_compute_at_stage_id = candidates[choice].first;
  new_compute_at_iter_id = candidates[choice].second;

  // Replay a new state.
  State tmp_s = task->compute_dag.GetInitState();
  for (size_t s = 0; s < old_state->transform_steps.size(); ++s) {
    if (s == step_id) {
      tmp_s.CopyOnWrite()->transform_steps.push_back(
          ComputeAtStep(ps->stage_id, new_compute_at_stage_id - stage_inc, new_compute_at_iter_id));
    } else {
      tmp_s.CopyOnWrite()->transform_steps.push_back(old_state->transform_steps[s]);
    }
    try {
      tmp_s.DoStep(tmp_s->transform_steps.back(), task->compute_dag);
    } catch (dmlc::Error &e) {
      return State();
    }
  }

  return tmp_s;
}

// Return whether a state has nested parallel, which is invalid on CPUs
bool HasNestedParallel(const State& state) {
  std::function<void(int stage_id, size_t*)> count_parallel_ct;

  count_parallel_ct = [&state, &count_parallel_ct](int stage_id, size_t* parallel_ct) {
    const Stage& stage = state->stages[stage_id];

    if (stage->compute_at == kInlined) {
      return;
    }

    for (size_t i = 0; i < stage->iters.size(); ++i) {
      if (stage->iters[i]->annotation == kParallel) {
        (*parallel_ct)++;
      }

      AttachMap::IterKey iter_key(stage_id, i);
      auto pair = state->attach_map->iter_to_attached_stages.find(iter_key);
      if (pair != state->attach_map->iter_to_attached_stages.end()) {
        for (const auto& attach_stage_id : pair->second) {
          count_parallel_ct(attach_stage_id, parallel_ct);
        }
      }
    }
  };
  
  for (size_t stage_id = 0; stage_id < state->stages.size(); ++stage_id) {
    size_t parallel_ct = 0;

    if (state->stages[stage_id]->compute_at == kRoot) {
      count_parallel_ct(stage_id, &parallel_ct);
      if (parallel_ct >= 2) {
        return true;
      }
    }
  }

  return false;
}

void PruneInvalidState(const SearchTask& task, std::vector<State>* states) {
  size_t pt = 0;
  for (size_t i = 0; i < states->size(); ++i) {
    if (!(*states)[i].defined()) {
      continue;
    }

    if (!IsGPUTask(task) && HasNestedParallel((*states)[i])) {
      continue;
    }

    if (i != pt) {
      (*states)[pt] = std::move((*states)[i]);
    }
    pt++;
  }

  if (pt == 0) {
    LOG(FATAL) << "All states are invalid.";
  } else {
    states->resize(pt);
  }
}

// Return stage ID given the stage name, or -1 if not found.
int GetStageIdByName(const State& state, const std::string& name) {
  for (size_t sid = 0; sid < state->stages.size(); ++sid) {
    if (name == state->stages[sid]->op->name) {
      return sid;
    }
  }
  return -1;
}

// Clone and apply one step from the reference state to a new state.
State ApplyStepToNewState(const SearchTask& task, const State& state, const State& ref_s,
                          const Step& step) {
  State tmp_s = state;

  int curr_stage_id = step->stage_id; 
  /*
  if (tmp_s->stages[curr_stage_id]->op->name != ref_s->stages[curr_stage_id]->op->name) {
    // Relocate stage_id by matching the name, so the step can work on the same stage
    curr_stage_id = GetStageIdByName(tmp_s, ref_s->stages[step->stage_id]->op->name);
    CHECK_NE(curr_stage_id, -1);
  }*/

  // The default case: Simply append the step to the history and do it
  std::function<State(const Step&)> simple_apply = [&task, &tmp_s, &curr_stage_id]
      (Step new_step) {
    if (new_step->stage_id != curr_stage_id) {
      new_step = new_step->CloneWithStageID(curr_stage_id);
    }

    tmp_s.CopyOnWrite()->transform_steps.push_back(new_step);
    try {
      tmp_s.DoStep(new_step, task->compute_dag);
    } catch (dmlc::Error &ex) {
      return State();
    }
    return tmp_s;
  };

  // Deal with special steps
  if (auto follow_step = step.as<FollowSplitStepNode>()) {
    // Follow split step specifies a source split step ID that might be changed when
    // crossover, so we try to identify the corresponding split step and use an offset
    // to adjust the source step id.
    const std::string& target_stage_name =
        ref_s->stages[ref_s->transform_steps[follow_step->src_step_id]->stage_id]->op->name;

    // Looking for the first SplitStep to the target stage.
    size_t tmp_s_ptr = 0;
    while (tmp_s_ptr < tmp_s->transform_steps.size() &&
           (!tmp_s->transform_steps[tmp_s_ptr]->IsInstance<SplitStepNode>() ||
            tmp_s->stages[tmp_s->transform_steps[tmp_s_ptr]->stage_id]->op->name !=
                target_stage_name)) {
      tmp_s_ptr++;
    }

    size_t ref_s_ptr = 0;
    while (ref_s_ptr < ref_s->transform_steps.size() &&
           (!ref_s->transform_steps[ref_s_ptr]->IsInstance<SplitStepNode>() ||
            ref_s->stages[ref_s->transform_steps[ref_s_ptr]->stage_id]->op->name !=
                target_stage_name)) {
      ref_s_ptr++;
    }

    int follow_split_src_offset = tmp_s_ptr - ref_s_ptr;
    int new_src_split_step_id = follow_step->src_step_id + follow_split_src_offset;

    // Check the validity of the new_src_split_step_id
    if (tmp_s_ptr == tmp_s->transform_steps.size() || ref_s_ptr == ref_s->transform_steps.size() ||
        new_src_split_step_id < 0 || new_src_split_step_id >= static_cast<int>(tmp_s->transform_steps.size())) {
      return State();
    }
    auto ps = tmp_s->transform_steps[new_src_split_step_id].as<SplitStepNode>();
    if (ps == nullptr || static_cast<int>(ps->lengths.size()) <= follow_step->n_split) {
      return State();
    }

    CHECK_LT(follow_step->iter_id, tmp_s->stages[curr_stage_id]->iters.size());
    auto iter = tmp_s->stages[curr_stage_id]->iters[follow_step->iter_id];

    tmp_s.follow_split(curr_stage_id, iter, new_src_split_step_id, follow_step->n_split);
  } else if (auto follow_step = step.as<FollowFusedSplitStepNode>()) {
    // Follow fused split step specifies a source split/fuse step IDs that might be changed when
    // crossover, so we update the source step ID list.

    // Get target stage ID and spatial split steps.
    std::set<int> consumers = GetConsumers(task, tmp_s, curr_stage_id);
    if (consumers.size() == 0) {
      return State();
    }
    int target_stage_id = *consumers.begin();

    // Get split steps.
    std::vector<int> spatial_split_step_ids;
    GetSpaceSplitStepIds(tmp_s, target_stage_id, &spatial_split_step_ids);
    if (spatial_split_step_ids.size() == 0 ||
        spatial_split_step_ids.size() != follow_step->src_step_ids.size()) {
      return State();
    }

    // Locate the new state iter by name.
    const std::string& iter_name = ref_s->stages[follow_step->stage_id]->iters[follow_step->iter_id]->name;
    Iterator iter;
    for (size_t i = 0; i < tmp_s->stages[curr_stage_id]->iters.size(); ++i) {
      if (tmp_s->stages[curr_stage_id]->iters[i]->name == iter_name) {
        iter = tmp_s->stages[curr_stage_id]->iters[i];
        break;
      }
    }

    if (!iter.defined()) {
      return State();
    }

    tmp_s.follow_fused_split(curr_stage_id, iter, spatial_split_step_ids, follow_step->level,
                             follow_step->factor_or_nparts);
  } else if (auto compute_at_step = step.as<ComputeAtStepNode>()) {
    // ComputeAt target stage may be missing in the new state so we need to update target stage ID
    // and iter ID before applying ComputeAtStep to a new state.
    // We use stage name and iter name from the reference state to update IDs. If any of names is
    // missing in the new state, we give up this crossover to avoid creating an invalid state.
    const std::string& target_stage_name =
        ref_s->stages[compute_at_step->target_stage_id]->op->name;
    const std::string& target_iter_name = ref_s->stages[compute_at_step->target_stage_id]
                                               ->iters[compute_at_step->target_iter_id]
                                               ->name;

    // Find the target stage and iter to apply the compute at step.
    for (size_t stage_id = 0; stage_id < tmp_s->stages.size(); ++stage_id) {
      if (target_stage_name == tmp_s->stages[stage_id]->op->name) {
        for (size_t iter_id = 0; iter_id < tmp_s->stages[stage_id]->iters.size(); ++iter_id) {
          if (target_iter_name == tmp_s->stages[stage_id]->iters[iter_id]->name) {
            auto clone_step = ComputeAtStep(curr_stage_id, stage_id, iter_id);
            tmp_s.CopyOnWrite()->transform_steps.push_back(clone_step);
            tmp_s.DoStep(clone_step, task->compute_dag);
            return tmp_s;
          }
        }
      }
    }
    return State();
  } else if (auto annotation_step = step.as<AnnotationStepNode>()) {
    if (annotation_step->annotation == kParallel &&
        tmp_s->stages[curr_stage_id]->compute_at != kRoot) {
      // Do not produce nested parallel
      return simple_apply(AnnotationStep(annotation_step->stage_id,
                                         annotation_step->iter_id,
                                         kNone));
    } else {
      return simple_apply(step);
    }
  } else {
    return simple_apply(step);
  }
  return tmp_s;
}

State CrossOverState(const SearchTask& task, std::mt19937* random_gen, const State& p1,
  const State& p2, std::vector<int>* fail_counters){
  // An internal class that replays a parent state to make the stage ID consist.
  class SyncingState {
   public:
    int id;
    State sync_state;
    const std::vector<Step>& steps;
    int stage_change_cnt;
    size_t step_ptr;

    SyncingState(const SearchTask& task, int id, const State& ref_state)
        : steps(ref_state->transform_steps) {
      this->id = id;
      this->sync_state = task->compute_dag.GetInitState();
      this->stage_change_cnt = 0;
      this->step_ptr = 0;
    }

    // Indicate if the state is up-to-date (all steps are applied).
    bool IsSynced() { return step_ptr == steps.size(); }

    // Number of applied steps that changed stages.
    int StageChangeCount() { return stage_change_cnt; }

    // Get the target stage name of the step to be applied.
    std::string GetCurrStageName() {
      if (IsSynced()) {
        return "";
      }
      return sync_state->stages[steps[step_ptr]->stage_id]->op->name;
    }

    // Apply one step to the syncing state. Do nothing if all steps are applied already.
    void ApplyOneStep(const SearchTask& task) {
      if (IsSynced()) {
        return;
      }

      const Step& step = steps[this->step_ptr];
      this->sync_state.CopyOnWrite()->transform_steps.push_back(step);
      this->sync_state.DoStep(step, task->compute_dag);

      if (IsStageNumberChangingStep(step)) {
        this->stage_change_cnt++;
      }
      this->step_ptr++;
    }
  };

  // Don't do crossover when the stage numbers are different
  if (p1->stages.size() != p2->stages.size()) {
    (*fail_counters)[0]++;
    return State();
  }

  // Create sync states to match the stages.
  SyncingState sync_p1(task, 1, p1);
  SyncingState sync_p2(task, 2, p2);
  std::vector<SyncingState*> sync_states = {&sync_p1, &sync_p2};

  // Stage index to the selected state. Default to p1.
  std::unordered_map<std::string, int> stage_out_to_states;
  int p1_selected = 0, p2_selected = 0;

  //doublex----p2+p1+p2
  int length=static_cast<int>(p1->stages.size());
  int one_point=rand() % length;
  int two_point=rand() % length;
  while(two_point==one_point){
    two_point=rand() % length;
  }
  int cnt=0;
  for (int t=length-1; t >= 0; --t) {

    // Don't do crossover only when the stage names are different
    if (p1->stages[t]->op->name != p2->stages[t]->op->name) {
      (*fail_counters)[1]++;
      return State();
    }

    // This stage is already been assigned
    if (stage_out_to_states.count(p1->stages[t]->op->name)) {
      continue;
    }

    if (p1->stages[t]->op_type == kPlaceholder) {
      // Since CacheRead steps target to placeholder stage, we assign all placeholders to p1.
      stage_out_to_states[p1->stages[t]->op->name] = sync_p1.id;
      continue;
    } 

    if(t==one_point || t==two_point) ++cnt;

    if(cnt==0 || cnt==2){
      stage_out_to_states[p2->stages[t]->op->name] = sync_p2.id;
      if (p2->stages[t]->compute_at != kInlined) {
        p2_selected++;
      }
    }

    if(cnt==1){
      stage_out_to_states[p1->stages[t]->op->name] = sync_p1.id;
      if (p1->stages[t]->compute_at != kInlined) {
        p1_selected++;
      }
    }

    if (IsGPUTask(task)) {
      int id = stage_out_to_states[p1->stages[t]->op->name];
      const State& parent = (id == 1 ? p1 : p2);

      // On GPU, if we choose a root stage, all stages in this GPU kernel should also be chosen.
      // This can fix some fatal dependency problems.
      if (parent->stages[t]->compute_at == kRoot) {
        std::function<void(int)> assign_attached_stages;
        assign_attached_stages = [&assign_attached_stages, id, &parent, &stage_out_to_states](int stage_id) {
          const Stage& stage = parent->stages[stage_id];
          for (size_t i = 0; i < stage->iters.size(); ++i) {
            AttachMap::IterKey iter_key(stage_id, i);
            auto res = parent->attach_map->iter_to_attached_stages.find(iter_key);
            if (res != parent->attach_map->iter_to_attached_stages.end()) {
              for (const auto& attach_stage_id : res->second) {
                stage_out_to_states[parent->stages[attach_stage_id]->op->name] = id;
                assign_attached_stages(attach_stage_id);
              }
            }
          }
        };
        assign_attached_stages(t);
      }
    } else {
      // If a rfactor stage is chosen, all stages related to this rfactor should be chosen.
      // This can fix some fatal dependency problems.
      if (StrEndsWith(p1->stages[t]->op->name, ".repl")) {
        int id = stage_out_to_states[p1->stages[t]->op->name];
        std::string raw_name = p1->stages[t]->op->name.substr(0, p1->stages[t]->op->name.size() - 5);
        stage_out_to_states[raw_name] = id;
        stage_out_to_states[raw_name + ".rf"] = id;
      }
    }
  }

  // If all stages are coming from the same state, then no need to crossover.
  if (p1_selected == 0 || p2_selected == 0) {
    (*fail_counters)[2]++;
    return State();
  }

  // Create a new state.
  State tmp_s = task->compute_dag.GetInitState();

  // Apply steps. Meanwhile we also re-apply steps to p1 and p2 to make sure
  // the stage ID is matched.
  while (!sync_states[0]->IsSynced() && !sync_states[1]->IsSynced()) {
    SyncingState* sync_s = nullptr;

    // Determine which state we will focus on this round.
    // If a state has changed its stages more times than another state, we prior to another state to
    // make their stages synced. Otherwise we simply go for the one with smaller step pointer.
    if (sync_states[0]->StageChangeCount() < sync_states[1]->StageChangeCount()) {
      sync_s = sync_states[0];
    } else if (sync_states[0]->StageChangeCount() > sync_states[1]->StageChangeCount()) {
      sync_s = sync_states[1];
    } else {
      sync_s = sync_states[(sync_states[0]->step_ptr <= sync_states[1]->step_ptr) ? 0 : 1];
    }
    const std::string& curr_stage_name = sync_s->GetCurrStageName();

    // Check if we want to apply this step.
    std::string target_stage_name = curr_stage_name;
    if (auto ps = sync_s->steps[sync_s->step_ptr].as<ComputeAtStepNode>()) {
      // Whether to apply Compute_at step depends on the target stage instead of self stage.
      target_stage_name = sync_s->sync_state->stages[ps->target_stage_id]->op->name;
    }

    // If the target stage of the current state is selected, we apply this step to the new state.
    if (stage_out_to_states[target_stage_name] == sync_s->id) {
      tmp_s = ApplyStepToNewState(task, tmp_s, sync_s->sync_state, sync_s->steps[sync_s->step_ptr]);
      if (!tmp_s.defined()) {
        (*fail_counters)[3]++;
        return tmp_s;
      }
    }

    sync_s->ApplyOneStep(task);
  }

  // Process tails.
  for (size_t i = 0; i < sync_states.size(); ++i) {
    SyncingState* sync_s = sync_states[i];
    while (!sync_s->IsSynced()) {
      const std::string& stage_name = sync_s->GetCurrStageName();

      // Check if we want to apply this step.
      std::string target_stage_name = stage_name;
      if (auto ps = sync_s->steps[sync_s->step_ptr].as<ComputeAtStepNode>()) {
        // Whether to apply Compute_at step depends on the target stage instead of self stage.
        target_stage_name = sync_s->sync_state->stages[ps->target_stage_id]->op->name;
      }

      // If the target stage of the current state is selected, we apply this step to the new state.
      if (stage_out_to_states[target_stage_name] == sync_s->id) {
        tmp_s = ApplyStepToNewState(task, tmp_s, sync_s->sync_state, sync_s->steps[sync_s->step_ptr]);
        if (!tmp_s.defined()) {
          (*fail_counters)[4]++;
          return tmp_s;
        }
      }

      sync_s->ApplyOneStep(task);
    }
  }

  // Check wheter the crossover creates a new state
  //tmp_s = task->compute_dag.InferBound(tmp_s);
  //std::string s1 = p1.ToStr();
  //std::string s2 = p2.ToStr();
  //std::string s3 = tmp_s.ToStr();

  //if (s1 == s2 || s1 == s3 || s2 == s3) {
  //  return State();
  //}

  return tmp_s;
}

}  // namespace ansor
}  // namespace tvm

