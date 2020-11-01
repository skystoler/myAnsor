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

"""TaskScheduler that allocates the time resources when tuning multiple tasks together"""
from typing import List, Union, Callable
import time
import math
import logging

import numpy as np

from .auto_schedule import SearchTask, SearchPolicy, SketchSearchPolicy, TuneOption
from .cost_model import RandomModel, XGBModel
from .measure import ProgramMeasurer
from .utils import array_mean, to_str_round
from .serialization import LogReader

logger = logging.getLogger('ansor')

class TaskScheduler:
    """Allocate the time resources when tuning multiple tasks together"""
    def __init__(self,
                 tasks: List[SearchTask],
                 objective_func: Callable = None):
        self.tasks = tasks
        self.objective_func = objective_func or sum

    def compute_score(self, costs: List[float]) -> float:
        return self.objective_func(costs)


def get_search_policies(search_policy: Union[str, List[SearchPolicy]], tasks: List[SearchTask],
                        num_measure_per_iter, load_model_file=None, load_log_file=None):
    if search_policy == 'default':
        search_policy = 'sketch.xgb'

    if isinstance(search_policy, str):
        policy_type, model_type = search_policy.split('.')
        if model_type == 'xgb':
            cost_model = XGBModel(num_warmup_sample=len(tasks) * num_measure_per_iter)
            if load_model_file:
                logger.info("Load pretrained model...")
                cost_model.load(load_model_file)
            elif load_log_file:
                cost_model.load_log_file(load_log_file)
        elif model_type == 'random':
            cost_model = RandomModel()
        else:
            raise ValueError("Invalid search policy: " + search_policy)

        if policy_type == 'sketch':
            search_policies = [SketchSearchPolicy(cost_model) for _ in range(len(tasks))]
        elif policy_type == 'limit-space':
            search_policies = [SketchSearchPolicy(cost_model,
                                                  params={'cpu_multi_level_tiling_structure': 'SRS',
                                                          'disable_change_compute_location': 1,
                                                          'limit_inner_most_tile_size': 16})
                               for _ in range(len(tasks))]
        elif policy_type == 'beam-search':
            search_policies = [SketchSearchPolicy(cost_model,
                                                  params={'use_beam_search': 1})
                               for _ in range(len(tasks))]
        else:
            raise ValueError("Invalid search policy: " + search_policy)
    else:
        # check type
        assert isinstance(search_policy, (tuple, list))
        for item in search_policy:
            assert isinstance(item, SearchPolicy)
        search_policies = search_policy

    return search_policies


def derive_similarity_tag(dag, log_base=1.618):
    """Derive the tag for similarity check from one computation DAG.
    The DAGs with the same tag are considered as similar tasks"""
    ret = ""
    for op in dag.ops:
        tag = op.attrs.get('ansor_task_scheduler_tag', None)
        if tag:
            ret += op.attrs['ansor_task_scheduler_tag'] + "_"
    if ret != "":
        ret += "%d" % int(math.log(dag.flop_ct + 1, log_base))
    return ret


class SimpleTaskScheduler(TaskScheduler):
    """The default task scheduler with several strategies

    Parameters
    ----------
    tasks: List[SearchTask]
        All workloads to tune
    weights: List[float]
        Weights of tasks   (i.e. the number of occurrence of a task in the whole network)
    strategy: str
        The joint tuning strategy.
        "sequential" : Tune tasks sequentially. Divide n_trials equally to every task.
        "round-robin": Tune tasks in round robin order.
        "gradient" : Tune tasks with gradient descent.
    load_log_file: str
        Load history log file to pre-train cost model
    eps-random: float
        Always allocate this percent of n_trials to select tasks randomly.
        This is for encouraging exploration.
    verbose: int
        The level of verbosity. 0 means silent.
    alpha: float
        The parameter used for 'gradient' strategy
    beta: float
        The parameter used for 'gradient' strategy
    backward_window_size: int
        The parameter used for 'gradient' strategy
    """
    def __init__(self,
                 tasks: List[SearchTask],
                 objective_func: Callable = None,
                 strategy: str = 'gradient',
                 load_log_file: str = None,
                 load_model_file: str = None,
                 eps_random: float = 0.05,
                 verbose: int = 1,
                 alpha: float = 0.2,
                 beta: float = 2,
                 gamma: float = 0.5,
                 backward_window_size: int = 3,
                 use_debug_measurement_simulator=None):
        super().__init__(tasks, objective_func)
        self.strategy = strategy
        self.eps_random = eps_random
        self.verbose = verbose
        self.load_log_file = load_log_file
        self.load_model_file = load_model_file
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.backward_window_size = backward_window_size
        self.use_debug_measurement_simulator = use_debug_measurement_simulator

        assert self.strategy in ['round-robin', 'gradient']

        # task_cts[i] saves how many times task i is tuned
        self.task_cts = [0 for _ in range(len(self.tasks))]

        # task_costs_history[i] saves the latency history of task i
        self.task_costs_history = [[] for _ in range(len(self.tasks))]

        # best_costs[i] saves the best latency of task i
        self.best_costs = 1e10 * np.ones(len(self.tasks))

        self.tune_option = self.measurer = self.search_policies = self.ct = self.tic = None
        self.num_measure_per_iter = None
        self.dead_tasks = set()
        self.sequential_now_task_idx = 0
        self.sequential_now_task_begin_ct = 0

        assert len(tasks) != 0, "No tasks"

        # build similarity group
        self.task_tags = []
        self.tag_to_group_id = {}
        self.group_task_ids = []
        self.flop_cts = []
        for i, task in enumerate(self.tasks):
            tag = derive_similarity_tag(task.compute_dag)
            self.task_tags.append(tag)
            self.flop_cts.append(task.compute_dag.flop_ct)
            if tag == "":
                continue

            if tag not in self.tag_to_group_id:
                self.tag_to_group_id[tag] = len(self.tag_to_group_id)
                self.group_task_ids.append([])
            self.group_task_ids[self.tag_to_group_id[tag]].append(i)

    def tune(self, tune_option: TuneOption,
             search_policy: Union[str, List[SearchPolicy]] = 'default'):
        """ Tune tasks.

        Notice: This method does not have return value, make sure to set `LogToFile`
        measure callback in `tune_option`.

        Parameters
        ----------
        tune_option: TuneOption
        search_policy: Str or List[SearchPolicy]
        """
        # init members
        self.tune_option = tune_option
        if self.use_debug_measurement_simulator is None:
            self.measurer = ProgramMeasurer(tune_option.builder, tune_option.runner,
                                            tune_option.measure_callbacks, tune_option.verbose)
        self.ct = 0
        self.tic = time.time()
        self.sequential_now_task_idx = 0
        self.sequential_now_task_begin_ct = 0
        # reset num_measure_per_iter to make sure every task is tuned at least once
        self.num_measure_per_iter = min(tune_option.num_measure_per_iter,
                                        tune_option.n_trials // len(self.tasks))
        assert self.num_measure_per_iter > 0, "n_trials is too small"

        # restore the status of the task scheduler from a log file
        self.restore_status(self.load_log_file, self.num_measure_per_iter)

        # make one search policy for one task
        self.search_policies = get_search_policies(search_policy, self.tasks,
                                                   self.num_measure_per_iter,
                                                   self.load_model_file,
                                                   self.load_log_file)
        for i in range(len(self.tasks)):
            search_policy = self.search_policies[i]
            task = self.tasks[i]
            search_policy.set_task(task)
            search_policy.set_verbose(tune_option.verbose)
            search_policy.run_callbacks(tune_option.pre_search_callbacks)

        # do a round robin first
        if self.strategy != 'sequential':
            for i in range(len(self.tasks)):
                self.tune_task(i)

        # use the specific strategy to choose workload to tune
        task_idx = -1
        while self.ct < tune_option.n_trials and len(self.dead_tasks) < len(self.tasks):
            if self.strategy == 'sequential':
                allocated_total_ct = ((tune_option.n_trials - self.sequential_now_task_begin_ct)
                                      / (len(self.tasks) - self.sequential_now_task_idx))
                used_ct = self.ct - self.sequential_now_task_begin_ct

                if self.sequential_now_task_idx in self.dead_tasks or used_ct >= allocated_total_ct:
                    self.sequential_now_task_idx += 1
                    self.sequential_now_task_begin_ct = self.ct
                task_idx = self.sequential_now_task_idx
                if task_idx >= len(self.tasks):
                    break
            elif self.strategy == 'round-robin':
                task_idx = (task_idx + 1) % len(self.tasks)
                while task_idx in self.dead_tasks:
                    task_idx = (task_idx + 1) % len(self.tasks)
            elif self.strategy == 'gradient':
                gradients = []
                for i in range(len(self.tasks)):
                    if i in self.dead_tasks:
                        gradients.append(0)
                        continue

                    # compute gradient from chain rule : (delta f / delta g_i)
                    delta = 1e-7
                    new_costs = list(self.best_costs)
                    new_costs[i] -= delta
                    chain_grad = (self.compute_score(self.best_costs) - self.compute_score(new_costs)) / delta

                    # compute (g_i(t_i) - g(t_i - \Delta t)) / (\Delta t)
                    if (self.task_cts[i] - 1 < len(self.task_costs_history[i]) and
                        self.task_cts[i] - 1 - self.backward_window_size >= 0):
                        backward_grad = (self.task_costs_history[i][self.task_cts[i] - 1]
                                         - self.task_costs_history[i][self.task_cts[i] - 1 - self.backward_window_size]) \
                                        / self.backward_window_size
                    else:
                        backward_grad = 0

                    # compute (g_i(t_i + \Delta t) - g(t_i)) / (\Delta t)
                    g_next_1 = self.best_costs[i] - (self.best_costs[i] / self.task_cts[i])

                    g_next_2 = self.beta * 1e30
                    group_id = self.tag_to_group_id.get(self.task_tags[i], None)
                    if group_id is not None and len(self.group_task_ids[group_id]) > 1:
                        best_flops = max([self.flop_cts[j] / self.best_costs[j]
                                for j in self.group_task_ids[group_id]])
                        g_next_2 = self.beta * self.flop_cts[i] / best_flops

                    g_next = min(g_next_1, g_next_2)
                    forward_grad = g_next - self.best_costs[i]

                    # combine all grads
                    grad = chain_grad * (self.alpha * backward_grad + (1 - self.alpha) * forward_grad)
                    assert grad <= 0
                    gradients.append(grad)

                if max(gradients) == min(gradients):
                    task_idx = np.random.choice(len(gradients))
                else:
                    task_idx = np.argmin(gradients)
            else:
                raise ValueError("Invalid strategy: " + self.strategy)

            self.tune_task(task_idx)
            self.adjust_similarity_group(task_idx)

    def tune_task(self, task_idx):
        if self.verbose >= 1:
            logger.info("TaskScheduler: task id:\t%d" % task_idx)
        if self.use_debug_measurement_simulator is not None:
            measure_inputs, measure_results = \
                self.use_debug_measurement_simulator.get_next_batch(
                    self.tasks[task_idx],
                    self.num_measure_per_iter,
                )
        else:
            measure_inputs, measure_results = \
                self.search_policies[task_idx].continue_search(
                    self.tasks[task_idx],
                    self.num_measure_per_iter,
                    self.tune_option.verbose,
                    self.measurer)

        for inp, res in zip(measure_inputs, measure_results):
            cost = array_mean(res.costs)
            if cost < self.best_costs[task_idx]:
                self.best_costs[task_idx] = cost

        if len(measure_inputs) == 0:
            self.dead_tasks.add(task_idx)

        self.task_cts[task_idx] += 1
        self.task_costs_history[task_idx].append(self.best_costs[task_idx])

        self.ct += len(measure_inputs)
        self.cur_score = self.compute_score(self.best_costs)

        if self.verbose >= 1:
            logger.info(("TaskScheduler\tct: %d\testimated cost (ms): %.3f\ttime elapsed: %.2f\t" +
                        "best_costs (ms): %s\ttask_ct: %s") %
                        (self.ct, self.cur_score * 1e3, time.time() - self.tic,
                         to_str_round(self.best_costs * 1e3, decimal=3),
                         self.task_cts))

    def adjust_similarity_group(self, task_idx):
        group_id = self.tag_to_group_id.get(self.task_tags[task_idx], None)
        if group_id is None or len(self.group_task_ids[group_id]) <= 1:
            return

        group_ids = self.group_task_ids[group_id]
        best_group_flops = max([self.flop_cts[j] / self.best_costs[j] for j in group_ids])
        cur_flops = self.flop_cts[task_idx] / self.best_costs[task_idx]
 
        # if we tune a task for many times but it still cannot achieve
        # a similar speed to the fastest one in its group, this means this task
        # is actually not similar to other tasks in its group.
        # So we will reomve it from its original group.
        if (cur_flops < best_group_flops / self.beta and 
            self.task_cts[task_idx] > 5 + max(self.task_cts[j] for j in group_ids if j != task_idx)):
            self.task_tags[task_idx] = None
            group_ids.remove(task_idx)

    def restore_status(self, log_file, num_measure_per_iter):
        # restore task_cts and best_costs from a log file
        if log_file:
            str_target = str(self.tasks[0].target)
            workload_key_to_task_id = {t.workload_key : i for i, t in enumerate(self.tasks)}
            total_ct = -1

            for total_ct, (inp, res) in enumerate(LogReader(log_file)):
                if str(inp.task.target) != str_target:
                    continue
                task_idx = workload_key_to_task_id.get(inp.task.workload_key, None)
                if task_idx is None:
                    continue

                if res.error_no == 0:
                    self.best_costs[task_idx] = min(self.best_costs[task_idx], array_mean(res.costs))

                self.task_cts[task_idx] += 1

            for i in range(len(self.tasks)):
                # The computation of taks_cts is just an estimation.
                # The estimation may not be accurate if the log file is changed externally or
                # `num_measure_per_iter` is different from the last tuning.
                self.task_cts[i] = int(self.task_cts[i] / num_measure_per_iter + 0.5)
                self.task_costs_history[i].append(self.best_costs[i])

            logger.info("TaskScheduler: Loaded %d measurement records from %s",
                        total_ct + 1, log_file)

        self.cur_score = self.compute_score(self.best_costs)

