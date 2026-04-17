# simulator.py
import heapq
from collections import Counter, defaultdict
import logging
import math
import os
import numpy as np # For RL state arrays

from tasks import EmergencyTask, SimulationEvent
from rescuers import Rescuer, distance
from schedulers import NearestScheduler, UrgencyScheduler, HybridScheduler, DQNScheduler, QMIXScheduler, euclidean_distance
from single_agent import DQNAgent
from multi_agent import QMIXAgent

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

SIM_X_MIN, SIM_X_MAX = -20, 20
SIM_Y_MIN, SIM_Y_MAX = -20, 20

class EmergencySimulator:
    def __init__(self, params: dict, task_rng_instance, sim_rng_instance, agent_instance=None):
        self.params = params
        self.task_rng = task_rng_instance
        self.sim_rng = sim_rng_instance

        self.end_time = params['T']
        self.k = params.get('k', 1.0)
        self.strategy_name = params['strategy'].lower()
        self.lambda_rate = params['lambda_rate']
        self.scale_config = params['scale_config']

        self.event_queue = []
        self.current_time = 0.0

        self.tasks = []
        self.active_tasks = set()
        self.completed_tasks_stats = set()
        self.expired_tasks_stats = set()
        self.scale_count = Counter()
        self.task_id_counter = 0
        self.steps = 0

        self.replay_interval = params.get('replay_interval', 1)
        self.rl_agent_update_interval = params.get('rl_agent_update_interval', 10)
        self.reward_components = params.get('reward_components', {
            'shaping_coeff': 0.1, 'completion_bonus_coeff': 0.5,
            'fast_finish_threshold': 60.0, 'expire_penalty_coeff': 2.0
        })
        self.load_model_path = params.get('load_model_path', None)
        self.eval_mode = params.get('eval_mode', False)

        self.rescuers = self._init_rescuers()
        self.num_rescuers = len(self.rescuers)

        self.agent = None
        self.is_rl_strategy = self.strategy_name in ['dqn', 'qmix']

        if agent_instance:
            self.agent = agent_instance
            if self.eval_mode and hasattr(self.agent, 'set_eval_mode'):
                 self.agent.set_eval_mode()
            logger.info(f"Using provided RL agent instance for strategy: {self.strategy_name}")
        elif self.is_rl_strategy:
            self._init_rl_agent()

        if self.strategy_name == 'nearest':
            self.scheduler = NearestScheduler(self)
        elif self.strategy_name == 'urgent':
            self.scheduler = UrgencyScheduler(self)
        elif self.strategy_name == 'hybrid':
            hybrid_params_cfg = params.get('hybrid_params', {})
            self.scheduler = HybridScheduler(self, **hybrid_params_cfg)
        elif self.strategy_name == 'dqn':
            if not self.agent: self._init_rl_agent()
            self.scheduler = DQNScheduler(self, self.agent)
        elif self.strategy_name == 'qmix':
            if not self.agent: self._init_rl_agent()
            self.scheduler = QMIXScheduler(self, self.agent, self.num_rescuers)
        else:
            raise ValueError(f"Invalid or unhandled strategy name: {self.strategy_name}")

        self.fatigue_update_interval = params.get('fatigue_update_interval', 1.0)

        self.last_total_urgency_for_reward = self._calculate_current_total_urgency()
        self.qmix_pending_transition = {
            'individual_obs': [None] * self.num_rescuers,
            'joint_actions': [None] * self.num_rescuers,
            'global_state': None,
            'decision_rescuers_mask': [False] * self.num_rescuers
        }
        self.dqn_pending_experiences = {}
        self.accumulated_reward_for_current_rl_step = 0.0

        self._schedule_next_task_arrival()
        self._schedule_fatigue_recovery_updates()

    def _init_rl_agent(self):
        logger.info(f"Initializing new RL agent for strategy: {self.strategy_name}")
        agent_rng_instance = self.sim_rng

        if self.strategy_name == 'dqn':
            self.agent = DQNAgent(
                state_dim=self.params.get('dqn_state_dim', 42),
                action_dim=self.params.get('dqn_action_dim', 10),
                rng_instance=agent_rng_instance
            )
        elif self.strategy_name == 'qmix':
            if self.num_rescuers <= 0:
                logger.error("QMIX strategy requires num_rescuers > 0. Agent not created.")
                self.is_rl_strategy = False
                return
            self.agent = QMIXAgent(
                individual_state_dim=self.params.get('qmix_individual_obs_dim', 42),
                action_dim=self.params.get('qmix_action_dim', 10),
                num_agents=self.num_rescuers,
                global_state_dim=self.params.get('qmix_global_state_dim', 42 * self.num_rescuers),
                rng_instance=agent_rng_instance
            )

        if self.agent and self.load_model_path and os.path.exists(self.load_model_path):
            logger.info(f"Loading RL model from: {self.load_model_path}")
            self.agent.load_model(self.load_model_path)

        if self.agent and self.eval_mode and hasattr(self.agent, 'set_eval_mode'):
            self.agent.set_eval_mode()
        elif self.agent and hasattr(self.agent, 'set_train_mode'):
             self.agent.set_train_mode()


    def _init_rescuers(self):
        rescuers_list = []
        current_rescuer_id = 1
        rescuer_configs_from_params = self.params.get('rescuers_types', [])
        num_total_rescuers_param = self.params.get('num_rescuers', sum(c.get('count',0) for c in rescuer_configs_from_params))

        if not rescuer_configs_from_params and num_total_rescuers_param > 0 :
             rescuer_configs_from_params = [{'type': 'DefaultGeneric', 'skills': {'general'},
                                            'speed': self.params.get('rescuer_speed',0.8), 'work_rate': 1.0,
                                            'work_force':1, 'count': num_total_rescuers_param,
                                            'fatigue_threshold':150, 'fatigue_recovery_rate':4.0}]
             logger.warning("No specific rescuer types in params, creating generic ones based on num_rescuers.")

        main_station_pos = (0.0, 0.0)
        # Use sub_station_distance from params, with a fallback default
        sub_station_dist = self.params.get('sub_station_distance', 12.0) 

        # Define 4 cardinal sub-stations, consistent with typical visualization
        sub_stations_options = [
            (sub_station_dist, 0.0),
            (-sub_station_dist, 0.0),
            (0.0, sub_station_dist),
            (0.0, -sub_station_dist),
        ]
        
        all_station_locations = [main_station_pos] + sub_stations_options # Total 5 stations
        
        station_idx_counter = 0
        processed_rescuers_count = 0

        for config in rescuer_configs_from_params:
            count_for_type = config.get('count', 0)
            for _ in range(count_for_type):
                if processed_rescuers_count >= num_total_rescuers_param: break

                # Cycle through the defined station locations
                pos_to_assign = all_station_locations[station_idx_counter % len(all_station_locations)]
                station_idx_counter +=1

                type_name = config.get('type', f'GenericType{current_rescuer_id}')
                type_to_role_map = {
                    'Medical Team': 'medical',
                    'Engineering Unit': 'engineering',
                    'Search & Logistics': 'search',
                }
                primary_role = type_to_role_map.get(type_name)
                if primary_role is None and config.get('skills'):
                    primary_role = list(config.get('skills'))[0]

                rescuer = Rescuer(
                    rescuer_id=current_rescuer_id,
                    speed=config.get('speed', self.params.get('rescuer_speed', 0.8)),
                    work_rate=config.get('work_rate', 1.0),
                    pos=pos_to_assign,
                    skills=config.get('skills', {'general'}),
                    work_force=config.get('work_force', 1),
                    primary_role=primary_role,
                    fatigue_threshold=config.get('fatigue_threshold', 100.0),
                    fatigue_recovery_rate=config.get('fatigue_recovery_rate', 5.0),
                    fatigue_coeff_moving=config.get('fatigue_coeff_moving', 1.0),
                    fatigue_coeff_working=config.get('fatigue_coeff_working', 1.5)
                )
                setattr(rescuer, 'type_name', type_name)
                rescuers_list.append(rescuer)
                current_rescuer_id += 1
                processed_rescuers_count += 1
            if processed_rescuers_count >= num_total_rescuers_param: break
        
        if not rescuers_list and num_total_rescuers_param > 0 :
            logger.error(f"Rescuer initialization failed to create rescuers, though {num_total_rescuers_param} were requested.")
        elif rescuers_list:
            logger.info(f"Initialized {len(rescuers_list)} rescuers. Positions assigned from {len(all_station_locations)} station locations.")
            # For debugging, log the first few rescuer positions
            for r_idx, r_val in enumerate(rescuers_list[:max(5, len(all_station_locations))]): # Log up to 5 or number of stations
                logger.debug(f"Rescuer {r_val.id} initialized at {r_val.pos}")


        rescuers_list.sort(key=lambda r: r.id)
        return rescuers_list

    def _calculate_current_total_urgency(self):
        return sum(task.get_urgency(self.current_time) for task in self.active_tasks if not task.completed)

    def _schedule_next_task_arrival(self):
        if self.lambda_rate <= 0: return
        time_to_next_arrival = self.task_rng.expovariate(self.lambda_rate)
        next_arrival_time = self.current_time + time_to_next_arrival
        if next_arrival_time <= self.end_time:
            event = SimulationEvent(time=next_arrival_time, etype='task_arrive_signal')
            heapq.heappush(self.event_queue, event)

    def _schedule_fatigue_recovery_updates(self):
        if self.fatigue_update_interval > 0:
            next_recovery_time = self.current_time + self.fatigue_update_interval
            if next_recovery_time <= self.end_time:
                event = SimulationEvent(
                    time=next_recovery_time, etype='fatigue_recovery_update',
                    data={'interval': self.fatigue_update_interval}
                )
                heapq.heappush(self.event_queue, event)

    def _create_new_task(self, arrival_time):
        self.task_id_counter += 1
        scale_keys = list(self.scale_config.keys())
        weights = [self.params.get('task_scale_weights',{}).get(s,1) for s in scale_keys]
        if sum(weights) == len(weights):
            weights = [4 if s == "small" else 3 if s == "medium" else 2 if s == "large" else 1 for s in scale_keys]
        scale = self.task_rng.choices(scale_keys, weights=weights, k=1)[0]
        cfg = self.scale_config[scale]
        init_urgency = self.task_rng.uniform(*cfg['urgency_range'])
        decay_rate_param = self.task_rng.uniform(*cfg['decay_range'])
        workload = self.task_rng.uniform(*cfg['workload_range'])
        x = self.task_rng.uniform(SIM_X_MIN, SIM_X_MAX)
        y = self.task_rng.uniform(SIM_Y_MIN, SIM_Y_MAX)
        possible_skills = set()
        if self.params.get('rescuers_types'):
            for r_type_cfg in self.params['rescuers_types']:
                possible_skills.update(r_type_cfg.get('skills', set()))
        sorted_possible_skills = sorted(list(possible_skills)) if possible_skills else []
        role_requirements = {}
        required_skill = None
        if scale == 'small':
            if sorted_possible_skills:
                if 'medical' in sorted_possible_skills and 'search' in sorted_possible_skills:
                    role_requirements = self.task_rng.choice([{'medical': 1}, {'search': 1}])
                elif 'medical' in sorted_possible_skills:
                    role_requirements = {'medical': 1}
                elif 'search' in sorted_possible_skills:
                    role_requirements = {'search': 1}
                else:
                    role_requirements = {sorted_possible_skills[0]: 1}
            else:
                role_requirements = {'search': 1}
        elif scale == 'medium':
            role_requirements = {'medical': 1, 'search': 1}
            if sorted_possible_skills and 'engineering' in sorted_possible_skills:
                if self.task_rng.random() < 0.3:
                    role_requirements['engineering'] = 1
        elif scale == 'large':
            role_requirements = {'engineering': 2, 'search': 1}
            # heavy_rescue 没有对应 primary_role 的救援人员，不纳入 role_requirements
        elif scale == 'extra_large':
            role_requirements = {'engineering': 3, 'medical': 2, 'search': 2}
            # logistics 没有对应 primary_role 的救援人员，不纳入 role_requirements
        decay_m = 'exponential' if scale in ['large', 'extra_large'] else 'linear'
        # max_rescuers = 角色需求总和 + 冗余（small/medium/large +1, extra_large +2）
        base_max_r = sum(role_requirements.values()) if role_requirements else 1
        redundancy = 2 if scale == 'extra_large' else 1
        max_r_for_task = base_max_r + redundancy
        max_r_map = self.params.get('max_rescuers_map', {"small": 2, "medium": 3, "large": 5, "extra_large": 7})
        max_r_for_task = max(max_r_for_task, max_r_map.get(scale, 1))
        # required_skill 用于 RL 状态向量兼容性，取需求最大的角色作为代表
        required_skill = max(role_requirements, key=role_requirements.get) if role_requirements else None
        return EmergencyTask(
            task_id=self.task_id_counter, generation_time=arrival_time, x=x, y=y,
            initial_urgency=init_urgency, decay_rate=decay_rate_param, scale=scale,
            workload=workload, max_rescuers=max_r_for_task, decay_mode=decay_m,
            required_skill=required_skill, role_requirements=role_requirements
        )

    def _process_rl_decision_outcomes(self, is_terminal_step=False):
        if not self.is_rl_strategy or not self.agent:
            return
        if self.strategy_name == 'qmix':
            if any(self.qmix_pending_transition['decision_rescuers_mask']):
                all_next_individual_obs = []
                for r_idx, rescuer in enumerate(self.rescuers):
                    tasks_for_state = list(self.active_tasks) if not is_terminal_step else []
                    next_ind_obs = self.scheduler._get_state(rescuer, self.current_time, tasks_for_state)
                    all_next_individual_obs.append(next_ind_obs)
                tasks_for_global_state = list(self.active_tasks) if not is_terminal_step else []
                next_global_s = self.scheduler._get_global_state(self.current_time, self.rescuers, tasks_for_global_state)
                self.agent.remember(
                    self.qmix_pending_transition['individual_obs'],
                    self.qmix_pending_transition['joint_actions'],
                    self.accumulated_reward_for_current_rl_step,
                    all_next_individual_obs,
                    self.qmix_pending_transition['global_state'],
                    next_global_s,
                    is_terminal_step
                )
                self.qmix_pending_transition = {
                    'individual_obs': [None] * self.num_rescuers,
                    'joint_actions': [None] * self.num_rescuers,
                    'global_state': None,
                    'decision_rescuers_mask': [False] * self.num_rescuers
                }
        elif self.strategy_name == 'dqn':
            for rescuer_id, (prev_s, prev_a, _) in list(self.dqn_pending_experiences.items()):
                rescuer = next((r for r in self.rescuers if r.id == rescuer_id), None)
                if rescuer:
                    tasks_for_state = list(self.active_tasks) if not is_terminal_step else []
                    next_s = self.scheduler._get_state(rescuer, self.current_time, tasks_for_state)
                    self.agent.remember(prev_s, prev_a, self.accumulated_reward_for_current_rl_step, next_s, is_terminal_step)
            self.dqn_pending_experiences.clear()
        self.accumulated_reward_for_current_rl_step = 0.0

    def step(self):
        if not self.event_queue:
            self._finalize_simulation_state(is_terminal_step=True)
            return False
        event = heapq.heappop(self.event_queue)
        if event.time > self.end_time:
            heapq.heappush(self.event_queue, event)
            self._finalize_simulation_state(is_terminal_step=True)
            return False
        if event.time < self.current_time:
            logger.error(f"Time inconsistency: current_time={self.current_time:.2f}, event_time={event.time:.2f}. Event: {event}. Skipping.")
            return True
        self.current_time = event.time
        for task_obj in list(self.active_tasks):
            task_obj.update_urgency(self.current_time)
            if task_obj.get_urgency(self.current_time) <= 0 and \
               task_obj not in self.expired_tasks_stats and \
               task_obj not in self.completed_tasks_stats:
                 is_expire_event_imminent = any(
                     e.etype == 'task_expire' and e.task == task_obj and e.time <= self.current_time
                     for e in self.event_queue + [event]
                 )
                 if not is_expire_event_imminent:
                    expire_event_now = SimulationEvent(time=self.current_time, etype='task_expire', task=task_obj)
                    heapq.heappush(self.event_queue, expire_event_now)
        if event.etype == 'task_arrive_signal': self._handle_task_arrive_signal(event)
        elif event.etype == 'rescuer_arrive': self._handle_rescuer_arrive(event)
        elif event.etype == 'task_work_update': self._handle_task_work_update(event)
        elif event.etype == 'task_expire': self._handle_task_expire(event)
        elif event.etype == 'fatigue_recovery_update': self._handle_fatigue_recovery_update(event)
        else: logger.warning(f"Unknown event type: {event.etype} at T={self.current_time:.2f}")
        self._schedule_rescuers_and_process_rl_decisions()
        self._process_rl_decision_outcomes(is_terminal_step=(self.current_time >= self.end_time and not self.event_queue))
        self.steps += 1
        if self.is_rl_strategy and self.agent and hasattr(self.agent, 'replay'):
            if self.steps % self.replay_interval == 0:
                if hasattr(self.agent, 'model') and self.agent.model.training:
                    self.agent.replay()
            if self.steps % self.rl_agent_update_interval == 0:
                if hasattr(self.agent, 'model') and self.agent.model.training:
                    if hasattr(self.agent, '_update_target_networks'):
                        self.agent._update_target_networks()
                    elif hasattr(self.agent, 'update_target_model'):
                        self.agent.update_target_model()
        return True

    def _schedule_rescuers_and_process_rl_decisions(self):
        self.accumulated_reward_for_current_rl_step = 0.0
        # 强制检查：任何 idle 且疲劳 >= 阈值的救援人员立即进入 resting
        for r in self.rescuers:
            if r.status == 'idle' and r.fatigue >= r.fatigue_threshold:
                r.status = 'resting'
        idle_rescuers_for_scheduling = [r for r in self.rescuers if r.status == 'idle' and r.fatigue < r.fatigue_threshold]
        if not idle_rescuers_for_scheduling:
            return
        schedulable_tasks = [t for t in self.active_tasks if t.get_urgency(self.current_time) > 0 and not t.completed]
        idle_rescuers_for_scheduling.sort(key=lambda r: r.id)

        if self.strategy_name == 'dqn':
            for rescuer in idle_rescuers_for_scheduling:
                if rescuer.status != 'idle': continue
                if rescuer.id in self.dqn_pending_experiences:
                    prev_s, prev_a, _ = self.dqn_pending_experiences.pop(rescuer.id)
                chosen_task, state_for_decision, action_taken = self.scheduler.get_task_for_rl(
                    rescuer, self.current_time, schedulable_tasks
                )
                if self.is_rl_strategy and state_for_decision is not None and action_taken is not None:
                     self.dqn_pending_experiences[rescuer.id] = (state_for_decision, action_taken, self.current_time)

                if chosen_task:
                    self._assign_rescuer_to_task(rescuer, chosen_task)
        elif self.strategy_name == 'qmix':
            current_ind_obs_list = [None] * self.num_rescuers
            agent_indices_making_decision = []
            for r_idx, rescuer in enumerate(self.rescuers):
                if rescuer.status == 'idle':
                    obs = self.scheduler._get_state(rescuer, self.current_time, list(self.active_tasks))
                    current_ind_obs_list[r_idx] = obs
                    agent_indices_making_decision.append(r_idx)
                    self.qmix_pending_transition['decision_rescuers_mask'][r_idx] = True
                else:
                    current_ind_obs_list[r_idx] = np.zeros(self.agent.individual_state_dim, dtype=np.float32)
            if not agent_indices_making_decision:
                return
            self.qmix_pending_transition['individual_obs'] = current_ind_obs_list
            self.qmix_pending_transition['global_state'] = self.scheduler._get_global_state(
                self.current_time, self.rescuers, list(self.active_tasks)
            )
            joint_actions = self.agent.get_joint_action(current_ind_obs_list)
            self.qmix_pending_transition['joint_actions'] = joint_actions
            for r_idx in agent_indices_making_decision:
                rescuer = self.rescuers[r_idx]
                action_for_rescuer = joint_actions[r_idx]
                candidate_tasks_for_rescuer = [t for t in schedulable_tasks if rescuer.can_handle_task(t) and t.can_assign(rescuer)]
                sorted_candidate_tasks_for_action = sorted(
                    candidate_tasks_for_rescuer,
                    key=lambda t: (-t.get_urgency(self.current_time),
                                   euclidean_distance(rescuer.pos, (t.x, t.y)),
                                   t.id)
                )
                chosen_task = None
                if 0 <= action_for_rescuer < min(len(sorted_candidate_tasks_for_action), self.agent.action_dim):
                    chosen_task = sorted_candidate_tasks_for_action[action_for_rescuer]
                if chosen_task:
                    self._assign_rescuer_to_task(rescuer, chosen_task)
        else:
            for rescuer in idle_rescuers_for_scheduling:
                if rescuer.status != 'idle': continue
                chosen_task = self.scheduler.get_task(rescuer, self.current_time, schedulable_tasks)

                if chosen_task:
                    self._assign_rescuer_to_task(rescuer, chosen_task)
        current_total_urgency = self._calculate_current_total_urgency()
        urgency_reduction_shaping_reward = (self.last_total_urgency_for_reward - current_total_urgency) * \
                                           self.reward_components.get('shaping_coeff', 0.1)
        self.accumulated_reward_for_current_rl_step += urgency_reduction_shaping_reward
        self.last_total_urgency_for_reward = current_total_urgency

    def _assign_rescuer_to_task(self, rescuer: Rescuer, task: EmergencyTask):
        travel_time = rescuer.assign(task, self.current_time)
        # 如果分配失败（返回 None），直接跳出
        if travel_time is None:
            return

        # 1) 记录移动起止时间／位置，供可视化做插值
        rescuer.move_start_time   = self.current_time
        rescuer.move_arrival_time = self.current_time + travel_time
        rescuer.move_start_pos    = rescuer.pos
        rescuer.move_dest_pos     = (task.x, task.y)

        # 2) 负向旅行惩罚
        travel_penalty = -travel_time * self.reward_components.get('travel_penalty_coeff', 0.01)
        self.accumulated_reward_for_current_rl_step += travel_penalty

        # 3) 调度到达事件
        arrival_event = SimulationEvent(
            time=rescuer.move_arrival_time,
            etype='rescuer_arrive',
            task=task,
            rescuer=rescuer,
            data={'travel_time': travel_time}
        )
        heapq.heappush(self.event_queue, arrival_event)

        # 4) 完成分配：把 rescuer 加入 task
        task.assign_rescuer(rescuer, self.current_time)

        # 5) 如果任务已经在进行中，补调度一次工作更新
        if task.in_progress and task.assigned_rescuers_list:
            self._schedule_task_work_update(task, is_new_rescuer_added=True)

    def run(self):
        logger.info(f"Sim started. Strategy: {self.strategy_name.upper()}, End Time: {self.end_time:.2f}, Rescuers: {self.num_rescuers}, Seed: {self.sim_rng.getstate()[1][0] if hasattr(self.sim_rng,'getstate') else 'N/A'}")
        while self.step():
            pass
        logger.info(f"Sim run completed at T={self.current_time:.2f}.")
        self.print_results()

    def run_for_training(self):
        while self.current_time < self.end_time and (self.event_queue or self.lambda_rate > 0) :
            if not self.step():
                break
        self._finalize_simulation_state(is_terminal_step=True)
        self._process_rl_decision_outcomes(is_terminal_step=True)

    def _handle_task_arrive_signal(self, event):
        actual_task = self._create_new_task(event.time)
        self.tasks.append(actual_task)
        self.active_tasks.add(actual_task)
        self.scale_count[actual_task.scale] += 1
        if actual_task.expire_time <= self.end_time and actual_task.expire_time > actual_task.generation_time:
            expire_event = SimulationEvent(time=actual_task.expire_time, etype='task_expire', task=actual_task)
            heapq.heappush(self.event_queue, expire_event)
        elif actual_task.expire_time <= actual_task.generation_time:
             expire_event_now = SimulationEvent(time=actual_task.generation_time, etype='task_expire', task=actual_task)
             heapq.heappush(self.event_queue, expire_event_now)
        self._schedule_next_task_arrival()
        self.last_total_urgency_for_reward = self._calculate_current_total_urgency()

        # ===== 最近任务/规模优先策略的途中抢占式重定向 =====
        # 当新任务到达时，检查是否有移动中的救援人员应该被重定向到这个新任务
        if self.strategy_name in ('nearest', 'urgent') and hasattr(self.scheduler, 'should_redirect_moving_rescuer'):
            self._check_and_redirect_moving_rescuers(actual_task)

    def _check_and_redirect_moving_rescuers(self, new_task):
        """
        检查所有移动中的救援人员，判断是否应该重定向到新任务

        重定向规则：
        1. 救援人员正在移动中（status='moving'）
        2. 新任务距离显著更近（满足 scheduler.should_redirect_moving_rescuer 的条件）
        3. 新任务仍有分配空间
        """
        moving_rescuers = [r for r in self.rescuers if r.status == 'moving']

        if not moving_rescuers:
            return

        redirect_candidates = []

        for rescuer in moving_rescuers:
            should_redirect, reason = self.scheduler.should_redirect_moving_rescuer(
                rescuer, new_task, self.current_time
            )
            if should_redirect:
                # 计算节省的距离
                dist_to_current = euclidean_distance(rescuer.pos, (rescuer.task.x, rescuer.task.y))
                dist_to_new = euclidean_distance(rescuer.pos, (new_task.x, new_task.y))
                distance_savings = dist_to_current - dist_to_new
                redirect_candidates.append((rescuer, distance_savings, reason))

        # 按距离节省量排序，优先重定向节省最多的救援人员
        redirect_candidates.sort(key=lambda x: -x[1])

        # 执行重定向，但要确保新任务不会超过最大救援人数
        for rescuer, savings, reason in redirect_candidates:
            # 再次检查任务是否仍可分配（可能已被其他重定向填满）
            if new_task.can_assign(rescuer) and rescuer.can_handle_task(new_task):
                self._redirect_rescuer_to_new_task(rescuer, new_task)
                logger.info(f"Rescuer {rescuer.id} redirected to new Task {new_task.id} (saves {savings:.2f}km). Reason: {reason}")

    def _redirect_rescuer_to_new_task(self, rescuer, new_task):
        """
        执行救援人员重定向：从当前目标任务切换到新任务

        步骤：
        1. 从旧任务释放救援人员
        2. 取消旧的到达事件
        3. 重新分配到新任务
        4. 创建新的到达事件
        5. 更新移动状态属性
        """
        old_task = rescuer.task

        # 1. 从旧任务释放
        if old_task:
            old_task.release_rescuer(rescuer)
            # 清理旧任务的工作更新事件（如果旧任务因为这次释放而无人工作）
            if not old_task.assigned_rescuers_list and old_task.in_progress:
                old_task.in_progress = False
                # 取消旧任务的待处理工作更新事件
                self._cancel_task_work_update_events(old_task)

        # 2. 累积已移动时间对应的疲劳（原到达事件被取消，尚未在 arrival 中扣减）
        if hasattr(rescuer, 'move_start_time'):
            time_already_moving = self.current_time - rescuer.move_start_time
            if time_already_moving > 0:
                rescuer.update_fatigue(time_already_moving * rescuer.fatigue_coeff_moving)

        # 3. 取消旧的到达事件
        self._cancel_rescuer_arrival_event(rescuer, old_task)

        # 4. 计算到新任务的行程
        dist_to_new = euclidean_distance(rescuer.pos, (new_task.x, new_task.y))
        travel_time = dist_to_new / rescuer.speed if rescuer.speed > 0 else float('inf')

        # 5. 分配新任务
        rescuer.task = new_task
        rescuer.status = 'moving'
        rescuer.arrival_time = self.current_time + travel_time

        # 6. 更新移动状态属性（供可视化使用）
        rescuer.move_start_time = self.current_time
        rescuer.move_arrival_time = self.current_time + travel_time
        rescuer.move_start_pos = rescuer.pos
        rescuer.move_dest_pos = (new_task.x, new_task.y)

        # 7. 将救援人员加入新任务
        new_task.assign_rescuer(rescuer, self.current_time)

        # 8. 创建新的到达事件
        new_arrival_event = SimulationEvent(
            time=rescuer.move_arrival_time,
            etype='rescuer_arrive',
            task=new_task,
            rescuer=rescuer,
            data={'travel_time': travel_time}
        )
        heapq.heappush(self.event_queue, new_arrival_event)

        # 9. 如果新任务已经有其他救援人员在工作，调度工作更新
        if new_task.in_progress and new_task.assigned_rescuers_list:
            self._schedule_task_work_update(new_task, is_new_rescuer_added=True)

    def _cancel_rescuer_arrival_event(self, rescuer, target_task):
        """
        取消救援人员的特定到达事件
        """
        if not target_task:
            return

        new_event_queue = []
        for e in self.event_queue:
            if e.etype == 'rescuer_arrive' and e.rescuer == rescuer and e.task == target_task:
                continue  # 跳过要取消的事件
            new_event_queue.append(e)

        self.event_queue = new_event_queue
        heapq.heapify(self.event_queue)

    def _cancel_task_work_update_events(self, task):
        """
        取消任务的待处理工作更新事件
        """
        new_event_queue = []
        for e in self.event_queue:
            if e.etype == 'task_work_update' and e.task == task:
                continue  # 跳过要取消的事件
            new_event_queue.append(e)

        self.event_queue = new_event_queue
        heapq.heapify(self.event_queue)

    def _check_and_preempt_for_urgent_task(self, new_task):
        """
        检查所有救援人员，判断是否应该被抢占派往紧急灾情

        抢占规则（由scheduler.should_preempt_for_critical_task判断）：
        1. 移动中的救援人员（redirect）：常规重定向
        2. 正在执行普通任务的救援人员（interrupt）：中断执行并派往紧急灾情
        3. 正在执行紧急灾情的救援人员：不中断

        抢占优先级：优先抢占评分差异最大的救援人员
        """
        # 获取可被抢占的救援人员候选
        redirect_candidates = []  # 移动中重定向候选
        interrupt_candidates = []  # 工作中中断候选

        for rescuer in self.rescuers:
            if rescuer.status in ['moving', 'working']:
                should_preempt, reason, preempt_type = self.scheduler.should_preempt_for_critical_task(
                    rescuer, new_task, self.current_time
                )
                if should_preempt and preempt_type == 'redirect':
                    redirect_candidates.append((rescuer, reason))
                elif should_preempt and preempt_type == 'interrupt':
                    interrupt_candidates.append((rescuer, reason))

        # 优先执行中断（紧急灾情优先）
        # 按评分差异排序，优先中断评分差异最大的救援人员
        interrupt_candidates.sort(key=lambda x: self.scheduler._calculate_task_priority_score(
            new_task, x[0], self.current_time), reverse=True)

        for rescuer, reason in interrupt_candidates:
            if new_task.can_assign(rescuer) and rescuer.can_handle_task(new_task):
                old_task_id = rescuer.task.id if rescuer.task else 'N/A'
                self._interrupt_working_rescuer(rescuer, new_task)
                logger.info(f"[URGENT INTERRUPT] Rescuer {rescuer.id} interrupted from Task {old_task_id} to urgent Task {new_task.id}. Reason: {reason}")

        # 然后执行重定向
        redirect_candidates.sort(key=lambda x: self.scheduler._calculate_task_priority_score(
            new_task, x[0], self.current_time), reverse=True)

        for rescuer, reason in redirect_candidates:
            if new_task.can_assign(rescuer) and rescuer.can_handle_task(new_task):
                self._redirect_rescuer_to_new_task(rescuer, new_task)
                logger.info(f"[URGENT REDIRECT] Rescuer {rescuer.id} redirected to urgent Task {new_task.id}. Reason: {reason}")

    def _interrupt_working_rescuer(self, rescuer, new_task):
        """
        中断正在执行任务的救援人员，并派往新的紧急灾情

        步骤：
        1. 从当前任务释放救援人员，记录已完成的工作量
        2. 取消当前任务的后续工作更新事件
        3. 如果当前任务仍有其他救援人员，重新调度工作更新
        4. 将救援人员派往新任务
        """
        old_task = rescuer.task
        if not old_task:
            return

        # 1. 从旧任务释放救援人员
        old_task.release_rescuer(rescuer)

        # 3. 取消旧任务的后续工作更新事件
        # 注意：如果旧任务仍有其他救援人员工作，需要重新调度
        self._cancel_task_work_update_events(old_task)

        if old_task.assigned_rescuers_list and old_task.in_progress:
            # 旧任务仍有救援人员，重新调度工作更新
            self._schedule_task_work_update(old_task)
        elif not old_task.assigned_rescuers_list and old_task.in_progress:
            # 旧任务无救援人员了，标记为暂停
            old_task.in_progress = False
            logger.info(f"Task {old_task.id} paused due to rescuer interruption (remaining workload: {old_task.remaining_workload:.1f})")

        # 4. 清除救援人员的当前任务引用
        rescuer.task = None
        rescuer.status = 'idle'

        # 5. 计算到新任务的行程
        dist_to_new = euclidean_distance(rescuer.pos, (new_task.x, new_task.y))
        travel_time = dist_to_new / rescuer.speed if rescuer.speed > 0 else float('inf')

        # 6. 分配新任务
        rescuer.task = new_task
        rescuer.status = 'moving'
        rescuer.arrival_time = self.current_time + travel_time

        # 7. 更新移动状态属性（供可视化使用）
        rescuer.move_start_time = self.current_time
        rescuer.move_arrival_time = self.current_time + travel_time
        rescuer.move_start_pos = rescuer.pos
        rescuer.move_dest_pos = (new_task.x, new_task.y)

        # 8. 将救援人员加入新任务
        new_task.assign_rescuer(rescuer, self.current_time)

        # 9. 创建新的到达事件
        new_arrival_event = SimulationEvent(
            time=rescuer.move_arrival_time,
            etype='rescuer_arrive',
            task=new_task,
            rescuer=rescuer,
            data={'travel_time': travel_time}
        )
        heapq.heappush(self.event_queue, new_arrival_event)

        # 10. 如果新任务已经有其他救援人员在工作，调度工作更新
        if new_task.in_progress and new_task.assigned_rescuers_list:
            self._schedule_task_work_update(new_task, is_new_rescuer_added=True)

    def _handle_rescuer_arrive(self, event):
        rescuer: Rescuer = event.rescuer
        task: EmergencyTask = event.task
        travel_duration = event.data.get('travel_time', 0)

        # 检查任务状态是否仍然有效
        if task.completed or task in self.expired_tasks_stats:
            # 任务已完成或过期，救援人员无需继续
            if rescuer.task == task:
                rescuer.complete_task_actions()
            # 清理可能残留的移动状态属性（防止可视化异常）
            self._clear_rescuer_move_attributes(rescuer)
            return

        # 检查救援人员是否仍被分配到此任务（可能中途被重新分配）
        if rescuer.task != task:
            # 救援人员已被重新分配到其他任务，清理此事件的残留状态
            self._clear_rescuer_move_attributes(rescuer)
            return

        rescuer.pos = (task.x, task.y)
        rescuer.update_fatigue(travel_duration * rescuer.fatigue_coeff_moving)

        # 若移动后疲劳达到阈值，强制休息，不开始工作
        if rescuer.fatigue >= rescuer.fatigue_threshold:
            task.release_rescuer(rescuer)
            rescuer.complete_task_actions()
            self._clear_rescuer_move_attributes(rescuer)
            if not task.assigned_rescuers_list and task.in_progress:
                task.in_progress = False
                self._cancel_task_work_update_events(task)
            return

        if task.get_urgency(self.current_time) <= 0:
            if task not in self.expired_tasks_stats:
                self._handle_task_expire(SimulationEvent(time=self.current_time, etype='task_expire', task=task))
            rescuer.complete_task_actions()
            return
        # 防御性检查：如果该角色的槽位已被更早到达的救援人员填满，则释放当前救援人员
        # 注意：需要排除当前救援人员自身，因为他们刚刚才要变为 working
        current_working_same_role = sum(
            1 for r in task.assigned_rescuers_list
            if r != rescuer and getattr(r, 'status', None) == 'working' and getattr(r, 'primary_role', None) == rescuer.primary_role
        )
        role_req = task.role_requirements.get(rescuer.primary_role, 0)
        if current_working_same_role >= role_req:
            task.release_rescuer(rescuer)
            rescuer.complete_task_actions()
            self._clear_rescuer_move_attributes(rescuer)
            if not task.assigned_rescuers_list and task.in_progress:
                task.in_progress = False
                self._cancel_task_work_update_events(task)
            elif task.assigned_rescuers_list:
                self._schedule_task_work_update(task)
            return

        rescuer.status = 'working'
        if not task.in_progress:
            task.in_progress = True
            if task.start_time is None: task.start_time = self.current_time
        self._schedule_task_work_update(task)

    def _clear_rescuer_move_attributes(self, rescuer):
        """清理救援人员的移动状态属性，防止可视化或状态查询异常"""
        if hasattr(rescuer, 'move_start_time'):
            delattr(rescuer, 'move_start_time')
        if hasattr(rescuer, 'move_arrival_time'):
            delattr(rescuer, 'move_arrival_time')
        if hasattr(rescuer, 'move_start_pos'):
            delattr(rescuer, 'move_start_pos')
        if hasattr(rescuer, 'move_dest_pos'):
            delattr(rescuer, 'move_dest_pos')

    def _schedule_task_work_update(self, task, is_new_rescuer_added=False):
        new_event_queue = [e for e in self.event_queue if not (e.etype == 'task_work_update' and e.task == task)]
        self.event_queue = new_event_queue
        heapq.heapify(self.event_queue)
        if not task.in_progress or not task.assigned_rescuers_list or task.completed or task in self.expired_tasks_stats:
            return
        effective_rescuers = task.get_effective_rescuers()
        num_effective_rescuers = len(effective_rescuers)
        if num_effective_rescuers == 0: return
        base_total_work_rate = sum(r.work_rate for r in effective_rescuers)
        effective_work_rate = base_total_work_rate * (num_effective_rescuers ** max(0, self.k - 1.0))
        if effective_work_rate <= 0: return
        work_update_granularity = self.params.get('work_update_granularity', 1.0)
        time_to_complete_remaining = task.remaining_workload / effective_work_rate
        duration_for_this_step: float
        if time_to_complete_remaining <= work_update_granularity:
            duration_for_this_step = time_to_complete_remaining
        else:
            duration_for_this_step = work_update_granularity
        work_to_be_done_this_step = effective_work_rate * duration_for_this_step
        next_event_time = self.current_time + duration_for_this_step
        if next_event_time <= self.end_time and duration_for_this_step > 1e-6 :
            work_event = SimulationEvent(
                time=next_event_time, etype='task_work_update', task=task,
                data={'work_to_do': work_to_be_done_this_step, 'duration': duration_for_this_step}
            )
            heapq.heappush(self.event_queue, work_event)
        elif task.remaining_workload > 0 and duration_for_this_step <= 1e-6 :
             work_event_now = SimulationEvent(
                 time=self.current_time, etype='task_work_update', task=task,
                 data={'work_to_do': task.remaining_workload, 'duration': 0.0}
             )
             heapq.heappush(self.event_queue, work_event_now)
             heapq.heapify(self.event_queue)

    def _handle_task_work_update(self, event):
        task: EmergencyTask = event.task
        work_done_this_interval = event.data['work_to_do']
        duration_of_work_this_interval = event.data['duration']
        if task.completed or task in self.expired_tasks_stats or not task.in_progress:
            return
        task.complete_work(work_done_this_interval)
        if task.assigned_rescuers_list and duration_of_work_this_interval > 0:
            # 每个救援人员根据个体的工作疲劳系数累积疲劳
            for r in task.assigned_rescuers_list:
                if r.status == 'working' and r.task == task:
                    fatigue_accrued = duration_of_work_this_interval * r.fatigue_coeff_working
                    r.update_fatigue(fatigue_accrued)

        # 强制休息检查：工作过程中达到疲劳阈值的救援人员退出任务
        exhausted_rescuers = [r for r in list(task.assigned_rescuers_list)
                              if r.status == 'working' and r.fatigue >= r.fatigue_threshold]
        for r in exhausted_rescuers:
            task.release_rescuer(r)
            r.complete_task_actions()
        if exhausted_rescuers:
            if not task.assigned_rescuers_list and task.in_progress:
                task.in_progress = False
                self._cancel_task_work_update_events(task)
            elif task.assigned_rescuers_list:
                self._schedule_task_work_update(task)

        if task.remaining_workload <= 0:
            task.completed = True
            task.finish_time = self.current_time
            self.completed_tasks_stats.add(task)
            if task in self.active_tasks: self.active_tasks.remove(task)
            logger.info(f"Task {task.id} COMPLETED T={self.current_time:.2f}.")
            completion_reward = task.initial_urgency
            if task.start_time is not None:
                time_taken = task.finish_time - task.start_time
                if time_taken < self.reward_components.get('fast_finish_threshold', 60.0):
                    completion_reward += (self.reward_components.get('fast_finish_threshold', 60.0) - time_taken) * \
                                         self.reward_components.get('completion_bonus_coeff', 0.5)
            self.accumulated_reward_for_current_rl_step += completion_reward
            for rescuer in list(task.assigned_rescuers_list):
                rescuer.complete_task_actions()
                task.release_rescuer(rescuer)
            self.last_total_urgency_for_reward = self._calculate_current_total_urgency()
        elif task.get_urgency(self.current_time) <= 0:
            if task not in self.expired_tasks_stats:
                self._handle_task_expire(SimulationEvent(time=self.current_time, etype='task_expire', task=task))
        else:
            self._schedule_task_work_update(task)

    def _handle_task_expire(self, event):
        task: EmergencyTask = event.task
        if task.completed or task in self.expired_tasks_stats: return
        task.current_urgency = 0
        self.expired_tasks_stats.add(task)
        if task in self.active_tasks: self.active_tasks.remove(task)
        logger.info(f"Task {task.id} EXPIRED T={self.current_time:.2f}.")
        expiration_penalty = -task.initial_urgency * self.reward_components.get('expire_penalty_coeff', 2.0)
        self.accumulated_reward_for_current_rl_step += expiration_penalty
        if task.assigned_rescuers_list:
            for rescuer in list(task.assigned_rescuers_list):
                rescuer.complete_task_actions()
                task.release_rescuer(rescuer)
        new_event_queue = [e for e in self.event_queue if not (e.etype == 'task_work_update' and e.task == task)]
        self.event_queue = new_event_queue
        heapq.heapify(self.event_queue)
        self.last_total_urgency_for_reward = self._calculate_current_total_urgency()

    def _handle_fatigue_recovery_update(self, event):
        interval_duration = event.data.get('interval', self.fatigue_update_interval)
        if interval_duration <= 0: return
        for rescuer in self.rescuers:
            # idle 和 resting 状态都可以恢复疲劳
            if rescuer.status in ['idle', 'resting']:
                rescuer.recover_fatigue(interval_duration)
        self._schedule_fatigue_recovery_updates()

    def _finalize_simulation_state(self, is_terminal_step=False):
        final_check_time = self.current_time
        for task in list(self.active_tasks):
            if task.completed or task in self.expired_tasks_stats: continue
            task.update_urgency(final_check_time)
            if task.current_urgency <= 0:
                self._handle_task_expire(SimulationEvent(time=final_check_time, etype='task_expire', task=task))
        if is_terminal_step:
            self._process_rl_decision_outcomes(is_terminal_step=True)

    def print_results(self):
        total_tasks_generated = len(self.tasks)
        num_completed = len(self.completed_tasks_stats)
        num_expired = len(self.expired_tasks_stats)
        num_unfinished_active = 0
        total_remaining_workload_unfinished = 0.0
        for task in self.tasks:
             if not task.completed and task not in self.expired_tasks_stats:
                  num_unfinished_active +=1
                  total_remaining_workload_unfinished += task.remaining_workload
        print(f"\n{' SIMULATION RESULTS ':=^60}")
        print(f"Strategy: {self.strategy_name.upper()}")
        print(f"Simulation End Time: {self.current_time:.2f} / {self.end_time:.2f} minutes")
        print(f"-"*60)
        print(f"Total Tasks Generated: {total_tasks_generated}")
        print(f"  Completed Tasks: {num_completed} ({num_completed / total_tasks_generated:.1%})" if total_tasks_generated > 0 else f"  Completed Tasks: {num_completed} (N/A)")
        print(f"  Expired Tasks: {num_expired} ({num_expired / total_tasks_generated:.1%})" if total_tasks_generated > 0 else f"  Expired Tasks: {num_expired} (N/A)")
        print(f"  Unfinished Active Tasks at Sim End: {num_unfinished_active}")
        if num_unfinished_active > 0:
            print(f"    (Total remaining workload for unfinished: {total_remaining_workload_unfinished:.2f})")
        print(f"-"*60)
        completion_details = []
        total_initial_urgency_completed = 0.0
        total_workload_completed = 0.0
        for task_stat in self.completed_tasks_stats:
            response_time = -1.0; service_time = -1.0
            if task_stat.start_time is not None:
                response_time = task_stat.start_time - task_stat.generation_time
                if task_stat.finish_time is not None:
                    service_time = task_stat.finish_time - task_stat.start_time
            completion_details.append({'response_time': max(0, response_time),
                                       'service_time': max(0, service_time),
                                       'initial_urgency': task_stat.initial_urgency,
                                       'workload': task_stat.workload,
                                       'scale': task_stat.scale})
            total_initial_urgency_completed += task_stat.initial_urgency
            total_workload_completed += task_stat.workload
        if completion_details:
            avg_response_time = sum(d['response_time'] for d in completion_details) / len(completion_details)
            avg_service_time = sum(d['service_time'] for d in completion_details) / len(completion_details)
            print(f"Avg. Response Time (completed tasks): {avg_response_time:.2f} min")
            print(f"Avg. Service Time (completed tasks): {avg_service_time:.2f} min")
            print(f"Total Initial Urgency of Completed Tasks: {total_initial_urgency_completed:.2f}")
            print(f"Total Workload of Completed Tasks: {total_workload_completed:.2f}")
        else:
            print("No tasks completed.")
        print(f"-"*60); print("Completed Tasks by Scale:")
        completed_scales_counter = Counter(d['scale'] for d in completion_details)
        for scale_type_key in self.scale_config.keys():
            print(f"  {scale_type_key.capitalize()}: {completed_scales_counter.get(scale_type_key, 0)}")
        print(f"-"*60); print("Generated Tasks by Scale (for reference):")
        # 输出顺序：small -> medium -> large -> extra_large
        for scale_type_key in self.scale_config.keys():
            print(f"  {scale_type_key.capitalize()}: {self.scale_count.get(scale_type_key, 0)}")
        print(f"-"*60); print("Rescuer Fatigue Summary:")
        if self.rescuers:
            total_fatigue_all_rescuers = sum(r.fatigue for r in self.rescuers)
            num_resting_at_simulation_end = sum(1 for r in self.rescuers if r.status == 'resting')
            avg_final_fatigue = total_fatigue_all_rescuers / len(self.rescuers) if self.rescuers else 0
            print(f"Average Final Fatigue: {avg_final_fatigue:.2f}")
            print(f"Rescuers Resting at End: {num_resting_at_simulation_end}")
        else: print("  No rescuers in simulation.")
        print("=" * 60)