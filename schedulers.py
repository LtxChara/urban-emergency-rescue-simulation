# schedulers.py
import math
from abc import ABC, abstractmethod
import numpy as np

def euclidean_distance(pos1, pos2):
    """
    计算两点之间的欧氏距离。
    参数:
        pos1 (tuple/list): 第一个点的坐标 (x1, y1)。
        pos2 (tuple/list): 第二个点的坐标 (x2, y2)。
    返回:
        float: 欧氏距离。
    """
    return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])

class SchedulerBase(ABC):
    """
    所有调度策略的抽象基类。
    """
    def __init__(self, simulator_ref=None):
        self.simulator = simulator_ref
        self.tasks_known_to_scheduler = []

    def update_known_tasks(self, tasks_list):
        self.tasks_known_to_scheduler = tasks_list

    @abstractmethod
    def get_task(self, rescuer, current_time, available_tasks_from_simulator):
        pass

    def get_task_for_rl(self, rescuer, current_time, available_tasks_from_simulator):
        chosen_task = self.get_task(rescuer, current_time, available_tasks_from_simulator)
        return chosen_task, None, None

    def _get_state(self, rescuer, current_time, valid_tasks_for_state_representation):
        map_boundary_abs_sim = self.simulator.params.get('map_boundary_for_norm', 20.0) if self.simulator else 20.0
        state_representation = [
            rescuer.pos[0] / map_boundary_abs_sim,
            rescuer.pos[1] / map_boundary_abs_sim
        ]
        sorted_tasks_for_rl = sorted(
            valid_tasks_for_state_representation,
            key=lambda t: (-t.get_urgency(current_time),
                           euclidean_distance(rescuer.pos, (t.x, t.y)),
                           t.id)
        )
        num_task_features = 4
        strategy_for_dims = self.simulator.params.get('strategy', 'dqn') if self.simulator else 'dqn'
        action_dim_key = 'dqn_action_dim' if strategy_for_dims == 'dqn' else 'qmix_action_dim'
        max_tasks_in_state = self.simulator.params.get(action_dim_key, 10) if self.simulator else 10
        max_urgency_est = self.simulator.params.get('max_urgency_for_norm', 250.0) if self.simulator else 250.0
        max_workload_est = self.simulator.params.get('max_workload_for_norm', 180.0) if self.simulator else 180.0
        for i in range(max_tasks_in_state):
            if i < len(sorted_tasks_for_rl):
                task = sorted_tasks_for_rl[i]
                state_representation.extend([
                    task.x / map_boundary_abs_sim,
                    task.y / map_boundary_abs_sim,
                    task.get_urgency(current_time) / max_urgency_est,
                    task.remaining_workload / max_workload_est
                ])
            else:
                state_representation.extend([0.0] * num_task_features)
        state_dim_key = 'dqn_state_dim' if strategy_for_dims == 'dqn' else 'qmix_individual_obs_dim'
        default_expected_dim = 2 + max_tasks_in_state * num_task_features
        expected_dim = self.simulator.params.get(state_dim_key, default_expected_dim) if self.simulator else default_expected_dim
        if len(state_representation) > expected_dim:
            state_representation = state_representation[:expected_dim]
        elif len(state_representation) < expected_dim:
            state_representation.extend([0.0] * (expected_dim - len(state_representation)))
        return np.array(state_representation, dtype=np.float32)

class NearestScheduler(SchedulerBase):
    """
    最近任务优先策略

    策略定义：
    - 基于不完全信息的动态贪心策略，只基于当前已知灾情进行调度
    - 核心原则：选择距离救援人员当前位置最近的可用任务
    - 辅助优化：仅在距离完全相等时，按紧急程度和工作量作为次级排序

    重分配规则（由仿真器调用 check_and_redirect_moving_rescuers 实现）：
    - 非抢占式执行：已开始作业的救援人员(status='working')不可中断
    - 途中抢占式重定向：移动中的救援人员(status='moving')可被重定向到更近的新任务
    """

    def get_task(self, rescuer, current_time, available_tasks_from_simulator):
        if rescuer.fatigue >= rescuer.fatigue_threshold:
            return None
        if not available_tasks_from_simulator:
            return None

        # 筛选出救援人员能处理、可分配且紧急程度>0的任务
        valid_for_rescuer = [
            t for t in available_tasks_from_simulator
            if rescuer.can_handle_task(t) and t.can_assign(rescuer) and t.get_urgency(current_time) > 0
        ]

        if not valid_for_rescuer:
            return None

        # 纯粹的距离优先排序
        # 主排序：距离（升序）
        # 次排序：紧急程度（降序）- 仅用于距离完全相等时的决策
        # 三级排序：工作量（升序）- 用于距离和紧急程度都相等时
        # 四级排序：任务ID（升序）- 确保确定性
        def distance_key(task):
            return euclidean_distance(rescuer.pos, (task.x, task.y))

        valid_for_rescuer.sort(key=lambda t: (
            distance_key(t),                    # 主排序：距离（升序）
            -t.get_urgency(current_time),       # 次排序：紧急程度（降序）
            t.remaining_workload,               # 三级：工作量（升序）
            t.id                                # 四级：ID（升序，确定性）
        ))

        return valid_for_rescuer[0]

    def should_redirect_moving_rescuer(self, rescuer, new_task, current_time, distance_threshold_ratio=0.7):
        """
        判断是否应该对移动中的救援人员进行重定向

        重定向条件：
        1. 救援人员正在移动中(status='moving')
        2. 新任务可被救援人员处理
        3. 新任务可分配（未满员）
        4. 新任务紧急程度>0
        5. 新任务距离显著更近（当前目标距离 * threshold_ratio）

        参数：
        - distance_threshold_ratio: 重定向的距离阈值比例，默认0.7
          即新任务距离必须小于当前目标距离的70%才触发重定向
          这个阈值防止频繁的微小距离优化导致的抖动

        返回：
        - (should_redirect: bool, reason: str)
        """
        if rescuer.status != 'moving':
            return False, "救援人员不在移动状态"

        if rescuer.fatigue >= rescuer.fatigue_threshold:
            return False, "救援人员疲劳值已达阈值"

        if not rescuer.task:
            return False, "救援人员无当前目标任务"

        # 检查救援人员是否能处理新任务
        if not rescuer.can_handle_task(new_task):
            return False, f"救援人员技能/载荷不匹配新任务"

        # 检查新任务是否可分配
        if not new_task.can_assign(rescuer):
            return False, f"新任务已满员({len(new_task.assigned_rescuers_list)}/{new_task.max_rescuers})"

        # 检查新任务紧急程度
        if new_task.get_urgency(current_time) <= 0:
            return False, "新任务已过期或无紧急度"

        # 计算距离
        # 使用救援人员的当前位置进行计算
        current_pos = rescuer.pos

        # 到当前目标的剩余距离
        dist_to_current_target = euclidean_distance(current_pos, (rescuer.task.x, rescuer.task.y))

        # 到新任务的距离
        dist_to_new_task = euclidean_distance(current_pos, (new_task.x, new_task.y))

        # 重定向阈值：新任务距离必须显著更近
        redirect_threshold = dist_to_current_target * distance_threshold_ratio

        if dist_to_new_task < redirect_threshold:
            savings = dist_to_current_target - dist_to_new_task
            return True, f"新任务距离{dist_to_new_task:.2f}km < 当前目标{dist_to_current_target:.2f}km的{distance_threshold_ratio*100:.0f}%，节省{savings:.2f}km"

        return False, f"新任务距离{dist_to_new_task:.2f}km不满足重定向阈值({redirect_threshold:.2f}km)"

class UrgencyScheduler(SchedulerBase):
    """
    灾情规模优先的贪心调度策略

    策略定义：
    - 基于不完全信息的动态贪心策略，核心原则：优先调度大规模灾情任务
    - 主排序：任务规模（降序）—— extra_large > large > medium > small
    - 次排序：距离（升序）—— 同规模下选择最近的任务
    - 三级排序：紧急程度（降序）—— 规模和距离都相同时选更紧急的
    - 四级排序：任务ID（升序）—— 确保确定性

    重分配规则（非抢占式）：
    - 已开始作业的救援人员(status='working')不可中断
    - 移动中的救援人员(status='moving')仅允许途中的scale-aware重定向：
        1) 新任务规模更大且距离在可接受范围内（不超过当前目标距离的1.5倍）
        2) 新任务规模相同且距离显著更近（小于当前目标距离的70%）
    """

    _SCALE_RANK = {
        'small': 1,
        'medium': 2,
        'large': 3,
        'extra_large': 4
    }

    def __init__(self, simulator_ref=None, distance_threshold_ratio=0.7,
                 scale_upgrade_distance_ratio=1.5):
        super().__init__(simulator_ref)
        self.distance_threshold_ratio = distance_threshold_ratio
        self.scale_upgrade_distance_ratio = scale_upgrade_distance_ratio

    def get_task(self, rescuer, current_time, available_tasks_from_simulator):
        if rescuer.fatigue >= rescuer.fatigue_threshold:
            return None
        if not available_tasks_from_simulator:
            return None

        valid_for_rescuer = [
            t for t in available_tasks_from_simulator
            if rescuer.can_handle_task(t) and t.can_assign(rescuer) and t.get_urgency(current_time) > 0
        ]

        if not valid_for_rescuer:
            return None

        def distance_key(task):
            return euclidean_distance(rescuer.pos, (task.x, task.y))

        valid_for_rescuer.sort(key=lambda t: (
            -self._SCALE_RANK.get(t.scale, 0),  # 主排序：规模降序
            distance_key(t),                     # 次排序：距离升序
            -t.get_urgency(current_time),        # 三级：紧急程度降序
            t.id                                 # 四级：ID升序（确定性）
        ))

        return valid_for_rescuer[0]

    def should_redirect_moving_rescuer(self, rescuer, new_task, current_time):
        """
        判断是否应该对移动中的救援人员进行scale-aware重定向
        """
        if rescuer.status != 'moving':
            return False, "救援人员不在移动状态"

        if rescuer.fatigue >= rescuer.fatigue_threshold:
            return False, "救援人员疲劳值已达阈值"

        if not rescuer.task:
            return False, "救援人员无当前目标任务"

        if not rescuer.can_handle_task(new_task):
            return False, "救援人员技能/载荷不匹配新任务"

        if not new_task.can_assign(rescuer):
            return False, f"新任务已满员({len(new_task.assigned_rescuers_list)}/{new_task.max_rescuers})"

        if new_task.get_urgency(current_time) <= 0:
            return False, "新任务已过期或无紧急度"

        current_task = rescuer.task
        if current_task == new_task:
            return False, "新任务与当前目标相同"

        current_rank = self._SCALE_RANK.get(current_task.scale, 0)
        new_rank = self._SCALE_RANK.get(new_task.scale, 0)

        dist_to_current = euclidean_distance(rescuer.pos, (current_task.x, current_task.y))
        dist_to_new = euclidean_distance(rescuer.pos, (new_task.x, new_task.y))

        if new_rank > current_rank:
            # 新任务规模更大：允许一定距离代价的重定向，但不可跨地图拉人
            if dist_to_new <= dist_to_current * self.scale_upgrade_distance_ratio:
                return True, (
                    f"新任务规模更大({new_task.scale}>{current_task.scale})且距离可接受"
                    f"({dist_to_new:.2f}km <= {dist_to_current:.2f}km*{self.scale_upgrade_distance_ratio:.1f})"
                )
            return False, (
                f"新任务规模更大但距离过远"
                f"({dist_to_new:.2f}km > {dist_to_current:.2f}km*{self.scale_upgrade_distance_ratio:.1f})"
            )

        if new_rank == current_rank:
            # 同规模：必须距离显著更近，避免抖动
            redirect_threshold = dist_to_current * self.distance_threshold_ratio
            if dist_to_new < redirect_threshold:
                savings = dist_to_current - dist_to_new
                return True, (
                    f"同规模下新任务距离显著更近"
                    f"({dist_to_new:.2f}km < {redirect_threshold:.2f}km, 节省{savings:.2f}km)"
                )
            return False, f"同规模下距离不满足重定向阈值({dist_to_new:.2f}km >= {redirect_threshold:.2f}km)"

        # new_rank < current_rank
        return False, f"新任务规模更小({new_task.scale}<{current_task.scale})，不触发重定向"


class HybridScheduler(SchedulerBase):
    def __init__(self, simulator_ref=None,
                 alpha=1.2, beta=1.2, delta=1.2, epsilon=0.5, zeta=0.2,
                 epsilon_dist=0.1, epsilon_etc=0.1):
        super().__init__(simulator_ref)
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.epsilon = epsilon
        self.zeta = zeta
        self.epsilon_dist = epsilon_dist
        self.epsilon_etc = epsilon_etc

    def get_task(self, rescuer, current_time, available_tasks_from_simulator):
        if rescuer.fatigue >= rescuer.fatigue_threshold:
            return None
        if not available_tasks_from_simulator:
            return None
        valid_for_rescuer = [
            t for t in available_tasks_from_simulator
            if rescuer.can_handle_task(t) and t.can_assign(rescuer) and t.get_urgency(current_time) > 0
        ]
        if not valid_for_rescuer:
            return None
        def hybrid_score(task, current_rescuer):
            urgency = task.get_urgency(current_time)
            if urgency <= 1e-6:
                return -float('inf')
            dist = euclidean_distance(current_rescuer.pos, (task.x, task.y)) + self.epsilon_dist
            wait_time = current_time - task.generation_time
            etc = task.remaining_workload / current_rescuer.work_rate if current_rescuer.work_rate > 0 else float('inf')
            etc += self.epsilon_etc
            urgency_factor = urgency ** self.beta
            distance_penalty = dist ** self.delta
            etc_penalty = etc ** self.epsilon
            if distance_penalty < 1e-6: distance_penalty = 1e-6
            if etc_penalty < 1e-6: etc_penalty = 1e-6
            main_score_component = urgency_factor / (distance_penalty * etc_penalty)
            score = (self.alpha * main_score_component) + (self.zeta * wait_time)
            return score
        valid_for_rescuer.sort(key=lambda t: (-hybrid_score(t, rescuer), t.id))
        return valid_for_rescuer[0]

class DQNScheduler(SchedulerBase):
    def __init__(self, simulator_ref, agent):
        super().__init__(simulator_ref)
        self.agent = agent

    def get_task_for_rl(self, rescuer, current_time, available_tasks_from_simulator):
        if rescuer.fatigue >= rescuer.fatigue_threshold:
            return None, None, None
        candidate_tasks_for_rescuer = [
            t for t in available_tasks_from_simulator
            if rescuer.can_handle_task(t) and t.can_assign(rescuer) and t.get_urgency(current_time) > 0
        ]
        all_active_schedulable_tasks = [
            t for t in self.simulator.active_tasks
            if t.get_urgency(current_time) > 0 and not t.completed
        ]
        current_s = self._get_state(rescuer, current_time, all_active_schedulable_tasks)
        action_idx = self.agent.act(current_s)
        sorted_candidate_tasks_for_action = sorted(
            candidate_tasks_for_rescuer,
            key=lambda t: (-t.get_urgency(current_time),
                           euclidean_distance(rescuer.pos, (t.x, t.y)),
                           t.id)
        )
        if 0 <= action_idx < len(sorted_candidate_tasks_for_action) and action_idx < self.agent.action_dim:
            chosen_task = sorted_candidate_tasks_for_action[action_idx]
            return chosen_task, current_s, action_idx
        return None, current_s, action_idx

    def get_task(self, rescuer, current_time, available_tasks_from_simulator):
        if rescuer.fatigue >= rescuer.fatigue_threshold:
            return None
        chosen_task, _, _ = self.get_task_for_rl(rescuer, current_time, available_tasks_from_simulator)
        return chosen_task

class QMIXScheduler(SchedulerBase):
    def __init__(self, simulator_ref, agent, num_agents_in_qmix):
        super().__init__(simulator_ref)
        self.agent = agent
        self.num_agents_in_qmix = num_agents_in_qmix

    def _get_global_state(self, current_time, all_rescuers_list, active_tasks_list):
        if not all_rescuers_list and self.num_agents_in_qmix > 0:
            expected_global_dim = self.agent.global_state_dim if hasattr(self.agent, 'global_state_dim') \
                                  else self.simulator.params.get('qmix_global_state_dim', 0)
            return np.zeros(expected_global_dim, dtype=np.float32)
        global_state_parts = []
        sorted_rescuers = sorted(all_rescuers_list, key=lambda r: r.id)
        for r_idx in range(self.num_agents_in_qmix):
            if r_idx < len(sorted_rescuers):
                rescuer = sorted_rescuers[r_idx]
                ind_obs = self._get_state(rescuer, current_time, active_tasks_list)
                global_state_parts.extend(ind_obs)
            else:
                ind_obs_dim = self.simulator.params.get('qmix_individual_obs_dim', 42)
                global_state_parts.extend([0.0] * ind_obs_dim)
        final_global_state = np.array(global_state_parts, dtype=np.float32)
        expected_global_dim = self.agent.global_state_dim
        if len(final_global_state) > expected_global_dim:
            final_global_state = final_global_state[:expected_global_dim]
        elif len(final_global_state) < expected_global_dim:
            padding = np.zeros(expected_global_dim - len(final_global_state), dtype=np.float32)
            final_global_state = np.concatenate((final_global_state, padding))
        return final_global_state

    def get_task_for_rl(self, rescuer, current_time, available_tasks_from_simulator):
        if rescuer.fatigue >= rescuer.fatigue_threshold:
            return None, None, None
        candidate_tasks_for_rescuer = [
            t for t in available_tasks_from_simulator
            if rescuer.can_handle_task(t) and t.can_assign(rescuer) and t.get_urgency(current_time) > 0
        ]
        all_active_schedulable_tasks = [
            t for t in self.simulator.active_tasks
            if t.get_urgency(current_time) > 0 and not t.completed
        ]
        current_ind_obs = self._get_state(rescuer, current_time, all_active_schedulable_tasks)
        agent_id_for_qmix = rescuer.id - 1
        if not (0 <= agent_id_for_qmix < self.num_agents_in_qmix):
            action_idx = self.agent.action_dim - 1
            return None, current_ind_obs, action_idx
        action_idx = self.agent.act_individual(current_ind_obs, agent_id_for_qmix)
        sorted_candidate_tasks_for_action = sorted(
            candidate_tasks_for_rescuer,
            key=lambda t: (-t.get_urgency(current_time),
                           euclidean_distance(rescuer.pos, (t.x, t.y)),
                           t.id)
        )
        chosen_task_if_individual = None
        if 0 <= action_idx < len(sorted_candidate_tasks_for_action) and action_idx < self.agent.action_dim:
            chosen_task_if_individual = sorted_candidate_tasks_for_action[action_idx]
        return chosen_task_if_individual, current_ind_obs, action_idx

    def get_task(self, rescuer, current_time, available_tasks_from_simulator):
        if rescuer.fatigue >= rescuer.fatigue_threshold:
            return None
        chosen_task_if_individual, _, _ = self.get_task_for_rl(
            rescuer, current_time, available_tasks_from_simulator
        )
        return chosen_task_if_individual

    def record_qmix_transition(self, all_individual_obs, joint_actions, reward,
                               all_next_individual_obs, global_state, next_global_state, done):
        if self.agent and hasattr(self.agent, 'remember'):
            self.agent.remember(all_individual_obs, joint_actions, reward,
                                all_next_individual_obs, global_state, next_global_state, done)