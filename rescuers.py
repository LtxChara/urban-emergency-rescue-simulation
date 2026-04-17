# rescuers.py
import math

def distance(x1, y1, x2, y2):
    # 计算两点之间的欧氏距离
    return math.hypot(x2 - x1, y2 - y1)

class Rescuer:
    def __init__(self, rescuer_id, speed=0.8, work_rate=1.0, pos=(0.0, 0.0),
                 skills=None, work_force=1, primary_role=None,
                 fatigue_threshold=100.0, fatigue_recovery_rate=5.0,
                 fatigue_coeff_moving=1.0, fatigue_coeff_working=1.5):
        self.id = rescuer_id
        self.speed = speed # 移动速度 (km/minute)
        self.work_rate = work_rate # 工作效率 (units of workload processed per minute)
        self.pos = pos # 当前位置 (x, y)
        self.status = 'idle'  # 状态: 'idle', 'moving', 'working', 'resting'
        self.task = None # 当前分配的任务对象
        self.arrival_time = 0.0 # 预计到达任务地点的时间

        self.skills = set(skills) if skills else set() # 掌握的技能集合

        # primary_role: 该救援人员的主要角色，用于角色槽位匹配
        # 例如 'medical', 'engineering', 'search', 'logistics', 'heavy_rescue'
        self.primary_role = primary_role

        # work_force: 该救援人员贡献的工作力单位（向后兼容，当前版本中主要用 primary_role 匹配）
        self.work_force = work_force

        self.fatigue = 0.0 # 当前疲劳值
        self.fatigue_threshold = fatigue_threshold # 疲劳阈值，超过后需要休息
        self.fatigue_recovery_rate = fatigue_recovery_rate # 每分钟恢复的疲劳值

        # 个体疲劳系数：不同救援类型有不同的疲劳累积速率
        self.fatigue_coeff_moving = fatigue_coeff_moving    # 移动时的疲劳累积系数
        self.fatigue_coeff_working = fatigue_coeff_working  # 工作时的疲劳累积系数

    def can_handle_task(self, task):
        # 检查该救援人员是否能处理指定任务（基于技能和角色槽位）
        # 技能检查：如果任务需要特定技能，救援者必须拥有该技能
        if task.required_skill and task.required_skill not in self.skills:
            return False
        # 角色槽位检查：如果任务有角色需求，救援者必须拥有匹配的主要角色
        if getattr(task, 'role_requirements', None):
            if self.primary_role not in task.role_requirements:
                return False
        return True

    def assign(self, task, current_time):
        # 分配任务给该救援人员
        # task: 要分配的 EmergencyTask 对象
        # current_time: 当前仿真时间
        # 返回: 到达任务所需的行程时间，如果无法分配则返回 None

        if task is None: # 防御性检查
            return None
        # can_handle_task 应该由调度器在选择此救援者前调用，但可以再次检查
        if not self.can_handle_task(task):
            # This case should ideally be caught by the scheduler before calling assign
            print(f"Warning: Rescuer {self.id} assigned task {task.id} it cannot handle. Skills: {self.skills}, Task req skill: {task.required_skill}.")
            return None

        # 注意：因为 work_force 是贡献量而非消耗量
        # 一个救援人员一次只能执行一个任务，通过 task 属性管理

        dist_to_task = distance(self.pos[0], self.pos[1], task.x, task.y)
        travel_time = dist_to_task / self.speed if self.speed > 0 else float('inf')

        self.task = task
        self.status = 'moving'
        self.arrival_time = current_time + travel_time # 记录预计到达时间

        # 注意：疲劳因移动产生的累积应在实际移动发生后（即到达时）更新
        return travel_time

    def update_fatigue(self, amount):
        # 更新疲劳值
        self.fatigue += amount


    def recover_fatigue(self, duration):
        # 在空闲/休息状态下恢复疲劳
        # duration: 此次恢复过程持续的时间 (e.g., 仿真中的一个时间步长)
        # 返回: True 如果疲劳完全恢复到阈值以下并变为空闲，False 如果仍在恢复中

        # idle 和 resting 状态都可以恢复疲劳
        # idle状态：空闲等待，相当于轻度休息，可以恢复疲劳
        # resting状态：强制休息，恢复速率更高
        if self.status not in ['idle', 'resting']:
            return False  # 只有在空闲或休息状态才能恢复

        # idle 和 resting 状态均按 1.5 倍速率恢复疲劳
        # 设计依据：空闲状态视为"轻度休息"，恢复速率与主动休息相同
        recovery_multiplier = 1.5

        recovered_amount = self.fatigue_recovery_rate * duration * recovery_multiplier
        self.fatigue = max(0.0, self.fatigue - recovered_amount)

        # 对于resting状态，检查是否可以退出休息
        if self.status == 'resting' and self.fatigue < self.fatigue_threshold:
            self.status = 'idle'
            return True  # 完全恢复，可以重新被调度

        # 对于idle状态，疲劳降低但不改变状态（保持空闲等待任务）
        return True  # 恢复成功

    def complete_task_actions(self, task_workload_contribution_for_fatigue=0):
        # 当救援人员完成（或被从）一个任务释放时的动作
        # task_workload_contribution_for_fatigue: 该救援人员在此任务中贡献的工作量（用于计算疲劳）
        # 注意：疲劳现在主要在 simulator._handle_task_work_update 中基于工作时长累积。
        # 此处的参数可以用于在任务结束时进行一次性的额外疲劳调整（如果需要）。
        # 目前，我们假设工作疲劳已在工作时段内累积。

        if task_workload_contribution_for_fatigue > 0: # 如果有额外的疲劳需要基于此参数计算
             self.update_fatigue(task_workload_contribution_for_fatigue)

        # 任务完成后，更新位置到任务地点
        if self.task: # 确保 self.task 存在
             self.pos = (self.task.x, self.task.y) # 更新位置到任务地点

        self.task = None # 清除当前任务

        # 根据当前疲劳值决定状态是休息还是空闲
        if self.fatigue >= self.fatigue_threshold:
            self.status = 'resting'
            # print(f"Rescuer {self.id} completed task actions, fatigue {self.fatigue:.2f}, now RESTING.")
        else:
            self.status = 'idle'
            # print(f"Rescuer {self.id} completed task actions, fatigue {self.fatigue:.2f}, now IDLE.")


    def __repr__(self):
        # 救援人员对象的字符串表示形式，用于调试和日志记录
        task_id_repr = self.task.id if self.task and hasattr(self.task, 'id') else 'None'
        return (f"Rescuer(id={self.id}, pos=({self.pos[0]:.1f},{self.pos[1]:.1f}), "
                f"status='{self.status}', task_id={task_id_repr}, "
                f"spd={self.speed:.1f}, work_rt={self.work_rate:.1f}, skills={list(self.skills)}, "
                f"role={self.primary_role}, work_force={self.work_force}, "
                f"fatigue={self.fatigue:.1f}/{self.fatigue_threshold:.1f})")