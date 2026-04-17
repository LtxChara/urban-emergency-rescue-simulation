# tasks.py
import math

class EmergencyTask:
    def __init__(
        self,
        task_id,
        generation_time,
        x,
        y,
        initial_urgency,
        decay_rate,
        scale,
        workload,
        max_rescuers,
        decay_mode=None, # Default to None, will be set based on scale
        required_skill=None,
        role_requirements=None  # 角色槽位需求，例如 {'medical': 1, 'engineering': 2}
    ):
        self.id = task_id
        self.generation_time = generation_time
        self.x = x
        self.y = y
        self.initial_urgency = initial_urgency
        self.current_urgency = initial_urgency # Initially, current_urgency is initial_urgency
        self.decay_rate = decay_rate
        self.scale = scale
        self.workload = workload # Total units of work required to complete the task
        self.remaining_workload = workload # Workload remaining, decreases as rescuers work
        self.max_rescuers = max_rescuers # Maximum number of rescuers that can be assigned simultaneously

        # Determine decay_mode if not explicitly provided
        if decay_mode is None:
            self.decay_mode = 'exponential' if self.scale in ['large', 'extra_large'] else 'linear'
        else:
            self.decay_mode = decay_mode.lower() # Ensure consistent casing

        self.required_skill = required_skill # Specific skill needed for this task (e.g., 'medical', 'engineering')

        # role_requirements: 角色槽位需求模型
        # 例如 {'medical': 1, 'engineering': 2} 表示需要1名医疗和2名工程救援人员
        if role_requirements is None:
            self.role_requirements = {}
        else:
            self.role_requirements = dict(role_requirements)
        
        self.in_progress = False # True if at least one rescuer is assigned and working/moving to it
        self.completed = False # True if the task's workload has been fully addressed
        self.assigned_rescuers_list = [] # List of Rescuer objects currently assigned to this task
        
        # start_time: Time when the first rescuer actually starts working on the task (after arrival)
        # This will be set by the simulator.
        self.start_time = None 
        self.finish_time = None # Time when the task is completed

        # Calculate theoretical expire_time based on decay parameters
        # This is when urgency would drop to a minimal threshold (e.g., 1.0 for exp, 0 for linear)
        if self.decay_rate > 0:
            if self.decay_mode == 'linear':
                # Time = Urgency / Rate
                self.expire_time = self.generation_time + (self.initial_urgency / self.decay_rate if self.decay_rate > 0 else float('inf'))
            elif self.decay_mode == 'exponential':
                urgency_threshold = 1.0 # Define a small positive threshold for practical expiration
                if self.initial_urgency > urgency_threshold and self.decay_rate > 0:
                    # U(t) = U0 * exp(-lambda*t) => t = -ln(U(t)/U0) / lambda = ln(U0/U(t)) / lambda
                    time_to_expire = (math.log(self.initial_urgency / urgency_threshold)) / self.decay_rate
                    self.expire_time = self.generation_time + time_to_expire
                else: # If initial urgency is already below threshold or no decay
                    self.expire_time = self.generation_time # Effectively expires immediately or never if U0=0
            else: # Unknown decay mode
                self.expire_time = float('inf')
        else: # No decay
            self.expire_time = float('inf')

    def update_urgency(self, current_time):
        # Updates the task's current_urgency based on the current_simulation_time.
        if self.completed or current_time < self.generation_time:
            # If completed or time is before generation, urgency doesn't change from its last state or initial state
            return self.current_urgency

        if self.current_urgency == 0.0: # Already fully decayed or marked as zero
            return 0.0
            
        # Ensure urgency doesn't go up if current_time somehow jumped back (defensive)
        # current_time = max(current_time, self.generation_time) # Not strictly needed if simulator time is monotonic

        # If current_time is at or beyond the calculated expire_time, set urgency to 0
        if current_time >= self.expire_time :
             self.current_urgency = 0.0
             return self.current_urgency

        # Calculate time elapsed since generation for decay calculation
        dt = current_time - self.generation_time
        
        if self.decay_mode == 'linear':
            self.current_urgency = max(0.0, self.initial_urgency - self.decay_rate * dt)
        elif self.decay_mode == 'exponential':
            self.current_urgency = max(0.0, self.initial_urgency * math.exp(-self.decay_rate * dt))
        # Add other decay modes here if necessary
        
        return self.current_urgency

    def get_urgency(self, current_time):
        # Returns the current urgency after updating it.
        return self.update_urgency(current_time)

    def get_effective_rescuers(self):
        """
        返回实际对任务工作量有贡献的救援人员列表。
        只有状态为 'working' 且填补未满足角色槽位的救援人员才有效。
        """
        effective = []
        role_counts = {}
        for r in self.assigned_rescuers_list:
            if getattr(r, 'status', None) != 'working':
                continue
            role = getattr(r, 'primary_role', None)
            if role is None or role not in self.role_requirements:
                continue
            required = self.role_requirements[role]
            current = role_counts.get(role, 0)
            if current < required:
                effective.append(r)
                role_counts[role] = current + 1
        return effective

    def needs_role(self, role):
        """
        检查任务是否仍需要指定角色的救援人员（基于当前 working 状态的人员计数）。
        """
        if role not in self.role_requirements:
            return False
        current_working_count = sum(
            1 for r in self.assigned_rescuers_list
            if getattr(r, 'status', None) == 'working' and getattr(r, 'primary_role', None) == role
        )
        return current_working_count < self.role_requirements[role]

    def can_assign(self, rescuer):
        # Checks if a new rescuer can be assigned to this task.
        # Considers completion status, max rescuer limits, and role requirements.
        # Note: Current urgency check is typically done by the simulator/scheduler before calling this.
        if self.completed:
            return False
        if len(self.assigned_rescuers_list) >= self.max_rescuers:
            return False
        # Role-slot check
        role = getattr(rescuer, 'primary_role', None)
        if not self.role_requirements:
            # If no specific role requirements, allow any rescuer (fallback)
            return True
        if role is None or role not in self.role_requirements:
            return False
        current_count = sum(
            1 for r in self.assigned_rescuers_list
            if getattr(r, 'primary_role', None) == role
        )
        if current_count >= self.role_requirements[role]:
            return False
        return True

    def assign_rescuer(self, rescuer, current_time_of_assignment):
        # Assigns a rescuer to this task.
        # current_time_of_assignment is when the decision to assign is made.
        # Actual work start time (self.start_time) is handled by the simulator upon rescuer arrival.
        if not self.can_assign(rescuer): # Double check, e.g. max rescuers and role slots
            return False
        if rescuer in self.assigned_rescuers_list: # Prevent double assignment
            return False # Or log a warning

        self.assigned_rescuers_list.append(rescuer)
        # self.in_progress and self.start_time are now set by the simulator
        # when the first rescuer arrives and starts working.
        return True
    
    def release_rescuer(self, rescuer):
        # Removes a rescuer from the task's assigned list.
        if rescuer in self.assigned_rescuers_list:
            self.assigned_rescuers_list.remove(rescuer)

        # If no rescuers are left and task is not completed, its 'in_progress' status
        # might change. This is handled by the simulator based on its logic
        # (e.g., if all assigned rescuers are released because task expired or completed).
        # If a rescuer just finishes their part or rests, task might remain in_progress if others are there.
        # Note: We don't set in_progress=False here because rescuers might still be en route (status='moving').
        # The simulator should check if there are any rescuers working OR moving to this task
        # before deciding that the task is no longer in progress.
        pass

    def complete_work(self, work_amount_done):
        # Applies an amount of work to the task and updates remaining_workload.
        # Returns True if the task is now fully completed, False otherwise.
        self.remaining_workload -= work_amount_done
        if self.remaining_workload <= 0:
            self.remaining_workload = 0
            self.completed = True # Mark as completed internally
            return True 
        return False

    def __lt__(self, other):
        # Comparison for priority queue (heapq).
        # Default to generation_time for tasks if no other event-specific time.
        # SimulationEvent class will handle more complex event prioritization.
        return self.generation_time < other.generation_time

    def __repr__(self):
        # String representation for logging and debugging.
        return (
            f"Task(id={self.id:03d} scale={self.scale}, pos=({self.x:.1f},{self.y:.1f}), "
            f"urg={self.current_urgency:.1f}/{self.initial_urgency:.1f} (gen@T={self.generation_time:.1f}), "
            f"decay={self.decay_mode[:3]}.@{self.decay_rate:.2f}, "
            f"exp@T={self.expire_time:.1f}, load={self.remaining_workload:.1f}/{self.workload:.1f}, "
            f"skill='{self.required_skill}', roles={self.role_requirements}, "
            f"assigned={len(self.assigned_rescuers_list)}/{self.max_rescuers}, "
            f"status={'Cmpl' if self.completed else 'InProg' if self.in_progress else 'Wait'}, "
            f"start_T={self.start_time:.1f if self.start_time else 'N/A'}, fin_T={self.finish_time:.1f if self.finish_time else 'N/A'})"
        )


class SimulationEvent:
    # Represents an event in the simulation's priority queue.
    def __init__(self, time, etype, task=None, rescuer=None, data=None):
        self.time = time # Event occurrence time
        self.etype = etype # Event type (e.g., 'task_arrive_signal', 'rescuer_arrive')
        self.task = task # Associated EmergencyTask, if any
        self.rescuer = rescuer # Associated Rescuer, if any
        self.data = data # Optional dictionary for additional event data

    def __lt__(self, other):
        # Comparison for heapq. Prioritizes by time, then by event type for determinism.
        if self.time != other.time:
            return self.time < other.time
        
        # Define a priority for event types if they occur at the exact same time.
        # Lower numbers have higher priority.
        etype_priority = {
            'task_expire': 0,           # Process expirations first
            'task_work_update': 1,      # Then work updates (which might lead to completion)
            # 'task_complete': 1,       # (Covered by task_work_update leading to completion)
            'rescuer_arrive': 2,        # Then rescuer arrivals
            'fatigue_recovery_update': 3,# Then fatigue recovery
            'task_arrive_signal': 4,    # Finally, new task arrivals
        }
        # Assign a default high number for unknown event types to give them lower priority
        return etype_priority.get(self.etype, 99) < etype_priority.get(other.etype, 99)

    def __repr__(self):
        task_id_repr = self.task.id if self.task and hasattr(self.task, 'id') else 'N/A'
        resc_id_repr = self.rescuer.id if self.rescuer and hasattr(self.rescuer, 'id') else 'N/A'
        return (
            f"Event(T={self.time:.2f}, type='{self.etype}', "
            f"task_id={task_id_repr}, resc_id={resc_id_repr}, data={self.data})"
        )