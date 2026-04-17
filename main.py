# main.py
"""
城市地震紧急救援仿真系统 - 主入口模块

配置参数设计依据：
================================================================================
一、任务规模配置 (DEFAULT_SCALE_CONFIG)

紧急度(urgency): 表示任务的紧急程度，单位为"生命值"或"救援紧迫度指数"
- 紧急度随时间衰减，当降至0时任务过期（代表救援窗口关闭）
- 设计依据：地震救援的"黄金72小时"原则，不同规模灾情的生存概率衰减不同

衰减率(decay_rate): 紧急度每分钟下降的速率
- 线性衰减: U(t) = U0 - λ*t，适用于小型/中型任务（如伤员救治、物资分发）
- 指数衰减: U(t) = U0 * exp(-λ*t)，适用于大型/超大型任务（如建筑倒塌、火灾蔓延）
- 设计依据：大型灾害的紧迫度衰减更复杂，呈指数级恶化

工作量(workload): 完成任务所需的工作量单位（人·分钟）
- 设计依据：参考实际救援行动的典型时长
  * 小型任务（伤员转运）: 1-2人，15-30分钟 → 15-60工作量
  * 中型任务（物资分发）: 2-4人，30-60分钟 → 60-240工作量
  * 大型任务（废墟清理）: 4-8人，60-120分钟 → 240-960工作量
  * 超大型任务（建筑救援）: 8-12人，120-240分钟 → 960-2880工作量

二、救援人员配置 (DEFAULT_RESCUER_TYPES)

速度(speed): 移动速度 (km/min)
- 设计依据：城市应急车辆实际速度
  * 步行/徒步: 0.08 km/min (5 km/h)
  * 医疗救护车: 0.5-0.8 km/min (30-50 km/h，考虑拥堵)
  * 工程车辆: 0.3-0.5 km/min (20-30 km/h)
  * 搜救车辆: 0.6-1.0 km/min (35-60 km/h)

工作效率(work_rate): 每分钟完成的工作量单位
- 设计依据：救援行动的实际产出
  * 医疗队: 处理1名伤员约需15-30分钟 → work_rate 0.8-1.2
  * 工程队: 清理废墟效率较高 → work_rate 1.0-1.5
  * 搜救队: 搜索效率中等 → work_rate 0.8-1.0

疲劳机制(fatigue_threshold, fatigue_recovery_rate, fatigue_coeff_moving, fatigue_coeff_working):
- 设计依据：人体工程学、不同救援类型的实际体力消耗特征
  * 工程队：重体力劳动，阈值最低(110)，恢复最慢(2.0/分钟)，工作系数最高(2.2)
  * 医疗队：精细+中等体力，阈值中等(170)，恢复中等(4.0/分钟)，工作系数中等(1.5)
  * 搜救队：长时间机动，耐力最好，阈值最高(250)，恢复最快(6.0/分钟)，工作系数较低(1.1)
  * 时间尺度：以分钟为单位，一个完整救援班次480分钟(8小时)
================================================================================
"""

import argparse
import math
import random
import time
import tkinter as tk
import os
import numpy
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from simulator import EmergencySimulator
from single_agent import DQNAgent
from multi_agent import QMIXAgent

# =============================================================================
# 任务规模配置
# =============================================================================
# 紧急度单位：救援紧迫度指数（无单位，用于排序和奖励计算）
# 衰减率单位：紧迫度指数/分钟（线性）或 1/分钟（指数）
# 工作量单位：人·分钟（1个救援人员工作1分钟完成的工作量）
DEFAULT_SCALE_CONFIG = {
    # 小型任务：高敏感度，但若快速响应仍可挽救
    # 过期窗口：35 ~ 130 分钟
    "small": {
        "urgency_range": (35, 65),
        "decay_range": (0.4, 0.8),       # 线性衰减，过期窗口 44~163 分钟
        "workload_range": (15, 40)       # 1人×15~40分钟
    },

    # 中型任务：窗口中等，调度不当会过期
    # 过期窗口：175 ~ 550 分钟
    "medium": {
        "urgency_range": (70, 110),
        "decay_range": (0.2, 0.4),       # 线性衰减
        "workload_range": (80, 200)      # 2-3人×40~67分钟
    },

    # 大型任务：抢救窗口有限，需多兵力协同
    # 指数衰减，过期：250 ~ 520 分钟
    "large": {
        "urgency_range": (100, 150),
        "decay_range": (0.008, 0.018),   # 指数衰减
        "workload_range": (250, 350)     # 4-5人×62~100分钟
    },

    # 超大型任务：窗口较长但工作量巨大，忽视则过期
    # 指数衰减，过期：460 ~ 920 分钟
    "extra_large": {
        "urgency_range": (130, 190),
        "decay_range": (0.004, 0.010),   # 指数衰减
        "workload_range": (400, 600)     # 7-8人×64~100分钟
    },
}

# =============================================================================
# 救援人员类型配置
# =============================================================================
# 速度单位：km/min（公里/分钟）
# 工作效率单位：工作量单位/分钟
# 疲劳阈值单位：疲劳点数
# 恢复速率单位：疲劳点数/分钟
DEFAULT_RESCUER_TYPES = [
    # 医疗救援队
    # 职责：伤员救治、医疗转运、现场急救
    # 装备：救护车、医疗设备、担架
    # 特点：速度较快，但工作效率受限于医疗程序复杂性
    # 疲劳特点：精细操作为主，体力消耗中等，恢复速度正常
    # 工作力：1单位（精细操作，单人作业为主）
    {
        'type': 'Medical Team',
        'skills': {'medical', 'search'},
        'speed': 0.8,                    # 48 km/h，救护车灾情应急速度
        'work_rate': 0.8,                # 医疗程序较复杂，效率中等
        'work_force': 1,                 # 精细医疗操作，单人工作力
        'fatigue_threshold': 160,        # 约1.5-2小时高强度医疗工作后需休息
        'fatigue_recovery_rate': 4.0,    # 休息约40-60分钟完全恢复
        'fatigue_coeff_moving': 1.0,     # 移动疲劳系数：标准
        'fatigue_coeff_working': 1.6,    # 工作疲劳系数：医疗操作强度中等
        'count_ratio': 0.35              # 占总救援力量的35%
    },

    # 工程救援队
    # 职责：废墟清理、道路疏通、建筑加固
    # 装备：挖掘机、起重机、破拆工具
    # 特点：速度较慢（重型设备），但工作效率高
    # 疲劳特点：重型机械操作，体力消耗最大，疲劳累积最快，恢复最慢
    # 工作力：2单位（重型机械，效率更高）
    {
        'type': 'Engineering Unit',
        'skills': {'engineering', 'heavy_rescue'},
        'speed': 0.5,                    # 30 km/h，重型车辆速度
        'work_rate': 1.5,                # 机械辅助，效率高
        'work_force': 2,                 # 重型机械，相当于2倍工作力
        'fatigue_threshold': 130,        # 约50-70分钟重体力劳动后需休息
        'fatigue_recovery_rate': 2.0,    # 恢复最慢，约50-80分钟完全恢复
        'fatigue_coeff_moving': 1.3,     # 移动疲劳系数：驾驶重型设备更累
        'fatigue_coeff_working': 1.9,    # 工作疲劳系数：重体力劳动，疲劳最快
        'count_ratio': 0.30              # 占总救援力量的30%
    },

    # 搜救物流队
    # 职责：人员搜救、物资配送、现场协调
    # 装备：搜救犬、无人机、通信设备、运输车辆
    # 特点：速度最快，灵活性高
    # 疲劳特点：机动性强，体力消耗相对较低，耐力最好
    # 工作力：1单位（标准效率）
    {
        'type': 'Search & Logistics',
        'skills': {'search', 'logistics'},
        'speed': 1.0,                    # 60 km/h，轻型车辆速度
        'work_rate': 1.0,                # 标准效率
        'work_force': 1,                 # 标准搜救/后勤工作力
        'fatigue_threshold': 200,        # 约2.5-3小时长时间机动后需休息
        'fatigue_recovery_rate': 6.0,    # 恢复最快，约30-40分钟完全恢复
        'fatigue_coeff_moving': 0.7,     # 移动疲劳系数：轻型车辆，疲劳较慢
        'fatigue_coeff_working': 1.3,    # 工作疲劳系数：搜救工作强度适中
        'count_ratio': 0.35              # 占总救援力量的35%
    },
]

# Global RNG instances for different parts of the simulation
task_rng = random.Random()
sim_rng = random.Random()

CHECKPOINTS_DIR = "checkpoints"

def set_all_random_seeds(seed_value):
    random.seed(seed_value)
    task_rng.seed(seed_value)
    sim_rng.seed(seed_value)
    try:
        numpy.random.seed(seed_value)
    except NameError:
        import numpy
        numpy.random.seed(seed_value)
    try:
        import torch
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
    except ImportError:
        pass
    print(f"All known RNGs (Python global, task_rng, sim_rng, numpy, torch) attempted to be seeded with: {seed_value}")


def set_main_seed(seed_choice_str, seed_value_str=None):
    actual_seed_value = None
    if seed_choice_str.lower() == "fixed":
        if seed_value_str is not None:
            try:
                actual_seed_value = int(seed_value_str)
            except ValueError:
                print(f"Warning: Invalid seed value '{seed_value_str}'. Using random seed based on time.")
        else:
            print("Warning: Seed choice 'fixed' but no seed_value provided. Using random seed based on time.")
    if actual_seed_value is None:
        actual_seed_value = int(time.time() * 1000) % (2**32)
        print(f"Using time-based random seed: {actual_seed_value}")
    set_all_random_seeds(actual_seed_value)
    return actual_seed_value


def distribute_rescuer_counts(total_num_rescuers, type_configs_templates):
    if not type_configs_templates and total_num_rescuers > 0:
        return [{'type': 'GenericFallback', 'skills': {'general'}, 'speed': 0.8, 'work_rate': 1.0,
                 'work_force': 1, 'count': total_num_rescuers, 'count_ratio':1.0,
                 'fatigue_threshold': 150, 'fatigue_recovery_rate': 4.0}]
    if total_num_rescuers == 0: return []
    type_configs = [cfg.copy() for cfg in type_configs_templates]
    total_ratio = sum(cfg.get('count_ratio', 0) for cfg in type_configs)
    if total_ratio <= 0:
        num_types = len(type_configs)
        if num_types > 0:
            for cfg in type_configs: cfg['count_ratio'] = 1.0 / num_types
            total_ratio = 1.0
        else:
            return []
    assigned_counts = []
    current_total_assigned = 0
    for i, config in enumerate(type_configs):
        ratio = config.get('count_ratio', 0)
        ideal_count = (ratio / total_ratio) * total_num_rescuers
        assigned_count = math.floor(ideal_count)
        assigned_counts.append(assigned_count)
        current_total_assigned += assigned_count
        config['count'] = assigned_count
    remainder = total_num_rescuers - current_total_assigned
    priority_for_remainder = []
    for i, config in enumerate(type_configs):
        priority_for_remainder.append((-config.get('count_ratio',0), i))
    priority_for_remainder.sort()
    for _ in range(remainder):
        target_idx_in_configs = priority_for_remainder[_ % len(priority_for_remainder)][1]
        type_configs[target_idx_in_configs]['count'] += 1
    return [c for c in type_configs if c.get('count', 0) > 0]


def cli_mode_run(params, task_rng_inst, sim_rng_inst, agent_to_pass=None):
    print(f"\n[Simulation Config for CLI Run]")
    for key, value in params.items():
        print(f"  {key}: {value}" if not isinstance(value, (dict, list)) else f"  {key}: (complex value)")
    simulator = EmergencySimulator(params, task_rng_instance=task_rng_inst, sim_rng_instance=sim_rng_inst, agent_instance=agent_to_pass)
    start_sim_real_time = time.time()
    simulator.run()
    end_sim_real_time = time.time()
    print(f"\nTotal real execution time for this run: {end_sim_real_time - start_sim_real_time:.2f} seconds.")


def main():
    """
    主函数：解析命令行参数，初始化仿真环境，运行仿真或训练

    参数设计依据：
    - 仿真时间480分钟(8小时)：代表一个救援班次的工作时长
    - 任务到达率0.08 tasks/min：地震发生后约每12分钟报告一个新灾情点
    - 救援人员40人：覆盖40x40km城市区域(1600平方公里)的基础救援力量
    - 协作系数k=1.05：多人协作效率略微提升（避免过度乐观的协作增益）
    - 子站点距离8km：覆盖城市核心区域，响应半径约15分钟车程
    """
    parser = argparse.ArgumentParser(
        description="城市地震紧急救援仿真系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本CLI运行
  python main.py --strategy nearest

  # 启动GUI
  python main.py --gui

  # DQN训练
  python main.py --strategy dqn --train_episodes 500

  # 使用固定种子（可复现结果）
  python main.py --seed_choice fixed --seed_value 42 --gui
        """
    )

    # === 仿真场景参数 ===
    parser.add_argument('--strategy', type=str, default='nearest',
                        choices=['nearest', 'urgent', 'hybrid', 'dqn', 'qmix'],
                        help='调度策略: nearest(最近优先), urgent(紧急优先), hybrid(混合评分), dqn(深度Q网络), qmix(多智能体QMIX)')
    parser.add_argument('--time', type=int, default=480,
                        help='仿真时长(分钟)，默认480分钟=8小时(一个救援班次)')
    parser.add_argument('--num_rescuers', type=int, default=40,
                        help='救援人员总数') 
    parser.add_argument('--lambda_rate', type=float, default=0.08,
                        help='任务到达率(任务/分钟)，地震场景典型值0.05-0.15')

    # === 随机种子参数 ===
    parser.add_argument('--seed_choice', type=str, choices=["random", "fixed"], default="fixed",
                        help="随机种子类型: random(基于时间), fixed(固定值，结果可复现)")
    parser.add_argument('--seed_value', type=str, default="415",
                        help="固定种子值(仅当seed_choice=fixed时使用)")

    # === 救援参数 ===
    parser.add_argument('--rescuer_speed', type=float, default=0.5,
                        help='默认救援人员速度(km/min)，约30km/h')
    parser.add_argument('--k', type=float, default=1.05,
                        help='协作系数(1.0-1.3)，k=1表示无协作增益，k>1表示多人协作效率提升')
    parser.add_argument('--sub_station_distance', type=float, default=12.0,
                        help='子站点距中心站距离(km)，覆盖响应半径')

    # === 强化学习训练参数 ===
    parser.add_argument('--train_episodes', type=int, default=0,
                        help='RL训练回合数，>0时启用训练模式')
    parser.add_argument('--save_model_prefix', type=str, default=CHECKPOINTS_DIR,
                        help='模型保存目录')
    parser.add_argument('--load_model_path', type=str, default=None,
                        help='预训练模型加载路径')
    parser.add_argument('--eval_mode', action='store_true',
                        help='评估模式(使用已训练模型)')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='模型保存间隔(回合数)')

    # === 奖励函数参数 ===
    parser.add_argument('--rew_shaping_coeff', type=float, default=0.05,
                        help='紧急度下降奖励系数')
    parser.add_argument('--rew_completion_bonus_coeff', type=float, default=0.3,
                        help='快速完成任务奖励系数')
    parser.add_argument('--rew_fast_finish_threshold', type=float, default=30.0,
                        help='快速完成阈值(分钟)')
    parser.add_argument('--rew_expire_penalty_coeff', type=float, default=2.0,
                        help='任务过期惩罚系数')
    parser.add_argument('--rew_travel_penalty_coeff', type=float, default=0.02,
                        help='行程时间惩罚系数')

    # === GUI参数 ===
    parser.add_argument('--gui', action='store_true',
                        help='启用图形界面')

    args = parser.parse_args()

    if not os.path.exists(args.save_model_prefix):
        os.makedirs(args.save_model_prefix, exist_ok=True)

    base_seed_used = set_main_seed(args.seed_choice, args.seed_value)

    final_rescuer_types_dist = distribute_rescuer_counts(args.num_rescuers, DEFAULT_RESCUER_TYPES)
    if not final_rescuer_types_dist and args.num_rescuers > 0:
        print(f"Warning: Rescuer type distribution resulted in no rescuers. Creating {args.num_rescuers} generic ones.")
        final_rescuer_types_dist = [{'type': 'GenericFallbackMain', 'skills': {'general'}, 'speed': args.rescuer_speed,
                                   'work_rate': 1.0, 'work_force': 1, 'count': args.num_rescuers,
                                   'fatigue_threshold': 150, 'fatigue_recovery_rate': 4.0}]
    elif not final_rescuer_types_dist and args.num_rescuers == 0:
        print("Notice: Number of rescuers is 0.")

    default_individual_state_dim = 42
    default_action_dim = 10

    sim_params = {
        'T': args.time,
        'num_rescuers': args.num_rescuers,
        'rescuer_speed': args.rescuer_speed,
        'k': args.k, 'strategy': args.strategy, 'lambda_rate': args.lambda_rate,
        'sub_station_distance': args.sub_station_distance,
        'scale_config': DEFAULT_SCALE_CONFIG,
        'rescuers_types': final_rescuer_types_dist,
        # 每种规模任务可分配的最大救援人数
        'max_rescuers_map': {
            "small": 2,          # 小型任务：1-2人足够
            "medium": 3,         # 中型任务：2-3人协作
            "large": 5,          # 大型任务：4-5人协作
            "extra_large": 8     # 超大型任务：7-8人协作
        },
        'fatigue_update_interval': 1.0,
        'dqn_state_dim': default_individual_state_dim,
        'dqn_action_dim': default_action_dim,
        'qmix_individual_obs_dim': default_individual_state_dim,
        'qmix_action_dim': default_action_dim,
        'qmix_global_state_dim': default_individual_state_dim * args.num_rescuers if args.num_rescuers > 0 else default_individual_state_dim,
        'reward_components': {
            'shaping_coeff': args.rew_shaping_coeff,
            'completion_bonus_coeff': args.rew_completion_bonus_coeff,
            'fast_finish_threshold': args.rew_fast_finish_threshold,
            'expire_penalty_coeff': args.rew_expire_penalty_coeff,
            'travel_penalty_coeff': args.rew_travel_penalty_coeff,
        },
        'load_model_path': args.load_model_path,
        'eval_mode': args.eval_mode,
        'replay_interval': 1 if args.train_episodes > 0 else 5,
        'rl_agent_update_interval': 10 if args.train_episodes > 0 else 100,
    }

    if args.train_episodes > 0:
        if args.strategy not in ['dqn', 'qmix']:
            print("Error: Training mode selected (--train_episodes > 0) but strategy is not 'dqn' or 'qmix'. Exiting.")
            return
        print(f"\n--- Starting RL Training: {args.strategy.upper()} for {args.train_episodes} episodes ---")
        print(f"Base Seed for training: {base_seed_used}")
        print(f"Models will be saved to '{args.save_model_prefix}' directory.")
        training_rl_agent = None
        if args.strategy == 'dqn':
            training_rl_agent = DQNAgent(
                state_dim=sim_params['dqn_state_dim'], action_dim=sim_params['dqn_action_dim'],
                rng_instance=sim_rng
            )
        elif args.strategy == 'qmix':
            if args.num_rescuers <= 0:
                print("Error: QMIX requires num_rescuers > 0 for training. Exiting.")
                return
            training_rl_agent = QMIXAgent(
                individual_state_dim=sim_params['qmix_individual_obs_dim'], action_dim=sim_params['qmix_action_dim'],
                num_agents=args.num_rescuers, global_state_dim=sim_params['qmix_global_state_dim'],
                rng_instance=sim_rng
            )
        if not training_rl_agent:
            print("Error: Could not initialize RL agent for training. Exiting.")
            return
        if args.load_model_path and os.path.exists(args.load_model_path):
            print(f"Attempting to load model from: {args.load_model_path} to continue training.")
            training_rl_agent.load_model(args.load_model_path)
        if hasattr(training_rl_agent, 'set_train_mode'): training_rl_agent.set_train_mode()
        else: training_rl_agent.model.train()

        for episode_num in range(1, args.train_episodes + 1):
            episode_specific_seed = base_seed_used + episode_num
            set_all_random_seeds(episode_specific_seed)
            episode_simulator = EmergencySimulator(
                sim_params.copy(),
                task_rng_instance=task_rng,
                sim_rng_instance=sim_rng,
                agent_instance=training_rl_agent
            )
            print(f"\nStarting Training Episode {episode_num}/{args.train_episodes} (Seed: {episode_specific_seed})...")
            episode_start_real_time = time.time()
            episode_simulator.run_for_training()
            episode_duration_real_time = time.time() - episode_start_real_time
            print(f"Episode {episode_num} finished in {episode_duration_real_time:.2f}s. Final Sim Time: {episode_simulator.current_time:.2f}")
            if episode_num % args.save_interval == 0 or episode_num == args.train_episodes:
                model_filename = f"{args.strategy}_episode_{episode_num}.pth"
                full_save_path = os.path.join(args.save_model_prefix, model_filename)
                training_rl_agent.save_model(full_save_path)
        print("\n--- RL Training Finished ---")
        final_model_filename = f"{args.strategy}_final_trained_ep{args.train_episodes}.pth"
        final_full_save_path = os.path.join(args.save_model_prefix, final_model_filename)
        training_rl_agent.save_model(final_full_save_path)
        return

    if args.gui:
        
        from visualization import EmergencySimulationApp
        root = tk.Tk()
        # Pass sim_params to GUI so it can use sub_station_distance
        app = EmergencySimulationApp(root, cli_params_for_sim=sim_params,
                                         task_rng_instance=task_rng, sim_rng_instance=sim_rng)
        root.mainloop()
    elif args.train_episodes == 0:
        print("Starting CLI-based emergency simulation...")
        cli_mode_run(sim_params, task_rng, sim_rng)
        

if __name__ == '__main__':
    main()