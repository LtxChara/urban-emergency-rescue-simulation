# Urban Earthquake Emergency Rescue Simulation System

# 城市地震紧急救援仿真系统

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.5.1-EE4C2C)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A discrete-event simulation framework for urban earthquake emergency response, featuring multiple scheduling strategies (rule-based and RL-based) and an optional real-time Tkinter GUI.

这是一个基于离散事件驱动的城市地震紧急救援仿真框架，支持多种调度策略（基于规则与基于强化学习）以及可选的实时 Tkinter 图形界面。

![地震救援仿真页面](pic/homepage.png)


---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Scheduling Strategies](#scheduling-strategies)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [License](#license)

---

## Features

- **Discrete-Event Simulation Core**Event-driven engine with a priority queue (`TASK_ARRIVAL`, `RESCUER_ARRIVAL`, `TASK_COMPLETED`, `FATIGUE_RECOVERY`).
- **Multi-Type Rescuer Model**Three archetypes with distinct speed, work rate, skills, and fatigue mechanics:

  - **Medical Team**: fast response, moderate work rate, fatigue threshold 160.
  - **Engineering Unit**: slow but powerful (work rate 1.5), heavy machinery, fatigue threshold 130.
  - **Search & Logistics**: highest speed (1.0 km/min), best endurance, fatigue threshold 200.
- **Dynamic Task Model**Tasks vary in scale (`small`, `medium`, `large`, `extra_large`) with:

  - Time-decaying urgency (linear or exponential).
  - Role-based collaboration requirements.
  - Workload measured in person-minutes.
- **Scheduling Strategies**

  - `nearest`: Greedy distance minimization.
  - `urgent`: Greedy urgency-per-time-budget maximization.
  - `hybrid`: Weighted score combining distance, urgency, and workload.
  - `dqn`: Single-agent Deep Q-Network.
  - `qmix`: Multi-agent QMIX.
- **Real-Time GUI**Tkinter + Matplotlib visualization showing city map, stations, rescuers, and tasks.
- **Reproducibility**
  Seeded RNGs for Python, NumPy, and PyTorch.

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/urban-emergency-rescue-simulation.git
cd urban-emergency-rescue-simulation
```

2. **Create a virtual environment (recommended)**

```bash
conda create -n urbanSim python=3.10
conda activate urbanSim
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

> **Note on PyTorch**: The `requirements.txt` pins core packages. If you need a specific CUDA version or are on a different platform, install PyTorch manually following the [official guide](https://pytorch.org/get-started/locally/).

---

## Quick Start

### 1. CLI Simulation

Run a basic simulation with the `nearest` strategy:

```bash
python main.py --strategy nearest
```

Run with a custom arrival rate and more rescuers:

```bash
python main.py --strategy urgent --num_rescuers 50 --lambda_rate 0.12 --time 480
```

### 2. Launch GUI

```bash
python main.py --gui
```

You can also set a fixed seed for reproducible visualizations:

```bash
python main.py --gui --seed_choice fixed --seed_value 42
```

### 3. Train RL Agents

Train a DQN agent for 500 episodes:

```bash
python main.py --strategy dqn --train_episodes 500
```

Train a QMIX multi-agent:

```bash
python main.py --strategy qmix --train_episodes 500
```

Trained models are saved to the `checkpoints/` directory (e.g. `checkpoints/dqn_final_trained_ep500.pth`).

### 4. Evaluate a Trained Model

```bash
python main.py --strategy dqn --load_model_path checkpoints/dqn_final_trained_ep500.pth --eval_mode
```

### Common CLI Arguments

| Argument                | Default         | Description                                                                |
| ----------------------- | --------------- | -------------------------------------------------------------------------- |
| `--strategy`          | `nearest`     | Scheduling strategy:`nearest`, `urgent`, `hybrid`, `dqn`, `qmix` |
| `--time`              | `480`         | Simulation duration in minutes (default = 8 hours)                         |
| `--num_rescuers`      | `40`          | Total number of rescuers                                                   |
| `--lambda_rate`       | `0.08`        | Poisson task arrival rate (tasks/minute)                                   |
| `--seed_choice`       | `fixed`       | `random` (time-based) or `fixed`                                       |
| `--seed_value`        | `415`         | Fixed random seed                                                          |
| `--gui`               | `False`       | Enable the Tkinter GUI                                                     |
| `--train_episodes`    | `0`           | RL training episodes (>0 enables training)                                 |
| `--save_model_prefix` | `checkpoints` | Directory to save trained models                                           |
| `--load_model_path`   | `None`        | Path to a pre-trained model for evaluation or continued training           |

---

## Scheduling Strategies

| Strategy    | Type       | Description                                                                                        |
| ----------- | ---------- | -------------------------------------------------------------------------------------------------- |
| `nearest` | Rule-based | Assigns the closest available task to an idle rescuer.                                             |
| `urgent`  | Rule-based | Ranks tasks by**urgency at arrival / (travel + solo remaining work + 0.1)**. Non-preemptive. |
| `hybrid`  | Rule-based | Combines distance, urgency, and workload into a single weighted score.                             |
| `dqn`     | RL-based   | Single-agent Deep Q-Network that learns a task-selection policy from state vectors.                |
| `qmix`    | RL-based   | Multi-agent QMIX that coordinates all rescuers via a centralized mixer and decentralized policies. |

---

## Project Structure

```
.
├── main.py              # Entry point: CLI args, seeding, training loops
├── simulator.py         # Discrete-event simulation core (EmergencySimulator)
├── tasks.py             # EmergencyTask model and urgency decay logic
├── rescuers.py          # Rescuer model, skills, and fatigue mechanics
├── schedulers.py        # Scheduler base + nearest / urgent / hybrid / DQN / QMIX wrappers
├── single_agent.py      # DQNAgent implementation
├── multi_agent.py       # QMIXAgent implementation
├── visualization.py     # Tkinter + Matplotlib real-time GUI
├── requirements.txt     # Python dependencies
├── pic/                 # Static assets (map.jpg, station.jpg)
│   ├── map.jpg
│   └── station.jpg
├── checkpoints/         # Saved RL models (generated at runtime)
├── CLAUDE.md            # Internal codebase documentation for contributors
└── README.md            # This file
```

---

## Customization

You can tune the simulation by editing `main.py`:

- **`DEFAULT_SCALE_CONFIG`**: Adjust urgency ranges, decay rates, and workload ranges for each task scale.
- **`DEFAULT_RESCUER_TYPES`**: Modify rescuer archetypes (speed, work rate, fatigue thresholds, count ratios).
- **`max_rescuers_map`**: Change the hard cap on how many rescuers can be assigned to each task scale.

For RL research, reward coefficients and network hyperparameters can be adjusted via CLI arguments or directly in `single_agent.py` / `multi_agent.py`.

---

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to use, modify, and share it for educational and research purposes.

---

## Acknowledgements

This project was originally developed as a Data Structure course project. It is now open-sourced in the hope that future students and programming enthusiasts can learn from it or build upon it.

If you find this project helpful, please consider giving it a star.
