import os
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors  # Ensure installed
import random  # GUI RNG
import time   # Default seed
import math   # Fallback logic
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.legend_handler import HandlerBase

import sys, logging
from tkinter.scrolledtext import ScrolledText


# Assuming EmergencySimulator is correctly imported.
from simulator import EmergencySimulator

# Import configs and helpers from main.py
# This allows the GUI to use the same defaults and logic as the CLI mode.
try:
    from main import (
        DEFAULT_SCALE_CONFIG as MAIN_DEFAULT_SCALE_CONFIG,
        DEFAULT_RESCUER_TYPES as MAIN_DEFAULT_RESCUER_TYPES,
        distribute_rescuer_counts,
        # set_main_seed is not directly called by GUI; GUI manages its own RNGs
        # or uses the ones passed from main.
    )
except ImportError:
    print("Warning (visualization.py): Could not import defaults from main.py. Using hardcoded defaults for GUI.")
    # Define fallback defaults if main.py is not accessible (e.g., when running visualization.py directly)
    MAIN_DEFAULT_SCALE_CONFIG = {
        "small": {"urgency_range": (30, 60), "decay_range": (0.1, 0.2), "workload_range": (10, 25)},
        "medium": {"urgency_range": (60, 100), "decay_range": (0.2, 0.4), "workload_range": (25, 60)},
        "large": {"urgency_range": (100, 150), "decay_range": (0.03, 0.06), "workload_range": (60, 100)},
        "extra_large": {"urgency_range": (150, 250), "decay_range": (0.05, 0.08), "workload_range": (100, 180)},
    }
    MAIN_DEFAULT_RESCUER_TYPES = [
        {'type': 'Medical Team', 'skills': {'medical', 'search'}, 'speed': 0.7, 'work_rate': 0.8,
         'fatigue_threshold': 90, 'fatigue_recovery_rate': 6.0, 'count_ratio': 0.4},
        {'type': 'Engineering Unit', 'skills': {'engineering', 'heavy_rescue'}, 'speed': 0.5, 'work_rate': 1.2, 
         'fatigue_threshold': 120, 'fatigue_recovery_rate': 4.0, 'count_ratio': 0.25},
        {'type': 'Search & Logistics', 'skills': {'search', 'logistics'}, 'speed': 0.8, 'work_rate': 1.0,
         'fatigue_threshold': 100, 'fatigue_recovery_rate': 5.0, 'count_ratio': 0.35},
    ]
    def distribute_rescuer_counts(total_num_rescuers, type_configs):
        distributed_configs = []
        remaining_rescuers = total_num_rescuers
        if not type_configs:
            if total_num_rescuers > 0:
                return [{'type': 'GenericFallback', 'skills': {'general'}, 'speed': 0.8, 'work_rate': 1.0,
                'count': total_num_rescuers,
                         'fatigue_threshold': 100, 'fatigue_recovery_rate': 5.0}]
            return []
        for i, config in enumerate(type_configs):
            ratio = config.get('count_ratio', 0)
            if i == len(type_configs) - 1: count = remaining_rescuers
            else:
                count = math.floor(total_num_rescuers * ratio)
                count = min(count, remaining_rescuers)
            new_config = config.copy(); new_config['count'] = count
            distributed_configs.append(new_config)
            remaining_rescuers -= count
        if remaining_rescuers > 0 and distributed_configs:
            ratios = [c.get('count_ratio', 0) for c in type_configs]
            if any(r > 0 for r in ratios):
                 max_ratio_idx = ratios.index(max(ratios))
                 distributed_configs[max_ratio_idx]['count'] += remaining_rescuers
            elif distributed_configs: distributed_configs[0]['count'] += remaining_rescuers
        return [c for c in distributed_configs if c.get('count', 0) > 0]

MAP_EXTENT = [-20, 20, -20, 20]
MAP_IMAGE_PATH = "pic\map.jpg"
STATION_IMAGE_PATH = "pic\station.jpg"
DEFAULT_GUI_SEED_VALUE = "42"

# ============================================================================
# 颜色和样式配置
# ============================================================================
COLORS = {
    'primary': '#2C3E50',      # 深蓝灰 - 主色调
    'secondary': '#3498DB',    # 亮蓝色 - 强调色
    'success': '#27AE60',      # 绿色 - 成功/运行
    'warning': '#F39C12',      # 橙色 - 警告/暂停
    'danger': '#E74C3C',       # 红色 - 危险/错误
    'light': '#ECF0F1',        # 浅灰 - 背景
    'dark': '#2C3E50',         # 深色 - 文字
    'white': '#FFFFFF',
    'panel_bg': '#F8F9FA',     # 面板背景
}

# 任务颜色映射（与图例一致）
TASK_COLORS = {
    'waiting': '#FFD700',       # 金色 - 等待中
    'in_progress': '#FF8C00',   # 深橙色 - 进行中
    'completed': '#32CD32',     # 酸橙绿 - 已完成
    'expired': '#DC143C',       # 深红色 - 已过期
}

# 救援者颜色映射
RESCUER_COLORS = {
    'idle': '#808080',          # 灰色 - 空闲
    'moving': '#8A2BE2',        # 蓝紫色 - 移动中
    'working': '#1E90FF',       # 道奇蓝 - 工作中
    'resting': '#40E0D0',       # 绿松石 - 休息中
}


class ToolTip:
    """工具提示类，为控件添加悬停提示"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip_window:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify='left',
                        background=COLORS['dark'], foreground=COLORS['white'],
                        relief='solid', borderwidth=1, padx=8, pady=4,
                        font=('Microsoft YaHei UI', 9))
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class HandlerStationImage(HandlerBase):
    def __init__(self, station_img, zoom):
        super().__init__()
        self.station_img = station_img
        self.zoom = zoom
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        image = OffsetImage(self.station_img, zoom=self.zoom)
        ab = AnnotationBbox(image,
                            ((xdescent + width/2), (ydescent + height/2)),
                            frameon=False, transform=trans)
        return [ab]


class EmergencySimulationApp:
    def __init__(self, root, cli_params_for_sim=None, task_rng_instance=None, sim_rng_instance=None):
        self.root = root
        self.root.title("城市地震紧急救援仿真系统")
        self.simulator = None
        self.paused = False
        self.after_id = None
        self._current_sel_annotation = None
        self.station_annots = []

        self.initial_params = cli_params_for_sim if cli_params_for_sim else {}

        if task_rng_instance and sim_rng_instance:
            self.task_rng = task_rng_instance
            self.sim_rng = sim_rng_instance
            self.blink_flag = False
            print("GUI: Using RNG instances passed from main.")
        else:
            print("GUI: Creating new RNG instances for standalone run.")
            self.task_rng = random.Random()
            self.sim_rng = random.Random()
            initial_seed_for_standalone_str = self.initial_params.get('seed_value', DEFAULT_GUI_SEED_VALUE)
            try:
                initial_seed_for_standalone = int(initial_seed_for_standalone_str)
            except ValueError:
                initial_seed_for_standalone = int(DEFAULT_GUI_SEED_VALUE) # Fallback

            self.task_rng.seed(initial_seed_for_standalone)
            self.sim_rng.seed(initial_seed_for_standalone)
            random.seed(initial_seed_for_standalone)
            print(f"GUI Standalone: RNGs seeded with {initial_seed_for_standalone}")

        self.strategy_var = tk.StringVar(value=self.initial_params.get('strategy', "hybrid"))
        self.time_var = tk.IntVar(value=self.initial_params.get('T', 480))

        default_num_rescuers = 20
        if self.initial_params and 'rescuers_types' in self.initial_params and self.initial_params['rescuers_types']:
            actual_num_rescuers = sum(rt.get('count', 0) for rt in self.initial_params['rescuers_types'])
            if actual_num_rescuers > 0: default_num_rescuers = actual_num_rescuers
        elif 'num_rescuers' in self.initial_params:
             default_num_rescuers = self.initial_params.get('num_rescuers', 20)
        self.num_rescuers_var = tk.IntVar(value=default_num_rescuers)

        self.speed_var = tk.DoubleVar(value=self.initial_params.get('rescuer_speed', 0.8))
        self.k_var = tk.DoubleVar(value=self.initial_params.get('k', 1.1))
        self.lambda_var = tk.DoubleVar(value=self.initial_params.get('lambda_rate', 0.1))

        self.seed_choice_var = tk.StringVar(value=self.initial_params.get('seed_choice', "fixed"))
        self.seed_value_var = tk.StringVar(value=str(self.initial_params.get('seed_value', DEFAULT_GUI_SEED_VALUE)))

        self._configure_styles()
        self._build_ui()
        self._init_visualization_plot()

    def _configure_styles(self):
        """配置ttk样式主题"""
        self.style = ttk.Style()

        # 使用clam主题作为基础
        try:
            self.style.theme_use('clam')
        except:
            pass

        # 配置通用字体
        default_font = ('Microsoft YaHei UI', 10)
        heading_font = ('Microsoft YaHei UI', 11, 'bold')

        # 配置Frame样式
        self.style.configure('TFrame', background=COLORS['panel_bg'])
        self.style.configure('Card.TFrame', background=COLORS['white'])

        # 配置Label样式
        self.style.configure('TLabel',
                           font=default_font,
                           background=COLORS['panel_bg'],
                           foreground=COLORS['dark'])
        self.style.configure('Header.TLabel',
                           font=heading_font,
                           background=COLORS['primary'],
                           foreground=COLORS['white'])
        self.style.configure('Status.TLabel',
                           font=('Microsoft YaHei UI', 10, 'bold'),
                           background=COLORS['light'],
                           foreground=COLORS['dark'])

        # 配置LabelFrame样式
        self.style.configure('Card.TLabelframe',
                           background=COLORS['white'],
                           bordercolor=COLORS['light'])
        self.style.configure('Card.TLabelframe.Label',
                           font=('Microsoft YaHei UI', 10, 'bold'),
                           background=COLORS['white'],
                           foreground=COLORS['primary'])

        # 配置按钮样式
        self.style.configure('Action.TButton',
                           font=('Microsoft YaHei UI', 10),
                           padding=(12, 8))
        self.style.configure('Start.TButton',
                           font=('Microsoft YaHei UI', 10, 'bold'),
                           padding=(12, 8))

        # 配置Entry样式
        self.style.configure('TEntry',
                           font=default_font,
                           padding=5)

        # 配置进度条样式
        self.style.configure("Custom.Horizontal.TProgressbar",
                           troughcolor=COLORS['light'],
                           background=COLORS['success'],
                           borderwidth=0,
                           lightcolor=COLORS['success'],
                           darkcolor=COLORS['success'])

    def _build_ui(self):
        """构建用户界面"""
        # 主控制面板
        control_frame = ttk.Frame(self.root, padding=10, style='TFrame')
        control_frame.pack(side="left", fill="y", padx=5, pady=5)
        control_frame.pack_propagate(False)
        control_frame.config(width=280)

        row_idx = 0

        # ==================== 标题区域 ====================
        title_frame = ttk.Frame(control_frame, style='Card.TFrame')
        title_frame.grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        title_label = tk.Label(title_frame,
                              text="城市地震紧急救援仿真系统",
                              font=('Microsoft YaHei UI', 12, 'bold'),
                              bg=COLORS['primary'],
                              fg=COLORS['white'],
                              padx=15, pady=10)
        title_label.pack(fill='x')
        row_idx += 1

        # ==================== 参数设置区域 ====================
        params_frame = ttk.LabelFrame(control_frame, text=" 参数设置 ",
                                      padding=10, style='Card.TLabelframe')
        params_frame.grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=5)
        params_frame.columnconfigure(1, weight=1)

        param_row = 0

        # 辅助函数：添加参数行
        def add_param_row(label_text, widget_creator, tooltip_text=None):
            nonlocal param_row
            label = ttk.Label(params_frame, text=label_text)
            label.grid(row=param_row, column=0, sticky="w", pady=4, padx=(0, 10))
            widget = widget_creator(params_frame)
            widget.grid(row=param_row, column=1, sticky="ew", pady=4)
            if tooltip_text:
                ToolTip(widget, tooltip_text)
                ToolTip(label, tooltip_text)
            param_row += 1
            return widget

        # 调度策略
        strategy_combo = add_param_row(
            "调度策略:",
            lambda p: ttk.Combobox(p, textvariable=self.strategy_var,
                                   values=["nearest", "urgent", "hybrid", "dqn", "qmix"],
                                   state="readonly", width=15),
        )

        # 仿真时长
        add_param_row(
            "仿真时长(分钟):",
            lambda p: ttk.Entry(p, textvariable=self.time_var, width=18),
            "设置仿真的总运行时间(分钟)"
        )

        # 救援人员数量
        add_param_row(
            "救援人员数量:",
            lambda p: ttk.Entry(p, textvariable=self.num_rescuers_var, width=18),
            "设置参与救援的人员总数"
        )

        # 移动速度
        add_param_row(
            "移动速度(km/min):",
            lambda p: ttk.Entry(p, textvariable=self.speed_var, width=18),
            "救援人员的基准移动速度"
        )

        # 协作系数
        add_param_row(
            "协作系数(k):",
            lambda p: ttk.Entry(p, textvariable=self.k_var, width=18),
            "多人协作效率提升系数\nk>1表示协作增益"
        )

        # 任务到达率
        add_param_row(
            "任务到达率(λ):",
            lambda p: ttk.Entry(p, textvariable=self.lambda_var, width=18),
            "泊松分布的任务到达率\n(任务/分钟)"
        )

        # 随机种子
        seed_frame = ttk.Frame(params_frame)
        ttk.Label(seed_frame, text="随机种子:").pack(side="left")
        seed_combo = ttk.Combobox(seed_frame, textvariable=self.seed_choice_var,
                                  values=["random", "fixed"], state="readonly", width=8)
        seed_combo.pack(side="left", padx=5)
        seed_entry = ttk.Entry(seed_frame, textvariable=self.seed_value_var, width=8)
        seed_entry.pack(side="left")
        seed_frame.grid(row=param_row, column=0, columnspan=2, sticky="w", pady=4)
        ToolTip(seed_combo, "random: 随机种子\nfixed: 固定种子(可复现)")
        param_row += 1

        row_idx += 1

        # ==================== 操作区域 ====================
        action_frame = ttk.LabelFrame(control_frame, text=" 仿真控制 ",
                                      padding=10, style='Card.TLabelframe')
        action_frame.grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=10)

        # 按钮样式
        btn_style = 'Action.TButton'

        # 开始按钮
        start_btn = ttk.Button(action_frame, text="▶ 开始仿真",
                               command=self.start_simulation, style='Start.TButton')
        start_btn.pack(fill='x', pady=3)
        ToolTip(start_btn, "启动救援仿真(快捷键: Enter)")
        self.root.bind('<Return>', lambda e: self.start_simulation())

        # 暂停/继续按钮
        pause_btn = ttk.Button(action_frame, text="⏸ 暂停 / 继续",
                               command=self.toggle_pause, style=btn_style)
        pause_btn.pack(fill='x', pady=3)
        ToolTip(pause_btn, "暂停或继续仿真运行(快捷键: Space)")
        self.root.bind('<space>', lambda e: self.toggle_pause())

        # 重置按钮
        reset_btn = ttk.Button(action_frame, text="⟳ 重置仿真",
                               command=self.reset_simulation, style=btn_style)
        reset_btn.pack(fill='x', pady=3)
        ToolTip(reset_btn, "重置仿真状态，恢复初始设置(快捷键: R)")
        self.root.bind('<r>', lambda e: self.reset_simulation())
        self.root.bind('<R>', lambda e: self.reset_simulation())

        row_idx += 1

        # ==================== 状态显示区域 ====================
        status_frame = ttk.LabelFrame(control_frame, text=" 运行状态 ",
                                      padding=10, style='Card.TLabelframe')
        status_frame.grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=5)
        status_frame.columnconfigure(0, weight=1)

        # 进度条
        self.progress = ttk.Progressbar(status_frame, orient="horizontal",
                                        mode="determinate",
                                        style="Custom.Horizontal.TProgressbar")
        self.progress.pack(fill='x', pady=(0, 8))

        # 状态标签
        self.status_label = tk.Label(status_frame, text="就绪 - 请配置参数并开始仿真",
                                      font=('Microsoft YaHei UI', 10),
                                      bg=COLORS['light'], fg=COLORS['dark'],
                                      padx=10, pady=5)
        self.status_label.pack(fill='x')

        row_idx += 1

        # ==================== 日志输出区域 ====================
        log_frame = ttk.LabelFrame(control_frame, text=" 运行日志 ",
                                   padding=5, style='Card.TLabelframe')
        log_frame.grid(row=row_idx, column=0, columnspan=2, sticky="nsew", pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = ScrolledText(log_frame, height=8, state='disabled', wrap='word',
                                     font=('Consolas', 9), bg='#1E1E1E', fg='#D4D4D4',
                                     insertbackground='white')
        self.log_text.pack(fill='both', expand=True)

        control_frame.rowconfigure(row_idx, weight=1)
        control_frame.columnconfigure(0, weight=1)

        # 配置日志标签颜色
        self.log_text.tag_configure('info', foreground='#4EC9B0')
        self.log_text.tag_configure('warning', foreground='#FFD700')
        self.log_text.tag_configure('error', foreground='#F44747')
        self.log_text.tag_configure('success', foreground='#6A9955')

        # 重定向stdout/stderr
        class TextRedirector:
            def __init__(self, widget, tag='info'):
                self.widget = widget
                self.tag = tag
            def write(self, txt):
                self.widget.configure(state='normal')
                self.widget.insert('end', txt, self.tag)
                self.widget.see('end')
                self.widget.configure(state='disabled')
            def flush(self):
                pass

        sys.stdout = TextRedirector(self.log_text, 'info')
        sys.stderr = TextRedirector(self.log_text, 'error')
        text_handler = logging.StreamHandler(TextRedirector(self.log_text, 'info'))
        text_handler.setLevel(logging.INFO)
        text_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(text_handler)

    def _init_visualization_plot(self):
        """初始化可视化绘图区域"""
        vis_frame = ttk.Frame(self.root)
        vis_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        # 设置matplotlib支持中文显示
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        self.fig, self.ax = plt.subplots(figsize=(9, 9), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        self._draw_map_background()
        self.ax.set_title("城市救援仿真地图", fontsize=14, fontweight='bold', pad=15)
        self.ax.set_xlabel("X 坐标 (公里)", fontsize=11)
        self.ax.set_ylabel("Y 坐标 (公里)", fontsize=11)
        self._create_plot_legend_elements()
        self.fig.tight_layout(rect=[0, 0.05, 1, 1])

        # Save the initial axes position to restore on reset
        # This prevents cumulative position drift from repeated tight_layout calls
        self._initial_ax_position = self.ax.get_position()

        self.task_scatter_plot = None
        self.rescuer_scatter_plot = None
        self.station_scatter_plot = None
        self._init_hover_annotations_for_plot()

    def _draw_map_background(self):
        self.ax.clear()
        try:
            if os.path.exists(MAP_IMAGE_PATH):
                img = mpimg.imread(MAP_IMAGE_PATH)
                self.ax.imshow(img, extent=MAP_EXTENT,
                               aspect='auto', alpha=0.8)
        except:
            pass
        self.ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
        self.ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
        self.ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.5)
        self.ax.set_aspect('equal', adjustable='box')

        # Station coordinates
        coords = [(0.0, 0.0)]
        dist = self.initial_params.get('sub_station_distance', 6.0)
        coords += [(dist, 0), (-dist, 0), (0, dist), (0, -dist)]

        # Remove old station images
        # Note: AnnotationBbox.remove() may not be implemented in some matplotlib versions
        # Use try-except and alternative removal methods for robustness
        for ab in getattr(self, 'station_annots', []):
            try:
                # Try standard remove method
                ab.remove()
            except NotImplementedError:
                # Fallback: remove from axes artists list
                try:
                    if ab in self.ax.artists:
                        self.ax.artists.remove(ab)
                except Exception:
                    pass
            except Exception:
                pass
        self.station_annots.clear()

        # Add new station images with reduced zoom (1/32)
        try:
            station_img = mpimg.imread(STATION_IMAGE_PATH)
            for x, y in coords:
                icon = OffsetImage(station_img, zoom=1/100)
                ab = AnnotationBbox(icon, (x, y), frameon=False, zorder=5)
                self.ax.add_artist(ab)
                self.station_annots.append(ab)
        except:
            sx = [c[0] for c in coords]
            sy = [c[1] for c in coords]
            self.ax.scatter(sx, sy, c='navy', marker='H', s=25,
                            edgecolors='white', linewidth=1.0, zorder=5)

    def _create_plot_legend_elements(self):
        """创建图例元素，使用统一的颜色配置"""
        # 任务图例
        task_legend = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=TASK_COLORS['waiting'],
                   markersize=10, label='任务: 等待中',
                   markeredgecolor='#333333', markeredgewidth=0.5),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=TASK_COLORS['in_progress'],
                   markersize=10, label='任务: 进行中',
                   markeredgecolor='#333333', markeredgewidth=0.5),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=TASK_COLORS['completed'],
                   markersize=10, label='任务: 已完成',
                   markeredgecolor='#333333', markeredgewidth=0.5),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=TASK_COLORS['expired'],
                   markersize=10, label='任务: 已过期',
                   markeredgecolor='#333333', markeredgewidth=0.5),
        ]

        # 救援者图例
        rescuer_legend = [
            Line2D([0], [0], marker='^', color='w',
                   markerfacecolor=RESCUER_COLORS['idle'],
                   markersize=10, label='救援者: 空闲',
                   markeredgecolor='#333333', markeredgewidth=0.5),
            Line2D([0], [0], marker='^', color='w',
                   markerfacecolor=RESCUER_COLORS['moving'],
                   markersize=10, label='救援者: 移动中',
                   markeredgecolor='#333333', markeredgewidth=0.5),
            Line2D([0], [0], marker='^', color='w',
                   markerfacecolor=RESCUER_COLORS['working'],
                   markersize=10, label='救援者: 工作中',
                   markeredgecolor='#333333', markeredgewidth=0.5),
            Line2D([0], [0], marker='^', color='w',
                   markerfacecolor=RESCUER_COLORS['resting'],
                   markersize=10, label='救援者: 休息中',
                   markeredgecolor='#333333', markeredgewidth=0.5),
        ]

        legend_handles = task_legend + rescuer_legend
        labels = [h.get_label() for h in legend_handles]

        # 创建图例
        legend = self.ax.legend(
            handles=legend_handles,
            labels=labels,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.08),
            ncol=4,
            fontsize=9,
            frameon=True,
            facecolor='white',
            edgecolor='#CCCCCC',
            borderpad=1.0,
            columnspacing=2.0,
            handletextpad=0.5
        )

        # 设置图例标题
        # Note: legend_handles is only available in matplotlib 3.7+
        if hasattr(legend, 'legend_handles'):
            for handle in legend.legend_handles:
                if isinstance(handle, AnnotationBbox):
                    handle.set_transform(legend.axes.transAxes)
        
    def _init_hover_annotations_for_plot(self): # Renamed
        if hasattr(self, 'mpl_cursor_instance') and self.mpl_cursor_instance: # Renamed
            try: self.mpl_cursor_instance.remove()
            except Exception: pass
            
        artists_to_hover_on = [artist for artist in [self.task_scatter_plot, self.rescuer_scatter_plot, self.station_scatter_plot] if artist is not None]
        if not artists_to_hover_on: return

        self.mpl_cursor_instance = mplcursors.cursor(artists_to_hover_on, hover=mplcursors.HoverMode.Transient)
        self.mpl_cursor_instance.connect("add", self._hover_callback_for_plot_elements) # Renamed callback

    def _hover_callback_for_plot_elements(self, sel): # Renamed
        text_to_display = "No specific info"
        idx = sel.index
        
        if self.simulator and self.task_scatter_plot and sel.artist == self.task_scatter_plot:
            if idx < len(self.simulator.tasks):
                task = self.simulator.tasks[idx] 
                urg = task.get_urgency(self.simulator.current_time)
                rem_time_str = "N/A"
                if task.expire_time != float('inf') and not task.completed:
                     rem_time_val = max(task.expire_time - self.simulator.current_time, 0.0)
                     rem_time_str = f"{rem_time_val:.1f} min"
                elif task.completed: rem_time_str = "Completed"
                elif task.expire_time == float('inf'): rem_time_str = "No Expiry"
                status_str = "Waiting/Active" # Default
                if task in self.simulator.completed_tasks_stats: status_str = "Completed"
                elif task in self.simulator.expired_tasks_stats: status_str = "Expired"
                elif task.in_progress: status_str = "In Progress"
                text_to_display = (f"Task ID: {task.id}\nScale: {task.scale.capitalize()}\nStatus: {status_str}\n"
                                   f"Urgency: {urg:.1f} (Initial: {task.initial_urgency:.1f})\n"
                                   f"Time Left: {rem_time_str}\nWorkload: {task.remaining_workload:.1f}/{task.workload:.1f}\n"
                                   f"Skill Req: {task.required_skill if task.required_skill else 'Any'}\n"
                                   f"Assigned: {len(task.assigned_rescuers_list)}/{task.max_rescuers}")
        elif self.simulator and self.rescuer_scatter_plot and sel.artist == self.rescuer_scatter_plot:
            if idx < len(self.simulator.rescuers):
                rescuer = self.simulator.rescuers[idx] 
                task_id_display = rescuer.task.id if rescuer.task else "None"
                rescuer_type_name = getattr(rescuer, 'type_name', 'Generic') # Use type_name if set
                text_to_display = (f"Rescuer ID: {rescuer.id} (Type: {rescuer_type_name})\nStatus: {rescuer.status.capitalize()}\n"
                                   f"Task ID: {task_id_display}\nSpeed: {rescuer.speed:.1f}\nWork Rate: {rescuer.work_rate:.1f}\n"
                                   f"Skills: {', '.join(list(rescuer.skills)) if rescuer.skills else 'General'}\n"
                                   f"Fatigue: {rescuer.fatigue:.1f}/{rescuer.fatigue_threshold:.1f}")
        elif self.station_scatter_plot and sel.artist == self.station_scatter_plot:
             text_to_display = f"Station\nPos: ({sel.target[0]:.1f}, {sel.target[1]:.1f})"

        sel.annotation.set_text(text_to_display)
        sel.annotation.get_bbox_patch().set(alpha=0.92, fc="lightyellow") # Using my last style
        sel.annotation.arrow_patch.set(arrowstyle="->", fc="gray", alpha=0.7)


    def start_simulation(self):
        try:
            sim_time_val = self.time_var.get()
            num_rescuers_total_val = self.num_rescuers_var.get()
            if num_rescuers_total_val <= 0:
                messagebox.showerror("Input Error", "Number of rescuers must be > 0.")
                return
            if sim_time_val <=0:
                messagebox.showerror("Input Error", "Simulation time must be > 0.")
                return
        except tk.TclError:
            messagebox.showerror("Input Error", "Ensure inputs are valid numbers.")
            return

        seed_choice_val = self.seed_choice_var.get()
        seed_val_str_val = self.seed_value_var.get()
        
        current_seed_to_use = None
        if seed_choice_val == "fixed":
            try:
                current_seed_to_use = int(seed_val_str_val)
            except ValueError:
                messagebox.showerror("Seed Error", "Invalid fixed seed value. Must be an integer.")
                return
        else: # Random seed
            current_seed_to_use = int(time.time())
        
        # Seed the RNG instances that this GUI app holds/manages
        self.task_rng.seed(current_seed_to_use)
        self.sim_rng.seed(current_seed_to_use)
        random.seed(current_seed_to_use) 
        print(f"GUI: Simulation started with seed: {current_seed_to_use} (Choice: {seed_choice_val})")

        final_gui_rescuer_types = distribute_rescuer_counts(num_rescuers_total_val, MAIN_DEFAULT_RESCUER_TYPES)
        if not final_gui_rescuer_types and num_rescuers_total_val > 0 :
             final_gui_rescuer_types = [{'type': 'Generic', 'skills': {'general'}, 
                                    'speed': self.speed_var.get(), 'work_rate': 1.0,
                                    'count': num_rescuers_total_val,
                                    'fatigue_threshold': 100, 'fatigue_recovery_rate': 5.0}]

        gui_sim_params = {
            'strategy': self.strategy_var.get(), 'T': sim_time_val,
            'rescuer_speed': self.speed_var.get(), 'k': self.k_var.get(),
            'lambda_rate': self.lambda_var.get(),
            'scale_config': MAIN_DEFAULT_SCALE_CONFIG, 
            'rescuers_types': final_gui_rescuer_types, 
            'max_rescuers_map': self.initial_params.get('max_rescuers_map', 
                                 {"small": 2, "medium": 3, "large": 5, "extra_large": 7}),
            'fatigue_update_interval': self.initial_params.get('fatigue_update_interval', 1.0),
            'dqn_state_dim': self.initial_params.get('dqn_state_dim', 42),
            'dqn_action_dim': self.initial_params.get('dqn_action_dim', 10),
            'qmix_state_dim': self.initial_params.get('qmix_state_dim', 42), 
            'qmix_individual_obs_dim': self.initial_params.get('qmix_individual_obs_dim', 42),
            'qmix_action_dim': self.initial_params.get('qmix_action_dim', 10),
        }
        
        self.simulator = EmergencySimulator(gui_sim_params, 
                                            task_rng_instance=self.task_rng, 
                                            sim_rng_instance=self.sim_rng)
        self.paused = False
        self.progress['value'] = 0
        self.status_label.config(text="仿真运行中...", bg=COLORS['success'], fg=COLORS['white'])

        if self.after_id: self.root.after_cancel(self.after_id)
        self._simulation_tick_loop()

    def _simulation_tick_loop(self):
        if not self.simulator or self.paused:
            if self.paused: self.status_label.config(text="仿真已暂停", bg=COLORS['warning'], fg=COLORS['white'])
            return


        is_sim_still_running = self.simulator.step()
        self.update_plot_contents()
        self._update_gui_progressbar()

        if is_sim_still_running and self.simulator.current_time < self.simulator.end_time :
            self.after_id = self.root.after(1, self._simulation_tick_loop)
        else:
            self.status_label.config(text="仿真已完成", bg=COLORS['primary'], fg=COLORS['white'])
            self.after_id = None
            if self.simulator:
                messagebox.showinfo("仿真完成",
                    f"仿真于 T={self.simulator.current_time:.2f} 分钟时完成。\n请查看控制台获取详细结果。")
                self.simulator.print_results()

    def toggle_pause(self):
        if not self.simulator:
            messagebox.showwarning("提示", "请先启动仿真。")
            return
        self.paused = not self.paused
        if self.paused:
            self.status_label.config(text="仿真已暂停", bg=COLORS['warning'], fg=COLORS['white'])
            if self.after_id: self.root.after_cancel(self.after_id); self.after_id = None
        else:
            self.status_label.config(text="仿真运行中...", bg=COLORS['success'], fg=COLORS['white'])
            if not self.after_id : self._simulation_tick_loop()

    def reset_simulation(self):
        if self.after_id: self.root.after_cancel(self.after_id); self.after_id = None
        self.simulator = None
        self.paused = False
        self.progress['value'] = 0
        self.status_label.config(text="就绪 - 请配置参数并开始仿真", bg=COLORS['light'], fg=COLORS['dark'])
        if self.task_scatter_plot: self.task_scatter_plot.remove(); self.task_scatter_plot = None
        if self.rescuer_scatter_plot: self.rescuer_scatter_plot.remove(); self.rescuer_scatter_plot = None
        self._draw_map_background()
        self.ax.set_title("城市救援仿真地图", fontsize=14, fontweight='bold', pad=15)
        self.ax.set_xlabel("X 坐标 (公里)", fontsize=11)
        self.ax.set_ylabel("Y 坐标 (公里)", fontsize=11)
        self._create_plot_legend_elements()

        # Restore the initial axes position to prevent position drift
        if hasattr(self, '_initial_ax_position'):
            self.ax.set_position(self._initial_ax_position)

        self.canvas.draw_idle()
        self._init_hover_annotations_for_plot()

    def update_plot_contents(self):
        """更新地图上的任务和救援者显示"""
        if not self.simulator: return
        self.blink_flag = not self.blink_flag

        if self.task_scatter_plot: self.task_scatter_plot.remove()
        if self.rescuer_scatter_plot: self.rescuer_scatter_plot.remove()

        # 更新标题显示当前时间
        self.ax.set_title(
            f"仿真时间: {self.simulator.current_time:.1f} / {self.simulator.end_time:.1f} 分钟",
            fontsize=14, fontweight='bold', pad=15
        )

        # 绘制任务点
        task_x_coords, task_y_coords, task_colors_list, task_sizes_list = [], [], [], []
        for task in self.simulator.tasks:
            task_x_coords.append(task.x)
            task_y_coords.append(task.y)
            task_scale_cfg_details = self.simulator.scale_config.get(
                task.scale, {"workload_range": (10,10)}
            )
            base_size = 30 + task_scale_cfg_details['workload_range'][0] / 2.0

            # 等待中的任务闪烁效果
            if task in self.simulator.active_tasks and not task.in_progress:
                size = base_size if self.blink_flag else 0
            else:
                size = base_size
            task_sizes_list.append(size)

            # 使用统一的颜色配置
            if task in self.simulator.completed_tasks_stats:
                task_colors_list.append(TASK_COLORS['completed'])
            elif task in self.simulator.expired_tasks_stats:
                task_colors_list.append(TASK_COLORS['expired'])
            elif task.in_progress:
                task_colors_list.append(TASK_COLORS['in_progress'])
            elif task in self.simulator.active_tasks:
                task_colors_list.append(TASK_COLORS['waiting'])
            else:
                task_colors_list.append('#333333')

        self.task_scatter_plot = self.ax.scatter(
            task_x_coords, task_y_coords, c=task_colors_list, s=task_sizes_list,
            edgecolors='#333333', alpha=0.85, linewidths=1.0, zorder=10
        )

        # 绘制救援者
        resc_x_coords, resc_y_coords, resc_colors_list = [], [], []
        for r_obj in self.simulator.rescuers:
            if r_obj.status == 'moving' and hasattr(r_obj, 'move_start_time'):
                # 插值计算移动中的位置
                t0 = r_obj.move_start_time
                t1 = r_obj.move_arrival_time if hasattr(r_obj, 'move_arrival_time') else t0
                if t1 > t0:
                    frac = (self.simulator.current_time - t0) / (t1 - t0)
                    frac = max(0.0, min(1.0, frac))
                else:
                    frac = 1.0
                sx, sy = r_obj.move_start_pos
                dx, dy = r_obj.move_dest_pos
                x = sx + frac * (dx - sx)
                y = sy + frac * (dy - sy)
                resc_colors_list.append(RESCUER_COLORS['moving'])
            else:
                x, y = r_obj.pos
                if r_obj.status == 'idle':
                    resc_colors_list.append(RESCUER_COLORS['idle'])
                elif r_obj.status == 'working':
                    resc_colors_list.append(RESCUER_COLORS['working'])
                elif r_obj.status == 'resting':
                    resc_colors_list.append(RESCUER_COLORS['resting'])
                else:
                    resc_colors_list.append('#FF69B4')
            resc_x_coords.append(x)
            resc_y_coords.append(y)

        self.rescuer_scatter_plot = self.ax.scatter(
            resc_x_coords, resc_y_coords, c=resc_colors_list, marker='^', s=80,
            edgecolors='#333333', linewidths=1.0, zorder=12
        )

        self.canvas.draw_idle()
        self._init_hover_annotations_for_plot()

    def _update_gui_progressbar(self):
        """更新进度条显示"""
        if self.simulator and self.simulator.end_time > 0:
            elapsed_time_val = self.simulator.current_time
            total_time_val = self.simulator.end_time
            self.progress['value'] = (elapsed_time_val / total_time_val) * 100 if total_time_val > 0 else 0
        else: self.progress['value'] = 0

if __name__ == '__main__':
    print("Running EmergencySimulationApp directly for UI testing...")
    test_root = tk.Tk()
    app_instance = EmergencySimulationApp(test_root) 
    test_root.mainloop()
