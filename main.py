"""
游标卡尺识别 - 主程序 (GUI) v4.0
流水线架构：预处理 → ROI → 矫正 → 分离 → 主尺识别 → 游标识别 → 合并
支持查看每一步处理的中间输出图像
"""

import os
import sys
import ctypes


def _bootstrap_tk_paths():
    """Make tkinter use the bundled Tcl/Tk scripts on Windows."""
    base_dir = os.path.dirname(sys.executable)
    tcl_dll = os.path.join(base_dir, "DLLs", "tcl86t.dll")
    if os.path.exists(tcl_dll):
        try:
            tcl = ctypes.CDLL(tcl_dll)
            tcl.Tcl_FindExecutable.argtypes = [ctypes.c_wchar_p]
            tcl.Tcl_FindExecutable.restype = None
            tcl.Tcl_FindExecutable(sys.executable)
        except Exception:
            pass


_bootstrap_tk_paths()

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading

from caliper import CaliperPipeline, CaliperResult


class CaliperApp:
    """游标卡尺识别应用"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("游标卡尺读数识别 v4.0")
        self.root.geometry("1200x750")
        self.root.minsize(950, 620)

        # 主题色
        self.bg_color = "#1e1e2e"
        self.fg_color = "#cdd6f4"
        self.accent_color = "#89b4fa"
        self.success_color = "#a6e3a1"
        self.warn_color = "#fab387"
        self.card_color = "#313244"

        self.root.configure(bg=self.bg_color)

        # 流水线实例
        self.pipeline = CaliperPipeline()

        # 当前图像 & 结果
        self.current_image: np.ndarray = None
        self.current_result: CaliperResult = None
        self._is_processing = False
        self._latest_file_path = ""
        self._auto_follow_progress = True

        # 动态标签页
        self.tab_widgets = {}
        self.tab_keys = []
        self.current_tab = ""

        # ---- 样式 ----
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Microsoft YaHei', 10), padding=8)
        style.configure('TLabel', font=('Microsoft YaHei', 10),
                        background=self.bg_color, foreground=self.fg_color)

        self._build_ui()

    def _build_ui(self):
        """构建界面"""
        # 顶部标题栏
        title_frame = tk.Frame(self.root, bg=self.card_color, height=55)
        title_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        title_frame.pack_propagate(False)

        tk.Label(
            title_frame, text="📏 游标卡尺读数识别",
            font=('Microsoft YaHei', 18, 'bold'),
            bg=self.card_color, fg=self.accent_color
        ).pack(side=tk.LEFT, padx=20, pady=10)

        # 版本信息
        tk.Label(
            title_frame, text="v4.0 | 流水线架构 · 可查看每步中间结果",
            font=('Microsoft YaHei', 9),
            bg=self.card_color, fg="#6c7086"
        ).pack(side=tk.RIGHT, padx=20, pady=10)

        # 主内容区
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 左：控制面板
        self._build_control_panel(main_frame)

        # 右：图像显示区
        self._build_image_panel(main_frame)

        # 底部状态栏
        status_frame = tk.Frame(self.root, bg=self.card_color, height=30)
        status_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(
            status_frame, text="就绪 — 请打开一张游标卡尺图像",
            font=('Microsoft YaHei', 9), bg=self.card_color, fg="#6c7086"
        )
        self.status_label.pack(side=tk.LEFT, padx=15, pady=5)

    def _build_control_panel(self, parent: tk.Frame):
        """构建左侧控制面板"""
        panel = tk.Frame(parent, bg=self.card_color, width=270)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        panel.pack_propagate(False)

        inner = tk.Frame(panel, bg=self.card_color)
        inner.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # 打开图像按钮
        self.btn_open = tk.Button(
            inner, text="📂 打开图像文件", font=('Microsoft YaHei', 11, 'bold'),
            bg=self.accent_color, fg="#1e1e2e",
            activebackground="#74c7ec", activeforeground="#1e1e2e",
            relief=tk.FLAT, cursor="hand2", padx=15, pady=10,
            command=self._open_image
        )
        self.btn_open.pack(fill=tk.X, pady=(0, 10))

        # 结果展示区
        tk.Label(
            inner, text="📋 识别结果", font=('Microsoft YaHei', 13, 'bold'),
            bg=self.card_color, fg=self.fg_color
        ).pack(anchor=tk.W, pady=(10, 10))

        result_card = tk.Frame(inner, bg=self.bg_color, relief=tk.FLAT,
                               highlightthickness=1, highlightbackground="#45475a")
        result_card.pack(fill=tk.X, pady=(0, 10))
        result_inner = tk.Frame(result_card, bg=self.bg_color)
        result_inner.pack(fill=tk.X, padx=12, pady=12)

        # 精度
        tk.Label(result_inner, text="支持精度", font=('Microsoft YaHei', 9),
                 bg=self.bg_color, fg="#6c7086").pack(anchor=tk.W)
        self.lbl_precision = tk.Label(
            result_inner, text="-- mm", font=('Consolas', 16, 'bold'),
            bg=self.bg_color, fg=self.warn_color
        )
        self.lbl_precision.pack(anchor=tk.W, pady=(2, 8))

        # 主尺读数
        tk.Label(result_inner, text="主尺读数（整数部分）", font=('Microsoft YaHei', 9),
                 bg=self.bg_color, fg="#6c7086").pack(anchor=tk.W)
        self.lbl_main = tk.Label(
            result_inner, text="-- mm", font=('Consolas', 14),
            bg=self.bg_color, fg=self.fg_color
        )
        self.lbl_main.pack(anchor=tk.W, pady=(2, 8))

        # 游标读数
        tk.Label(result_inner, text="游标读数（小数部分）", font=('Microsoft YaHei', 9),
                 bg=self.bg_color, fg="#6c7086").pack(anchor=tk.W)
        self.lbl_vernier = tk.Label(
            result_inner, text="-- mm", font=('Consolas', 14),
            bg=self.bg_color, fg=self.fg_color
        )
        self.lbl_vernier.pack(anchor=tk.W, pady=(2, 8))

        ttk.Separator(result_inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)

        # 总读数
        tk.Label(result_inner, text="总读数", font=('Microsoft YaHei', 9),
                 bg=self.bg_color, fg="#6c7086").pack(anchor=tk.W, pady=(6, 0))
        self.lbl_total = tk.Label(
            result_inner, text="-- mm",
            font=('Consolas', 28, 'bold'),
            bg=self.bg_color, fg=self.success_color
        )
        self.lbl_total.pack(anchor=tk.W, pady=(0, 5))

        # 置信度
        tk.Label(result_inner, text="置信度", font=('Microsoft YaHei', 9),
                 bg=self.bg_color, fg="#6c7086").pack(anchor=tk.W)
        self.confidence_bar = ttk.Progressbar(
            result_inner, length=200, mode='determinate'
        )
        self.confidence_bar.pack(fill=tk.X, pady=(2, 2))
        self.lbl_confidence = tk.Label(
            result_inner, text="0%", font=('Microsoft YaHei', 9),
            bg=self.bg_color, fg="#6c7086"
        )
        self.lbl_confidence.pack(anchor=tk.W)

        # OCR 引擎状态
        self.lbl_ocr_engine = tk.Label(
            result_inner, text="OCR: 检测中…", font=('Microsoft YaHei', 8),
            bg=self.bg_color, fg="#585b70"
        )
        self.lbl_ocr_engine.pack(anchor=tk.W, pady=(6, 0))
        self.lbl_ocr_result = tk.Label(
            result_inner, text="OCR result: --", font=('Consolas', 9),
            bg=self.bg_color, fg="#6c7086"
        )
        self.lbl_ocr_result.pack(anchor=tk.W, pady=(2, 0))

        ttk.Separator(inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # 使用说明
        tk.Label(
            inner, text="💡 使用说明", font=('Microsoft YaHei', 11, 'bold'),
            bg=self.card_color, fg=self.fg_color
        ).pack(anchor=tk.W, pady=(5, 5))

        tips = [
            "1. 点击「打开图像文件」选择图片",
            "2. 确保游标卡尺刻度清晰可见",
            "3. 尽量垂直拍摄，光线均匀",
            "4. 支持 jpg / png / bmp 格式",
            "5. 固定支持 0.02 mm 精度",
            "6. 右侧标签页可查看每步中间结果",
            "7. 安装 EasyOCR 可启用数字识别",
        ]
        for tip in tips:
            tk.Label(
                inner, text=tip, font=('Microsoft YaHei', 9),
                bg=self.card_color, fg="#a6adc8", justify=tk.LEFT
            ).pack(anchor=tk.W, pady=1)

        # 保存结果按钮
        self.btn_save = tk.Button(
            inner, text="💾 保存标注图像", font=('Microsoft YaHei', 10),
            bg="#45475a", fg=self.fg_color,
            activebackground="#585b70", activeforeground=self.fg_color,
            relief=tk.FLAT, cursor="hand2", padx=10, pady=8,
            command=self._save_result, state=tk.DISABLED
        )
        self.btn_save.pack(fill=tk.X, pady=(15, 0))

    def _build_image_panel(self, parent: tk.Frame):
        """构建右侧图像显示区域（支持缩放 + 滚动条）"""
        panel = tk.Frame(parent, bg=self.card_color)
        panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 标签页容器
        self.tab_container = tk.Frame(panel, bg=self.card_color)
        self.tab_container.pack(fill=tk.X, padx=10, pady=(10, 0))

        # 默认标签页占位
        placeholder = tk.Label(
            self.tab_container, text="（识别后将显示各步骤标签页）",
            font=('Microsoft YaHei', 9),
            bg=self.card_color, fg="#585b70"
        )
        placeholder.pack(side=tk.LEFT, padx=5)

        # ── 缩放控制条 ──
        zoom_bar = tk.Frame(panel, bg=self.card_color)
        zoom_bar.pack(fill=tk.X, padx=10, pady=(2, 0))

        self.btn_zoom_out = tk.Button(
            zoom_bar, text="−", font=('Consolas', 12, 'bold'),
            bg="#45475a", fg=self.fg_color, relief=tk.FLAT,
            width=3, cursor="hand2", command=self._zoom_out
        )
        self.btn_zoom_out.pack(side=tk.LEFT, padx=(0, 2))

        self.btn_zoom_fit = tk.Button(
            zoom_bar, text="⊡ 适应窗口", font=('Microsoft YaHei', 8),
            bg="#45475a", fg=self.fg_color, relief=tk.FLAT,
            padx=6, cursor="hand2", command=self._zoom_fit
        )
        self.btn_zoom_fit.pack(side=tk.LEFT, padx=2)

        self.btn_zoom_100 = tk.Button(
            zoom_bar, text="1:1", font=('Microsoft YaHei', 8),
            bg="#45475a", fg=self.fg_color, relief=tk.FLAT,
            padx=6, cursor="hand2", command=self._zoom_100
        )
        self.btn_zoom_100.pack(side=tk.LEFT, padx=2)

        self.btn_zoom_in = tk.Button(
            zoom_bar, text="+", font=('Consolas', 12, 'bold'),
            bg="#45475a", fg=self.fg_color, relief=tk.FLAT,
            width=3, cursor="hand2", command=self._zoom_in
        )
        self.btn_zoom_in.pack(side=tk.LEFT, padx=(2, 0))

        self.lbl_zoom = tk.Label(
            zoom_bar, text="适应窗口", font=('Microsoft YaHei', 8),
            bg=self.card_color, fg="#6c7086", width=12, anchor=tk.W
        )
        self.lbl_zoom.pack(side=tk.LEFT, padx=8)

        # 提示
        self.lbl_zoom_hint = tk.Label(
            zoom_bar, text="Ctrl+滚轮缩放  |  拖拽平移  |  双击切换 1:1/适应",
            font=('Microsoft YaHei', 7), bg=self.card_color, fg="#45475a"
        )
        self.lbl_zoom_hint.pack(side=tk.RIGHT, padx=4)

        # ── 图像显示区（Canvas + Scrollbar）──
        img_frame = tk.Frame(panel, bg=self.bg_color)
        img_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(2, 10))

        self.img_canvas = tk.Canvas(
            img_frame, bg=self.bg_color, highlightthickness=0,
            cursor="hand1"  # 抓手光标提示可拖拽
        )

        # 滚动条
        self.v_scrollbar = tk.Scrollbar(img_frame, orient=tk.VERTICAL,
                                         command=self.img_canvas.yview)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scrollbar = tk.Scrollbar(img_frame, orient=tk.HORIZONTAL,
                                         command=self.img_canvas.xview)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.img_canvas.configure(
            yscrollcommand=self._on_scroll_y,
            xscrollcommand=self._on_scroll_x
        )
        self.img_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Canvas 初始占位文字
        self.canvas_text_id = self.img_canvas.create_text(
            370, 260, text="暂无图像\n\n请点击左侧「打开图像文件」选择图片",
            font=('Microsoft YaHei', 12), fill="#45475a"
        )
        self.canvas_img_id = None
        self.canvas_rect_id = None

        # ── 缩放状态 ──
        self._zoom_level: float = 1.0      # 当前缩放比例
        self._zoom_fit_mode: bool = True   # True = 自动适应窗口
        self._show_original: np.ndarray = None    # 原始图（cv2 BGR）
        self._pan_start_x: int = 0
        self._pan_start_y: int = 0
        self._pan_canvas_x: float = 0.0
        self._pan_canvas_y: float = 0.0

        # ── 鼠标事件 ──
        self.img_canvas.bind("<Configure>", self._on_canvas_resize)
        self.img_canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.img_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.img_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.img_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.img_canvas.bind("<Double-Button-1>", self._on_double_click)
        # Linux / Mac 滚轮
        self.img_canvas.bind("<Button-4>", lambda e: self._on_mousewheel_alt(1))
        self.img_canvas.bind("<Button-5>", lambda e: self._on_mousewheel_alt(-1))

    def _build_dynamic_tabs(self, debug_images: dict, reset: bool = False):
        """根据 debug_images 更新标签页；reset=True 时重建整组标签。"""
        if reset:
            for w in self.tab_container.winfo_children():
                w.destroy()
            self.tab_widgets.clear()
            self.tab_keys.clear()

        all_keys = ['原图'] if self.current_image is not None else []
        all_keys.extend(sorted(debug_images.keys()))
        all_keys.append('最终标注')

        tab_style = {
            'font': ('Microsoft YaHei', 8),
            'padx': 8, 'pady': 3,
            'cursor': 'hand2',
            'bg': '#45475a',
            'fg': self.fg_color,
        }

        for key in all_keys:
            if key in self.tab_widgets:
                continue
            # 缩短显示名
            display = key.replace('_', ' ').replace('a', '').replace('b', '').replace('c', '')
            if len(display) > 10:
                display = display[:9] + '…'

            lbl = tk.Label(self.tab_container, text=display, **tab_style)
            lbl.pack(side=tk.LEFT, padx=(0, 2))
            lbl.bind("<Button-1>", lambda e, k=key: self._switch_tab(k))
            self.tab_widgets[key] = lbl
            self.tab_keys.append(key)

        if reset and self.tab_keys:
            self._switch_tab(self.tab_keys[0])

    def _refresh_dynamic_tabs(self, select_tab: str = None):
        """Rebuild tabs while keeping the current selection when possible."""
        previous_tab = self.current_tab
        self._build_dynamic_tabs(
            self.current_result.debug_images if self.current_result else {},
            reset=False,
        )
        target = select_tab or previous_tab
        if target in self.tab_widgets:
            self._switch_tab(target)

    def _make_pending_result(self) -> CaliperResult:
        """Create a temporary result object while the pipeline is still running."""
        return CaliperResult(
            main_scale=0.0,
            vernier_scale=0.0,
            total=0.0,
            precision=0.0,
            confidence=0.0,
            image_annotated=None,
            debug_images={},
        )

    def _on_pipeline_progress(self, step_key: str, image: np.ndarray, status: str):
        """Receive a finished pipeline step from the worker thread."""
        if self.current_result is None:
            self.current_result = self._make_pending_result()
        self.current_result.debug_images[step_key] = image
        should_follow = self._auto_follow_progress or self.current_tab in ("", "原图")
        if step_key not in self.tab_widgets:
            self._build_dynamic_tabs(self.current_result.debug_images, reset=False)
        if should_follow:
            self._switch_tab(step_key)
        elif self.current_tab == step_key:
            self._display_image(image)
        if should_follow:
            self._auto_follow_progress = True
        self.status_label.config(text=f"处理中: {status}")
        self.root.update_idletasks()

    def _switch_tab(self, tab: str):
        """切换标签页"""
        self.current_tab = tab

        for key, w in self.tab_widgets.items():
            if key == tab:
                w.config(bg=self.accent_color, fg="#1e1e2e",
                         font=('Microsoft YaHei', 8, 'bold'))
            else:
                w.config(bg="#45475a", fg=self.fg_color,
                         font=('Microsoft YaHei', 8))

        # 显示对应图像
        if tab == '原图':
            self._display_image(self.current_image)
        elif tab == '最终标注' and self.current_result:
            self._display_image(self.current_result.image_annotated)
        elif self.current_result and tab in self.current_result.debug_images:
            img = self.current_result.debug_images[tab]
            if img is not None:
                self._display_image(img)
            else:
                self._display_image(self.current_image)
        else:
            self._display_image(self.current_image)

    def _open_image(self):
        """打开图像文件"""
        if self._is_processing:
            return

        file_path = filedialog.askopenfilename(
            title="选择游标卡尺图像",
            filetypes=[
                ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("所有文件", "*.*")
            ]
        )
        if not file_path:
            return

        self._is_processing = True
        self._latest_file_path = file_path
        self._auto_follow_progress = True
        self.btn_open.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED, bg="#45475a", fg=self.fg_color)
        self.status_label.config(text=f"⏳ 正在识别: {os.path.basename(file_path)} ...")
        self.root.update_idletasks()

        def worker():
            try:
                img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("无法读取图像文件，请确认文件格式正确。")

                def progress(step_key, image, status):
                    self.root.after(
                        0,
                        lambda k=step_key, im=image, s=status:
                            self._on_pipeline_progress(k, im, s)
                    )

                self.root.after(0, lambda: self._on_image_loaded_for_progress(img))
                result = self.pipeline.run(img, progress_callback=progress)
                self.root.after(0, lambda: self._on_image_processed(img, result, file_path))
            except Exception as e:
                self.root.after(0, lambda: self._on_image_process_error(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_image_loaded_for_progress(self, img: np.ndarray):
        self.current_image = img
        self.current_result = self._make_pending_result()
        self._build_dynamic_tabs(self.current_result.debug_images, reset=True)
        self._switch_tab('原图')

    def _on_image_processed(self, img: np.ndarray, result: CaliperResult, file_path: str):
        self.current_image = img
        self.current_result = result

        self._show_result(result)
        self._build_dynamic_tabs(result.debug_images, reset=False)

        self.btn_save.config(state=tk.NORMAL, bg=self.accent_color, fg="#1e1e2e")
        self.btn_open.config(state=tk.NORMAL)
        self._is_processing = False

        self.status_label.config(
            text=f"✅ 识别完成 — {os.path.basename(file_path)} | "
                 f"精度: {result.precision:.2f}mm | "
                 f"结果: {result.total:.3f}mm"
        )

    def _on_image_process_error(self, error: Exception):
        import traceback
        traceback.print_exc()
        messagebox.showerror("识别错误", f"识别过程中出现错误:\n{str(error)}")
        self.btn_open.config(state=tk.NORMAL)
        self._is_processing = False
        self.status_label.config(text="❌ 识别失败")

    def _show_result(self, result: CaliperResult):
        """显示识别结果"""
        self.lbl_precision.config(text=f"{result.precision:.2f} mm")
        self.lbl_main.config(text=f"{result.main_scale:.1f} mm")
        self.lbl_vernier.config(text=f"{result.vernier_scale:.2f} mm")
        self.lbl_total.config(text=f"{result.total:.2f} mm")

        pct = int(result.confidence * 100)
        self.confidence_bar['value'] = pct
        self.lbl_confidence.config(text=f"{pct}%")

        if result.confidence > 0.6:
            self.lbl_total.config(fg=self.success_color)
        elif result.confidence > 0.3:
            self.lbl_total.config(fg=self.warn_color)
        else:
            self.lbl_total.config(fg="#f38ba8")

        # ── OCR 引擎状态 ──
        from caliper.ocr import DigitReader
        from caliper.main_scale import get_ocr_reader
        reader = get_ocr_reader()
        status = reader.engine_status() if hasattr(reader, 'engine_status') else reader.engine_name()
        if 'fallback' in status.lower() or '无' in status:
            self.lbl_ocr_engine.config(text=f"OCR: {status}", fg="#f38ba8")
        elif 'tesseract' in status.lower():
            self.lbl_ocr_engine.config(text=f"OCR: {status}", fg=self.success_color)
        elif 'easyocr' in status.lower():
            self.lbl_ocr_engine.config(text=f"OCR: {status}", fg=self.accent_color)
        else:
            self.lbl_ocr_engine.config(text=f"OCR: {status}", fg="#a6adc8")

        extra = result.extra_info
        deriv = extra.get('main_derivation', {}) if isinstance(extra, dict) else {}
        ocr_text = deriv.get('ocr_text') if isinstance(deriv, dict) else None
        ocr_conf = deriv.get('ocr_confidence') if isinstance(deriv, dict) else None
        ocr_engine = deriv.get('ocr_engine') if isinstance(deriv, dict) else None
        if ocr_text is not None:
            conf_text = f"{ocr_conf:.2f}" if isinstance(ocr_conf, (int, float)) else "?"
            self.lbl_ocr_result.config(text=f"OCR: {ocr_text}  conf={conf_text}")
        else:
            self.lbl_ocr_result.config(text=f"OCR: --  ({ocr_engine or 'no result'})")

        # 显示额外信息
        if extra:
            info_parts = []
            if extra.get('main_ticks_count'):
                info_parts.append(f"主尺刻线: {extra['main_ticks_count']}条")
            if extra.get('vernier_ticks_count'):
                info_parts.append(f"游标刻线: {extra['vernier_ticks_count']}条")
            if extra.get('main_gap_px'):
                info_parts.append(f"间距: {extra['main_gap_px']:.1f}px")

    # ═════════════════════════════════════════════════
    #  缩放 / 平移图像显示
    # ═════════════════════════════════════════════════

    def _display_image(self, img: np.ndarray):
        """在 Canvas 中显示图像（支持缩放）"""
        if img is None:
            return
        self._show_original = img
        self._zoom_fit_mode = True
        self._zoom_level = 1.0
        self._render_image()

    def _render_image(self, keep_position: bool = False):
        """根据当前缩放级别渲染图像到 Canvas"""
        img = self._show_original
        if img is None:
            return

        self.img_canvas.update_idletasks()
        h, w = img.shape[:2]
        cw = self.img_canvas.winfo_width()
        ch = self.img_canvas.winfo_height()
        # Canvas 尺寸未就绪时，延迟重试，避免用错误尺寸渲染导致模糊
        if cw < 50 or ch < 50:
            self.root.after(50, self._render_image)
            return

        # ── 计算缩放 ──
        if self._zoom_fit_mode:
            fit_scale = min(cw / w, ch / h)
            self._zoom_level = fit_scale
            self.lbl_zoom.config(text="适应窗口")
        else:
            pct = int(self._zoom_level * 100)
            self.lbl_zoom.config(text=f"{pct}%")

        dw, dh = int(w * self._zoom_level), int(h * self._zoom_level)
        dw = max(dw, 1); dh = max(dh, 1)

        # v6.5: 下采样用 INTER_AREA（避免摩尔纹），上采样用 INTER_LINEAR（更清晰）
        interp = cv2.INTER_AREA if self._zoom_level < 1.0 else cv2.INTER_LINEAR
        display = cv2.resize(img, (dw, dh), interpolation=interp)
        if len(display.shape) == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self._tk_img_ref = ImageTk.PhotoImage(pil_img)

        # ── 更新 Canvas ──
        self.img_canvas.delete("all")
        self.canvas_img_id = self.img_canvas.create_image(
            0, 0, anchor=tk.NW, image=self._tk_img_ref
        )
        self.canvas_text_id = None
        self.canvas_rect_id = None

        # Scrollregion
        self.img_canvas.configure(scrollregion=(0, 0, dw, dh))

    def _on_canvas_resize(self, event=None):
        """Canvas 尺寸变化 → 重新渲染（保持缩放模式）"""
        if self._show_original is None:
            return
        if self._zoom_fit_mode:
            self._render_image()

    def _on_scroll_y(self, *args):
        self.v_scrollbar.set(*args)

    def _on_scroll_x(self, *args):
        self.h_scrollbar.set(*args)

    def _on_canvas_press(self, event):
        """鼠标按下 → 开始拖拽平移"""
        self._pan_start_x = event.x
        self._pan_start_y = event.y
        self._pan_canvas_x = self.img_canvas.canvasx(0)
        self._pan_canvas_y = self.img_canvas.canvasy(0)

    def _on_canvas_drag(self, event):
        """拖拽平移"""
        if self._show_original is None or self._zoom_fit_mode:
            return  # 适应窗口时无需拖拽
        dx = self._pan_start_x - event.x
        dy = self._pan_start_y - event.y
        self.img_canvas.xview_scroll(int(dx), "units")
        self.img_canvas.yview_scroll(int(dy), "units")
        self._pan_start_x = event.x
        self._pan_start_y = event.y

    def _on_canvas_release(self, event):
        return

    def _on_mousewheel(self, event):
        """Ctrl+鼠标滚轮 → 缩放"""
        if not (event.state & 0x4):  # Ctrl 键
            return
        if event.delta > 0:
            self._zoom_in()
        else:
            self._zoom_out()

    def _on_mousewheel_alt(self, direction):
        """Linux/Mac 滚轮缩放 (Button-4/5)"""
        # 简化：仅在没有 Ctrl 检测的平台上触发
        pass

    def _on_double_click(self, event):
        """双击切换 1:1 / 适应窗口"""
        if self._zoom_fit_mode:
            self._zoom_100()
        else:
            self._zoom_fit()

    def _zoom_in(self):
        if self._show_original is None:
            return
        self._zoom_fit_mode = False
        self._zoom_level = min(self._zoom_level * 1.25, 10.0)
        self._render_image()

    def _zoom_out(self):
        if self._show_original is None:
            return
        self._zoom_fit_mode = False
        self._zoom_level = max(self._zoom_level / 1.25, 0.05)
        self._render_image()

    def _zoom_fit(self):
        if self._show_original is None:
            return
        self._zoom_fit_mode = True
        self._render_image()

    def _zoom_100(self):
        if self._show_original is None:
            return
        self._zoom_fit_mode = False
        self._zoom_level = 1.0
        self._render_image()

    def _display_image_with_overlay(self, img: np.ndarray):
        """兼容旧接口"""
        self._display_image(img)

    def _save_result(self):
        """保存标注结果图像"""
        if self.current_result is None:
            return

        file_path = filedialog.asksaveasfilename(
            title="保存标注图像",
            defaultextension=".png",
            filetypes=[
                ("PNG 图像", "*.png"),
                ("JPEG 图像", "*.jpg"),
                ("所有文件", "*.*")
            ]
        )
        if not file_path:
            return

        try:
            _, buf = cv2.imencode(os.path.splitext(file_path)[1],
                                   self.current_result.image_annotated)
            if buf is not None:
                buf.tofile(file_path)
            else:
                cv2.imwrite(file_path, self.current_result.image_annotated)
            self.status_label.config(text=f"💾 已保存: {os.path.basename(file_path)}")
            messagebox.showinfo("保存成功", f"标注图像已保存到:\n{file_path}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))


def main():
    root = tk.Tk()
    app = CaliperApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
