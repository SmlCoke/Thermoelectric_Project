"""
Sky Image Viewer (Left=Original, Right=Selectable) - Strict Probe Logic + Sidebar UI + Pixel-metric Probes

Key behaviors:
1) Colorbar only shown for prob views (prob_overlay / prob_heatmap). Hidden otherwise to enlarge right view.
2) Probe marker draws NO text on image (avoids clutter). Values go to Probe table & Log.
3) Bottom area uses Notebook tabs: Summary / Probes / Log.
4) Probe strict logic:
   - Image-like view: record Cloud_Prob(pixel) + selected PIXEL metric value, and draw marker.
   - Chart-like view: ONLY record selected IMAGE-level metric value (no marker, no Cloud_Prob).
5) Mouse wheel zoom for right image views.
6) Summary always contains fixed "[Metric meanings]" section (does not disappear when switching images).
7) Right view initial size matches left (auto-fit). Each time you load a NEW image, zoom resets to auto-fit again.
   ✅ Fix: For ALL folders/images, initial left & right sizes match by using true auto-fit on right (NOT clipped by zoom_min).
      Also keep a per-image zoom_fit floor so zoom_out never "jumps bigger" when auto-fit < zoom_min.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from PIL import Image, ImageTk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import matplotlib as mpl


# =========================
# 0) Windows/中文路径鲁棒读图
# =========================
def imread_unicode(path: str) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


# =========================
# 1) 云概率 + 特征
# =========================
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def soft_score(x: np.ndarray, center: float, scale: float) -> np.ndarray:
    scale = max(scale, 1e-6)
    return sigmoid((x - center) / scale)


@dataclass
class AlgoCfg:
    rbr_center: float = 0.95
    rbr_scale: float = 0.12
    bright_center: float = 0.62
    bright_scale: float = 0.18
    w_rbr: float = 0.65
    w_bright: float = 0.35
    prob_blur_ksize: int = 5
    overlay_alpha: float = 0.45


def cloud_probability_map(img_bgr: np.ndarray, cfg: AlgoCfg) -> np.ndarray:
    img = img_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(img)
    rbr = r / (b + 1e-6)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    p_rbr = soft_score(rbr, cfg.rbr_center, cfg.rbr_scale)
    p_v = soft_score(v, cfg.bright_center, cfg.bright_scale)

    prob = cfg.w_rbr * p_rbr + cfg.w_bright * p_v
    prob = np.clip(prob, 0.0, 1.0)

    k = cfg.prob_blur_ksize
    if k and k >= 3 and k % 2 == 1:
        prob = cv2.GaussianBlur(prob, (k, k), 0)

    return prob


def extract_features(img_bgr: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    cloud_fraction = float(np.mean(prob))
    opacity_proxy = float(np.std(gray))

    lap = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_32F)
    texture_lap_var = float(np.var(lap))

    edges = cv2.Canny((gray * 255).astype(np.uint8), 40, 120)
    edge_density = float(np.mean(edges > 0))

    sky_uniformity = float(1.0 / (1.0 + np.var(gray)))
    valid_pixels = int(gray.size)

    return {
        "cloud_fraction": cloud_fraction,
        "opacity_proxy": opacity_proxy,
        "texture_lap_var": texture_lap_var,
        "edge_density": edge_density,
        "sky_uniformity": sky_uniformity,
        "valid_pixels": valid_pixels,
    }


def make_overlay(img_bgr: np.ndarray, prob: np.ndarray, alpha: float) -> np.ndarray:
    heat = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heat, alpha, 0)
    return overlay


def compute_pixel_maps(img_bgr: np.ndarray, prob: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Pixel-level maps you can probe.
    All maps are HxW float32 (except edge: 0/1 float32).
    """
    img = img_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(img)
    rbr = r / (b + 1e-6)  # can be >1
    rbr = np.clip(rbr, 0.0, 3.0).astype(np.float32)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = (hsv[:, :, 2].astype(np.float32) / 255.0).astype(np.float32)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    lap = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_32F)
    lap_abs = np.clip(np.abs(lap) / 255.0, 0.0, 1.0).astype(np.float32)

    edges = cv2.Canny((gray * 255).astype(np.uint8), 40, 120)
    edge01 = (edges > 0).astype(np.float32)

    return {
        "cloud_prob": prob.astype(np.float32),
        "rbr": rbr,
        "brightness_v": v,
        "gray": gray.astype(np.float32),
        "lap_abs": lap_abs,
        "edge01": edge01,
    }


# =========================
# 2) 时间戳解析
# =========================
def build_timestamp_from_path(path: str, year: int) -> Optional[pd.Timestamp]:
    parent = os.path.basename(os.path.dirname(path))
    m = re.fullmatch(r"(\d{2})(\d{2})", parent)
    if not m:
        return None
    month, day = int(m.group(1)), int(m.group(2))

    base = os.path.basename(path)
    t = re.search(r"(\d{1,2})[ _-]?[:_]?(\d{2})", base)
    if not t:
        return None
    hh, mm = int(t.group(1)), int(t.group(2))

    try:
        return pd.Timestamp(year=year, month=month, day=day, hour=hh, minute=mm, second=0)
    except Exception:
        return None


# =========================
# 3) 文件扫描 + CSV
# =========================
def scan_date_folders(root_dir: str) -> List[str]:
    if not os.path.isdir(root_dir):
        return []
    folders = []
    for name in os.listdir(root_dir):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p) and re.fullmatch(r"\d{4}", name):
            folders.append(name)
    return sorted(folders)


def list_images_in_folder(folder_path: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    if not os.path.isdir(folder_path):
        return []
    files = []
    for fn in sorted(os.listdir(folder_path)):
        if fn.lower().endswith(exts):
            files.append(os.path.join(folder_path, fn))
    return files


def load_csv_index(csv_path: str) -> Optional[pd.DataFrame]:
    if not csv_path or not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if "path" in df.columns:
            df["path"] = df["path"].astype(str)
        return df
    except Exception:
        return None


# =========================
# 4) Colorbar + 文本图
# =========================
def build_colorbar_image(height: int = 520, width: int = 180, cmap: str = "jet") -> Image.Image:
    dpi = 150
    fig_w = width / dpi
    fig_h = height / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0.38, 0.06, 0.22, 0.90])
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    cb = mpl.colorbar.ColorbarBase(
        ax,
        cmap=plt.get_cmap(cmap),
        norm=norm,
        orientation="vertical",
        ticks=[0.0, 0.5, 1.0],
    )
    cb.ax.yaxis.set_ticks_position("right")
    cb.ax.tick_params(labelsize=10)
    cb.set_label("Cloud Prob.", fontsize=12, rotation=90, labelpad=18)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def render_text_image(text: str, w: int = 900, h: int = 600) -> Image.Image:
    dpi = 120
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
    ax.axis("off")
    ax.text(0, 0.95, text, fontsize=12, va="top")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# =========================
# 5) GUI
# =========================
class SkyGUI:
    VIEW_OPTIONS = [
        ("Prob Overlay", "prob_overlay"),
        ("Prob Heatmap", "prob_heatmap"),
        ("Original (Right)", "right_original"),
        ("Gray", "right_gray"),
        ("Edges (Canny)", "right_edges"),
        ("Laplacian (Texture)", "right_laplacian"),
        ("CSV Bar (current)", "csv_bar_current"),
        ("CSV Trend (folder)", "csv_trend_folder"),
    ]

    IMAGE_VIEWS = {
        "prob_overlay",
        "prob_heatmap",
        "right_original",
        "right_gray",
        "right_edges",
        "right_laplacian",
    }
    CHART_VIEWS = {"csv_bar_current", "csv_trend_folder"}
    PROB_VIEWS = {"prob_overlay", "prob_heatmap"}  # only these show colorbar

    # --- Image-level metrics (for summary + charts) ---
    IMG_METRICS = [
        "cloud_fraction",
        "opacity_proxy",
        "texture_lap_var",
        "edge_density",
        "sky_uniformity",
        "valid_pixels",
    ]
    METRIC_MEANINGS = {
        "cloud_fraction": "云覆盖强度（图像级）：Cloud Prob. 在整幅图上的平均值（越大代表整体越“多云/更可能有云”）。",
        "opacity_proxy": "不透明/雾感（图像级）：灰度图 std(gray)（越大说明亮度起伏更明显）。",
        "texture_lap_var": "纹理强度（图像级）：var(Laplacian(gray))（越大高频细节越多）。",
        "edge_density": "边缘密度（图像级）：mean(Canny>0)（越大边缘/结构越多）。",
        "sky_uniformity": "天空均匀度（图像级）：1/(1+var(gray))，越接近 1 越均匀。",
        "valid_pixels": "有效像素数（图像级）：参与计算的像素总数（基本等于整幅图像像素数）。",
    }

    # --- Pixel-level metrics (for probing) ---
    PIXEL_METRICS = [
        ("cloud_prob", "Cloud Prob (pixel)"),
        ("rbr", "R/B Ratio (pixel)"),
        ("brightness_v", "Brightness V (pixel)"),
        ("gray", "Gray (pixel)"),
        ("lap_abs", "Laplacian |.| (pixel)"),
        ("edge01", "Edge 0/1 (pixel)"),
    ]

    PIXEL_METRIC_MEANINGS = {
        "cloud_prob": "像素级云概率：prob map 在该像素处的值（0~1）。",
        "rbr": "像素级红蓝比：R/(B+eps)，可用于区分偏红云/偏蓝天空等（可能 >1）。",
        "brightness_v": "像素级亮度：HSV 的 V 通道（0~1）。",
        "gray": "像素级灰度强度：gray(0~1)。",
        "lap_abs": "像素级纹理强度：|Laplacian(gray)| 归一化（0~1）。",
        "edge01": "像素级边缘标记：Canny 边缘（0 或 1）。",
    }

    def __init__(self, root: tk.Tk, root_dir: str, year: int, csv_path: Optional[str] = None):
        self.root = root
        self.root.title("Sky Image Viewer (Left=Original, Right=Selectable)")
        self.root.geometry("1500x860")

        self.root_dir = root_dir
        self.year = year
        self.csv_path = csv_path

        self.algo_cfg = AlgoCfg()
        self.csv_df = load_csv_index(csv_path)

        self.current_folder: Optional[str] = None
        self.current_files: List[str] = []

        self.current_path: Optional[str] = None
        self.current_img_bgr: Optional[np.ndarray] = None
        self.current_prob: Optional[np.ndarray] = None
        self.current_overlay_bgr: Optional[np.ndarray] = None

        self.current_feats_rt: Optional[Dict[str, float]] = None
        self.current_feats_csv: Optional[Dict[str, float]] = None
        self.current_ts: Optional[pd.Timestamp] = None

        self.current_pixel_maps: Optional[Dict[str, np.ndarray]] = None

        # zoom only for right image-like views
        self.zoom: Optional[float] = None          # auto-fit when None
        self.zoom_fit: Optional[float] = None      # ✅ per-image auto-fit floor (can be < zoom_min)
        self.zoom_min: float = 0.2                 # user zoom-out floor for “normal cases”
        self.zoom_max: float = 6.0
        self.zoom_step: float = 1.15

        # probes store (ox, oy) in original image coordinates
        self.probes: List[Tuple[int, int]] = []

        # Tk image refs
        self._tk_left_img = None
        self._tk_right_img = None
        self._tk_cbar_img = None

        self._left_canvas_img_id = None
        self._right_canvas_img_id = None

        # mapping info for right canvas (probe coordinate mapping)
        self._right_disp_info = None

        self._summary_fixed_intro = self._build_summary_fixed_intro()

        self._build_ui()
        self._refresh_dates()

        self.root.bind("<KeyPress>", self._on_keypress)

    def _build_summary_fixed_intro(self) -> str:
        lines = []
        lines.append("[Pixel metric meanings (for probes)]\n")
        for k, _name in self.PIXEL_METRICS:
            meaning = self.PIXEL_METRIC_MEANINGS.get(k, "(No description)")
            lines.append(f"- {k}: {meaning}\n")
        lines.append("-" * 60 + "\n\n")

        lines.append("[Image-level metric meanings]\n")
        for k in self.IMG_METRICS:
            meaning = self.METRIC_MEANINGS.get(k, "(No description)")
            lines.append(f"- {k}: {meaning}\n")
        lines.append("-" * 60 + "\n\n")

        return "".join(lines)

    # ---------- UI ----------
    def _build_ui(self):
        style = ttk.Style()
        style.configure("Big.TButton", padding=(12, 8), font=("Segoe UI", 10))
        style.configure("Big.TLabel", font=("Segoe UI", 10))
        style.configure("Group.TLabelframe", padding=(10, 8))
        style.configure("Group.TLabelframe.Label", font=("Segoe UI", 10, "bold"))

        root_wrap = ttk.Frame(self.root, padding=(10, 8))
        root_wrap.pack(fill=tk.BOTH, expand=True)

        root_wrap.grid_rowconfigure(0, weight=1)
        root_wrap.grid_columnconfigure(0, weight=0)
        root_wrap.grid_columnconfigure(1, weight=1)

        # ===== Left Sidebar =====
        sidebar = ttk.Frame(root_wrap)
        sidebar.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        sidebar.grid_rowconfigure(99, weight=1)

        # --- System Status ---
        self.grp_status = ttk.LabelFrame(sidebar, text="系统状态", style="Group.TLabelframe")
        self.grp_status.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.grp_status.grid_columnconfigure(1, weight=1)

        ttk.Label(self.grp_status, text="当前文件夹:", style="Big.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.lbl_folder = ttk.Label(self.grp_status, text="-", style="Big.TLabel")
        self.lbl_folder.grid(row=0, column=1, sticky="w")

        ttk.Label(self.grp_status, text="图像数量:", style="Big.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 8))
        self.lbl_count = ttk.Label(self.grp_status, text="-", style="Big.TLabel")
        self.lbl_count.grid(row=1, column=1, sticky="w")

        ttk.Label(self.grp_status, text="当前索引 i:", style="Big.TLabel").grid(row=2, column=0, sticky="w", padx=(0, 8))
        self.lbl_idx = ttk.Label(self.grp_status, text="-", style="Big.TLabel")
        self.lbl_idx.grid(row=2, column=1, sticky="w")

        ttk.Label(self.grp_status, text="当前文件:", style="Big.TLabel").grid(row=3, column=0, sticky="w", padx=(0, 8))
        self.lbl_file = ttk.Label(self.grp_status, text="-", style="Big.TLabel")
        self.lbl_file.grid(row=3, column=1, sticky="w")

        ttk.Label(self.grp_status, text="指标来源:", style="Big.TLabel").grid(row=4, column=0, sticky="w", padx=(0, 8))
        self.lbl_src = ttk.Label(self.grp_status, text="-", style="Big.TLabel")
        self.lbl_src.grid(row=4, column=1, sticky="w")

        ttk.Label(self.grp_status, text="解析时间:", style="Big.TLabel").grid(row=5, column=0, sticky="w", padx=(0, 8))
        self.lbl_time = ttk.Label(self.grp_status, text="-", style="Big.TLabel")
        self.lbl_time.grid(row=5, column=1, sticky="w")

        # --- Data Selection ---
        self.grp_data = ttk.LabelFrame(sidebar, text="数据选择", style="Group.TLabelframe")
        self.grp_data.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.grp_data.grid_columnconfigure(0, weight=1)

        ttk.Label(self.grp_data, text="日期文件夹(mmdd):", style="Big.TLabel").grid(row=0, column=0, sticky="w")
        self.date_var = tk.StringVar()
        self.date_combo = ttk.Combobox(self.grp_data, textvariable=self.date_var, state="readonly", width=16)
        self.date_combo.grid(row=1, column=0, sticky="ew", pady=(4, 10))
        self.date_combo.bind("<<ComboboxSelected>>", lambda e: self._on_date_selected())

        ttk.Label(self.grp_data, text="图像索引 i (从 1 开始):", style="Big.TLabel").grid(row=2, column=0, sticky="w")
        self.idx_var = tk.StringVar(value="1")
        self.idx_entry = ttk.Entry(self.grp_data, textvariable=self.idx_var, width=10)
        self.idx_entry.grid(row=3, column=0, sticky="ew", pady=(4, 10))
        self.idx_entry.bind("<Return>", lambda e: self.on_show())

        btn_row = ttk.Frame(self.grp_data)
        btn_row.grid(row=4, column=0, sticky="ew")
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)

        self.btn_show = ttk.Button(btn_row, text="Show", command=self.on_show, style="Big.TButton")
        self.btn_show.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.btn_refresh = ttk.Button(btn_row, text="Refresh", command=self._refresh_dates, style="Big.TButton")
        self.btn_refresh.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        # --- View Settings ---
        self.grp_view = ttk.LabelFrame(sidebar, text="视图设置", style="Group.TLabelframe")
        self.grp_view.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.grp_view.grid_columnconfigure(0, weight=1)

        ttk.Label(self.grp_view, text="Right view:", style="Big.TLabel").grid(row=0, column=0, sticky="w")
        self.view_mode_var = tk.StringVar(value="prob_overlay")
        self.view_combo = ttk.Combobox(
            self.grp_view, state="readonly", width=18,
            values=[name for name, _key in self.VIEW_OPTIONS]
        )
        self.view_combo.grid(row=1, column=0, sticky="ew", pady=(4, 10))
        self._view_name_to_key = {name: key for name, key in self.VIEW_OPTIONS}
        self._view_key_to_name = {key: name for name, key in self.VIEW_OPTIONS}
        self.view_combo.set(self._view_key_to_name[self.view_mode_var.get()])
        self.view_combo.bind("<<ComboboxSelected>>", lambda e: self._on_view_change())

        ttk.Label(self.grp_view, text="Pixel metric (for probes):", style="Big.TLabel").grid(row=2, column=0, sticky="w")
        self.pixel_metric_var = tk.StringVar(value="cloud_prob")
        self.pixel_metric_combo = ttk.Combobox(
            self.grp_view, textvariable=self.pixel_metric_var,
            state="readonly", width=18,
            values=[k for k, _name in self.PIXEL_METRICS]
        )
        self.pixel_metric_combo.grid(row=3, column=0, sticky="ew", pady=(4, 10))

        ttk.Label(self.grp_view, text="Chart metric (image-level):", style="Big.TLabel").grid(row=4, column=0, sticky="w")
        self.chart_metric_var = tk.StringVar(value="sky_uniformity")
        self.chart_metric_combo = ttk.Combobox(
            self.grp_view, textvariable=self.chart_metric_var,
            state="readonly", width=18,
            values=self.IMG_METRICS
        )
        self.chart_metric_combo.grid(row=5, column=0, sticky="ew", pady=(4, 0))
        self.chart_metric_combo.bind("<<ComboboxSelected>>", lambda e: self._redraw_right())

        # --- Zoom ---
        self.grp_zoom = ttk.LabelFrame(sidebar, text="缩放", style="Group.TLabelframe")
        self.grp_zoom.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self.grp_zoom.grid_columnconfigure(0, weight=1)
        self.zoom_label = ttk.Label(self.grp_zoom, text="Zoom: auto-fit", style="Big.TLabel")
        self.zoom_label.grid(row=0, column=0, sticky="w", pady=(0, 8))

        zoom_row = ttk.Frame(self.grp_zoom)
        zoom_row.grid(row=1, column=0, sticky="ew")
        zoom_row.grid_columnconfigure(0, weight=1)
        zoom_row.grid_columnconfigure(1, weight=1)

        self.btn_zoomin = ttk.Button(zoom_row, text="Zoom +", command=self.zoom_in, style="Big.TButton")
        self.btn_zoomin.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.btn_zoomout = ttk.Button(zoom_row, text="Zoom -", command=self.zoom_out, style="Big.TButton")
        self.btn_zoomout.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        # --- Actions ---
        self.grp_actions = ttk.LabelFrame(sidebar, text="操作", style="Group.TLabelframe")
        self.grp_actions.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        self.grp_actions.grid_columnconfigure(0, weight=1)

        self.btn_clear_markers = ttk.Button(self.grp_actions, text="Clear probes (markers)", command=self.clear_probes, style="Big.TButton")
        self.btn_clear_markers.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.btn_clear_table = ttk.Button(self.grp_actions, text="Clear probe table", command=self.clear_probe_table, style="Big.TButton")
        self.btn_clear_table.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        self.btn_copy = ttk.Button(self.grp_actions, text="Copy selected", command=self.copy_selected_probes, style="Big.TButton")
        self.btn_copy.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        self.btn_export = ttk.Button(self.grp_actions, text="Export CSV", command=self.export_probes_csv, style="Big.TButton")
        self.btn_export.grid(row=3, column=0, sticky="ew")

        ttk.Frame(sidebar).grid(row=99, column=0, sticky="nsew")

        # ===== Right Main Area =====
        main = ttk.Frame(root_wrap)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(0, weight=3)
        main.grid_rowconfigure(1, weight=2)
        main.grid_columnconfigure(0, weight=1)

        view_frame = ttk.Frame(main)
        view_frame.grid(row=0, column=0, sticky="nsew")
        view_frame.grid_rowconfigure(0, weight=1)
        view_frame.grid_columnconfigure(0, weight=1, uniform="img")
        view_frame.grid_columnconfigure(1, weight=1, uniform="img")
        view_frame.grid_columnconfigure(2, weight=0)

        self.left_panel = ttk.LabelFrame(view_frame, text="Original Image", padding=6)
        self.right_panel = ttk.LabelFrame(view_frame, text="Right View", padding=6)
        self.cbar_panel = ttk.LabelFrame(view_frame, text="Colorbar", padding=6)

        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 10))
        self.cbar_panel.grid(row=0, column=2, sticky="ns")

        self.cbar_panel.configure(width=200)
        self.cbar_panel.grid_propagate(False)

        self.left_canvas = tk.Canvas(self.left_panel, highlightthickness=0, bg="#f0f0f0")
        self.left_canvas.pack(fill=tk.BOTH, expand=True)
        self.left_canvas.bind("<Configure>", lambda e: self._redraw_left())

        self.right_canvas = tk.Canvas(self.right_panel, highlightthickness=0, bg="#f0f0f0")
        self.right_canvas.pack(fill=tk.BOTH, expand=True)
        self.right_canvas.bind("<Configure>", lambda e: self._redraw_right())
        self.right_canvas.bind("<Button-1>", self._on_right_click)

        self.right_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.right_canvas.bind("<Button-4>", lambda e: self._on_mousewheel_linux(+1))
        self.right_canvas.bind("<Button-5>", lambda e: self._on_mousewheel_linux(-1))

        self.cbar_label = ttk.Label(self.cbar_panel)
        self.cbar_label.pack(fill=tk.BOTH, expand=True)
        self._update_colorbar()

        bottom_frame = ttk.Frame(main)
        bottom_frame.grid(row=1, column=0, sticky="nsew")

        nb = ttk.Notebook(bottom_frame)
        nb.pack(fill=tk.BOTH, expand=True)

        self.tab_summary = ttk.Frame(nb)
        self.tab_probes = ttk.Frame(nb)
        self.tab_log = ttk.Frame(nb)

        nb.add(self.tab_summary, text="Summary")
        nb.add(self.tab_probes, text="Probes")
        nb.add(self.tab_log, text="Log")

        self.summary_text = tk.Text(self.tab_summary, height=10, wrap="word")
        self.summary_text.pack(fill=tk.BOTH, expand=True)

        table_wrap = ttk.Frame(self.tab_probes)
        table_wrap.pack(fill=tk.BOTH, expand=True)

        cols = ("No", "x", "y", "Cloud_Prob(pixel)", "Metric(name)", "Metric(pixel)", "RightView")
        self.probe_tree = ttk.Treeview(table_wrap, columns=cols, show="headings", height=10)
        for c in cols:
            self.probe_tree.heading(c, text=c)
            self.probe_tree.column(c, width=140 if c != "No" else 60, anchor="center")
        self.probe_tree.column("Metric(name)", width=160, anchor="center")
        self.probe_tree.column("RightView", width=140, anchor="center")

        vsb = ttk.Scrollbar(table_wrap, orient="vertical", command=self.probe_tree.yview)
        self.probe_tree.configure(yscrollcommand=vsb.set)
        self.probe_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self._probe_counter = 0

        self.log_text = tk.Text(self.tab_log, height=10, wrap="word")
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self._apply_view_dependent_ui()
        self._update_status_panel()

    def _update_colorbar(self):
        pil_cb = build_colorbar_image(height=300, width=120, cmap="jet")
        tk_cb = ImageTk.PhotoImage(pil_cb)
        self.cbar_label.configure(image=tk_cb)
        self._tk_cbar_img = tk_cb

    # ---------- Utilities ----------
    def _append_log(self, msg: str):
        self.log_text.insert(tk.END, msg.rstrip() + "\n")
        self.log_text.see(tk.END)

    def _fmt6(self, x) -> str:
        try:
            if x is None:
                return "N/A"
            if pd.isna(x):
                return "NaN"
            if isinstance(x, int):
                return str(x)
            return f"{float(x):.6g}"
        except Exception:
            return str(x)

    def _get_mode(self) -> str:
        return self.view_mode_var.get()

    def _get_feats_prefer_csv(self) -> Optional[Dict[str, float]]:
        return self.current_feats_csv if self.current_feats_csv is not None else self.current_feats_rt

    def _get_current_source_name(self) -> str:
        if self.current_feats_csv is not None:
            return "CSV"
        if self.current_feats_rt is not None:
            return "Real-time"
        return "-"

    def _update_status_panel(self):
        self.lbl_folder.config(text=str(self.current_folder) if self.current_folder else "-")
        self.lbl_count.config(text=str(len(self.current_files)) if self.current_files else "-")
        try:
            self.lbl_idx.config(text=str(int(self.idx_var.get())))
        except Exception:
            self.lbl_idx.config(text="-")
        self.lbl_file.config(text=os.path.basename(self.current_path) if self.current_path else "-")
        self.lbl_src.config(text=self._get_current_source_name())
        self.lbl_time.config(text="-" if self.current_ts is None else str(self.current_ts))

    # ---------- View dependent UI ----------
    def _apply_view_dependent_ui(self):
        mode = self._get_mode()

        if mode in self.PROB_VIEWS:
            if not self.cbar_panel.winfo_ismapped():
                self.cbar_panel.grid(row=0, column=2, sticky="ns")
        else:
            if self.cbar_panel.winfo_ismapped():
                self.cbar_panel.grid_remove()

        if mode in self.IMAGE_VIEWS:
            self.btn_zoomin.state(["!disabled"])
            self.btn_zoomout.state(["!disabled"])
            self.zoom_label.config(text="Zoom: auto-fit" if self.zoom is None else f"Zoom: {self.zoom:.2f}x")
        else:
            self.btn_zoomin.state(["disabled"])
            self.btn_zoomout.state(["disabled"])
            self.zoom_label.config(text="Zoom: N/A")

        name = self._view_key_to_name.get(mode, mode)
        self.right_panel.config(text=f"Right View - {name}")

        if mode in self.CHART_VIEWS:
            self.chart_metric_combo.state(["!disabled"])
        else:
            self.chart_metric_combo.state(["disabled"])

    # ---------- Data ----------
    def _refresh_dates(self):
        folders = scan_date_folders(self.root_dir)
        if not folders:
            messagebox.showwarning("Warning", f"No date folders found under:\n{self.root_dir}")
            self.date_combo["values"] = []
            return
        self.date_combo["values"] = folders
        if self.date_var.get() not in folders:
            self.date_var.set(folders[0])
        self._on_date_selected()

    def _on_date_selected(self):
        folder = self.date_var.get()
        folder_path = os.path.join(self.root_dir, folder)
        self.current_folder = folder
        self.current_files = list_images_in_folder(folder_path)
        self._update_status_panel()

    def _on_view_change(self):
        name = self.view_combo.get()
        key = self._view_name_to_key.get(name, "prob_overlay")
        self.view_mode_var.set(key)
        self._apply_view_dependent_ui()
        self._redraw_right()

    # ---------- Keybind ----------
    def _on_keypress(self, e: tk.Event):
        if e.char in ['+', '=']:
            self.zoom_in()
        elif e.char == '-':
            self.zoom_out()
        elif e.char in ['c', 'C']:
            self.clear_probes()

    # ---------- Zoom ----------
    def _ensure_zoom_initialized(self):
        if self.zoom is not None:
            return
        if self.current_img_bgr is None:
            return

        panel_w = self.right_canvas.winfo_width()
        panel_h = self.right_canvas.winfo_height()
        if panel_w <= 2 or panel_h <= 2:
            return

        H, W = self.current_img_bgr.shape[:2]
        fit = min(panel_w / max(1, W), panel_h / max(1, H))
        fit = float(min(fit, self.zoom_max))   # ✅ do NOT raise to zoom_min on auto-fit
        self.zoom = fit
        self.zoom_fit = fit

    def zoom_in(self):
        if self._get_mode() not in self.IMAGE_VIEWS:
            return
        self._ensure_zoom_initialized()
        if self.zoom is None:
            return
        self.zoom = min(self.zoom_max, self.zoom * self.zoom_step)
        self.zoom_label.config(text=f"Zoom: {self.zoom:.2f}x")
        self._redraw_right()

    def zoom_out(self):
        if self._get_mode() not in self.IMAGE_VIEWS:
            return
        self._ensure_zoom_initialized()
        if self.zoom is None:
            return
        floor = self.zoom_fit if self.zoom_fit is not None else self.zoom_min  # ✅ per-image floor
        self.zoom = max(floor, self.zoom / self.zoom_step)
        self.zoom_label.config(text=f"Zoom: {self.zoom:.2f}x")
        self._redraw_right()

    def _on_mousewheel(self, event: tk.Event):
        if self._get_mode() not in self.IMAGE_VIEWS:
            return
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def _on_mousewheel_linux(self, direction: int):
        if self._get_mode() not in self.IMAGE_VIEWS:
            return
        if direction > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    # ---------- Probe controls ----------
    def clear_probes(self):
        self.probes.clear()
        self._redraw_right()
        self._append_log("Probes cleared (markers).")

    def clear_probe_table(self):
        for item in self.probe_tree.get_children():
            self.probe_tree.delete(item)
        self._probe_counter = 0
        self._append_log("Probe table cleared.")

    def _insert_probe_row(self, x, y, cloud_prob, metric_name, metric_val, right_view_key):
        self._probe_counter += 1
        self.probe_tree.insert(
            "", "end",
            values=(self._probe_counter, x, y, cloud_prob, metric_name, metric_val, right_view_key)
        )

    def copy_selected_probes(self):
        items = self.probe_tree.selection()
        if not items:
            messagebox.showinfo("Info", "No rows selected.")
            return
        lines = []
        header = ["No", "x", "y", "Cloud_Prob(pixel)", "Metric(name)", "Metric(pixel)", "RightView"]
        lines.append("\t".join(header))
        for it in items:
            vals = self.probe_tree.item(it, "values")
            lines.append("\t".join([str(v) for v in vals]))
        text = "\n".join(lines)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._append_log(f"Copied {len(items)} selected probe rows to clipboard.")

    def export_probes_csv(self):
        items = self.probe_tree.get_children()
        if not items:
            messagebox.showinfo("Info", "Probe table is empty.")
            return
        save_path = filedialog.asksaveasfilename(
            title="Save probes to CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not save_path:
            return
        rows = []
        for it in items:
            vals = self.probe_tree.item(it, "values")
            rows.append(vals)
        df = pd.DataFrame(rows, columns=["No", "x", "y", "Cloud_Prob(pixel)", "Metric(name)", "Metric(pixel)", "RightView"])
        try:
            df.to_csv(save_path, index=False, encoding="utf-8-sig")
            self._append_log(f"Exported probes to: {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save CSV:\n{e}")

    # ---------- Show ----------
    def on_show(self):
        if not self.current_files:
            messagebox.showerror("Error", "No images in selected folder.")
            return
        try:
            i = int(self.idx_var.get())
        except ValueError:
            messagebox.showerror("Error", "Index i must be an integer.")
            return
        if i < 1 or i > len(self.current_files):
            messagebox.showerror("Error", f"i out of range: 1~{len(self.current_files)}")
            return

        path = self.current_files[i - 1]
        img = imread_unicode(path)
        if img is None:
            messagebox.showerror("Error", f"Cannot read image:\n{path}")
            return

        prob = cloud_probability_map(img, self.algo_cfg)
        overlay = make_overlay(img, prob, self.algo_cfg.overlay_alpha)
        feats_rt = extract_features(img, prob)
        feats_csv = self._lookup_csv_metrics(path)
        ts = build_timestamp_from_path(path, self.year)

        self.current_path = path
        self.current_img_bgr = img
        self.current_prob = prob
        self.current_overlay_bgr = overlay
        self.current_feats_rt = feats_rt
        self.current_feats_csv = feats_csv
        self.current_ts = ts

        self.current_pixel_maps = compute_pixel_maps(img, prob)

        # ✅ 每次新图：zoom 回到 auto-fit（并清空 per-image floor）
        self.zoom = None
        self.zoom_fit = None
        self.probes.clear()

        self._redraw_left()
        self._redraw_right()

        feats_use = feats_csv if feats_csv is not None else feats_rt
        self._show_summary(path, ts, feats_use, from_csv=(feats_csv is not None))
        self._append_log(f"Loaded image: {os.path.basename(path)}  | Source: {'CSV' if feats_csv else 'Real-time'}")

        self._update_status_panel()

    def _lookup_csv_metrics(self, img_path: str) -> Optional[Dict[str, float]]:
        if self.csv_df is None or "path" not in self.csv_df.columns:
            return None

        p_abs = os.path.normpath(img_path)
        p_base = os.path.basename(img_path)

        df = self.csv_df
        df_paths = df["path"].astype(str).apply(lambda x: os.path.normpath(x))

        hit = df[df_paths == p_abs]
        if hit.empty:
            hit = df[df["path"].astype(str).str.endswith(p_base)]
        if hit.empty:
            return None

        row = hit.iloc[0].to_dict()
        out = {k: row.get(k) for k in self.IMG_METRICS if k in row}
        return out if out else None

    # ---------- Summary ----------
    def _show_summary(self, path: str, ts: Optional[pd.Timestamp], feats: Dict[str, float], from_csv: bool):
        self.summary_text.delete("1.0", tk.END)

        header = f"File: {path}\n"
        header += f"Time: {ts if ts is not None else 'Parse failed'}\n"
        header += f"Source: {'CSV' if from_csv else 'Real-time'}\n"
        header += "-" * 60 + "\n"
        self.summary_text.insert(tk.END, header)

        self.summary_text.insert(tk.END, self._summary_fixed_intro)

        self.summary_text.insert(tk.END, "[Image-level Values]\n")
        for k in self.IMG_METRICS:
            if k in feats:
                self.summary_text.insert(tk.END, f"{k:16s} = {self._fmt6(feats.get(k)):>10s}\n")

        pm = self.pixel_metric_var.get()
        meaning = self.PIXEL_METRIC_MEANINGS.get(pm, "")
        if meaning:
            self.summary_text.insert(tk.END, "\n" + "-" * 60 + "\n")
            self.summary_text.insert(tk.END, "[Current probe pixel metric]\n")
            self.summary_text.insert(tk.END, f"- {pm}: {meaning}\n")

    # ---------- CSV trend/bar rendering ----------
    def _get_csv_folder_df(self) -> Optional[pd.DataFrame]:
        if self.csv_df is None or self.current_folder is None:
            return None
        if "path" not in self.csv_df.columns:
            return None

        folder = self.current_folder
        df = self.csv_df.copy()
        s = df["path"].astype(str)
        mask = s.str.contains(rf"[\\/]{re.escape(folder)}[\\/]", regex=True)
        df = df[mask].copy()
        if df.empty:
            return None

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        else:
            df["timestamp"] = pd.NaT

        need = df["timestamp"].isna()
        if need.any():
            def _parse_ts(p):
                ts = build_timestamp_from_path(p, self.year)
                return ts if ts is not None else pd.NaT
            df.loc[need, "timestamp"] = df.loc[need, "path"].apply(_parse_ts)

        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        return df if not df.empty else None

    def _build_csv_bar_pil(self) -> Image.Image:
        feats = self._get_feats_prefer_csv()
        if feats is None:
            return render_text_image("No image loaded.\nClick 'Show' first.")

        keys = [k for k in self.IMG_METRICS if k in feats and k != "valid_pixels"]
        if not keys:
            return render_text_image("No metrics available for bar chart.")

        vals = []
        for k in keys:
            try:
                vals.append(float(feats.get(k)))
            except Exception:
                vals.append(np.nan)

        dpi = 120
        fig = plt.figure(figsize=(9, 5), dpi=dpi)
        ax = fig.add_axes([0.08, 0.18, 0.88, 0.72])
        ax.bar(range(len(keys)), vals)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=20, ha="right")
        ax.set_ylabel("Value")
        title_src = "CSV" if self.current_feats_csv is not None else "Real-time"
        ax.set_title(f"Metrics Bar ({title_src})")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    def _build_csv_trend_pil(self) -> Image.Image:
        df = self._get_csv_folder_df()
        if df is None:
            return render_text_image(
                "CSV Trend unavailable.\n"
                "- Check CSV_PATH exists and has 'path' (and ideally 'timestamp') columns.\n"
                "- Ensure CSV paths include the mmdd folder name."
            )

        metric = self.chart_metric_var.get()
        if metric not in df.columns:
            return render_text_image(f"Metric '{metric}' not found in CSV columns.")

        x = df["timestamp"]
        y = pd.to_numeric(df[metric], errors="coerce")

        dpi = 120
        fig = plt.figure(figsize=(9, 5), dpi=dpi)
        ax = fig.add_axes([0.10, 0.18, 0.86, 0.72])
        ax.plot(x, y, marker="o")
        ax.set_title(f"{metric} trend in folder {self.current_folder}")
        ax.set_xlabel("Time")
        ax.set_ylabel(metric)
        fig.autofmt_xdate(rotation=25)

        if self.current_path is not None:
            p_abs = os.path.normpath(self.current_path)
            dfp = df["path"].astype(str).apply(lambda s: os.path.normpath(s))
            hit = df[dfp == p_abs]
            if hit.empty:
                hit = df[df["path"].astype(str).str.endswith(os.path.basename(self.current_path))]
            if not hit.empty:
                row = hit.iloc[0]
                try:
                    ax.scatter([row["timestamp"]], [float(row[metric])], s=100)
                    ax.text(row["timestamp"], float(row[metric]), "  current", va="center")
                except Exception:
                    pass

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    # ---------- Draw left ----------
    def _redraw_left(self):
        if self.current_img_bgr is None:
            return
        rgb = cv2.cvtColor(self.current_img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        panel_w = self.left_canvas.winfo_width()
        panel_h = self.left_canvas.winfo_height()
        if panel_w <= 2 or panel_h <= 2:
            return

        disp_w, disp_h = self._fit_size(pil.size, (panel_w, panel_h))
        pil_resized = pil.resize((disp_w, disp_h), Image.BILINEAR)

        x0 = (panel_w - disp_w) // 2
        y0 = (panel_h - disp_h) // 2

        canvas_img = Image.new("RGB", (panel_w, panel_h), (240, 240, 240))
        canvas_img.paste(pil_resized, (x0, y0))

        tk_img = ImageTk.PhotoImage(canvas_img)
        self._tk_left_img = tk_img

        if self._left_canvas_img_id is None:
            self._left_canvas_img_id = self.left_canvas.create_image(0, 0, anchor="nw", image=tk_img)
        else:
            self.left_canvas.itemconfig(self._left_canvas_img_id, image=tk_img)

    # ---------- Draw right ----------
    def _redraw_right(self):
        mode = self._get_mode()
        self._apply_view_dependent_ui()

        if mode == "csv_bar_current":
            pil = self._build_csv_bar_pil()
            self._render_right_fit(pil)
            self._right_disp_info = None
            return

        if mode == "csv_trend_folder":
            pil = self._build_csv_trend_pil()
            self._render_right_fit(pil)
            self._right_disp_info = None
            return

        if self.current_img_bgr is None:
            pil = render_text_image("No image loaded.\nClick 'Show' first.")
            self._render_right_fit(pil)
            self._right_disp_info = None
            return

        img_bgr = self.current_img_bgr
        prob = self.current_prob
        overlay_bgr = self.current_overlay_bgr

        if mode == "prob_overlay":
            base_bgr = overlay_bgr if overlay_bgr is not None else img_bgr
        elif mode == "prob_heatmap":
            base_bgr = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_JET) if prob is not None else img_bgr
        elif mode == "right_original":
            base_bgr = img_bgr
        elif mode == "right_gray":
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            base_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif mode == "right_edges":
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 40, 120)
            base_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif mode == "right_laplacian":
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_32F)
            lap_abs = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
            base_bgr = cv2.cvtColor(lap_abs, cv2.COLOR_GRAY2BGR)
        else:
            base_bgr = overlay_bgr if overlay_bgr is not None else img_bgr

        draw_bgr = base_bgr.copy()

        if prob is not None and self.probes:
            H, W = prob.shape[:2]
            for (x, y) in self.probes:
                x = int(np.clip(x, 0, W - 1))
                y = int(np.clip(y, 0, H - 1))
                cv2.drawMarker(
                    draw_bgr, (x, y), (255, 255, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=16,
                    thickness=2,
                    line_type=cv2.LINE_AA
                )

        rgb = cv2.cvtColor(draw_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        self._render_right_true_zoom(pil, orig_wh=(draw_bgr.shape[1], draw_bgr.shape[0]))
        self.zoom_label.config(text="Zoom: auto-fit" if self.zoom is None else f"Zoom: {self.zoom:.2f}x")

    def _render_right_fit(self, pil: Image.Image):
        panel_w = self.right_canvas.winfo_width()
        panel_h = self.right_canvas.winfo_height()
        if panel_w <= 2 or panel_h <= 2:
            return

        disp_w, disp_h = self._fit_size(pil.size, (panel_w, panel_h))
        pil_resized = pil.resize((disp_w, disp_h), Image.BILINEAR)
        x0 = (panel_w - disp_w) // 2
        y0 = (panel_h - disp_h) // 2

        canvas_img = Image.new("RGB", (panel_w, panel_h), (240, 240, 240))
        canvas_img.paste(pil_resized, (x0, y0))

        tk_img = ImageTk.PhotoImage(canvas_img)
        self._tk_right_img = tk_img

        if self._right_canvas_img_id is None:
            self._right_canvas_img_id = self.right_canvas.create_image(0, 0, anchor="nw", image=tk_img)
        else:
            self.right_canvas.itemconfig(self._right_canvas_img_id, image=tk_img)

    # Right image: auto-fit initially, then true zoom (center crop/pad). Keeps probe mapping correct.
    def _render_right_true_zoom(self, pil: Image.Image, orig_wh: Tuple[int, int]):
        panel_w = self.right_canvas.winfo_width()
        panel_h = self.right_canvas.winfo_height()
        if panel_w <= 2 or panel_h <= 2:
            return

        W, H = orig_wh

        # ✅ auto-fit once when zoom is None (do NOT clamp to zoom_min here)
        if self.zoom is None:
            fit = min(panel_w / max(1, W), panel_h / max(1, H))
            fit = float(min(fit, self.zoom_max))
            self.zoom = fit
            self.zoom_fit = fit  # ✅ record per-image auto-fit floor (can be < zoom_min)

        disp_w = max(1, int(W * float(self.zoom)))
        disp_h = max(1, int(H * float(self.zoom)))
        pil_resized = pil.resize((disp_w, disp_h), Image.BILINEAR)

        if disp_w >= panel_w:
            crop_x = (disp_w - panel_w) // 2
            region_w = panel_w
            x0 = 0
        else:
            crop_x = 0
            region_w = disp_w
            x0 = (panel_w - disp_w) // 2

        if disp_h >= panel_h:
            crop_y = (disp_h - panel_h) // 2
            region_h = panel_h
            y0 = 0
        else:
            crop_y = 0
            region_h = disp_h
            y0 = (panel_h - disp_h) // 2

        region = pil_resized.crop((crop_x, crop_y, crop_x + region_w, crop_y + region_h))
        canvas_img = Image.new("RGB", (panel_w, panel_h), (240, 240, 240))
        canvas_img.paste(region, (x0, y0))

        self._right_disp_info = (x0, y0, region_w, region_h, crop_x, crop_y, disp_w, disp_h, W, H)

        tk_img = ImageTk.PhotoImage(canvas_img)
        self._tk_right_img = tk_img
        if self._right_canvas_img_id is None:
            self._right_canvas_img_id = self.right_canvas.create_image(0, 0, anchor="nw", image=tk_img)
        else:
            self.right_canvas.itemconfig(self._right_canvas_img_id, image=tk_img)

    # ---------- strict probe click ----------
    def _on_right_click(self, event: tk.Event):
        mode = self._get_mode()

        # Chart views: STRICT (no Cloud_Prob, no markers)
        if mode in self.CHART_VIEWS:
            feats = self._get_feats_prefer_csv()
            metric_name = self.chart_metric_var.get()
            if feats is None:
                self._append_log("[Chart view] Click ignored: no image loaded.")
                return

            metric_val = feats.get(metric_name, None)
            if metric_val is None:
                self._append_log(f"[Chart view] {mode} click -> metric '{metric_name}' not available.")
                self._insert_probe_row("-", "-", "N/A", metric_name, "N/A", mode)
            else:
                self._append_log(f"[Chart view] {mode} click -> {metric_name}(image-level) = {self._fmt6(metric_val)}")
                self._insert_probe_row("-", "-", "N/A", metric_name, self._fmt6(metric_val), mode)
            return

        # Image views: pixel probe + pixel metric
        if mode not in self.IMAGE_VIEWS:
            return
        if self.current_prob is None or self._right_disp_info is None or self.current_pixel_maps is None:
            return

        x0, y0, region_w, region_h, crop_x, crop_y, disp_w, disp_h, W, H = self._right_disp_info
        x = event.x
        y = event.y

        if x < x0 or x >= x0 + region_w or y < y0 or y >= y0 + region_h:
            return

        ux = crop_x + (x - x0)
        uy = crop_y + (y - y0)

        ox = int(np.clip(round(ux / max(1, disp_w - 1) * (W - 1)), 0, W - 1))
        oy = int(np.clip(round(uy / max(1, disp_h - 1) * (H - 1)), 0, H - 1))

        cloud_p = float(self.current_prob[oy, ox])

        metric_key = self.pixel_metric_var.get()
        metric_map = self.current_pixel_maps.get(metric_key, None)
        metric_val = None if metric_map is None else float(metric_map[oy, ox])

        self.probes.append((ox, oy))

        self._insert_probe_row(
            ox, oy,
            self._fmt6(cloud_p),
            metric_key,
            self._fmt6(metric_val),
            mode
        )

        self._append_log(
            f"[Image view] Probe (x={ox}, y={oy}) -> Cloud_Prob(pixel)={self._fmt6(cloud_p)}; "
            f"{metric_key}(pixel)={self._fmt6(metric_val)}"
        )

        self._redraw_right()

    # ---------- util ----------
    @staticmethod
    def _fit_size(src_size: Tuple[int, int], dst_box: Tuple[int, int]) -> Tuple[int, int]:
        sw, sh = src_size
        dw, dh = dst_box
        if sw <= 0 or sh <= 0:
            return (1, 1)
        scale = min(dw / sw, dh / sh)
        return (max(1, int(sw * scale)), max(1, int(sh * scale)))


# =========================
# 入口：只改这里
# =========================
if __name__ == "__main__":
    ROOT_DIR = r"D:\\Courses\\芯片发电\\data_clean\\Figures"
    YEAR = 2025
    CSV_PATH = os.path.join(ROOT_DIR, "sky_features_all.csv")  # 或 None

    root = tk.Tk()
    app = SkyGUI(root, root_dir=ROOT_DIR, year=YEAR, csv_path=CSV_PATH)
    root.mainloop()
