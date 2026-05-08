import os
import sys
import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, QObject, QThread, QTimer, pyqtSignal, pyqtSlot, QRectF
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QDoubleSpinBox,
    QCheckBox, QMessageBox, QListWidget, QListWidgetItem,
    QAbstractItemView, QTabWidget, QSpinBox, QSizePolicy, QProgressBar,
    QScrollArea, QGridLayout, QFrame, QToolButton, QLineEdit,
    QDialog, QDialogButtonBox, QFormLayout, QColorDialog,
    QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt5.QtGui import QPixmap, QPainter, QColor, QIcon
# --- PyQtGraph: interactive / fast plotting ---
try:
    import pyqtgraph as pg
    _HAVE_PG = True
except Exception:
    pg = None
    _HAVE_PG = False

try:
    import pyqtgraph.opengl as gl
    _HAVE_PG_GL = True
except Exception:
    gl = None
    _HAVE_PG_GL = False

# --- Matplotlib: export/report only ---
import matplotlib
import matplotlib.colors as mcolors
matplotlib.use("Agg")
matplotlib.rcParams["path.simplify"] = True
matplotlib.rcParams["path.simplify_threshold"] = 1.0
matplotlib.rcParams["agg.path.chunksize"] = 10000
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# --- TDMS (npTDMS) ---
try:
    from nptdms import TdmsFile
    _HAVE_TDMS = True
except Exception:
    TdmsFile = None
    _HAVE_TDMS = False

# --- nptdms write support ---
try:
    from nptdms import TdmsWriter, RootObject as TdmsRootObject, GroupObject as TdmsGroupObject, ChannelObject as TdmsChannelObject
    _HAVE_TDMS_WRITE = True
except Exception:
    TdmsWriter = None
    _HAVE_TDMS_WRITE = False


# --- SciPy (Butterworth + zero-phase) ---
try:
    from scipy.signal import butter, sosfiltfilt
    _HAVE_SCIPY = True
except Exception:
    butter = None
    sosfiltfilt = None
    _HAVE_SCIPY = False

# --- Optional SciPy interpolation for higher-quality waterfall rendering ---
try:
    from scipy.interpolate import PchipInterpolator, interp1d
    _HAVE_SCIPY_INTERP = True
except Exception:
    PchipInterpolator = None
    interp1d = None
    _HAVE_SCIPY_INTERP = False

if _HAVE_PG:
    pg.setConfigOptions(antialias=False, imageAxisOrder="row-major")

G0 = 9.80665           # m/s^2 per g
VEL_SCALE = 1000.0     # m/s -> mm/s
VEL_UNIT = "mm/s"


def get_amp_conv_factor_label(amp_conv: str, signal_kind: str):
    sk = str(signal_kind).strip().lower()
    unit = VEL_UNIT if sk == "vel" else "g"
    sel = str(amp_conv).strip().lower()

    if sel == "peak":
        return 1.0, f"{unit} (peak)"
    if sel == "rms":
        return 1.0 / np.sqrt(2.0), f"{unit} RMS"
    if sel in ("peak-to-peak", "peak to peak", "p-p", "pk-pk", "peak_to_peak"):
        return 2.0, f"{unit} p-p"

    return 1.0, f"{unit} (peak)"


def build_spectrum_ylabel(output_type: str, scale: str, unit_lbl: str) -> str:
    out = str(output_type).strip() or "Amplitude"
    scl = str(scale).strip().lower()
    if scl == "db":
        return f"{out} ({unit_lbl}) [dB]"
    return f"{out} ({unit_lbl})"


def compute_avg_spectrum_from_signal(
    y: np.ndarray,
    fs: float,
    nfft: int,
    overlap_pct: float,
    output_type: str,
    scale: str,
    amp_conv: str,
    signal_kind: str,
):
    y = np.nan_to_num(np.asarray(y, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    fs = float(fs)
    nfft = max(2, int(nfft))

    if y.size == 0 or not np.isfinite(fs) or fs <= 0.0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    ovl = np.clip(float(overlap_pct) / 100.0, 0.0, 0.95)
    hop = max(1, int(round(nfft * (1.0 - ovl))))

    w = np.hanning(nfft).astype(np.float64)
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        w_sum = float(nfft)

    if y.size < nfft:
        n_frames = 1
    else:
        n_frames = max(1, 1 + (y.size - nfft) // hop)

    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    spec_acc = np.zeros(freqs.size, dtype=np.float64)
    conv, _ = get_amp_conv_factor_label(amp_conv, signal_kind)
    out_type = str(output_type).strip()

    for i in range(n_frames):
        start = i * hop
        frame = np.zeros(nfft, dtype=np.float64)
        take = max(0, min(nfft, y.size - start))
        if take > 0:
            frame[:take] = y[start:start + take]

        X = np.fft.rfft(frame * w, n=nfft)
        mag = np.abs(X)
        A_peak = (2.0 * mag) / w_sum
        A_peak[0] *= 0.5
        if (nfft % 2 == 0) and (A_peak.size > 1):
            A_peak[-1] *= 0.5

        A = A_peak * conv
        P = A if out_type == "Amplitude" else (A * A)
        spec_acc += P

    spec = spec_acc / max(n_frames, 1)

    if str(scale).strip().lower() == "db":
        eps = 1e-20
        spec = 20.0 * np.log10(spec + eps) if out_type == "Amplitude" else 10.0 * np.log10(spec + eps)

    return freqs, spec

def resolve_color_limits(S: np.ndarray, scale: str, color_cfg: dict):
    S = np.asarray(S, dtype=np.float64)
    finite = S[np.isfinite(S)]

    auto_vmin = float(np.min(finite)) if finite.size else 0.0
    auto_vmax = float(np.max(finite)) if finite.size else 1.0

    if auto_vmax <= auto_vmin:
        auto_vmax = auto_vmin + 1e-12

    manual = bool(color_cfg.get("manual_clim", False))
    dyn_range_db = max(1e-9, float(color_cfg.get("dyn_range_db", 80.0)))
    is_db = str(scale).strip().lower() == "db"

    if manual:
        vmin = float(color_cfg.get("vmin", auto_vmin))
        vmax = float(color_cfg.get("vmax", auto_vmax))
    elif is_db:
        vmax = auto_vmax
        vmin = vmax - dyn_range_db
    else:
        vmin = auto_vmin
        vmax = auto_vmax

    if not np.isfinite(vmin):
        vmin = auto_vmin
    if not np.isfinite(vmax):
        vmax = auto_vmax

    if vmax <= vmin:
        vmax = vmin + 1e-12

    return vmin, vmax, auto_vmin, auto_vmax


def get_pg_colormap(name: str, reverse: bool = False):
    desired = f"{name}_r" if reverse else name
    if not _HAVE_PG:
        return None
    try:
        return pg.colormap.get(desired, source="matplotlib")
    except Exception:
        try:
            return pg.colormap.get(desired)
        except Exception:
            fallback = "viridis_r" if reverse else "viridis"
            try:
                return pg.colormap.get(fallback, source="matplotlib")
            except Exception:
                return pg.colormap.get("viridis")


def qcolor_to_hex(color) -> str:
    qc = QColor(color)
    if not qc.isValid():
        qc = QColor("#ffffff")
    return qc.name().upper()


def default_custom_color_bands():
    return [
        {"min": 0.0, "max": 200.0, "color": "#0000FF"},
        {"min": 200.0, "max": 500.0, "color": "#00FF00"},
        {"min": 500.0, "max": 800.0, "color": "#FFFF00"},
        {"min": 800.0, "max": 1000.0, "color": "#FF0000"},
    ]


def normalize_custom_bands(bands):
    if not bands:
        return []

    out = []
    for i, band in enumerate(bands):
        try:
            lo = float(band.get("min"))
            hi = float(band.get("max"))
        except Exception:
            raise ValueError(f"Band {i + 1}: invalid numeric limits.")

        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"Band {i + 1}: limits must be finite.")
        if hi <= lo:
            raise ValueError(f"Band {i + 1}: max must be greater than min.")

        color_hex = qcolor_to_hex(band.get("color", "#FFFFFF"))
        out.append({"min": lo, "max": hi, "color": color_hex})

    out.sort(key=lambda b: (b["min"], b["max"]))
    for i in range(1, len(out)):
        prev = out[i - 1]
        cur = out[i]
        if cur["min"] < prev["max"] - 1e-12:
            raise ValueError("Custom color bands must not overlap.")

    return out


def build_segmented_pg_colormap(bands):
    bands = normalize_custom_bands(bands)
    if not bands:
        return get_pg_colormap("viridis", reverse=False)

    low = float(bands[0]["min"])
    high = float(bands[-1]["max"])
    span = max(high - low, 1e-12)

    pos = []
    colors = []
    for band in bands:
        color = QColor(band["color"])
        rgba = (color.red(), color.green(), color.blue(), 255)
        p0 = float(np.clip((band["min"] - low) / span, 0.0, 1.0))
        p1 = float(np.clip((band["max"] - low) / span, 0.0, 1.0))
        pos.extend([p0, p1])
        colors.extend([rgba, rgba])

    if len(pos) < 2 or abs(pos[-1] - pos[0]) < 1e-12:
        pos = [0.0, 1.0]
        color = QColor(bands[0]["color"])
        rgba = (color.red(), color.green(), color.blue(), 255)
        colors = [rgba, rgba]

    return pg.ColorMap(np.asarray(pos, dtype=np.float64), colors)


def resolve_image_style(S: np.ndarray, scale: str, color_cfg: dict):
    color_mode = str(color_cfg.get("color_mode", "Continuous")).strip()
    reverse = bool(color_cfg.get("reverse", False))

    if color_mode == "Custom Bands":
        bands = normalize_custom_bands(color_cfg.get("custom_bands", []))
        if bands:
            vmin = float(bands[0]["min"])
            vmax = float(bands[-1]["max"])
            if vmax <= vmin:
                vmax = vmin + 1e-12
            return {
                "mode": "bands",
                "bands": bands,
                "vmin": vmin,
                "vmax": vmax,
                "cm": build_segmented_pg_colormap(bands),
                "interactive": False,
            }

    vmin, vmax, auto_vmin, auto_vmax = resolve_color_limits(S, scale, color_cfg)
    return {
        "mode": "continuous",
        "bands": [],
        "vmin": vmin,
        "vmax": vmax,
        "auto_vmin": auto_vmin,
        "auto_vmax": auto_vmax,
        "cm": get_pg_colormap(color_cfg.get("cmap", "viridis"), reverse=reverse),
        "interactive": bool(color_cfg.get("manual_clim", False)),
    }


def resolve_mpl_image_style(S: np.ndarray, scale: str, color_cfg: dict):
    color_mode = str(color_cfg.get("color_mode", "Continuous")).strip()
    reverse = bool(color_cfg.get("reverse", False))

    if color_mode == "Custom Bands":
        bands = normalize_custom_bands(color_cfg.get("custom_bands", []))
        if bands:
            colors = [band["color"] for band in bands]
            boundaries = [bands[0]["min"]] + [band["max"] for band in bands]
            cmap = mcolors.ListedColormap(colors)
            norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
            return {
                "mode": "bands",
                "bands": bands,
                "boundaries": boundaries,
                "cmap": cmap,
                "norm": norm,
            }

    vmin, vmax, auto_vmin, auto_vmax = resolve_color_limits(S, scale, color_cfg)
    cmap_name = color_cfg.get("cmap", "viridis")
    cmap = matplotlib.colormaps.get_cmap(f"{cmap_name}_r" if reverse else cmap_name)
    return {
        "mode": "continuous",
        "vmin": vmin,
        "vmax": vmax,
        "auto_vmin": auto_vmin,
        "auto_vmax": auto_vmax,
        "cmap": cmap,
        "norm": None,
    }


def make_colormap_preview_icon(name: str, width: int = 72, height: int = 14):
    pix = QPixmap(width, height)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    try:
        cmap = matplotlib.colormaps.get_cmap(name)
    except Exception:
        cmap = matplotlib.colormaps.get_cmap("viridis")

    for x in range(width):
        frac = x / max(width - 1, 1)
        rgba = cmap(frac)
        painter.setPen(QColor(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)))
        painter.drawLine(x, 0, x, height)

    painter.setPen(QColor(80, 80, 80))
    painter.drawRect(0, 0, width - 1, height - 1)
    painter.end()
    return QIcon(pix)

def _read_csv_df(path: str, nrows=None):
    kwargs = {"sep": None, "engine": "python"}
    if nrows is not None:
        kwargs["nrows"] = nrows
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        kwargs.pop("sep", None)
        kwargs.pop("engine", None)
        return pd.read_csv(path, **kwargs)


def _series_to_numeric_array(series) -> np.ndarray:
    s = series if isinstance(series, pd.Series) else pd.Series(series)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64, copy=False)

    s2 = s.astype(str).str.strip().str.replace("\u00a0", "", regex=False)
    has_comma = s2.str.contains(",", regex=False, na=False).any()
    has_dot = s2.str.contains(".", regex=False, na=False).any()
    if has_comma and not has_dot:
        s2 = s2.str.replace(",", ".", regex=False)
    return pd.to_numeric(s2, errors="coerce").to_numpy(dtype=np.float64, copy=False)


def detect_file_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return "excel"
    if ext == ".tdms":
        return "tdms"
    if ext == ".csv":
        return "csv"
    raise RuntimeError("Unsupported file extension.")


def read_source_container_names(path: str, file_type: str):
    if file_type == "excel":
        xls = pd.ExcelFile(path)
        names = list(xls.sheet_names)
        if not names:
            raise RuntimeError("No sheets found in the Excel file.")
        return names

    if file_type == "tdms":
        if not _HAVE_TDMS:
            raise RuntimeError("npTDMS is not available. Install with: pip install npTDMS")
        tdms = TdmsFile.read_metadata(path) if hasattr(TdmsFile, "read_metadata") else TdmsFile.read(path)
        names = [g.name for g in tdms.groups()]
        if not names:
            raise RuntimeError("No groups found in the TDMS file.")
        return names

    if file_type == "csv":
        return ["CSV Data"]

    raise RuntimeError("Unknown file type.")


def read_source_signal_names(path: str, file_type: str, container_name: str):
    if file_type == "excel":
        df0 = pd.read_excel(path, sheet_name=container_name, header=0, nrows=0)
        cols = list(df0.columns)
        if not cols:
            raise RuntimeError("No columns found in the selected sheet.")
        return [str(c) for c in cols]

    if file_type == "tdms":
        if not _HAVE_TDMS:
            raise RuntimeError("npTDMS is not available. Install with: pip install npTDMS")
        tdms = TdmsFile.read_metadata(path) if hasattr(TdmsFile, "read_metadata") else TdmsFile.read(path)
        group = tdms[container_name]
        ch_names = [ch.name for ch in group.channels()]
        if not ch_names:
            raise RuntimeError("No channels found in the TDMS file.")
        return ch_names

    if file_type == "csv":
        df0 = _read_csv_df(path, nrows=0)
        cols = list(df0.columns)
        if not cols:
            raise RuntimeError("No columns found in the CSV file.")
        return [str(c) for c in cols]

    raise RuntimeError("Unknown file type.")


def read_signal_from_source(path: str, file_type: str, container_name: str, signal_name: str) -> np.ndarray:
    if file_type == "excel":
        df = pd.read_excel(path, sheet_name=container_name, header=0)
        if signal_name not in df.columns:
            raise KeyError(f"Column not found: {signal_name}")
        return _series_to_numeric_array(df[signal_name])

    if file_type == "tdms":
        if not _HAVE_TDMS:
            raise RuntimeError("npTDMS is not available. Install with: pip install npTDMS")
        tdms = TdmsFile.read(path)
        group = tdms[container_name]
        if signal_name not in group:
            raise KeyError(f"Channel not found in group '{container_name}': {signal_name}")
        return np.asarray(group[signal_name][:], dtype=np.float64)

    if file_type == "csv":
        df = _read_csv_df(path)
        if signal_name not in df.columns:
            raise KeyError(f"Column not found: {signal_name}")
        return _series_to_numeric_array(df[signal_name])

    raise RuntimeError("Unknown file type.")


class InteractivePlotWidget(QWidget):
    clicked = pyqtSignal()
    color_levels_changed = pyqtSignal(float, float)
    color_levels_change_finished = pyqtSignal(float, float)

    def __init__(self, mini=False):
        super().__init__()
        self.mini = bool(mini)
        self.plot_state = None
        self.plot_key = None

        self._plot_widget = None
        self._plot_item = None
        self._view_box = None
        self._mouse_proxy = None
        self._home_range = None
        self._rect_zoom_active = False
        self._image_item = None
        self._color_bar = None
        self._suspend_colorbar_sync = False

        self.enable_click_annotations = not self.mini
        self._click_annotations = []

        self.hover_label = QLabel(self)
        self.hover_label.setVisible(False)
        self.hover_label.setStyleSheet(
            "background: rgba(255,255,255,230); border: 1px solid #666; padding: 2px 4px;"
        )

        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(0, 0, 0, 0)

    def clear_plot(self):
        self._clear_click_annotations()

        self.plot_state = None
        self.plot_key = None
        self._home_range = None
        self._image_item = None
        self._color_bar = None
        self.hover_label.setVisible(False)

        if self._plot_widget is not None:
            try:
                if self._plot_item is not None:
                    self._plot_item.clear()
            except Exception:
                pass
            self.root.removeWidget(self._plot_widget)
            self._plot_widget.deleteLater()
            self._plot_widget = None
            self._plot_item = None
            self._view_box = None
            self._mouse_proxy = None

    def _create_plot_widget(self):
        pw = pg.PlotWidget(background="w")
        pw.setMenuEnabled(False)
        pw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        pi = pw.getPlotItem()
        vb = pi.getViewBox()
        vb.setMouseMode(pg.ViewBox.RectMode if self._rect_zoom_active else pg.ViewBox.PanMode)
        vb.enableAutoRange(pg.ViewBox.XYAxes, False)
        vb.sigRangeChanged.connect(self._update_annotation_buttons)
        pi.showGrid(x=True, y=True, alpha=0.25)
        self.root.addWidget(pw)

        pw.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self._mouse_proxy = pg.SignalProxy(
            pw.scene().sigMouseMoved,
            rateLimit=35,
            slot=self._on_mouse_moved,
        )

        self._plot_widget = pw
        self._plot_item = pi
        self._view_box = vb

    def set_rect_zoom_active(self, active: bool):
        self._rect_zoom_active = bool(active)
        if self._view_box is not None:
            self._view_box.setMouseMode(pg.ViewBox.RectMode if active else pg.ViewBox.PanMode)

    def get_view_box(self):
        return self._view_box

    def get_view_range(self):
        if self._view_box is None:
            return None
        try:
            xr, yr = self._view_box.viewRange()
            return (float(xr[0]), float(xr[1])), (float(yr[0]), float(yr[1]))
        except Exception:
            return None

    def _cursor_top_y(self):
        span = max(self.rpm_max - self.rpm_min, 1e-12)
        return self.rpm_max - 0.035 * span

    def _cursor_label_x(self, x_value: float) -> float:
        span = max(self.freq_max - self.freq_min, 1e-12)
        return float(np.clip(x_value, self.freq_min + 0.02 * span, self.freq_max - 0.02 * span))

    def _on_upper_colorbar_levels_changed(self, *_):
        if self._suspend_colorbar_sync or self._upper_color_bar is None:
            return
        try:
            lo, hi = self._upper_color_bar.levels()
            self.color_levels_changed.emit(float(lo), float(hi))
        except Exception:
            pass

    def _on_upper_colorbar_levels_change_finished(self, *_):
        if self._suspend_colorbar_sync or self._upper_color_bar is None:
            return
        try:
            lo, hi = self._upper_color_bar.levels()
            self.color_levels_change_finished.emit(float(lo), float(hi))
        except Exception:
            pass

    def _on_mouse_clicked(self, event):
        if self._plot_widget is None or self._view_box is None:
            return

        try:
            if event.button() == Qt.LeftButton:
                self.clicked.emit()

                if self._rect_zoom_active or not self.enable_click_annotations:
                    return

                pos = event.scenePos()
                if not self._view_box.sceneBoundingRect().contains(pos):
                    return

                mp = self._view_box.mapSceneToView(pos)
                self._add_click_annotation(float(mp.x()), float(mp.y()))
        except Exception:
            self.clicked.emit()

    def _on_mouse_moved(self, evt):
        if self._plot_widget is None or self._view_box is None:
            return

        pos = evt[0] if isinstance(evt, tuple) else evt
        if pos is None:
            self.hover_label.setVisible(False)
            return

        try:
            scene_rect = self._plot_widget.sceneBoundingRect()
            if not scene_rect.contains(pos):
                self.hover_label.setVisible(False)
                return

            mp = self._view_box.mapSceneToView(pos)
            text = f"({mp.x():.6g}, {mp.y():.6g})"
            self.hover_label.setText(text)
            self.hover_label.adjustSize()

            local = self.mapFromGlobal(self.cursor().pos())
            x = min(max(0, local.x() + 14), max(0, self.width() - self.hover_label.width() - 4))
            y = min(max(0, local.y() + 14), max(0, self.height() - self.hover_label.height() - 4))
            self.hover_label.move(x, y)
            self.hover_label.setVisible(True)
            self.hover_label.raise_()
        except Exception:
            self.hover_label.setVisible(False)

    def _clear_click_annotations(self):
        for ann in self._click_annotations[:]:
            self._remove_click_annotation(ann)

    def _remove_click_annotation(self, ann):
        try:
            if self._plot_item is not None and ann.get("marker") is not None:
                self._plot_item.removeItem(ann["marker"])
        except Exception:
            pass

        try:
            if self._plot_item is not None and ann.get("label") is not None:
                self._plot_item.removeItem(ann["label"])
        except Exception:
            pass

        try:
            btn = ann.get("close_btn")
            if btn is not None:
                btn.hide()
                btn.deleteLater()
        except Exception:
            pass

        try:
            self._click_annotations.remove(ann)
        except ValueError:
            pass

    def _add_click_annotation(self, x: float, y: float):
        if self._plot_item is None or self._plot_widget is None:
            return

        marker = pg.ScatterPlotItem(
            [x], [y],
            symbol="o",
            size=8,
            pen=pg.mkPen((200, 40, 40), width=1.2),
            brush=pg.mkBrush(255, 255, 255, 220),
        )
        marker.setZValue(1000)

        html = (
            "<div style='background: rgba(255,255,255,230); "
            "border: 1px solid #666; padding: 2px 6px;'>"
            f"({x:.6g}, {y:.6g})"
            "</div>"
        )
        label = pg.TextItem(html=html, anchor=(0, 1))
        label.setPos(x, y)
        label.setZValue(1001)

        self._plot_item.addItem(marker)
        self._plot_item.addItem(label)

        btn = QToolButton(self._plot_widget.viewport())
        btn.setText("\u00d7")
        btn.setAutoRaise(True)
        btn.setFixedSize(16, 16)
        btn.setStyleSheet(
            "QToolButton {"
            "background: rgba(255,255,255,230);"
            "border: 1px solid #666;"
            "border-radius: 8px;"
            "font-weight: bold;"
            "padding: 0px;"
            "}"
        )

        ann = {
            "x": float(x),
            "y": float(y),
            "marker": marker,
            "label": label,
            "close_btn": btn,
        }

        btn.clicked.connect(lambda _=False, a=ann: self._remove_click_annotation(a))

        self._click_annotations.append(ann)
        btn.show()
        self._update_single_annotation_button(ann)

    def _update_single_annotation_button(self, ann):
        if self._plot_widget is None:
            return

        btn = ann.get("close_btn")
        label = ann.get("label")
        if btn is None or label is None:
            return

        try:
            sr = label.sceneBoundingRect()
            vp_pt = self._plot_widget.mapFromScene(sr.topRight())

            x = int(vp_pt.x()) - btn.width() // 2
            y = int(vp_pt.y()) - btn.height() // 2
            btn.move(x, y)
            btn.raise_()

            viewport_rect = self._plot_widget.viewport().rect()
            btn.setVisible(viewport_rect.adjusted(-20, -20, 20, 20).contains(vp_pt))
        except Exception:
            btn.setVisible(False)

    def _update_annotation_buttons(self, *args):
        for ann in self._click_annotations[:]:
            self._update_single_annotation_button(ann)

    def clear_click_annotations(self):
        self._clear_click_annotations()

    def get_export_annotations(self):
        out = []
        for ann in self._click_annotations:
            try:
                x = float(ann["x"])
                y = float(ann["y"])
                if np.isfinite(x) and np.isfinite(y):
                    out.append({
                        "x": x,
                        "y": y,
                        "text": f"({x:.6g}, {y:.6g})"
                    })
            except Exception:
                pass
        return out

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_annotation_buttons()

    def _set_home_range(self, x0, x1, y0, y1, padding=0.02):
        if not np.isfinite([x0, x1, y0, y1]).all():
            self._home_range = None
            return

        if x0 == x1:
            x0 -= 0.5
            x1 += 0.5
        if y0 == y1:
            y0 -= 0.5
            y1 += 0.5

        dx = x1 - x0
        dy = y1 - y0
        self._home_range = (
            x0 - padding * dx,
            x1 + padding * dx,
            y0 - padding * dy,
            y1 + padding * dy,
        )

    def store_home_view(self):
        if self._view_box is None:
            return
        try:
            xr, yr = self._view_box.viewRange()
            self._home_range = (xr[0], xr[1], yr[0], yr[1])
        except Exception:
            pass

    def reset_view(self):
        if self._view_box is None or self._home_range is None:
            return
        x0, x1, y0, y1 = self._home_range
        self._view_box.setXRange(x0, x1, padding=0.0)
        self._view_box.setYRange(y0, y1, padding=0.0)
        self.clear_click_annotations()

    def zoom_axes(self, factor: float):
        if self._view_box is None:
            return
        try:
            self._view_box.scaleBy((factor, factor))
        except Exception:
            pass

    @staticmethod
    def _safe_pen():
        return pg.mkPen((35, 110, 210), width=1)

    @staticmethod
    def _state_y_bounds(state: dict):
        if state["kind"] in ("time", "avg"):
            all_y = []
            for s in state.get("series", []):
                y = np.asarray(s["y"], dtype=np.float64)
                y = y[np.isfinite(y)]
                if y.size:
                    all_y.append((float(np.min(y)), float(np.max(y))))
            if not all_y:
                return -1.0, 1.0
            return min(v[0] for v in all_y), max(v[1] for v in all_y)

        S = np.asarray(state["S"], dtype=np.float64)
        S = S[np.isfinite(S)]
        if S.size == 0:
            return 0.0, 1.0
        return float(np.min(S)), float(np.max(S))

    @staticmethod
    def _pg_colormap(name: str, reverse: bool = False):
        return get_pg_colormap(name, reverse=reverse)

    def _on_colorbar_levels_changed(self, *_):
        if self._suspend_colorbar_sync or self._color_bar is None:
            return
        try:
            lo, hi = self._color_bar.levels()
            self.color_levels_changed.emit(float(lo), float(hi))
        except Exception:
            pass

    def _on_colorbar_levels_change_finished(self, *_):
        if self._suspend_colorbar_sync or self._color_bar is None:
            return
        try:
            lo, hi = self._color_bar.levels()
            self.color_levels_change_finished.emit(float(lo), float(hi))
        except Exception:
            pass

    def set_state(self, state: dict, color_cfg=None, preserve_view=False, preserve_annotations=False):
        old_range = None
        old_annotations = self.get_export_annotations() if preserve_annotations else []

        if preserve_view and self._view_box is not None:
            try:
                xr, yr = self._view_box.viewRange()
                old_range = (xr[0], xr[1], yr[0], yr[1])
            except Exception:
                old_range = None

        self.clear_plot()
        self._create_plot_widget()

        self.plot_state = state
        self.plot_key = state["key"]

        self._plot_item.setLabel("bottom", state.get("xlabel", ""))
        self._plot_item.setLabel("left", state.get("ylabel", ""))

        if state["kind"] in ("time", "avg"):
            self._draw_line_state(state)
        else:
            self._draw_image_state(state, color_cfg or {})

        if old_range is not None:
            x0, x1, y0, y1 = old_range
            self._view_box.setXRange(x0, x1, padding=0.0)
            self._view_box.setYRange(y0, y1, padding=0.0)
        else:
            self.reset_view()

        if old_annotations:
            for ann in old_annotations:
                self._add_click_annotation(ann["x"], ann["y"])
            self._update_annotation_buttons()

    def _draw_line_state(self, state: dict):
        x_bounds = []
        y_bounds = []

        for series in state.get("series", []):
            x = np.asarray(series["x"], dtype=np.float64)
            y = np.asarray(series["y"], dtype=np.float64)

            if self.mini and x.size > 3500:
                step = max(1, x.size // 3500)
                x = x[::step]
                y = y[::step]

            item = pg.PlotDataItem(
                x=x,
                y=y,
                pen=self._safe_pen(),
                antialias=False,
                connect="finite",
            )
            try:
                item.setSkipFiniteCheck(True)
            except Exception:
                pass

            self._plot_item.addItem(item)

            x = x[np.isfinite(x)]
            y = y[np.isfinite(y)]
            if x.size:
                x_bounds.append((float(np.min(x)), float(np.max(x))))
            if y.size:
                y_bounds.append((float(np.min(y)), float(np.max(y))))

        if x_bounds and y_bounds:
            self._set_home_range(
                min(v[0] for v in x_bounds),
                max(v[1] for v in x_bounds),
                min(v[0] for v in y_bounds),
                max(v[1] for v in y_bounds),
                padding=0.02,
            )
        else:
            self._set_home_range(0.0, 1.0, -1.0, 1.0)

    def _draw_image_state(self, state: dict, color_cfg: dict):
        times = np.asarray(state["times"], dtype=np.float64)
        freqs = np.asarray(state["freqs"], dtype=np.float64)
        S = np.asarray(state["S"], dtype=np.float64)

        if times.size == 0 or freqs.size == 0 or S.size == 0:
            self._set_home_range(0.0, 1.0, 0.0, 1.0)
            return

        if times.size > 1:
            dx = float(np.median(np.diff(times)))
        else:
            dx = 1.0
        if freqs.size > 1:
            dy = float(np.median(np.diff(freqs)))
        else:
            dy = 1.0

        x0 = float(times[0] - 0.5 * dx)
        x1 = float(times[-1] + 0.5 * dx)
        y0 = float(freqs[0] - 0.5 * dy)
        y1 = float(freqs[-1] + 0.5 * dy)

        style = resolve_image_style(S, state.get("scale", "Linear"), color_cfg)

        image_item = pg.ImageItem(axisOrder="row-major")
        image_item.setImage(S, autoLevels=False)
        try:
            image_item.setOpts(interpolate=True)
        except Exception:
            pass
        image_item.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        image_item.setLevels((style["vmin"], style["vmax"]))
        image_item.setColorMap(style["cm"])
        self._plot_item.addItem(image_item)

        cbar = pg.ColorBarItem(
            values=(style["vmin"], style["vmax"]),
            colorMap=style["cm"],
            label=state.get("colorbar_label", ""),
            interactive=bool(style.get("interactive", False)),
            width=12 if self.mini else 18,
        )

        self._suspend_colorbar_sync = True
        cbar.setImageItem(image_item, insert_in=self._plot_item)
        self._suspend_colorbar_sync = False

        self._image_item = image_item
        self._color_bar = cbar

        if bool(style.get("interactive", False)):
            try:
                cbar.sigLevelsChanged.connect(self._on_colorbar_levels_changed)
            except Exception:
                pass
            try:
                cbar.sigLevelsChangeFinished.connect(self._on_colorbar_levels_change_finished)
            except Exception:
                pass

        self._set_home_range(x0, x1, y0, y1, padding=0.0)


class WaterfallSetupDialog(QDialog):
    def __init__(self, channel_names, parent=None, rpm_guess="", acc_guess="",
                 default_acc_fs=10000.0, default_rpm_fs=10000.0):
        super().__init__(parent)
        self.setWindowTitle("Waterfall setup")
        self.setModal(True)

        self.cmb_rpm = QComboBox()
        self.cmb_acc = QComboBox()

        self.spin_acc_fs = QDoubleSpinBox()
        self.spin_acc_fs.setRange(0.1, 1_000_000.0)
        self.spin_acc_fs.setDecimals(2)
        self.spin_acc_fs.setValue(default_acc_fs)

        self.spin_rpm_fs = QDoubleSpinBox()
        self.spin_rpm_fs.setRange(0.1, 1_000_000.0)
        self.spin_rpm_fs.setDecimals(2)
        self.spin_rpm_fs.setValue(default_rpm_fs)

        self.cmb_resample_mode = QComboBox()
        self.cmb_resample_mode.addItems([
            "Resample RPM -> Acc timebase",
            "Resample Acc -> RPM timebase",
            "Trim only (same fs)",
        ])
        self.cmb_resample_mode.setCurrentText("Resample RPM -> Acc timebase")

        self.cmb_interp = QComboBox()
        self.cmb_interp.addItems(["Linear", "Nearest"])
        self.cmb_interp.setCurrentText("Linear")

        self.cmb_filter_target = QComboBox()
        self.cmb_filter_target.addItems(["None", "Acceleration", "RPM", "Both"])
        self.chk_open_3d = QCheckBox("Open 3D waterfall in separate window after plotting")
        self.chk_open_3d.setChecked(False)

        self.cmb_rpm.addItems(channel_names)
        self.cmb_acc.addItems(channel_names)

        if rpm_guess and rpm_guess in channel_names:
            self.cmb_rpm.setCurrentText(rpm_guess)
        if acc_guess and acc_guess in channel_names:
            self.cmb_acc.setCurrentText(acc_guess)

        form = QFormLayout(self)
        form.addRow("RPM channel:", self.cmb_rpm)
        form.addRow("Acceleration channel:", self.cmb_acc)

        fs_row = QHBoxLayout()
        fs_row.addWidget(QLabel("Acc fs (Hz):"))
        fs_row.addWidget(self.spin_acc_fs)
        fs_row.addSpacing(14)
        fs_row.addWidget(QLabel("RPM fs (Hz):"))
        fs_row.addWidget(self.spin_rpm_fs)
        form.addRow("Sampling rates:", fs_row)

        resample_row = QHBoxLayout()
        resample_row.addWidget(self.cmb_resample_mode)
        resample_row.addSpacing(10)
        resample_row.addWidget(QLabel("Interpolation:"))
        resample_row.addWidget(self.cmb_interp)
        form.addRow("Resample mode:", resample_row)

        form.addRow("Apply Butterworth filter to:", self.cmb_filter_target)
        form.addRow("", self.chk_open_3d)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _accept(self):
        if self.cmb_rpm.currentText().strip() == self.cmb_acc.currentText().strip():
            QMessageBox.warning(self, "Selection", "RPM and acceleration channels must be different.")
            return
        self.accept()

    def values(self):
        return {
            "rpm_name": self.cmb_rpm.currentText().strip(),
            "acc_name": self.cmb_acc.currentText().strip(),
            "acc_fs": float(self.spin_acc_fs.value()),
            "rpm_fs": float(self.spin_rpm_fs.value()),
            "resample_mode": self.cmb_resample_mode.currentText().strip(),
            "resample_interp": self.cmb_interp.currentText().strip(),
            "filter_target": self.cmb_filter_target.currentText().strip(),
            "open_3d": self.chk_open_3d.isChecked(),
        }


class BandEditorDialog(QDialog):
    def __init__(self, bands=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Color Bands")
        self.resize(560, 360)

        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(["Min", "Max", "Color"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)

        self.btn_add = QPushButton("Add Band")
        self.btn_remove = QPushButton("Remove Selected")
        self.btn_pick = QPushButton("Pick Color")
        self.btn_reset = QPushButton("Reset Defaults")

        row_btn = QHBoxLayout()
        row_btn.addWidget(self.btn_add)
        row_btn.addWidget(self.btn_remove)
        row_btn.addWidget(self.btn_pick)
        row_btn.addWidget(self.btn_reset)
        row_btn.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Define numeric intervals and colors. Double-click a color cell to change it."))
        lay.addWidget(self.table, 1)
        lay.addLayout(row_btn)
        lay.addWidget(buttons)

        self.btn_add.clicked.connect(lambda: self._add_row())
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_pick.clicked.connect(self._pick_selected_color)
        self.btn_reset.clicked.connect(self._reset_defaults)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

        initial = bands if bands else default_custom_color_bands()
        for band in initial:
            self._add_row(band)

    def _set_color_item(self, row: int, color_text: str):
        color_text = qcolor_to_hex(color_text)
        item = self.table.item(row, 2)
        if item is None:
            item = QTableWidgetItem()
            self.table.setItem(row, 2, item)
        item.setText(color_text)
        item.setBackground(QColor(color_text))
        item.setForeground(QColor("#000000") if QColor(color_text).lightness() > 128 else QColor("#FFFFFF"))

    def _add_row(self, band=None):
        band = band or {"min": 0.0, "max": 1.0, "color": "#FFFFFF"}
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(band.get("min", 0.0))))
        self.table.setItem(row, 1, QTableWidgetItem(str(band.get("max", 1.0))))
        self._set_color_item(row, band.get("color", "#FFFFFF"))

    def _remove_selected(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

    def _pick_color_for_row(self, row: int):
        if row < 0 or row >= self.table.rowCount():
            return
        current = self.table.item(row, 2)
        start = QColor(current.text()) if current is not None else QColor("#FFFFFF")
        color = QColorDialog.getColor(start, self, "Select band color")
        if color.isValid():
            self._set_color_item(row, color.name())

    def _pick_selected_color(self):
        row = self.table.currentRow()
        if row >= 0:
            self._pick_color_for_row(row)

    def _reset_defaults(self):
        self.table.setRowCount(0)
        for band in default_custom_color_bands():
            self._add_row(band)

    def _on_cell_double_clicked(self, row: int, column: int):
        if column == 2:
            self._pick_color_for_row(row)

    def bands(self):
        out = []
        for row in range(self.table.rowCount()):
            min_item = self.table.item(row, 0)
            max_item = self.table.item(row, 1)
            color_item = self.table.item(row, 2)
            out.append({
                "min": float(min_item.text()) if min_item is not None else 0.0,
                "max": float(max_item.text()) if max_item is not None else 1.0,
                "color": color_item.text() if color_item is not None else "#FFFFFF",
            })
        return normalize_custom_bands(out)

    def _accept(self):
        try:
            self.bands()
        except Exception as e:
            QMessageBox.warning(self, "Custom Bands", str(e))
            return
        self.accept()



class ExportCroppedTdmsDialog(QDialog):
    """Dialog for exporting time-cropped, zero-aligned signals to .tdms."""

    def __init__(self, channel_names, parent=None,
                 default_fs=10000.0, default_t0=0.0, default_t1=30.0,
                 default_group="CroppedData", skip_first_default=False):
        super().__init__(parent)
        self.setWindowTitle("Export Cropped TDMS")
        self.setMinimumWidth(560)
        self.setModal(True)

        self.list_channels = QListWidget()
        self.list_channels.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_channels.setMinimumHeight(120)
        for nm in channel_names:
            self.list_channels.addItem(QListWidgetItem(nm))
        self.list_channels.selectAll()

        self.btn_select_all = QPushButton("Select All")
        self.btn_clear_sel = QPushButton("Clear")
        self.btn_select_all.clicked.connect(self.list_channels.selectAll)
        self.btn_clear_sel.clicked.connect(self.list_channels.clearSelection)

        self.fs_spin = QDoubleSpinBox()
        self.fs_spin.setRange(0.1, 1_000_000.0)
        self.fs_spin.setDecimals(2)
        self.fs_spin.setValue(default_fs)

        self.t0_spin = QDoubleSpinBox()
        self.t0_spin.setRange(0.0, 1_000_000_000.0)
        self.t0_spin.setDecimals(3)
        self.t0_spin.setValue(default_t0)

        self.t1_spin = QDoubleSpinBox()
        self.t1_spin.setRange(0.0, 1_000_000_000.0)
        self.t1_spin.setDecimals(3)
        self.t1_spin.setValue(default_t1)

        self.chk_skip_first = QCheckBox("Skip first sample (y[1:])")
        self.chk_skip_first.setChecked(skip_first_default)

        self.edit_group = QLineEdit(default_group)

        self.edit_output = QLineEdit()
        self.edit_output.setPlaceholderText("Select output .tdms file path...")
        self.edit_output.setReadOnly(True)
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._browse_output)

        self.lbl_info = QLabel(
            "Exports the selected channels cropped to [t0, t1] with the time axis "
            "shifted to start at t=0. Metadata (wf_increment=1/fs, wf_start_offset=0) "
            "is preserved in the TDMS file."
        )
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color: #555;")

        self.lbl_preview = QLabel("")
        self.lbl_preview.setStyleSheet("color: #444; font-style: italic;")
        self.lbl_preview.setWordWrap(True)

        form = QFormLayout(self)
        form.addRow(self.lbl_info)
        form.addRow("Channels to export:", self.list_channels)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_select_all)
        btn_row.addWidget(self.btn_clear_sel)
        btn_row.addStretch(1)
        form.addRow("", btn_row)

        form.addRow("Sampling rate fs (Hz):", self.fs_spin)
        form.addRow("Crop start t0 (s):", self.t0_spin)
        form.addRow("Crop end t1 (s):", self.t1_spin)
        form.addRow("", self.chk_skip_first)
        form.addRow("TDMS group name:", self.edit_group)

        out_row = QHBoxLayout()
        out_row.addWidget(self.edit_output, 1)
        out_row.addWidget(self.btn_browse)
        form.addRow("Output file:", out_row)

        form.addRow(self.lbl_preview)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        self.t0_spin.valueChanged.connect(self._update_preview)
        self.t1_spin.valueChanged.connect(self._update_preview)
        self.fs_spin.valueChanged.connect(self._update_preview)
        self._update_preview()

    def _update_preview(self):
        t0 = self.t0_spin.value()
        t1 = self.t1_spin.value()
        fs = self.fs_spin.value()
        dt = max(0.0, t1 - t0)
        n_samples = int(round(dt * fs))
        self.lbl_preview.setText(
            f"Preview: crop [{t0:.3f}, {t1:.3f}] s -> zero-aligned [0, {dt:.3f}] s | "
            f"~{n_samples} samples | wf_increment = {1.0/fs:.8g} s"
        )

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save cropped TDMS file", "cropped_export.tdms",
            "TDMS Files (*.tdms);;All Files (*)"
        )
        if path:
            if not path.lower().endswith(".tdms"):
                path += ".tdms"
            self.edit_output.setText(path)

    def _accept(self):
        sel = self.list_channels.selectedItems()
        if not sel:
            QMessageBox.warning(self, "Export", "Please select at least one channel.")
            return

        if self.t1_spin.value() <= self.t0_spin.value():
            QMessageBox.warning(self, "Export", "End time (t1) must be greater than start time (t0).")
            return

        if not self.edit_output.text().strip():
            QMessageBox.warning(self, "Export", "Please select an output file path.")
            return

        group_name = self.edit_group.text().strip()
        if not group_name:
            QMessageBox.warning(self, "Export", "TDMS group name cannot be empty.")
            return

        self.accept()

    def values(self):
        return {
            "channel_names": [it.text() for it in self.list_channels.selectedItems()],
            "fs": float(self.fs_spin.value()),
            "t0": float(self.t0_spin.value()),
            "t1": float(self.t1_spin.value()),
            "skip_first": self.chk_skip_first.isChecked(),
            "group_name": self.edit_group.text().strip(),
            "output_path": self.edit_output.text().strip(),
        }


def crop_and_zero_align(y: np.ndarray, fs: float, t0: float, t1: float,
                        skip_first: bool = False) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)

    if skip_first and y.size > 0:
        y = y[1:]

    n = y.size
    if n == 0:
        return np.array([], dtype=np.float64)

    i0 = int(np.floor(t0 * fs))
    i1 = int(np.ceil(t1 * fs))

    i0 = max(0, min(i0, n))
    i1 = max(i0, min(i1, n))

    if i1 <= i0:
        return np.array([], dtype=np.float64)

    return np.nan_to_num(y[i0:i1], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)


def export_cropped_tdms(output_path: str,
                        group_name: str,
                        channels: dict,
                        fs: float,
                        t0: float,
                        t1: float,
                        skip_first: bool = False,
                        source_properties: dict = None):
    if not _HAVE_TDMS_WRITE:
        raise RuntimeError(
            "TDMS writing requires nptdms with write support.\n"
            "Install with: pip install nptdms"
        )

    if not channels:
        raise ValueError("No channels provided for export.")

    source_properties = source_properties or {}
    dt = t1 - t0
    wf_increment = 1.0 / float(fs)

    root_props = {
        "title": f"Cropped export | [{t0:.6g}, {t1:.6g}] s -> [0, {dt:.6g}] s",
        "original_t0": float(t0),
        "original_t1": float(t1),
        "fs_hz": float(fs),
    }
    root_obj = TdmsRootObject(properties=root_props)

    group_props = {
        "description": f"Time-cropped data, zero-aligned. Original window: [{t0:.6g}, {t1:.6g}] s",
        "original_crop_start_s": float(t0),
        "original_crop_end_s": float(t1),
        "sampling_rate_hz": float(fs),
    }
    group_obj = TdmsGroupObject(group_name, properties=group_props)

    channel_objects = []
    exported_info = []

    for ch_name, raw_data in channels.items():
        cropped = crop_and_zero_align(raw_data, fs, t0, t1, skip_first=skip_first)

        if cropped.size == 0:
            raise ValueError(
                f"Channel '{ch_name}': no samples remain after cropping to "
                f"[{t0:.6g}, {t1:.6g}] s. Signal length = {raw_data.size} samples "
                f"({raw_data.size / fs:.6g} s at fs={fs:.6g} Hz)."
            )

        ch_props = {
            "wf_increment": wf_increment,
            "wf_start_offset": 0.0,
            "NI_ChannelName": ch_name,
            "original_crop_start_s": float(t0),
            "original_crop_end_s": float(t1),
        }

        src_p = source_properties.get(ch_name, {})
        for k, v in src_p.items():
            if k not in ("wf_increment", "wf_start_offset", "wf_start_time"):
                ch_props[k] = v

        ch_obj = TdmsChannelObject(group_name, ch_name, cropped, properties=ch_props)
        channel_objects.append(ch_obj)

        exported_info.append({
            "name": ch_name,
            "original_samples": int(raw_data.size),
            "exported_samples": int(cropped.size),
            "duration_s": float(cropped.size * wf_increment),
        })

    with TdmsWriter(output_path) as writer:
        writer.write_segment([root_obj, group_obj] + channel_objects)

    return exported_info




class Waterfall2DStandaloneWindow(QDialog):
    """Standalone 2D waterfall colormap window.

    Renders only the upper waterfall heatmap (no order trace subplot).
    Captures the current viewport zoom and color limits from the main view if provided.
    """

    def __init__(self, state: dict, parent=None,
                 view_range=None, cmap_name="jet",
                 order_cursor_specs=None,
                 color_limits=None):
        super().__init__(parent)
        self.state = state
        self.view_range = view_range
        self.cmap_name = cmap_name
        self.order_cursor_specs = order_cursor_specs or []
        self.color_limits = color_limits
        self.setWindowTitle("Waterfall (standalone)")
        self.resize(1400, 800)

        lay = QVBoxLayout(self)

        toolbar_lay = QHBoxLayout()
        toolbar_lay.addWidget(QLabel("Colormap:"))
        self.cmb_cmap = QComboBox()
        for cm in ["jet", "turbo", "nipy_spectral", "gist_rainbow",
                    "viridis", "plasma", "inferno", "hot", "coolwarm"]:
            self.cmb_cmap.addItem(cm)
        self.cmb_cmap.setCurrentText(cmap_name)
        self.cmb_cmap.currentTextChanged.connect(self._rebuild)
        toolbar_lay.addWidget(self.cmb_cmap)

        self.chk_log = QCheckBox("Logarithmic colorbar")
        is_db = str(state.get("scale", "")).strip().lower() == "db"
        self.chk_log.setChecked(not is_db)
        self.chk_log.stateChanged.connect(self._rebuild)
        toolbar_lay.addWidget(self.chk_log)

        self.btn_save = QPushButton("Save Image...")
        self.btn_save.clicked.connect(self._save_image)
        toolbar_lay.addWidget(self.btn_save)

        self.btn_reset_zoom = QPushButton("Reset to full range")
        self.btn_reset_zoom.clicked.connect(self._reset_zoom)
        toolbar_lay.addWidget(self.btn_reset_zoom)

        self.btn_reset_colors = QPushButton("Reset Color Range")
        self.btn_reset_colors.setToolTip("Revert to auto-calculated color limits")
        self.btn_reset_colors.clicked.connect(self._reset_color_range)
        toolbar_lay.addWidget(self.btn_reset_colors)

        toolbar_lay.addStretch(1)
        lay.addLayout(toolbar_lay)

        self.fig = Figure(figsize=(14, 7), dpi=180, constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)

        self._rebuild()

    def _reset_zoom(self):
        self.view_range = None
        self._rebuild()

    def _reset_color_range(self):
        self.color_limits = None
        self._rebuild()

    def _rebuild(self):
        self.fig.clear()
        freqs = np.asarray(self.state.get("freqs_display", self.state["freqs"]), dtype=np.float64)
        rpms = np.asarray(self.state.get("rpms_display", self.state["rpms"]), dtype=np.float64)
        S = np.asarray(self.state.get("S_display", self.state["S"]), dtype=np.float64)

        ax = self.fig.add_subplot(111)

        if freqs.size == 0 or rpms.size == 0 or S.size == 0:
            ax.set_title("Waterfall (no data)")
            self.canvas.draw_idle()
            return

        cmap_name = self.cmb_cmap.currentText().strip() or "jet"
        use_log = self.chk_log.isChecked()
        is_db = str(self.state.get("scale", "")).strip().lower() == "db"

        try:
            cmap = matplotlib.colormaps.get_cmap(cmap_name)
        except Exception:
            cmap = matplotlib.colormaps.get_cmap("jet")

        # Determine color limits (inherit from parent if available)
        if self.color_limits is not None:
            vmin = float(self.color_limits.get("vmin", 0.0))
            vmax = float(self.color_limits.get("vmax", 1.0))
            if vmax <= vmin:
                vmax = vmin + 1.0
        else:
            finite = S[np.isfinite(S)]
            if finite.size == 0:
                vmin, vmax = 0.0, 1.0
            else:
                vmin = float(np.min(finite))
                vmax = float(np.max(finite))
                if vmax <= vmin:
                    vmax = vmin + 1.0

        # ───────────────── FIX 2 ─────────────────
        # Replace pcolormesh (flat-colored quads, no interpolation) with imshow
        # using bilinear interpolation. This matches the STFT rendering style
        # and removes the "blocky" appearance when zooming in.
        #
        # Matrix orientation: S[i, j] = amplitude at rpms[i], freqs[j].
        # imshow with origin='lower' and extent=[f0, f1, r0, r1] maps
        # row i -> y=rpms[i], col j -> x=freqs[j]. No transpose needed.
        if use_log and not is_db:
            S_plot = np.clip(S, max(vmin, 1e-20), None)
            vmin_plot = max(vmin, 1e-20)
            norm = mcolors.LogNorm(vmin=vmin_plot, vmax=vmax)
            S_to_display = S_plot
            norm_to_use = norm
        else:
            S_to_display = S
            norm_to_use = None

        mesh = ax.imshow(
            S_to_display,
            aspect='auto',
            origin='lower',
            extent=[float(freqs[0]), float(freqs[-1]),
                    float(rpms[0]), float(rpms[-1])],
            interpolation='bilinear',
            cmap=cmap,
            norm=norm_to_use,
            vmin=(None if norm_to_use is not None else vmin),
            vmax=(None if norm_to_use is not None else vmax),
        )
        # ─────────────── END FIX 2 ───────────────

        cbar = self.fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label(self.state.get("colorbar_label", "Amplitude"))

        # Draw order cursors on the waterfall
        for spec in self.order_cursor_specs:
            order_val = float(spec["order"])
            color_rgb = spec.get("color", (220, 60, 60))
            if isinstance(color_rgb, (tuple, list)):
                color = np.array(color_rgb, dtype=np.float64) / 255.0
            else:
                color = "red"
            fx = (order_val * rpms) / 60.0
            m = np.isfinite(fx) & (fx >= freqs[0]) & (fx <= freqs[-1])
            if np.any(m):
                ax.plot(fx[m], rpms[m], color=color, linewidth=1.8,
                        label=f"Order {order_val:.2f}x")

        if self.order_cursor_specs:
            ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

        ax.set_xlabel(self.state.get("xlabel", "Frequency (Hz)"))
        ax.set_ylabel(self.state.get("ylabel", "RPM"))
        ax.set_title(self.state.get("title", "Waterfall"))

        # Apply viewport zoom if specified
        if self.view_range is not None:
            try:
                (x0, x1), (y0, y1) = self.view_range
                if np.isfinite([x0, x1, y0, y1]).all() and x1 > x0 and y1 > y0:
                    ax.set_xlim(x0, x1)
                    ax.set_ylim(y0, y1)
            except Exception:
                pass

        self.canvas.draw_idle()

    def _save_image(self):
        name = self.state.get("name", "waterfall")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save standalone waterfall image",
            f"{name}_standalone.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if path:
            try:
                self.fig.savefig(path, dpi=300)
                QMessageBox.information(self, "Saved", f"Image saved:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Save failed:\n{e}")


class Waterfall3DWindow(QDialog):
    def __init__(self, state: dict, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWindowTitle(state.get("title", "3D Waterfall"))
        self.resize(1280, 840)

        lay = QVBoxLayout(self)

        self.fig = Figure(figsize=(11, 7), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)

        self._build_scene()

    def _build_scene(self):
        freqs = np.asarray(self.state.get("freqs_display", self.state["freqs"]), dtype=np.float64)
        rpms = np.asarray(self.state.get("rpms_display", self.state["rpms"]), dtype=np.float64)
        S = np.asarray(self.state.get("S_display", self.state["S"]), dtype=np.float64)

        ax = self.fig.add_subplot(111, projection="3d")
        ax.clear()

        if freqs.size == 0 or rpms.size == 0 or S.size == 0:
            ax.set_title("3D waterfall")
            self.canvas.draw_idle()
            return

        max_freq_points = 900
        max_rpm_lines = 55
        sx = max(1, int(np.ceil(freqs.size / max_freq_points)))
        sy = max(1, int(np.ceil(rpms.size / max_rpm_lines)))

        freqs_ds = np.asarray(freqs[::sx], dtype=np.float64)
        rpms_ds = np.asarray(rpms[::sy], dtype=np.float64)
        S_ds = np.asarray(S[::sy, ::sx], dtype=np.float64)

        finite = S_ds[np.isfinite(S_ds)]
        if finite.size == 0:
            z0, z1 = 0.0, 1.0
            S_plot = np.zeros_like(S_ds, dtype=np.float64)
        else:
            z0 = float(np.min(finite))
            z1 = float(np.max(finite))
            if z1 <= z0:
                z1 = z0 + 1.0
            S_plot = np.nan_to_num(S_ds, nan=z0, posinf=z1, neginf=z0)

        cmap = matplotlib.colormaps.get_cmap("Oranges")
        n_lines = max(1, rpms_ds.size)
        for i, rpm_val in enumerate(rpms_ds):
            frac = i / max(n_lines - 1, 1)
            color = cmap(0.35 + 0.55 * frac)
            yline = np.full(freqs_ds.shape, float(rpm_val), dtype=np.float64)
            ax.plot(freqs_ds, yline, S_plot[i, :], color=color, linewidth=1.0)

        if freqs_ds.size > 1 and rpms_ds.size > 1:
            fmesh, rmesh = np.meshgrid(freqs_ds, rpms_ds)
            ax.plot_surface(
                fmesh, rmesh, S_plot,
                cmap="viridis",
                linewidth=0,
                antialiased=False,
                alpha=0.22,
                shade=True,
            )

        ax.set_xlabel("Frequency (Hz)", labelpad=12)
        ax.set_ylabel("RPM", labelpad=12)
        ax.set_zlabel(self.state.get("colorbar_label", "Amplitude"), labelpad=10)
        ax.set_title(self.state.get("title", "3D Waterfall"), pad=16)

        ax.set_xlim(float(freqs_ds[0]), float(freqs_ds[-1]))
        ax.set_ylim(float(rpms_ds[0]), float(rpms_ds[-1]))
        ax.set_zlim(z0, z1)
        ax.view_init(elev=22, azim=-62)

        try:
            ax.xaxis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
            ax.yaxis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
            ax.zaxis.pane.set_facecolor((0.985, 0.985, 0.985, 1.0))
        except Exception:
            pass

        ax.grid(True)
        self.canvas.draw_idle()


class WaterfallOrderCursor(QObject):
    changed = pyqtSignal()

    def __init__(self, owner, order_value: float, color):
        super().__init__()
        self.owner = owner
        self.order = float(order_value)
        self.color = color
        self._guard = False

        self.line = pg.PlotCurveItem(pen=pg.mkPen(color, width=2))
        self.label = pg.TextItem(text="", color=color, anchor=(0.5, 0.5), fill=pg.mkBrush(255, 255, 255, 215), border=pg.mkPen(color, width=1))
        self.handle = pg.TargetItem(
            pos=(self.owner._cursor_label_x(self.owner._freq_at_top(self.order)), self.owner._cursor_top_y()),
            size=12,
            pen=pg.mkPen(color, width=2),
            brush=pg.mkBrush(255, 255, 255, 170),
            movable=True,
        )

        self.owner.upper_item.addItem(self.line)
        self.owner.upper_item.addItem(self.label)
        self.owner.upper_item.addItem(self.handle)

        self.handle.sigPositionChanged.connect(self._on_handle_moved)
        self.redraw()

    def _on_handle_moved(self):
        if self._guard:
            return
        self._guard = True
        try:
            pos = self.handle.pos()
            x = float(np.clip(pos.x(), self.owner.freq_min, self.owner.freq_max))
            self.handle.setPos(self.owner._cursor_label_x(x), self.owner._cursor_top_y())
            self.order = max(1e-6, 60.0 * x / max(self.owner.rpm_max, 1e-12))
            self.redraw()
            self.changed.emit()
        finally:
            self._guard = False

    def redraw(self):
        y = self.owner.rpms
        x = (self.order * y) / 60.0
        m = np.isfinite(x) & np.isfinite(y) & (x >= self.owner.freq_min) & (x <= self.owner.freq_max)
        self.line.setData(x[m], y[m])

        x_top = self.owner._cursor_label_x(self.owner._freq_at_top(self.order))
        self.label.setText(f"{self.order:.2f}x")
        self.label.setPos(x_top, self.owner._cursor_top_y())

    def remove(self):
        for item in (self.line, self.label, self.handle):
            try:
                self.owner.upper_item.removeItem(item)
            except Exception:
                pass


class WaterfallPlotWidget(QWidget):
    clicked = pyqtSignal()
    color_levels_changed = pyqtSignal(float, float)
    color_levels_change_finished = pyqtSignal(float, float)

    def __init__(self, mini=False, parent=None):
        super().__init__(parent)
        self.mini = bool(mini)
        self.state = None
        self.plot_key = None
        self._rect_zoom_active = False
        self._home_upper = None
        self._home_lower = None
        self._legend = None
        self._cursors = []
        self._upper_image_item = None
        self._upper_color_bar = None
        self._suspend_colorbar_sync = False
        self._last_color_cfg = {}
        self._cursor_colors = [
            (220, 60, 60),
            (70, 180, 90),
            (60, 120, 220),
            (210, 140, 30),
            (180, 70, 190),
        ]

        self.freqs = np.array([], dtype=np.float64)
        self.freqs_analysis = np.array([], dtype=np.float64)
        self.freqs_full_display = np.array([], dtype=np.float64)
        self.freqs_trace = np.array([], dtype=np.float64)
        self.rpms = np.array([], dtype=np.float64)
        self.S = np.empty((0, 0), dtype=np.float64)
        self.rpms_analysis = np.array([], dtype=np.float64)
        self.S_analysis = np.empty((0, 0), dtype=np.float64)
        self.frame_rpms = np.array([], dtype=np.float64)
        self.frame_spectra_linear = np.empty((0, 0), dtype=np.float64)
        self.rpms_full_display = np.array([], dtype=np.float64)
        self.S_full_display = np.empty((0, 0), dtype=np.float64)
        self.rpms_trace = np.array([], dtype=np.float64)
        self.S_trace = np.empty((0, 0), dtype=np.float64)
        self._viewport_dense_factor = 4
        self._viewport_min_points = 600
        self._viewport_max_points = 3000
        self._viewport_freq_dense_factor = 2.0
        self._viewport_min_cols = 900
        self._viewport_max_cols = 4096
        self._viewport_row_pixel_factor = 1.25
        self._viewport_col_pixel_factor = 1.25
        self._suspend_viewport_reinterp = False
        self._high_quality_zoom = False
        self._hq_supersample = 1
        self._hq_use_monotone = False
        self.freq_min = 0.0
        self.freq_max = 1.0
        self.rpm_min = 0.0
        self.rpm_max = 1.0
        self.full_freq_min = 0.0
        self.full_freq_max = 1.0
        self.full_rpm_min = 0.0
        self.full_rpm_max = 1.0

        self.enable_click_annotations = not self.mini
        self._upper_click_annotations = []
        self._lower_click_annotations = []
        self._upper_mouse_proxy = None
        self._lower_mouse_proxy = None

        self.btn_add_cursor = QPushButton("Add cursor")
        self.btn_clear_cursors = QPushButton("Clear cursors")
        self.btn_open_3d = QPushButton("Open 3D")
        self.btn_open_2d_standalone = QPushButton("Open 2D Standalone")
        self.btn_open_2d_standalone.setToolTip(
            "Open waterfall heatmap in a standalone matplotlib window. "
            "Inherits current colormap, color range (vmin/vmax), "
            "zoom viewport, and order cursors from the main view."
        )
        self.lbl_order_info = QLabel("Active orders: -")
        self.lbl_order_info.setStyleSheet("color: #333; font-weight: 600;")

        self.upper = pg.PlotWidget(background="w")
        self.upper_item = self.upper.getPlotItem()
        self.upper_item.showGrid(x=True, y=True, alpha=0.25)
        self.upper_item.setLabel("bottom", "Frequency (Hz)")
        self.upper_item.setLabel("left", "RPM")
        self.upper.setMenuEnabled(False)

        self.lower = pg.PlotWidget(background="w")
        self.lower_item = self.lower.getPlotItem()
        self.lower_item.showGrid(x=True, y=True, alpha=0.25)
        self.lower_item.setLabel("bottom", "RPM")
        self.lower_item.setLabel("left", "Amplitude")
        self.lower.setMenuEnabled(False)

        self.upper_hover_label = QLabel(self.upper.viewport())
        self.upper_hover_label.setVisible(False)
        self.upper_hover_label.setStyleSheet(
            "background: rgba(255,255,255,230); border: 1px solid #666; padding: 2px 4px;"
        )

        self.lower_hover_label = QLabel(self.lower.viewport())
        self.lower_hover_label.setVisible(False)
        self.lower_hover_label.setStyleSheet(
            "background: rgba(255,255,255,230); border: 1px solid #666; padding: 2px 4px;"
        )

        self.upper.scene().sigMouseClicked.connect(self._on_upper_mouse_clicked)
        self.lower.scene().sigMouseClicked.connect(self._on_lower_mouse_clicked)

        self._upper_mouse_proxy = pg.SignalProxy(
            self.upper.scene().sigMouseMoved,
            rateLimit=35,
            slot=self._on_upper_mouse_moved,
        )
        self._lower_mouse_proxy = pg.SignalProxy(
            self.lower.scene().sigMouseMoved,
            rateLimit=35,
            slot=self._on_lower_mouse_moved,
        )

        self.upper_item.getViewBox().sigRangeChanged.connect(self._update_upper_annotation_buttons)
        self.upper_item.getViewBox().sigRangeChanged.connect(self._on_upper_view_range_changed)
        self.lower_item.getViewBox().sigRangeChanged.connect(self._update_lower_annotation_buttons)

        self._visible_reinterp_timer = QTimer(self)
        self._visible_reinterp_timer.setSingleShot(True)
        self._visible_reinterp_timer.setInterval(90)
        self._visible_reinterp_timer.timeout.connect(self._rebuild_visible_waterfall)

        topbar = QHBoxLayout()
        topbar.addWidget(self.btn_add_cursor)
        topbar.addWidget(self.btn_clear_cursors)
        topbar.addWidget(self.btn_open_3d)
        topbar.addWidget(self.btn_open_2d_standalone)
        topbar.addSpacing(10)
        topbar.addWidget(self.lbl_order_info, 1)
        topbar.addStretch(1)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if not self.mini:
            lay.addLayout(topbar)
        lay.addWidget(self.upper, 3)
        if not self.mini:
            lay.addWidget(self.lower, 2)

        self.btn_add_cursor.clicked.connect(self._add_cursor_default)
        self.btn_clear_cursors.clicked.connect(self.clear_cursors)
        self.btn_open_3d.clicked.connect(self.open_3d_window)
        self.btn_open_2d_standalone.clicked.connect(self.open_2d_standalone_window)

        if self.mini:
            self.btn_add_cursor.hide()
            self.btn_clear_cursors.hide()
            self.btn_open_3d.hide()
            self.btn_open_2d_standalone.hide()
            self.lbl_order_info.hide()
            self.lower.hide()

    def _clear_upper_colorbar(self):
        cbar = self._upper_color_bar
        self._upper_color_bar = None
        if cbar is None:
            return
        try:
            cbar.setImageItem(None)
        except Exception:
            pass
        try:
            self.upper_item.layout.removeItem(cbar)
        except Exception:
            pass
        try:
            scene = cbar.scene()
            if scene is not None:
                scene.removeItem(cbar)
        except Exception:
            pass
        try:
            cbar.deleteLater()
        except Exception:
            pass

    @staticmethod
    def _aggregate_duplicate_rpms(rpms: np.ndarray, spectra: np.ndarray):
        rpms = np.asarray(rpms, dtype=np.float64)
        spectra = np.asarray(spectra, dtype=np.float64)

        if rpms.size == 0 or spectra.size == 0:
            return np.array([], dtype=np.float64), np.empty((0, 0), dtype=np.float64)

        uniq, inverse = np.unique(rpms, return_inverse=True)
        out = np.zeros((uniq.size, spectra.shape[1]), dtype=np.float64)
        counts = np.zeros(uniq.size, dtype=np.float64)

        for i, idx in enumerate(inverse):
            out[idx] += spectra[i]
            counts[idx] += 1.0

        counts[counts <= 0.0] = 1.0
        out /= counts[:, None]
        return uniq, out

    @staticmethod
    def _convert_waterfall_scale(S_linear: np.ndarray, output_type: str, scale: str) -> np.ndarray:
        S_linear = np.asarray(S_linear, dtype=np.float64)
        if str(scale).strip().lower() != "db":
            return S_linear

        eps = 1e-20
        if output_type == "Amplitude":
            return 20.0 * np.log10(S_linear + eps)
        return 10.0 * np.log10(S_linear + eps)

    def _interp_axis_monotone_or_linear(self, x_src: np.ndarray, values: np.ndarray,
                                      x_dst: np.ndarray, axis: int,
                                      prefer_monotone: bool = False) -> np.ndarray:
        x_src = np.asarray(x_src, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        x_dst = np.asarray(x_dst, dtype=np.float64)

        if x_src.size == 0 or values.size == 0 or x_dst.size == 0:
            shape = list(values.shape)
            if axis == 0:
                shape[0] = x_dst.size
            else:
                shape[1] = x_dst.size
            return np.empty(shape, dtype=np.float64)

        if x_src.size == 1:
            if axis == 0:
                out = np.repeat(values[:1, :], x_dst.size, axis=0)
            else:
                out = np.repeat(values[:, :1], x_dst.size, axis=1)
            return np.maximum(out, 0.0)

        if prefer_monotone and _HAVE_SCIPY_INTERP and PchipInterpolator is not None and x_src.size >= 3:
            try:
                out = PchipInterpolator(x_src, values, axis=axis, extrapolate=True)(x_dst)
                return np.maximum(np.asarray(out, dtype=np.float64), 0.0)
            except Exception:
                pass

        if _HAVE_SCIPY_INTERP and interp1d is not None:
            try:
                f = interp1d(
                    x_src, values, axis=axis, kind='linear',
                    bounds_error=False, fill_value='extrapolate', assume_sorted=True,
                )
                out = f(x_dst)
                return np.maximum(np.asarray(out, dtype=np.float64), 0.0)
            except Exception:
                pass

        if axis == 0:
            out = np.empty((x_dst.size, values.shape[1]), dtype=np.float64)
            for j in range(values.shape[1]):
                out[:, j] = np.interp(x_dst, x_src, values[:, j])
        else:
            out = np.empty((values.shape[0], x_dst.size), dtype=np.float64)
            for i in range(values.shape[0]):
                out[i, :] = np.interp(x_dst, x_src, values[i, :])
        return np.maximum(out, 0.0)

    def _bin_average_axis0(self, x_src: np.ndarray, values: np.ndarray,
                           bin_edges: np.ndarray) -> np.ndarray:
        x_src = np.asarray(x_src, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        bin_edges = np.asarray(bin_edges, dtype=np.float64)

        if x_src.size == 0 or values.size == 0 or bin_edges.size < 2:
            return np.empty((0, values.shape[1] if values.ndim == 2 else 0), dtype=np.float64)

        csum = np.vstack([
            np.zeros((1, values.shape[1]), dtype=np.float64),
            np.cumsum(values, axis=0, dtype=np.float64)
        ])
        lo_idx = np.searchsorted(x_src, bin_edges[:-1], side='left')
        hi_idx = np.searchsorted(x_src, bin_edges[1:], side='left')
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        out = np.empty((bin_edges.size - 1, values.shape[1]), dtype=np.float64)
        for i in range(bin_edges.size - 1):
            lo = int(lo_idx[i])
            hi = int(hi_idx[i])
            if hi > lo:
                out[i, :] = (csum[hi, :] - csum[lo, :]) / float(hi - lo)
            else:
                center = centers[i]
                idx = int(np.searchsorted(x_src, center, side='left'))
                idx = max(0, min(idx, x_src.size - 1))
                if idx > 0 and idx < x_src.size:
                    left = idx - 1
                    right = idx
                    if abs(center - x_src[left]) <= abs(x_src[right] - center):
                        idx = left
                out[i, :] = values[idx, :]
        return np.maximum(out, 0.0)

    def _bin_average_axis1(self, x_src: np.ndarray, values: np.ndarray,
                           bin_edges: np.ndarray) -> np.ndarray:
        x_src = np.asarray(x_src, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        bin_edges = np.asarray(bin_edges, dtype=np.float64)

        if x_src.size == 0 or values.size == 0 or bin_edges.size < 2:
            return np.empty((values.shape[0] if values.ndim == 2 else 0, 0), dtype=np.float64)

        csum = np.hstack([
            np.zeros((values.shape[0], 1), dtype=np.float64),
            np.cumsum(values, axis=1, dtype=np.float64)
        ])
        lo_idx = np.searchsorted(x_src, bin_edges[:-1], side='left')
        hi_idx = np.searchsorted(x_src, bin_edges[1:], side='left')
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        out = np.empty((values.shape[0], bin_edges.size - 1), dtype=np.float64)
        for j in range(bin_edges.size - 1):
            lo = int(lo_idx[j])
            hi = int(hi_idx[j])
            if hi > lo:
                out[:, j] = (csum[:, hi] - csum[:, lo]) / float(hi - lo)
            else:
                center = centers[j]
                idx = int(np.searchsorted(x_src, center, side='left'))
                idx = max(0, min(idx, x_src.size - 1))
                if idx > 0 and idx < x_src.size:
                    left = idx - 1
                    right = idx
                    if abs(center - x_src[left]) <= abs(x_src[right] - center):
                        idx = left
                out[:, j] = values[:, idx]
        return np.maximum(out, 0.0)

    @staticmethod
    def _downsample_mean_2d(values: np.ndarray, factor_r: int = 1, factor_c: int = 1) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        factor_r = max(1, int(factor_r))
        factor_c = max(1, int(factor_c))
        if factor_r == 1 and factor_c == 1:
            return values
        nr = (values.shape[0] // factor_r) * factor_r
        nc = (values.shape[1] // factor_c) * factor_c
        if nr <= 0 or nc <= 0:
            return values
        trimmed = values[:nr, :nc]
        out = trimmed.reshape(nr // factor_r, factor_r, nc // factor_c, factor_c).mean(axis=(1, 3))
        return np.maximum(out, 0.0)

    def _set_upper_image_matrix(self, freqs: np.ndarray, rpms: np.ndarray, S: np.ndarray):
        freqs = np.asarray(freqs, dtype=np.float64)
        rpms = np.asarray(rpms, dtype=np.float64)
        S = np.asarray(S, dtype=np.float64)
        if self._upper_image_item is None or freqs.size == 0 or rpms.size == 0 or S.size == 0:
            return

        if freqs.size > 1:
            dx = float(np.median(np.diff(freqs)))
        else:
            dx = 1.0
        if rpms.size > 1:
            dy = float(np.median(np.diff(rpms)))
        else:
            dy = 1.0

        x0 = float(freqs[0] - 0.5 * dx)
        x1 = float(freqs[-1] + 0.5 * dx)
        y0 = float(rpms[0] - 0.5 * dy)
        y1 = float(rpms[-1] + 0.5 * dy)

        # Note: the interpolate=True flag set on the ImageItem in set_state()
        # persists across setImage() calls, so adaptive re-renders maintain
        # bilinear texture filtering automatically.
        self._upper_image_item.setImage(S, autoLevels=False)
        self._upper_image_item.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))

        self.freqs = freqs
        self.rpms = rpms
        self.S = S
        self.freq_min = float(freqs[0]) if freqs.size else 0.0
        self.freq_max = float(freqs[-1]) if freqs.size else 1.0
        self.rpm_min = float(rpms[0]) if rpms.size else 0.0
        self.rpm_max = float(rpms[-1]) if rpms.size else 1.0

    def _restore_full_upper_display(self):
        if self.freqs_full_display.size == 0 or self.rpms_full_display.size == 0 or self.S_full_display.size == 0:
            return
        self._set_upper_image_matrix(self.freqs_full_display, self.rpms_full_display, self.S_full_display)

    def _on_upper_view_range_changed(self, *_):
        if self.mini or self._suspend_viewport_reinterp:
            return
        if self.state is None or self.frame_rpms.size == 0 or self.frame_spectra_linear.size == 0:
            return
        self._visible_reinterp_timer.start()

    def _render_waterfall_view_from_frames(self, view_f_lo: float, view_f_hi: float,
                                           view_r_lo: float, view_r_hi: float,
                                           pad_frac_f: float = 0.03,
                                           pad_frac_r: float = 0.05):
        frame_rpms = np.asarray(self.frame_rpms, dtype=np.float64)
        spectra_linear = np.asarray(self.frame_spectra_linear, dtype=np.float64)
        freqs_u = np.asarray(self.freqs_analysis, dtype=np.float64)

        if frame_rpms.size < 2 or spectra_linear.size == 0 or freqs_u.size < 2:
            return self.freqs_full_display, self.rpms_full_display, self.S_full_display

        m = np.isfinite(frame_rpms)
        frame_rpms = frame_rpms[m]
        spectra_linear = spectra_linear[m]
        if frame_rpms.size < 2 or spectra_linear.size == 0:
            return self.freqs_full_display, self.rpms_full_display, self.S_full_display

        order = np.argsort(frame_rpms)
        frame_rpms = frame_rpms[order]
        spectra_linear = spectra_linear[order]
        frame_rpms, spectra_linear = self._aggregate_duplicate_rpms(frame_rpms, spectra_linear)
        if frame_rpms.size < 2 or spectra_linear.size == 0:
            return self.freqs_full_display, self.rpms_full_display, self.S_full_display

        full_f_lo = float(freqs_u[0])
        full_f_hi = float(freqs_u[-1])
        full_r_lo = float(frame_rpms[0])
        full_r_hi = float(frame_rpms[-1])

        vf0 = float(min(view_f_lo, view_f_hi))
        vf1 = float(max(view_f_lo, view_f_hi))
        vr0 = float(min(view_r_lo, view_r_hi))
        vr1 = float(max(view_r_lo, view_r_hi))

        view_f_lo = float(np.clip(vf0, full_f_lo, full_f_hi))
        view_f_hi = float(np.clip(vf1, full_f_lo, full_f_hi))
        view_r_lo = float(np.clip(vr0, full_r_lo, full_r_hi))
        view_r_hi = float(np.clip(vr1, full_r_lo, full_r_hi))

        if view_f_hi <= view_f_lo or view_r_hi <= view_r_lo:
            return self.freqs_full_display, self.rpms_full_display, self.S_full_display

        vis_f_span = max(view_f_hi - view_f_lo, 1e-12)
        vis_r_span = max(view_r_hi - view_r_lo, 1e-12)
        full_f_span = max(full_f_hi - full_f_lo, 1e-12)
        full_r_span = max(full_r_hi - full_r_lo, 1e-12)

        high_quality = bool(self._high_quality_zoom)
        supersample = int(max(1, self._hq_supersample if high_quality else 1))
        prefer_monotone = bool(self._hq_use_monotone and high_quality)

        near_full_f = vis_f_span >= 0.98 * full_f_span
        near_full_r = vis_r_span >= 0.98 * full_r_span

        eff_pad_f = 0.012 if high_quality else float(pad_frac_f)
        eff_pad_r = 0.020 if high_quality else float(pad_frac_r)

        if near_full_f:
            render_f_lo, render_f_hi = full_f_lo, full_f_hi
        else:
            f_pad = max(eff_pad_f * vis_f_span, 0.0003 * full_f_span)
            render_f_lo = max(view_f_lo - f_pad, full_f_lo)
            render_f_hi = min(view_f_hi + f_pad, full_f_hi)

        if near_full_r:
            render_r_lo, render_r_hi = full_r_lo, full_r_hi
        else:
            r_pad = max(eff_pad_r * vis_r_span, 0.0005 * full_r_span)
            render_r_lo = max(view_r_lo - r_pad, full_r_lo)
            render_r_hi = min(view_r_hi + r_pad, full_r_hi)

        if render_f_hi <= render_f_lo or render_r_hi <= render_r_lo:
            return self.freqs_full_display, self.rpms_full_display, self.S_full_display

        rpm_sel = (frame_rpms >= render_r_lo) & (frame_rpms <= render_r_hi)
        if np.count_nonzero(rpm_sel) < 4:
            center = 0.5 * (render_r_lo + render_r_hi)
            idx = int(np.searchsorted(frame_rpms, center, side='left'))
            i0 = max(0, idx - 24)
            i1 = min(frame_rpms.size, idx + 24)
            local_rpms = frame_rpms[i0:i1]
            local_spectra = spectra_linear[i0:i1]
        else:
            local_rpms = frame_rpms[rpm_sel]
            local_spectra = spectra_linear[rpm_sel]

        if local_rpms.size < 2 or local_spectra.size == 0:
            return self.freqs_full_display, self.rpms_full_display, self.S_full_display

        freq_sel = (freqs_u >= render_f_lo) & (freqs_u <= render_f_hi)
        if np.count_nonzero(freq_sel) < 8:
            center = 0.5 * (render_f_lo + render_f_hi)
            idx = int(np.searchsorted(freqs_u, center, side='left'))
            j0 = max(0, idx - 96)
            j1 = min(freqs_u.size, idx + 96)
            local_freqs = freqs_u[j0:j1]
            local_spectra = local_spectra[:, j0:j1]
        else:
            local_freqs = freqs_u[freq_sel]
            local_spectra = local_spectra[:, freq_sel]

        if local_freqs.size < 2 or local_spectra.size == 0:
            return self.freqs_full_display, self.rpms_full_display, self.S_full_display

        pix_h = max(32, int(self.upper.viewport().height()))
        pix_w = max(32, int(self.upper.viewport().width()))

        target_rows = max(
            int(self._viewport_min_points),
            int(np.ceil(pix_h * float(self._viewport_row_pixel_factor))),
            int(np.ceil(local_rpms.size * max(1.0, float(self._viewport_dense_factor)))),
        )
        target_cols = max(
            int(self._viewport_min_cols),
            int(np.ceil(pix_w * float(self._viewport_col_pixel_factor))),
            int(np.ceil(local_freqs.size * max(1.0, float(self._viewport_freq_dense_factor)))),
        )

        if high_quality:
            target_rows = max(target_rows, int(np.ceil(pix_h * 1.4)), 900)
            target_cols = max(target_cols, int(np.ceil(pix_w * 1.4)), 1200)

        target_rows = int(min(int(self._viewport_max_points), max(2, target_rows)))
        target_cols = int(min(int(self._viewport_max_cols), max(2, target_cols)))

        internal_rows = int(min(int(self._viewport_max_points) * supersample, max(2, target_rows * supersample)))
        internal_cols = int(min(int(self._viewport_max_cols) * supersample, max(2, target_cols * supersample)))

        row_edges_internal = np.linspace(render_r_lo, render_r_hi, internal_rows + 1)
        rpm_centers_internal = 0.5 * (row_edges_internal[:-1] + row_edges_internal[1:])
        col_edges_internal = np.linspace(render_f_lo, render_f_hi, internal_cols + 1)
        freq_centers_internal = 0.5 * (col_edges_internal[:-1] + col_edges_internal[1:])

        zooming_in_r = internal_rows > int(np.ceil(local_rpms.size * 1.15))
        zooming_in_f = internal_cols > int(np.ceil(local_freqs.size * 1.05))

        if zooming_in_r:
            S_rows_linear = self._interp_axis_monotone_or_linear(
                local_rpms, local_spectra, rpm_centers_internal,
                axis=0, prefer_monotone=prefer_monotone,
            )
        else:
            S_rows_linear = self._bin_average_axis0(local_rpms, local_spectra, row_edges_internal)

        if zooming_in_f:
            S_internal_linear = self._interp_axis_monotone_or_linear(
                local_freqs, S_rows_linear, freq_centers_internal,
                axis=1, prefer_monotone=prefer_monotone,
            )
        else:
            S_internal_linear = self._bin_average_axis1(local_freqs, S_rows_linear, col_edges_internal)

        if supersample > 1:
            S_out_linear = self._downsample_mean_2d(S_internal_linear, supersample, supersample)
            row_edges_final = np.linspace(render_r_lo, render_r_hi, target_rows + 1)
            target_rpms = 0.5 * (row_edges_final[:-1] + row_edges_final[1:])
            col_edges_final = np.linspace(render_f_lo, render_f_hi, target_cols + 1)
            target_freqs = 0.5 * (col_edges_final[:-1] + col_edges_final[1:])
        else:
            S_out_linear = S_internal_linear
            target_rpms = rpm_centers_internal
            target_freqs = freq_centers_internal

        S_out_linear = np.maximum(np.asarray(S_out_linear, dtype=np.float64), 0.0)
        S_out = self._convert_waterfall_scale(
            S_out_linear,
            self.state.get('out_type', 'Amplitude'),
            self.state.get('scale', 'Linear'),
        )
        return target_freqs, target_rpms, S_out

    def _recompute_full_upper_display(self):
        if self.state is None or self.frame_rpms.size == 0 or self.frame_spectra_linear.size == 0 or self.freqs_analysis.size < 2:
            return False
        try:
            freqs_full, rpms_full, S_full = self._render_waterfall_view_from_frames(
                float(self.freqs_analysis[0]),
                float(self.freqs_analysis[-1]),
                float(np.min(self.frame_rpms)),
                float(np.max(self.frame_rpms)),
                pad_frac_f=0.0,
                pad_frac_r=0.0,
            )
        except Exception:
            return False

        if freqs_full.size < 2 or rpms_full.size < 2 or S_full.size == 0:
            return False

        self.freqs_full_display = freqs_full
        self.rpms_full_display = rpms_full
        self.S_full_display = S_full
        self.freqs_trace = freqs_full.copy()
        self.rpms_trace = rpms_full.copy()
        self.S_trace = S_full.copy()
        return True

    def _rebuild_visible_waterfall(self):
        if self.mini or self._suspend_viewport_reinterp:
            return
        if self.state is None or self._upper_image_item is None:
            return
        if self.frame_rpms.size == 0 or self.frame_spectra_linear.size == 0 or self.freqs_analysis.size < 2:
            return

        try:
            xr, yr = self.upper_item.getViewBox().viewRange()
            view_f_lo = float(min(xr[0], xr[1]))
            view_f_hi = float(max(xr[0], xr[1]))
            view_r_lo = float(min(yr[0], yr[1]))
            view_r_hi = float(max(yr[0], yr[1]))
        except Exception:
            return

        full_f_lo = self.full_freq_min
        full_f_hi = self.full_freq_max
        full_r_lo = self.full_rpm_min
        full_r_hi = self.full_rpm_max
        full_f_span = max(full_f_hi - full_f_lo, 1e-12)
        full_r_span = max(full_r_hi - full_r_lo, 1e-12)
        visible_f_span = max(view_f_hi - view_f_lo, 0.0)
        visible_r_span = max(view_r_hi - view_r_lo, 0.0)

        if visible_f_span >= 0.92 * full_f_span and visible_r_span >= 0.92 * full_r_span:
            self._restore_full_upper_display()
        else:
            freqs_vis, rpms_vis, S_vis = self._render_waterfall_view_from_frames(
                view_f_lo, view_f_hi, view_r_lo, view_r_hi
            )
            if freqs_vis.size >= 2 and rpms_vis.size >= 2 and S_vis.size:
                self._set_upper_image_matrix(freqs_vis, rpms_vis, S_vis)

        for c in self._cursors:
            try:
                c.redraw()
            except Exception:
                pass
        self._update_upper_annotation_buttons()

    def _freq_at_top(self, order_value: float) -> float:
        return float(order_value) * self.rpm_max / 60.0

    def _cursor_top_y(self):
        span = max(self.rpm_max - self.rpm_min, 1e-12)
        return self.rpm_max - 0.035 * span

    def _cursor_label_x(self, x_value: float) -> float:
        span = max(self.freq_max - self.freq_min, 1e-12)
        return float(np.clip(x_value, self.freq_min + 0.02 * span, self.freq_max - 0.02 * span))

    def _ann_html(self, text: str) -> str:
        return (
            "<div style='background: rgba(255,255,255,230); "
            "border: 1px solid #666; padding: 2px 6px;'>"
            f"{text}"
            "</div>"
        )

    def _annotation_templates(self):
        return [
            {"dx": 0.012, "dy": 0.030, "anchor": (0, 1)},
            {"dx": 0.012, "dy": -0.030, "anchor": (0, 0)},
            {"dx": 0.024, "dy": 0.052, "anchor": (0, 1)},
            {"dx": -0.024, "dy": 0.052, "anchor": (1, 1)},
            {"dx": 0.024, "dy": -0.052, "anchor": (0, 0)},
            {"dx": -0.024, "dy": -0.052, "anchor": (1, 0)},
            {"dx": 0.040, "dy": 0.020, "anchor": (0, 1)},
            {"dx": -0.040, "dy": 0.020, "anchor": (1, 1)},
        ]

    def _choose_annotation_layout(self, existing, x: float, y: float, x_span: float, y_span: float):
        x_span = max(float(x_span), 1e-12)
        y_span = max(float(y_span), 1e-12)

        x_tol = 0.035 * x_span
        y_tol = 0.060 * y_span

        nearby = 0
        for ann in existing:
            try:
                ax = float(ann["x"])
                ay = float(ann["y"])
                if abs(ax - x) <= x_tol and abs(ay - y) <= y_tol:
                    nearby += 1
            except Exception:
                pass

        tpl = self._annotation_templates()[nearby % len(self._annotation_templates())]
        return {
            "label_x": float(x + tpl["dx"] * x_span),
            "label_y": float(y + tpl["dy"] * y_span),
            "anchor": tpl["anchor"],
        }

    def _on_upper_mouse_clicked(self, event):
        try:
            if event.button() == Qt.LeftButton:
                self.clicked.emit()

                if self._rect_zoom_active or not self.enable_click_annotations:
                    return

                vb = self.upper_item.getViewBox()
                pos = event.scenePos()
                if not vb.sceneBoundingRect().contains(pos):
                    return

                mp = vb.mapSceneToView(pos)
                self._add_upper_click_annotation(float(mp.x()), float(mp.y()))
        except Exception:
            self.clicked.emit()

    def _on_lower_mouse_clicked(self, event):
        try:
            if event.button() == Qt.LeftButton:
                self.clicked.emit()

                if self.mini or self._rect_zoom_active or not self.enable_click_annotations:
                    return

                vb = self.lower_item.getViewBox()
                pos = event.scenePos()
                if not vb.sceneBoundingRect().contains(pos):
                    return

                mp = vb.mapSceneToView(pos)
                info = self._nearest_order_trace_info(float(mp.x()), float(mp.y()))
                if info is None:
                    return

                self._add_lower_click_annotation(
                    info["x"], info["y"], info["order"], info["color"]
                )
        except Exception:
            self.clicked.emit()

    def _on_upper_mouse_moved(self, evt):
        pos = evt[0] if isinstance(evt, tuple) else evt
        if pos is None:
            self.upper_hover_label.setVisible(False)
            return

        try:
            vb = self.upper_item.getViewBox()
            if not vb.sceneBoundingRect().contains(pos):
                self.upper_hover_label.setVisible(False)
                return

            mp = vb.mapSceneToView(pos)
            text = f"({mp.x():.6g}, {mp.y():.6g})"
            self.upper_hover_label.setText(text)
            self.upper_hover_label.adjustSize()

            vp = self.upper.mapFromScene(pos)
            x = min(max(0, int(vp.x()) + 14),
                    max(0, self.upper.viewport().width() - self.upper_hover_label.width() - 4))
            y = min(max(0, int(vp.y()) + 14),
                    max(0, self.upper.viewport().height() - self.upper_hover_label.height() - 4))

            self.upper_hover_label.move(x, y)
            self.upper_hover_label.setVisible(True)
            self.upper_hover_label.raise_()
        except Exception:
            self.upper_hover_label.setVisible(False)

    def _on_lower_mouse_moved(self, evt):
        if self.mini:
            self.lower_hover_label.setVisible(False)
            return

        pos = evt[0] if isinstance(evt, tuple) else evt
        if pos is None:
            self.lower_hover_label.setVisible(False)
            return

        try:
            vb = self.lower_item.getViewBox()
            if not vb.sceneBoundingRect().contains(pos):
                self.lower_hover_label.setVisible(False)
                return

            mp = vb.mapSceneToView(pos)
            info = self._nearest_order_trace_info(float(mp.x()), float(mp.y()))

            if info is None:
                text = f"({mp.x():.6g}, {mp.y():.6g})"
            else:
                text = f"{info['order']:.2f}x | ({info['x']:.6g}, {info['y']:.6g})"

            self.lower_hover_label.setText(text)
            self.lower_hover_label.adjustSize()

            vp = self.lower.mapFromScene(pos)
            x = min(max(0, int(vp.x()) + 14),
                    max(0, self.lower.viewport().width() - self.lower_hover_label.width() - 4))
            y = min(max(0, int(vp.y()) + 14),
                    max(0, self.lower.viewport().height() - self.lower_hover_label.height() - 4))

            self.lower_hover_label.move(x, y)
            self.lower_hover_label.setVisible(True)
            self.lower_hover_label.raise_()
        except Exception:
            self.lower_hover_label.setVisible(False)

    def _nearest_order_trace_info(self, rpm_value: float, amp_value: float = None):
        if self.mini or not self._cursors:
            return None

        best = None
        best_metric = np.inf

        for c in self._cursors:
            tr = self._extract_order_trace(c.order)
            m = np.isfinite(tr) & np.isfinite(self.rpms)
            if np.count_nonzero(m) < 2:
                continue

            xr = self.rpms[m]
            yr = tr[m]

            x_min = float(np.min(xr))
            x_max = float(np.max(xr))
            if rpm_value < x_min or rpm_value > x_max:
                continue

            y_curve = float(np.interp(rpm_value, xr, yr))
            metric = abs(y_curve - amp_value) if (amp_value is not None and np.isfinite(amp_value)) else 0.0

            if metric < best_metric:
                best_metric = metric
                best = {
                    "x": float(rpm_value),
                    "y": y_curve,
                    "order": float(c.order),
                    "color": c.color,
                }

        return best

    def _clear_upper_click_annotations(self):
        for ann in self._upper_click_annotations[:]:
            self._remove_upper_click_annotation(ann)

    def _clear_lower_click_annotations(self):
        for ann in self._lower_click_annotations[:]:
            self._remove_lower_click_annotation(ann)

    def clear_click_annotations(self):
        self._clear_upper_click_annotations()
        self._clear_lower_click_annotations()

    def _remove_upper_click_annotation(self, ann):
        try:
            if ann.get("marker") is not None:
                self.upper_item.removeItem(ann["marker"])
        except Exception:
            pass

        try:
            if ann.get("label") is not None:
                self.upper_item.removeItem(ann["label"])
        except Exception:
            pass

        try:
            btn = ann.get("close_btn")
            if btn is not None:
                btn.hide()
                btn.deleteLater()
        except Exception:
            pass

        try:
            self._upper_click_annotations.remove(ann)
        except ValueError:
            pass

    def _remove_lower_click_annotation(self, ann):
        try:
            if ann.get("marker") is not None:
                self.lower_item.removeItem(ann["marker"])
        except Exception:
            pass

        try:
            if ann.get("label") is not None:
                self.lower_item.removeItem(ann["label"])
        except Exception:
            pass

        try:
            btn = ann.get("close_btn")
            if btn is not None:
                btn.hide()
                btn.deleteLater()
        except Exception:
            pass

        try:
            self._lower_click_annotations.remove(ann)
        except ValueError:
            pass

    def _add_upper_click_annotation(self, x: float, y: float):
        x_span = max(self.freq_max - self.freq_min, 1e-12)
        y_span = max(self.rpm_max - self.rpm_min, 1e-12)
        layout = self._choose_annotation_layout(self._upper_click_annotations, x, y, x_span, y_span)

        marker = pg.ScatterPlotItem(
            [x], [y],
            symbol="o",
            size=8,
            pen=pg.mkPen((200, 40, 40), width=1.2),
            brush=pg.mkBrush(255, 255, 255, 220),
        )
        marker.setZValue(2000)

        label = pg.TextItem(
            html=self._ann_html(f"({x:.6g}, {y:.6g})"),
            anchor=layout["anchor"]
        )
        label.setPos(layout["label_x"], layout["label_y"])
        label.setZValue(2001)

        self.upper_item.addItem(marker)
        self.upper_item.addItem(label)

        btn = QToolButton(self.upper.viewport())
        btn.setText("\u00d7")
        btn.setAutoRaise(True)
        btn.setFixedSize(16, 16)
        btn.setStyleSheet(
            "QToolButton {"
            "background: rgba(255,255,255,230);"
            "border: 1px solid #666;"
            "border-radius: 8px;"
            "font-weight: bold;"
            "padding: 0px;"
            "}"
        )

        ann = {
            "x": float(x),
            "y": float(y),
            "label_x": float(layout["label_x"]),
            "label_y": float(layout["label_y"]),
            "marker": marker,
            "label": label,
            "close_btn": btn,
        }

        btn.clicked.connect(lambda _=False, a=ann: self._remove_upper_click_annotation(a))
        self._upper_click_annotations.append(ann)

        btn.show()
        self._update_single_upper_annotation_button(ann)

    def _add_lower_click_annotation(self, x: float, y: float, order_value: float, color):
        lower_finite = self.S[np.isfinite(self.S)]
        y0 = float(np.min(lower_finite)) if lower_finite.size else 0.0
        y1 = float(np.max(lower_finite)) if lower_finite.size else 1.0
        x_span = max(self.rpm_max - self.rpm_min, 1e-12)
        y_span = max(y1 - y0, 1e-12)

        layout = self._choose_annotation_layout(self._lower_click_annotations, x, y, x_span, y_span)

        marker = pg.ScatterPlotItem(
            [x], [y],
            symbol="o",
            size=8,
            pen=pg.mkPen(color, width=1.5),
            brush=pg.mkBrush(255, 255, 255, 220),
        )
        marker.setZValue(2000)

        text = f"{order_value:.2f}x | ({x:.6g}, {y:.6g})"
        label = pg.TextItem(
            html=self._ann_html(text),
            anchor=layout["anchor"]
        )
        label.setPos(layout["label_x"], layout["label_y"])
        label.setZValue(2001)

        self.lower_item.addItem(marker)
        self.lower_item.addItem(label)

        btn = QToolButton(self.lower.viewport())
        btn.setText("\u00d7")
        btn.setAutoRaise(True)
        btn.setFixedSize(16, 16)
        btn.setStyleSheet(
            "QToolButton {"
            "background: rgba(255,255,255,230);"
            "border: 1px solid #666;"
            "border-radius: 8px;"
            "font-weight: bold;"
            "padding: 0px;"
            "}"
        )

        ann = {
            "x": float(x),
            "y": float(y),
            "order": float(order_value),
            "text": text,
            "color": color,
            "label_x": float(layout["label_x"]),
            "label_y": float(layout["label_y"]),
            "marker": marker,
            "label": label,
            "close_btn": btn,
        }

        btn.clicked.connect(lambda _=False, a=ann: self._remove_lower_click_annotation(a))
        self._lower_click_annotations.append(ann)

        btn.show()
        self._update_single_lower_annotation_button(ann)

    def _update_single_upper_annotation_button(self, ann):
        btn = ann.get("close_btn")
        label = ann.get("label")
        if btn is None or label is None:
            return

        try:
            sr = label.sceneBoundingRect()
            vp_pt = self.upper.mapFromScene(sr.topRight())

            x = int(vp_pt.x()) - btn.width() // 2
            y = int(vp_pt.y()) - btn.height() // 2
            btn.move(x, y)
            btn.raise_()

            viewport_rect = self.upper.viewport().rect()
            btn.setVisible(viewport_rect.adjusted(-20, -20, 20, 20).contains(vp_pt))
        except Exception:
            btn.setVisible(False)

    def _update_single_lower_annotation_button(self, ann):
        btn = ann.get("close_btn")
        label = ann.get("label")
        if btn is None or label is None:
            return

        try:
            sr = label.sceneBoundingRect()
            vp_pt = self.lower.mapFromScene(sr.topRight())

            x = int(vp_pt.x()) - btn.width() // 2
            y = int(vp_pt.y()) - btn.height() // 2
            btn.move(x, y)
            btn.raise_()

            viewport_rect = self.lower.viewport().rect()
            btn.setVisible(viewport_rect.adjusted(-20, -20, 20, 20).contains(vp_pt))
        except Exception:
            btn.setVisible(False)

    def _update_upper_annotation_buttons(self, *args):
        for ann in self._upper_click_annotations[:]:
            self._update_single_upper_annotation_button(ann)

    def _update_lower_annotation_buttons(self, *args):
        for ann in self._lower_click_annotations[:]:
            self._update_single_lower_annotation_button(ann)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_upper_annotation_buttons()
        self._update_lower_annotation_buttons()
        # ───────────────── FIX 3 ─────────────────
        # Previously there was `if preserve_view:` here, which always raised
        # NameError (preserve_view is a parameter of set_state, not in scope here).
        # After a window resize, pix_h/pix_w change, which changes target_rows
        # and target_cols in _render_waterfall_view_from_frames. We must
        # always re-trigger the adaptive renderer so the output matrix
        # matches the new viewport pixel density.
        self._suspend_viewport_reinterp = False
        self._visible_reinterp_timer.start()
        # ─────────────── END FIX 3 ───────────────

    def set_rect_zoom_active(self, active: bool):
        self._rect_zoom_active = bool(active)
        mode = pg.ViewBox.RectMode if active else pg.ViewBox.PanMode
        self.upper_item.getViewBox().setMouseMode(mode)
        if not self.mini:
            self.lower_item.getViewBox().setMouseMode(mode)

    def zoom_axes(self, factor: float):
        try:
            self.upper_item.getViewBox().scaleBy((factor, factor))
        except Exception:
            pass
        if not self.mini:
            try:
                self.lower_item.getViewBox().scaleBy((factor, factor))
            except Exception:
                pass

    def reset_view(self):
        if self._home_upper is not None:
            x0, x1, y0, y1 = self._home_upper
            vb = self.upper_item.getViewBox()
            vb.setXRange(x0, x1, padding=0.0)
            vb.setYRange(y0, y1, padding=0.0)
        if self._home_lower is not None and not self.mini:
            x0, x1, y0, y1 = self._home_lower
            vb = self.lower_item.getViewBox()
            vb.setXRange(x0, x1, padding=0.0)
            vb.setYRange(y0, y1, padding=0.0)

        self.clear_click_annotations()

    def get_export_annotations(self):
        out = []
        for ann in self._upper_click_annotations:
            try:
                x = float(ann["x"])
                y = float(ann["y"])
                if np.isfinite(x) and np.isfinite(y):
                    out.append({
                        "x": x,
                        "y": y,
                        "text": f"({x:.6g}, {y:.6g})",
                    })
            except Exception:
                pass
        return out

    def get_export_lower_annotations(self):
        out = []
        for ann in self._lower_click_annotations:
            try:
                x = float(ann["x"])
                y = float(ann["y"])
                order_value = float(ann["order"])
                if np.isfinite(x) and np.isfinite(y):
                    out.append({
                        "x": x,
                        "y": y,
                        "order": order_value,
                        "text": ann.get("text", f"{order_value:.2f}x | ({x:.6g}, {y:.6g})"),
                        "color": ann.get("color", (200, 40, 40)),
                    })
            except Exception:
                pass
        return out

    def clear_cursors(self):
        self._clear_lower_click_annotations()

        for c in self._cursors[:]:
            c.remove()
        self._cursors.clear()

        if not self.mini:
            self.lbl_order_info.setText("Active orders: -")
        self._update_lower_plot()

    def _add_cursor_default(self):
        if self.rpms.size == 0:
            return
        if not self._cursors:
            order = 1.0
        else:
            order = float(self._cursors[-1].order + 1.0)
        self.add_cursor(order)

    def add_cursor(self, order_value: float):
        color = self._cursor_colors[len(self._cursors) % len(self._cursor_colors)]
        cursor = WaterfallOrderCursor(self, order_value, color)
        cursor.changed.connect(self._update_lower_plot)
        self._cursors.append(cursor)
        self._update_lower_plot()

    def export_order_cursor_specs(self):
        out = []
        for c in self._cursors:
            out.append({"order": float(c.order), "color": c.color})
        return out

    def _extract_order_trace(self, order_value: float):
        rpms_src = self.rpms_trace if self.rpms_trace.size else self.rpms
        S_src = self.S_trace if self.S_trace.size else self.S
        freqs_src = self.freqs_trace if self.freqs_trace.size else self.freqs
        if S_src.size == 0 or rpms_src.size == 0 or freqs_src.size == 0:
            return np.array([], dtype=np.float64)
        freq_min = float(freqs_src[0]) if freqs_src.size else 0.0
        freq_max = float(freqs_src[-1]) if freqs_src.size else 0.0
        freq_track = (float(order_value) * rpms_src) / 60.0
        out = np.full(rpms_src.shape, np.nan, dtype=np.float64)
        for i in range(rpms_src.size):
            f = freq_track[i]
            if f < freq_min or f > freq_max:
                continue
            out[i] = np.interp(f, freqs_src, S_src[i, :], left=np.nan, right=np.nan)
        return out

    def _set_home(self):
        upper_freqs = self.freqs_full_display if self.freqs_full_display.size else self.freqs
        upper_rpms = self.rpms_full_display if self.rpms_full_display.size else self.rpms
        upper_S = self.S_full_display if self.S_full_display.size else self.S
        upper_freq_min = float(upper_freqs[0]) if upper_freqs.size else self.freq_min
        upper_freq_max = float(upper_freqs[-1]) if upper_freqs.size else self.freq_max
        upper_rpm_min = float(upper_rpms[0]) if upper_rpms.size else self.rpm_min
        upper_rpm_max = float(upper_rpms[-1]) if upper_rpms.size else self.rpm_max
        self._home_upper = (upper_freq_min, upper_freq_max, upper_rpm_min, upper_rpm_max)
        self.full_freq_min = upper_freq_min
        self.full_freq_max = upper_freq_max
        self.full_rpm_min = upper_rpm_min
        self.full_rpm_max = upper_rpm_max
        if not self.mini:
            lower_S = self.S_trace if self.S_trace.size else upper_S
            lower_rpms = self.rpms_trace if self.rpms_trace.size else upper_rpms
            finite = lower_S[np.isfinite(lower_S)]
            y0 = float(np.min(finite)) if finite.size else 0.0
            y1 = float(np.max(finite)) if finite.size else 1.0
            if y1 <= y0:
                y1 = y0 + 1.0
            lower_rpm_min = float(lower_rpms[0]) if lower_rpms.size else upper_rpm_min
            lower_rpm_max = float(lower_rpms[-1]) if lower_rpms.size else upper_rpm_max
            self._home_lower = (lower_rpm_min, lower_rpm_max, y0, y1)

    def _update_lower_plot(self):
        if self.mini:
            return

        self._clear_lower_click_annotations()

        if self._cursors:
            txt = ", ".join(f"{c.order:.2f}x" for c in self._cursors)
            self.lbl_order_info.setText(f"Active orders: {txt}")
        else:
            self.lbl_order_info.setText("Active orders: -")

        self.lower_item.clear()
        if self._legend is not None:
            try:
                self._legend.scene().removeItem(self._legend)
            except Exception:
                pass
            self._legend = None
        self._legend = self.lower_item.addLegend(offset=(10, 10))

        self.lower_item.setLabel("bottom", "RPM")
        self.lower_item.setLabel("left", self.state.get("colorbar_label", "Amplitude") if self.state else "Amplitude")

        rpms_src = self.rpms_trace if self.rpms_trace.size else self.rpms
        ymins = []
        ymaxs = []
        for c in self._cursors:
            tr = self._extract_order_trace(c.order)
            m = np.isfinite(tr) & np.isfinite(rpms_src)
            if not np.any(m):
                continue
            self.lower_item.plot(
                rpms_src[m], tr[m],
                pen=pg.mkPen(c.color, width=2),
                name=f"Order {c.order:.2f}x",
            )
            ymins.append(float(np.min(tr[m])))
            ymaxs.append(float(np.max(tr[m])))

        if ymins and ymaxs:
            y0 = min(ymins)
            y1 = max(ymaxs)
            if y1 <= y0:
                y1 = y0 + 1.0
            lower_rpm_min = float(rpms_src[0]) if rpms_src.size else self.rpm_min
            lower_rpm_max = float(rpms_src[-1]) if rpms_src.size else self.rpm_max
            self.lower_item.getViewBox().setXRange(lower_rpm_min, lower_rpm_max, padding=0.0)
            self.lower_item.getViewBox().setYRange(y0, y1, padding=0.05)

    def open_3d_window(self):
        if self.state is None:
            return
        state_for_window = dict(self.state)
        state_for_window["freqs_display"] = self.freqs.copy()
        state_for_window["rpms_display"] = self.rpms.copy()
        state_for_window["S_display"] = self.S.copy()
        dlg = Waterfall3DWindow(state_for_window, self)
        dlg.exec_()

    def open_2d_standalone_window(self):
        """Open standalone 2D waterfall window with current colormap, color range, and viewport."""
        if self.state is None:
            return

        # Capture current zoom viewport from upper plot
        view_range = None
        try:
            vb = self.upper_item.getViewBox()
            xr, yr = vb.viewRange()
            view_range = (
                (float(xr[0]), float(xr[1])),
                (float(yr[0]), float(yr[1])),
            )
        except Exception:
            pass

        # Capture current order cursor specs
        cursor_specs = self.export_order_cursor_specs()

        # Capture current color limits from PyQtGraph colorbar
        color_limits = None
        try:
            if self._upper_color_bar is not None:
                lo, hi = self._upper_color_bar.levels()
                color_limits = {"vmin": float(lo), "vmax": float(hi)}
        except Exception:
            pass

        # Determine colormap name from stored color config
        cmap_name = "jet"
        try:
            stored_cmap = self._last_color_cfg.get("cmap", "")
            if stored_cmap:
                reverse = bool(self._last_color_cfg.get("reverse", False))
                cmap_name = f"{stored_cmap}_r" if reverse else stored_cmap
        except Exception:
            pass

        state_for_window = dict(self.state)
        state_for_window["freqs_display"] = self.freqs.copy()
        state_for_window["rpms_display"] = self.rpms.copy()
        state_for_window["S_display"] = self.S.copy()

        # Optional: pass raw frame data so future adaptive-render hooks
        # in the standalone window can use it. Safe to include unconditionally.
        if self.frame_rpms.size and self.frame_spectra_linear.size and self.freqs_analysis.size:
            state_for_window["frame_rpms"] = self.frame_rpms.copy()
            state_for_window["frame_spectra_linear"] = self.frame_spectra_linear.copy()
            state_for_window["freqs_analysis"] = self.freqs_analysis.copy()

        dlg = Waterfall2DStandaloneWindow(
            state=state_for_window,
            parent=self,
            view_range=view_range,
            cmap_name=cmap_name,
            order_cursor_specs=cursor_specs,
            color_limits=color_limits,
        )
        dlg.exec_()

    def set_state(self, state: dict, color_cfg=None, preserve_view=False, preserve_annotations=False):
        color_cfg = color_cfg or {}
        self._last_color_cfg = dict(color_cfg)

        old_upper = None
        old_lower = None
        old_orders = [float(c.order) for c in self._cursors] if preserve_annotations else []
        old_upper_points = self.get_export_annotations() if preserve_annotations else []
        old_lower_points = self.get_export_lower_annotations() if preserve_annotations else []

        if preserve_view:
            try:
                xr, yr = self.upper_item.getViewBox().viewRange()
                old_upper = (xr[0], xr[1], yr[0], yr[1])
            except Exception:
                old_upper = None
            try:
                if not self.mini:
                    xr, yr = self.lower_item.getViewBox().viewRange()
                    old_lower = (xr[0], xr[1], yr[0], yr[1])
            except Exception:
                old_lower = None

        self._suspend_viewport_reinterp = True
        self.state = state
        self.plot_key = state["key"]
        self.freqs_analysis = np.asarray(state.get("freqs_analysis", state["freqs"]), dtype=np.float64)
        self.freqs_full_display = np.asarray(state.get("freqs_display", state["freqs"]), dtype=np.float64)
        self.freqs = self.freqs_full_display.copy()
        self.rpms_full_display = np.asarray(state.get("rpms_display", state["rpms"]), dtype=np.float64)
        self.S_full_display = np.asarray(state.get("S_display", state["S"]), dtype=np.float64)
        self.rpms = self.rpms_full_display.copy()
        self.S = self.S_full_display.copy()
        self.freqs_trace = self.freqs_full_display.copy()
        self.rpms_trace = self.rpms_full_display.copy()
        self.S_trace = self.S_full_display.copy()
        self.rpms_analysis = np.asarray(state.get("rpms_analysis", state["rpms"]), dtype=np.float64)
        self.S_analysis = np.asarray(state.get("S_analysis", state["S"]), dtype=np.float64)
        self.frame_rpms = np.asarray(state.get("frame_rpms", []), dtype=np.float64)
        self.frame_spectra_linear = np.asarray(state.get("frame_spectra_linear", []), dtype=np.float64)
        self._viewport_dense_factor = float(max(1.0, state.get("viewport_dense_factor", 4)))
        self._viewport_min_points = int(max(200, state.get("viewport_min_points", 600)))
        self._viewport_max_points = int(max(self._viewport_min_points, state.get("viewport_max_points", 3000)))
        self._viewport_freq_dense_factor = float(max(1.0, state.get("viewport_freq_dense_factor", 2.0)))
        self._viewport_min_cols = int(max(128, state.get("viewport_min_cols", 900)))
        self._viewport_max_cols = int(max(self._viewport_min_cols, state.get("viewport_max_cols", 4096)))
        self._viewport_row_pixel_factor = float(max(1.0, state.get("viewport_row_pixel_factor", 1.25)))
        self._viewport_col_pixel_factor = float(max(1.0, state.get("viewport_col_pixel_factor", 1.25)))
        self._high_quality_zoom = bool(state.get("high_quality_waterfall_zoom", False))
        self._hq_supersample = int(max(1, state.get("waterfall_hq_supersample", 2 if self._high_quality_zoom else 1)))
        self._hq_use_monotone = bool(state.get("waterfall_hq_monotone_interp", True))
        self._visible_reinterp_timer.setInterval(120 if self._high_quality_zoom else 90)

        self.freq_min = float(self.freqs[0]) if self.freqs.size else 0.0
        self.freq_max = float(self.freqs[-1]) if self.freqs.size else 1.0
        self.rpm_min = float(self.rpms[0]) if self.rpms.size else 0.0
        self.rpm_max = float(self.rpms[-1]) if self.rpms.size else 1.0

        self.upper_hover_label.setVisible(False)
        self.lower_hover_label.setVisible(False)
        self.clear_click_annotations()

        self._clear_upper_colorbar()
        self.upper_item.clear()
        self.lower_item.clear()
        self._upper_image_item = None
        self.clear_cursors()

        if self.freqs.size == 0 or self.rpms.size == 0 or self.S.size == 0:
            return

        if self.freqs.size > 1:
            dx = float(np.median(np.diff(self.freqs)))
        else:
            dx = 1.0
        if self.rpms.size > 1:
            dy = float(np.median(np.diff(self.rpms)))
        else:
            dy = 1.0

        x0 = float(self.freqs[0] - 0.5 * dx)
        x1 = float(self.freqs[-1] + 0.5 * dx)
        y0 = float(self.rpms[0] - 0.5 * dy)
        y1 = float(self.rpms[-1] + 0.5 * dy)

        style = resolve_image_style(self.S, state.get("scale", "Linear"), color_cfg)

        img = pg.ImageItem(axisOrder="row-major")
        img.setImage(self.S, autoLevels=False)
        # ───────────────── FIX 1 ─────────────────
        # Use bilinear texture filtering so zooming produces smooth gradients
        # instead of the blocky nearest-neighbor look. This setting persists
        # across all subsequent setImage() calls on the same ImageItem, so
        # the adaptive re-render path (_set_upper_image_matrix) inherits it
        # automatically.
        try:
            img.setOpts(interpolate=True)
        except Exception:
            pass
        # ─────────────── END FIX 1 ───────────────
        img.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        img.setLevels((style["vmin"], style["vmax"]))
        img.setColorMap(style["cm"])
        self.upper_item.addItem(img)

        cbar = pg.ColorBarItem(
            values=(style["vmin"], style["vmax"]),
            colorMap=style["cm"],
            label=state.get("colorbar_label", "Amplitude"),
            interactive=bool(style.get("interactive", False)),
            width=18,
        )
        self._suspend_colorbar_sync = True
        cbar.setImageItem(img, insert_in=self.upper_item)
        self._suspend_colorbar_sync = False
        self._upper_image_item = img
        self._upper_color_bar = cbar

        if bool(style.get("interactive", False)):
            try:
                cbar.sigLevelsChanged.connect(self._on_upper_colorbar_levels_changed)
            except Exception:
                pass
            try:
                cbar.sigLevelsChangeFinished.connect(self._on_upper_colorbar_levels_change_finished)
            except Exception:
                pass

        self.upper_item.setLabel("bottom", state.get("xlabel", "Frequency (Hz)"))
        self.upper_item.setLabel("left", state.get("ylabel", "RPM"))

        if self._recompute_full_upper_display():
            self._set_upper_image_matrix(self.freqs_full_display, self.rpms_full_display, self.S_full_display)

        self._set_home()

        if old_upper is not None:
            self.upper_item.getViewBox().setXRange(old_upper[0], old_upper[1], padding=0.0)
            self.upper_item.getViewBox().setYRange(old_upper[2], old_upper[3], padding=0.0)
        else:
            if self._home_upper is not None:
                self.upper_item.getViewBox().setXRange(self._home_upper[0], self._home_upper[1], padding=0.0)
                self.upper_item.getViewBox().setYRange(self._home_upper[2], self._home_upper[3], padding=0.0)

        if old_orders:
            for od in old_orders:
                self.add_cursor(od)
        else:
            self.add_cursor(1.0)
            self.add_cursor(2.0)

        if old_lower is not None and not self.mini:
            try:
                self.lower_item.getViewBox().setXRange(old_lower[0], old_lower[1], padding=0.0)
                self.lower_item.getViewBox().setYRange(old_lower[2], old_lower[3], padding=0.0)
            except Exception:
                pass

        for ann in old_upper_points:
            self._add_upper_click_annotation(ann["x"], ann["y"])

        for ann in old_lower_points:
            self._add_lower_click_annotation(
                ann["x"],
                ann["y"],
                ann.get("order", 0.0),
                ann.get("color", (200, 40, 40)),
            )

        self._update_upper_annotation_buttons()
        self._update_lower_annotation_buttons()

        # After state is fully configured, re-enable viewport-adaptive
        # re-rendering and trigger an initial render so the visible region
        # is rendered at the current viewport pixel density.
        self._suspend_viewport_reinterp = False
        self._visible_reinterp_timer.start()


class PlotMiniWidget(QFrame):
    clicked = pyqtSignal(str)

    def __init__(self, plot_state):
        super().__init__()
        self.plot_state = plot_state
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setStyleSheet("PlotMiniWidget { background: #FAFAFA; }")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(240, 180)

        lbl = QLabel(plot_state["title"])
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("font-weight: 600; color: #222;")

        self.plot = InteractivePlotWidget(mini=True)
        self.plot.set_state(plot_state)
        self.plot.clicked.connect(self._emit_clicked)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        lay.addWidget(lbl)
        lay.addWidget(self.plot, 1)

    def _emit_clicked(self):
        self.clicked.emit(self.plot_state["key"])


class WaterfallMiniWidget(QFrame):
    clicked = pyqtSignal(str)

    def __init__(self, plot_state):
        super().__init__()
        self.plot_state = plot_state
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setStyleSheet("WaterfallMiniWidget { background: #FAFAFA; }")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(280, 200)

        lbl = QLabel(plot_state["title"])
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("font-weight: 600; color: #222;")

        self.plot = WaterfallPlotWidget(mini=True)
        self.plot.set_state(plot_state, color_cfg=plot_state.get("color_cfg", {}))
        self.plot.clicked.connect(self._emit_clicked)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        lay.addWidget(lbl)
        lay.addWidget(self.plot, 1)

    def _emit_clicked(self):
        self.clicked.emit(self.plot_state["key"])


class DetailPlotWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.plot = InteractivePlotWidget(mini=False)
        self.plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.title_label = QLabel("")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-weight: 700; color: #111; padding-top: 4px;")

        self.toolbar = QHBoxLayout()
        self.btn_rect_zoom = QToolButton()
        self.btn_rect_zoom.setText("Rectangle zoom")
        self.btn_rect_zoom.setCheckable(True)

        self.btn_reset = QPushButton("Reset View")
        self.btn_zoom_in = QPushButton("Zoom In")
        self.btn_zoom_out = QPushButton("Zoom Out")
        self.btn_export = QPushButton("Export as Image...")

        for w in [self.btn_rect_zoom, self.btn_reset, self.btn_zoom_in, self.btn_zoom_out, self.btn_export]:
            self.toolbar.addWidget(w)
        self.toolbar.addStretch(1)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self.title_label)
        lay.addLayout(self.toolbar)
        lay.addWidget(self.plot, 1)

        self.btn_rect_zoom.toggled.connect(self.plot.set_rect_zoom_active)
        self.btn_reset.clicked.connect(self.plot.reset_view)
        self.btn_zoom_in.clicked.connect(lambda: self.plot.zoom_axes(0.8))
        self.btn_zoom_out.clicked.connect(lambda: self.plot.zoom_axes(1.25))

    def set_state(self, state, color_cfg=None, preserve_view=False, preserve_annotations=False):
        self.title_label.setText(state.get("title", ""))
        self.plot.set_state(state, color_cfg=color_cfg,
                            preserve_view=preserve_view,
                            preserve_annotations=preserve_annotations)


class WaterfallDetailWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.plot = WaterfallPlotWidget(mini=False)
        self.plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.title_label = QLabel("")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-weight: 700; color: #111; padding-top: 4px;")

        self.toolbar = QHBoxLayout()
        self.btn_rect_zoom = QToolButton()
        self.btn_rect_zoom.setText("Rectangle zoom")
        self.btn_rect_zoom.setCheckable(True)

        self.btn_reset = QPushButton("Reset View")
        self.btn_zoom_in = QPushButton("Zoom In")
        self.btn_zoom_out = QPushButton("Zoom Out")
        self.btn_export = QPushButton("Export as Image...")

        for w in [self.btn_rect_zoom, self.btn_reset, self.btn_zoom_in,
                  self.btn_zoom_out, self.btn_export]:
            self.toolbar.addWidget(w)
        self.toolbar.addStretch(1)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self.title_label)
        lay.addLayout(self.toolbar)
        lay.addWidget(self.plot, 1)

        self.btn_rect_zoom.toggled.connect(self.plot.set_rect_zoom_active)
        self.btn_reset.clicked.connect(self.plot.reset_view)
        self.btn_zoom_in.clicked.connect(lambda: self.plot.zoom_axes(0.8))
        self.btn_zoom_out.clicked.connect(lambda: self.plot.zoom_axes(1.25))

    def set_state(self, state, color_cfg=None, preserve_view=False, preserve_annotations=False):
        self.title_label.setText(state.get("title", ""))
        self.plot.set_state(state, color_cfg=color_cfg,
                            preserve_view=preserve_view,
                            preserve_annotations=preserve_annotations)


class TimeDomainZoomFFTWidget(QDialog):
    """Interactive time-domain zoom window with on-the-fly FFT of the zoomed segment.

    Click 'Select FFT Range' to enable a LinearRegionItem on the time plot.
    Dragging the region updates the FFT on the right pane in real time.
    """

    def __init__(self, times: np.ndarray, ampls: np.ndarray, fs: float,
                 name: str, unit_lbl: str, parent=None):
        super().__init__(parent)
        self.times = np.asarray(times, dtype=np.float64)
        self.ampls = np.asarray(ampls, dtype=np.float64)
        self.fs = float(fs)
        self.name = str(name)
        self.unit_lbl = str(unit_lbl)

        self.setWindowTitle(f"Time-Domain Zoom & FFT - {self.name}")
        self.resize(1400, 700)

        lay_root = QVBoxLayout(self)

        toolbar = QHBoxLayout()
        self.btn_select_range = QPushButton("Select FFT Range")
        self.btn_select_range.setCheckable(True)
        self.btn_select_range.setToolTip(
            "Toggle region selector on the time-domain plot. "
            "Drag the colored region to update the FFT on the right."
        )
        self.btn_reset_time = QPushButton("Reset Time Axis")
        self.btn_reset_fft = QPushButton("Reset FFT Axis")
        self.btn_export = QPushButton("Export Both Plots...")
        self.btn_close = QPushButton("Close")

        self.lbl_info = QLabel("Range: -")
        self.lbl_info.setStyleSheet("color: #333; padding-left: 12px;")

        toolbar.addWidget(self.btn_select_range)
        toolbar.addWidget(self.btn_reset_time)
        toolbar.addWidget(self.btn_reset_fft)
        toolbar.addWidget(self.btn_export)
        toolbar.addWidget(self.lbl_info, 1)
        toolbar.addWidget(self.btn_close)

        lay_root.addLayout(toolbar)

        splitter_lay = QHBoxLayout()
        lay_root.addLayout(splitter_lay, 1)

        self.time_plot = pg.PlotWidget(background="w")
        self.time_item = self.time_plot.getPlotItem()
        self.time_item.setLabel("bottom", "Time (s)")
        self.time_item.setLabel("left", f"Amplitude ({self.unit_lbl})")
        self.time_item.setTitle("Time-domain (zoom/pan with mouse)")
        self.time_item.showGrid(x=True, y=True, alpha=0.25)
        self.time_plot.setMenuEnabled(False)

        self.fft_plot = pg.PlotWidget(background="w")
        self.fft_item = self.fft_plot.getPlotItem()
        self.fft_item.setLabel("bottom", "Frequency (Hz)")
        self.fft_item.setLabel("left", f"FFT ({self.unit_lbl})")
        self.fft_item.setTitle("FFT of selected range")
        self.fft_item.showGrid(x=True, y=True, alpha=0.25)
        self.fft_plot.setMenuEnabled(False)

        splitter_lay.addWidget(self.time_plot, 1)
        splitter_lay.addWidget(self.fft_plot, 1)

        self._draw_time_signal()

        self.region = None
        self._fft_curve = None
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(60)
        self._update_timer.timeout.connect(self._update_fft_for_region)

        self.btn_select_range.toggled.connect(self._toggle_region_selector)
        self.btn_reset_time.clicked.connect(self._reset_time_view)
        self.btn_reset_fft.clicked.connect(self._reset_fft_view)
        self.btn_export.clicked.connect(self._export_both)
        self.btn_close.clicked.connect(self.close)

    def _draw_time_signal(self):
        if self.times.size == 0 or self.ampls.size == 0:
            return
        n = self.times.size
        max_points = 200_000
        if n > max_points:
            step = max(1, n // max_points)
            x = self.times[::step]
            y = self.ampls[::step]
        else:
            x = self.times
            y = self.ampls
        curve = pg.PlotDataItem(x, y, pen=pg.mkPen((35, 110, 210), width=1), connect="finite")
        try:
            curve.setSkipFiniteCheck(True)
        except Exception:
            pass
        self.time_item.addItem(curve)

        finite_y = y[np.isfinite(y)]
        ymin = float(np.min(finite_y)) if finite_y.size else -1.0
        ymax = float(np.max(finite_y)) if finite_y.size else 1.0
        if ymax <= ymin:
            ymax = ymin + 1.0
        self.time_item.getViewBox().setXRange(float(self.times[0]), float(self.times[-1]), padding=0.01)
        self.time_item.getViewBox().setYRange(ymin, ymax, padding=0.05)

    def _toggle_region_selector(self, checked: bool):
        if checked:
            if self.region is None:
                t0 = float(self.times[0])
                t1 = float(self.times[-1])
                span = t1 - t0
                r0 = t0 + 0.25 * span
                r1 = t0 + 0.75 * span
                self.region = pg.LinearRegionItem(values=(r0, r1), brush=(30, 120, 240, 45))
                self.region.setZValue(100)
                self.time_item.addItem(self.region)
                self.region.sigRegionChanged.connect(lambda: self._update_timer.start())
            else:
                self.region.setVisible(True)
            self._update_fft_for_region()
        else:
            if self.region is not None:
                self.region.setVisible(False)
            self.fft_item.clear()
            self._fft_curve = None
            self.lbl_info.setText("Range: -")

    def _reset_time_view(self):
        if self.times.size == 0:
            return
        finite_y = self.ampls[np.isfinite(self.ampls)]
        ymin = float(np.min(finite_y)) if finite_y.size else -1.0
        ymax = float(np.max(finite_y)) if finite_y.size else 1.0
        if ymax <= ymin:
            ymax = ymin + 1.0
        self.time_item.getViewBox().setXRange(float(self.times[0]), float(self.times[-1]), padding=0.01)
        self.time_item.getViewBox().setYRange(ymin, ymax, padding=0.05)

    def _reset_fft_view(self):
        if self._fft_curve is not None:
            self.fft_item.getViewBox().autoRange()

    def _update_fft_for_region(self):
        if self.region is None or not self.region.isVisible():
            return
        t0, t1 = self.region.getRegion()
        if t1 <= t0:
            return

        i0 = int(np.searchsorted(self.times, t0, side="left"))
        i1 = int(np.searchsorted(self.times, t1, side="right"))
        if i1 - i0 < 4:
            self.fft_item.clear()
            self._fft_curve = None
            self.lbl_info.setText(
                f"Range: [{t0:.4g}, {t1:.4g}] s - too few samples"
            )
            return

        y_seg = self.ampls[i0:i1]
        y_seg = np.nan_to_num(y_seg, nan=0.0, posinf=0.0, neginf=0.0)
        n = y_seg.size
        w = np.hanning(n).astype(np.float64)
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            w_sum = float(n)

        spec = np.fft.rfft(y_seg * w)
        mag = np.abs(spec)
        A_peak = (2.0 * mag) / w_sum
        if A_peak.size > 0:
            A_peak[0] *= 0.5
        if (n % 2 == 0) and A_peak.size > 1:
            A_peak[-1] *= 0.5

        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)

        self.fft_item.clear()
        self._fft_curve = pg.PlotDataItem(
            freqs, A_peak, pen=pg.mkPen((200, 40, 40), width=1.4)
        )
        self.fft_item.addItem(self._fft_curve)
        self.fft_item.getViewBox().autoRange()

        self.lbl_info.setText(
            f"Range: [{t0:.4g}, {t1:.4g}] s | "
            f"Duration: {(t1 - t0):.4g} s | "
            f"Samples: {n} | df = {self.fs / n:.4g} Hz"
        )

    def _export_both(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Time + FFT",
            f"{self.name}_zoom_fft.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if not path:
            return

        try:
            fig = Figure(figsize=(16, 6), dpi=300, constrained_layout=True)
            ax_t = fig.add_subplot(121)
            ax_t.plot(self.times, self.ampls, "b-", linewidth=0.8)
            ax_t.set_xlabel("Time (s)")
            ax_t.set_ylabel(f"Amplitude ({self.unit_lbl})")
            ax_t.set_title(f"Time-domain signal")
            ax_t.grid(True, alpha=0.3)

            vr = self.time_item.getViewBox().viewRange()
            ax_t.set_xlim(vr[0][0], vr[0][1])
            ax_t.set_ylim(vr[1][0], vr[1][1])

            if self.region is not None and self.region.isVisible():
                t0, t1 = self.region.getRegion()
                ax_t.axvspan(t0, t1, alpha=0.25, color="orange", label=f"FFT range")
                ax_t.legend(loc="upper right")

            ax_f = fig.add_subplot(122)
            if self._fft_curve is not None:
                fx, fy = self._fft_curve.getData()
                ax_f.plot(fx, fy, "r-", linewidth=0.9)
                vr_f = self.fft_item.getViewBox().viewRange()
                ax_f.set_xlim(vr_f[0][0], vr_f[0][1])
                ax_f.set_ylim(vr_f[1][0], vr_f[1][1])
            ax_f.set_xlabel("Frequency (Hz)")
            ax_f.set_ylabel(f"FFT ({self.unit_lbl})")
            ax_f.set_title("FFT of selected range")
            ax_f.grid(True, alpha=0.3)

            fig.suptitle(f"{self.name} - Time-domain Zoom & FFT")
            fig.savefig(path, dpi=300)
            QMessageBox.information(self, "Export", f"Saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))


class FileOpenWorker(QObject):
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, path: str, file_type: str):
        super().__init__()
        self.path = path
        self.file_type = file_type

    @pyqtSlot()
    def run(self):
        try:
            containers = read_source_container_names(self.path, self.file_type)
            self.finished.emit({
                "path": self.path,
                "file_type": self.file_type,
                "containers": containers,
            })
        except Exception as e:
            self.failed.emit(str(e))


class SheetLoadWorker(QObject):
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, path: str, file_type: str, container: str):
        super().__init__()
        self.path = path
        self.file_type = file_type
        self.container = container

    @pyqtSlot()
    def run(self):
        try:
            names = read_source_signal_names(self.path, self.file_type, self.container)
            self.finished.emit({
                "path": self.path,
                "file_type": self.file_type,
                "container": self.container,
                "signals": names,
            })
        except Exception as e:
            self.failed.emit(str(e))


class PlotWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def _butterworth_filter(y: np.ndarray, fs: float, cutoff_hz: float,
                            order: int = 4, kind: str = "lowpass") -> np.ndarray:
        if not _HAVE_SCIPY or butter is None or sosfiltfilt is None:
            return y
        try:
            nyq = max(fs / 2.0, 1e-9)
            wn = float(cutoff_hz) / nyq
            wn = float(np.clip(wn, 1e-6, 0.999999))
            sos = butter(int(order), wn, btype=str(kind), output="sos")
            return sosfiltfilt(sos, np.asarray(y, dtype=np.float64))
        except Exception:
            return y

    @staticmethod
    def _resample_onto_timebase(values: np.ndarray, src_fs: float,
                                dst_n: int, dst_fs: float,
                                kind: str = "linear") -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0 or dst_n == 0:
            return np.zeros(dst_n, dtype=np.float64)

        src_t = np.arange(values.size, dtype=np.float64) / max(src_fs, 1e-12)
        dst_t = np.arange(dst_n, dtype=np.float64) / max(dst_fs, 1e-12)

        if kind.strip().lower() == "nearest":
            idx = np.clip(np.round(dst_t * src_fs).astype(np.int64), 0, values.size - 1)
            return values[idx]

        return np.interp(dst_t, src_t, values, left=values[0], right=values[-1])

    def _align_waterfall_signals(self, rpm_raw: np.ndarray, acc_raw: np.ndarray,
                                 rpm_fs: float, acc_fs: float,
                                 mode: str, interp: str):
        rpm_raw = np.nan_to_num(np.asarray(rpm_raw, dtype=np.float64), nan=0.0)
        acc_raw = np.nan_to_num(np.asarray(acc_raw, dtype=np.float64), nan=0.0)

        mode = (mode or "").strip().lower()

        if mode.startswith("resample rpm"):
            n_out = min(acc_raw.size,
                        int(np.floor(rpm_raw.size * (acc_fs / max(rpm_fs, 1e-12)))))
            rpm_on_acc = self._resample_onto_timebase(rpm_raw, rpm_fs, n_out, acc_fs, kind=interp)
            acc_out = acc_raw[:n_out]
            return rpm_on_acc, acc_out, float(acc_fs)

        if mode.startswith("resample acc"):
            n_out = min(rpm_raw.size,
                        int(np.floor(acc_raw.size * (rpm_fs / max(acc_fs, 1e-12)))))
            acc_on_rpm = self._resample_onto_timebase(acc_raw, acc_fs, n_out, rpm_fs, kind=interp)
            rpm_out = rpm_raw[:n_out]
            return rpm_out, acc_on_rpm, float(rpm_fs)

        n = min(rpm_raw.size, acc_raw.size)
        fs = float(acc_fs) if acc_fs > 0 else float(rpm_fs)
        return rpm_raw[:n], acc_raw[:n], fs

    def _compute_waterfall_map(self, y: np.ndarray, rpm: np.ndarray, fs: float,
                               nfft: int, overlap_pct: float,
                               output_type: str, scale: str,
                               amp_conv: str, signal_kind: str):
        y = np.nan_to_num(np.asarray(y, dtype=np.float64))
        rpm = np.nan_to_num(np.asarray(rpm, dtype=np.float64))

        n = min(y.size, rpm.size)
        y = y[:n]
        rpm = rpm[:n]

        fs = float(fs)
        nfft = max(2, int(nfft))
        ovl = np.clip(float(overlap_pct) / 100.0, 0.0, 0.95)
        hop = max(1, int(round(nfft * (1.0 - ovl))))

        w = np.hanning(nfft).astype(np.float64)
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            w_sum = float(nfft)

        if y.size < nfft:
            return (np.array([], dtype=np.float64),
                    np.array([], dtype=np.float64),
                    np.empty((0, 0), dtype=np.float64),
                    np.empty((0, 0), dtype=np.float64))

        n_frames = max(1, 1 + (y.size - nfft) // hop)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

        spectra_linear = np.zeros((n_frames, freqs.size), dtype=np.float64)
        rpm_axis = np.zeros(n_frames, dtype=np.float64)

        conv, _ = get_amp_conv_factor_label(amp_conv, signal_kind)
        out_type = str(output_type).strip()

        for i in range(n_frames):
            start = i * hop
            center_idx = start + (nfft // 2)
            frame = y[start:start + nfft]
            if frame.size < nfft:
                pad = np.zeros(nfft - frame.size, dtype=np.float64)
                frame = np.concatenate([frame, pad])

            X = np.fft.rfft(frame * w, n=nfft)
            mag = np.abs(X)
            A_peak = (2.0 * mag) / w_sum
            A_peak[0] *= 0.5
            if (nfft % 2 == 0) and (A_peak.size > 1):
                A_peak[-1] *= 0.5

            A = A_peak * conv
            P = A if out_type == "Amplitude" else (A * A)
            spectra_linear[i, :] = P

            rpm_axis[i] = float(rpm[min(center_idx, rpm.size - 1)])

        frame_rpms_raw = rpm_axis.copy()
        frame_spectra_linear = spectra_linear.copy()
        rpms_sorted, spectra_sorted = WaterfallPlotWidget._aggregate_duplicate_rpms(
            rpm_axis, spectra_linear
        )

        S_analysis_linear = spectra_sorted
        S_analysis = WaterfallPlotWidget._convert_waterfall_scale(S_analysis_linear, out_type, scale)
        return freqs, rpms_sorted, S_analysis, frame_rpms_raw, frame_spectra_linear

    def _build_dense_waterfall_display(self, rpms_sorted: np.ndarray,
                                       spectra_linear_sorted: np.ndarray,
                                       out_type: str, scale: str,
                                       dense_factor: int, max_points: int):
        rpms_sorted = np.asarray(rpms_sorted, dtype=np.float64)
        spectra_linear_sorted = np.asarray(spectra_linear_sorted, dtype=np.float64)

        if rpms_sorted.size < 2 or spectra_linear_sorted.size == 0:
            return rpms_sorted, WaterfallPlotWidget._convert_waterfall_scale(
                spectra_linear_sorted, out_type, scale
            )

        dense_factor = max(1, int(dense_factor))
        max_points = max(rpms_sorted.size, int(max_points))

        target_n = min(max_points, rpms_sorted.size * dense_factor)
        if target_n <= rpms_sorted.size:
            return rpms_sorted, WaterfallPlotWidget._convert_waterfall_scale(
                spectra_linear_sorted, out_type, scale
            )

        rpm_dense = np.linspace(float(rpms_sorted[0]), float(rpms_sorted[-1]), target_n)

        if _HAVE_SCIPY_INTERP and interp1d is not None:
            try:
                f = interp1d(rpms_sorted, spectra_linear_sorted, axis=0, kind='linear',
                             bounds_error=False, fill_value='extrapolate', assume_sorted=True)
                S_dense_linear = np.maximum(np.asarray(f(rpm_dense), dtype=np.float64), 0.0)
            except Exception:
                S_dense_linear = np.empty((target_n, spectra_linear_sorted.shape[1]), dtype=np.float64)
                for j in range(spectra_linear_sorted.shape[1]):
                    S_dense_linear[:, j] = np.interp(rpm_dense, rpms_sorted, spectra_linear_sorted[:, j])
                S_dense_linear = np.maximum(S_dense_linear, 0.0)
        else:
            S_dense_linear = np.empty((target_n, spectra_linear_sorted.shape[1]), dtype=np.float64)
            for j in range(spectra_linear_sorted.shape[1]):
                S_dense_linear[:, j] = np.interp(rpm_dense, rpms_sorted, spectra_linear_sorted[:, j])
            S_dense_linear = np.maximum(S_dense_linear, 0.0)

        S_dense = WaterfallPlotWidget._convert_waterfall_scale(S_dense_linear, out_type, scale)
        return rpm_dense, S_dense

    def _build_waterfall_state(self, name: str, y: np.ndarray, rpm: np.ndarray, fs: float, cfg: dict):
        display_dense_factor = int(cfg.get("waterfall_display_dense_factor", 4))
        display_max_points = int(cfg.get("waterfall_display_max_points", 8000))

        freqs, rpms_sorted, S_analysis, frame_rpms_raw, frame_spectra_linear = self._compute_waterfall_map(
            y, rpm, fs,
            int(cfg["nfft"]),
            float(cfg["overlap"]),
            str(cfg["output_type"]),
            str(cfg["scale"]),
            str(cfg["amp_conv"]),
            str(cfg.get("signal_kind", "acc")),
        )

        if freqs.size == 0 or rpms_sorted.size == 0 or S_analysis.size == 0:
            rpms_display = rpms_sorted
            S_display = S_analysis
        else:
            rpms_display, S_display = self._build_dense_waterfall_display(
                rpms_sorted,
                np.asarray(frame_spectra_linear, dtype=np.float64) if frame_spectra_linear.size
                else WaterfallPlotWidget._aggregate_duplicate_rpms(frame_rpms_raw, frame_spectra_linear)[1]
                if frame_spectra_linear.size else
                WaterfallPlotWidget._convert_waterfall_scale(S_analysis, "Amplitude", "Linear"),
                str(cfg["output_type"]),
                str(cfg["scale"]),
                display_dense_factor,
                display_max_points,
            )

        unit_lbl = "mm/s" if str(cfg.get("signal_kind", "acc")).strip().lower() == "vel" else "g"
        cbar_label = build_spectrum_ylabel(cfg["output_type"], cfg["scale"], unit_lbl)

        return {
            "kind": "waterfall",
            "key": f"waterfall::{name}",
            "name": name,
            "title": f"Waterfall: {name}",
            "xlabel": "Frequency (Hz)",
            "ylabel": "RPM",
            "colorbar_label": cbar_label,
            "scale": cfg["scale"],
            "out_type": cfg["output_type"],
            "freqs": freqs,
            "rpms": rpms_sorted,
            "S": S_analysis,
            "freqs_analysis": freqs,
            "rpms_analysis": rpms_sorted,
            "S_analysis": S_analysis,
            "freqs_display": freqs,
            "rpms_display": rpms_display,
            "S_display": S_display,
            "frame_rpms": frame_rpms_raw,
            "frame_spectra_linear": frame_spectra_linear,
            "high_quality_waterfall_zoom": bool(cfg.get("waterfall_high_quality_zoom", False)),
            "waterfall_hq_supersample": int(cfg.get("waterfall_hq_supersample",
                                                    2 if cfg.get("waterfall_high_quality_zoom", False) else 1)),
            "waterfall_hq_monotone_interp": bool(cfg.get("waterfall_hq_monotone_interp", True)),
            "viewport_dense_factor": int(cfg.get("waterfall_viewport_dense_factor", 4)),
            "viewport_min_points": int(cfg.get("waterfall_viewport_min_points", 600)),
            "viewport_max_points": int(cfg.get("waterfall_viewport_max_points", 3000)),
            "viewport_freq_dense_factor": float(cfg.get("waterfall_viewport_freq_dense_factor", 2.0)),
            "viewport_min_cols": int(cfg.get("waterfall_viewport_min_cols", 900)),
            "viewport_max_cols": int(cfg.get("waterfall_viewport_max_cols", 4096)),
            "viewport_row_pixel_factor": float(cfg.get("waterfall_viewport_row_pixel_factor", 1.25)),
            "viewport_col_pixel_factor": float(cfg.get("waterfall_viewport_col_pixel_factor", 1.25)),
            "color_cfg": dict(cfg.get("color_cfg", {})),
        }

    def _build_plot_states(self) -> dict:
        cfg = self.cfg
        signals_map = cfg["signals_map"]
        fs = float(cfg["fs"])
        nfft = int(cfg["nfft"])
        overlap_pct = float(cfg["overlap"])
        output_type = str(cfg["output_type"])
        scale = str(cfg["scale"])
        amp_conv = str(cfg["amp_conv"])
        signal_kind = str(cfg.get("signal_kind", "acc"))
        skip_first = bool(cfg.get("skip_first", False))

        conv, unit_lbl = get_amp_conv_factor_label(amp_conv, signal_kind)
        y_label = build_spectrum_ylabel(output_type, scale, unit_lbl)

        plots_time = {}
        plots_avg = {}
        plots_spec = {}

        for name, y_raw in signals_map.items():
            self.progress.emit(f"Processing {name}...")

            y = np.asarray(y_raw, dtype=np.float64)
            if skip_first and y.size > 0:
                y = y[1:]

            n = y.size
            if n == 0:
                continue
            t = np.arange(n, dtype=np.float64) / fs

            plots_time[name] = {
                "kind": "time",
                "key": f"time::{name}",
                "name": name,
                "title": f"Time-domain: {name}",
                "xlabel": "Time (s)",
                "ylabel": f"Amplitude ({unit_lbl})",
                "series": [{"x": t, "y": y * conv}],
            }

            freqs, spec = compute_avg_spectrum_from_signal(
                y, fs, nfft, overlap_pct, output_type, scale, amp_conv, signal_kind
            )
            plots_avg[name] = {
                "kind": "avg",
                "key": f"avg::{name}",
                "name": name,
                "title": f"Averaged spectrum: {name}",
                "xlabel": "Frequency (Hz)",
                "ylabel": y_label,
                "series": [{"x": freqs, "y": spec}],
            }

            self.progress.emit(f"STFT: {name}")
            freqs_stft, times_stft, S_stft = self._compute_stft(
                y, fs, nfft, overlap_pct, output_type, scale, amp_conv, signal_kind
            )
            plots_spec[name] = {
                "kind": "spectrogram",
                "key": f"spec::{name}",
                "name": name,
                "title": f"Spectrogram: {name}",
                "xlabel": "Time (s)",
                "ylabel": "Frequency (Hz)",
                "colorbar_label": y_label,
                "scale": scale,
                "times": times_stft,
                "freqs": freqs_stft,
                "S": S_stft,
            }

        return {
            "time": plots_time,
            "avg": plots_avg,
            "spec": plots_spec,
        }

    @staticmethod
    def _compute_stft(y, fs, nfft, overlap_pct, output_type, scale, amp_conv, signal_kind):
        y = np.nan_to_num(np.asarray(y, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        fs = float(fs)
        nfft = max(2, int(nfft))
        ovl = np.clip(float(overlap_pct) / 100.0, 0.0, 0.95)
        hop = max(1, int(round(nfft * (1.0 - ovl))))

        if y.size < nfft:
            return (np.array([], dtype=np.float64),
                    np.array([], dtype=np.float64),
                    np.empty((0, 0), dtype=np.float64))

        w = np.hanning(nfft).astype(np.float64)
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            w_sum = float(nfft)

        n_frames = max(1, 1 + (y.size - nfft) // hop)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
        times = ((np.arange(n_frames, dtype=np.float64) * hop) + (nfft / 2.0)) / fs

        conv, _ = get_amp_conv_factor_label(amp_conv, signal_kind)
        out_type = str(output_type).strip()

        S = np.zeros((freqs.size, n_frames), dtype=np.float64)

        for i in range(n_frames):
            start = i * hop
            frame = y[start:start + nfft]
            if frame.size < nfft:
                pad = np.zeros(nfft - frame.size, dtype=np.float64)
                frame = np.concatenate([frame, pad])

            X = np.fft.rfft(frame * w, n=nfft)
            mag = np.abs(X)
            A_peak = (2.0 * mag) / w_sum
            A_peak[0] *= 0.5
            if (nfft % 2 == 0) and (A_peak.size > 1):
                A_peak[-1] *= 0.5

            A = A_peak * conv
            P = A if out_type == "Amplitude" else (A * A)
            S[:, i] = P

        if str(scale).strip().lower() == "db":
            eps = 1e-20
            S = 20.0 * np.log10(S + eps) if out_type == "Amplitude" else 10.0 * np.log10(S + eps)

        return freqs, times, S

    @pyqtSlot()
    def run(self):
        try:
            states = self._build_plot_states()

            waterfall_cfg = self.cfg.get("waterfall_cfg")
            if waterfall_cfg is not None:
                self.progress.emit("Building waterfall...")

                rpm_raw = np.asarray(waterfall_cfg["rpm_data"], dtype=np.float64)
                acc_raw = np.asarray(waterfall_cfg["acc_data"], dtype=np.float64)

                ft = str(waterfall_cfg.get("filter_target", "None"))
                cutoff = float(waterfall_cfg.get("filter_cutoff_hz", 500.0))
                order = int(waterfall_cfg.get("filter_order", 4))
                kind = str(waterfall_cfg.get("filter_kind", "lowpass"))

                if ft in ("Acceleration", "Both"):
                    acc_raw = self._butterworth_filter(acc_raw, float(waterfall_cfg["acc_fs"]),
                                                       cutoff, order, kind)
                if ft in ("RPM", "Both"):
                    rpm_raw = self._butterworth_filter(rpm_raw, float(waterfall_cfg["rpm_fs"]),
                                                       cutoff, order, kind)

                rpm_aligned, acc_aligned, fs_eff = self._align_waterfall_signals(
                    rpm_raw, acc_raw,
                    float(waterfall_cfg["rpm_fs"]),
                    float(waterfall_cfg["acc_fs"]),
                    str(waterfall_cfg.get("resample_mode", "Resample RPM -> Acc timebase")),
                    str(waterfall_cfg.get("resample_interp", "Linear")),
                )

                cfg_for_wf = dict(self.cfg)
                cfg_for_wf["fs"] = fs_eff

                name = str(waterfall_cfg.get("acc_name", "Acceleration"))
                wf_state = self._build_waterfall_state(name, acc_aligned, rpm_aligned, fs_eff, cfg_for_wf)
                states["waterfall"] = {name: wf_state}
                states["waterfall_open_3d"] = bool(waterfall_cfg.get("open_3d", False))

            self.finished.emit(states)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal Plotter - FFT / STFT / Waterfall")
        self.resize(1500, 920)

        self.current_file_path = ""
        self.current_file_type = ""
        self.available_containers = []
        self.current_container = ""
        self.available_signals = []
        self.signal_data_cache = {}
        self.raw_signal_cache = {}

        self.plot_states = {
            "time": {},
            "avg": {},
            "spec": {},
            "waterfall": {},
        }
        self.color_cfg = {
            "cmap": "viridis",
            "reverse": False,
            "color_mode": "Continuous",
            "manual_clim": False,
            "vmin": 0.0,
            "vmax": 1.0,
            "dyn_range_db": 80.0,
            "custom_bands": default_custom_color_bands(),
        }

        self._plot_thread = None
        self._plot_worker = None
        self._open_thread = None
        self._open_worker = None
        self._sheet_thread = None
        self._sheet_worker = None

        self._build_ui()
        self._wire_signals()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        self.tabs = QTabWidget()
        lay = QVBoxLayout(central)
        lay.addWidget(self.tabs)

        self._build_controls_tab()
        self._build_plot_view_tab()
        self._build_waterfall_tab()

        self.status = self.statusBar()
        self.status.showMessage("Ready.")

    def _build_controls_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Controls")

        outer = QVBoxLayout(tab)

        file_box = QFrame()
        file_box.setFrameShape(QFrame.StyledPanel)
        file_lay = QFormLayout(file_box)

        row_file = QHBoxLayout()
        self.btn_open = QPushButton("Open file...")
        self.lbl_path = QLabel("(no file)")
        self.lbl_path.setStyleSheet("color: #444;")
        row_file.addWidget(self.btn_open)
        row_file.addWidget(self.lbl_path, 1)
        file_lay.addRow("File:", row_file)

        self.cmb_container = QComboBox()
        self.cmb_container.setEnabled(False)
        file_lay.addRow("Sheet / group:", self.cmb_container)

        self.list_signals = QListWidget()
        self.list_signals.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_signals.setMinimumHeight(120)
        file_lay.addRow("Signals to plot:", self.list_signals)

        outer.addWidget(file_box)

        proc_box = QFrame()
        proc_box.setFrameShape(QFrame.StyledPanel)
        proc_lay = QFormLayout(proc_box)

        row_fs = QHBoxLayout()
        self.spin_fs = QDoubleSpinBox()
        self.spin_fs.setRange(0.1, 1_000_000.0)
        self.spin_fs.setDecimals(2)
        self.spin_fs.setValue(10000.0)
        self.chk_skip_first = QCheckBox("Skip first sample of each signal")
        row_fs.addWidget(self.spin_fs)
        row_fs.addWidget(self.chk_skip_first)
        row_fs.addStretch(1)
        proc_lay.addRow("Sampling rate (Hz):", row_fs)

        self.cmb_signal_kind = QComboBox()
        self.cmb_signal_kind.addItems(["Acceleration (g)", "Velocity (mm/s)"])
        proc_lay.addRow("Signal type:", self.cmb_signal_kind)

        row_nfft = QHBoxLayout()
        self.spin_nfft = QSpinBox()
        self.spin_nfft.setRange(2, 1 << 24)
        self.spin_nfft.setValue(16384)
        self.spin_overlap = QDoubleSpinBox()
        self.spin_overlap.setRange(0.0, 95.0)
        self.spin_overlap.setValue(50.0)
        row_nfft.addWidget(QLabel("NFFT:"))
        row_nfft.addWidget(self.spin_nfft)
        row_nfft.addSpacing(16)
        row_nfft.addWidget(QLabel("Overlap (%):"))
        row_nfft.addWidget(self.spin_overlap)
        row_nfft.addStretch(1)
        proc_lay.addRow("FFT params:", row_nfft)

        self.cmb_output_type = QComboBox()
        self.cmb_output_type.addItems(["Amplitude", "Power"])
        proc_lay.addRow("Output:", self.cmb_output_type)

        self.cmb_scale = QComboBox()
        self.cmb_scale.addItems(["Linear", "dB"])
        proc_lay.addRow("Scale:", self.cmb_scale)

        self.cmb_amp_conv = QComboBox()
        self.cmb_amp_conv.addItems(["Peak", "RMS", "Peak-to-peak"])
        proc_lay.addRow("Amplitude conv.:", self.cmb_amp_conv)

        outer.addWidget(proc_box)

        color_box = QFrame()
        color_box.setFrameShape(QFrame.StyledPanel)
        color_lay = QFormLayout(color_box)

        self.cmb_cmap = QComboBox()
        for name in ["viridis", "plasma", "inferno", "magma", "cividis",
                     "jet", "turbo", "coolwarm", "seismic", "Spectral",
                     "hot", "nipy_spectral"]:
            self.cmb_cmap.addItem(make_colormap_preview_icon(name), name)
        self.cmb_cmap.setIconSize(self.cmb_cmap.iconSize())

        self.chk_cmap_reverse = QCheckBox("Reverse")
        cmap_row = QHBoxLayout()
        cmap_row.addWidget(self.cmb_cmap)
        cmap_row.addWidget(self.chk_cmap_reverse)
        color_lay.addRow("Colormap:", cmap_row)

        self.cmb_color_mode = QComboBox()
        self.cmb_color_mode.addItems(["Continuous", "Custom Bands"])
        color_lay.addRow("Color mode:", self.cmb_color_mode)

        self.btn_edit_bands = QPushButton("Edit custom bands...")
        color_lay.addRow("", self.btn_edit_bands)

        self.chk_manual_clim = QCheckBox("Manual color limits")
        self.spin_vmin = QDoubleSpinBox()
        self.spin_vmin.setRange(-1e9, 1e9)
        self.spin_vmin.setDecimals(4)
        self.spin_vmax = QDoubleSpinBox()
        self.spin_vmax.setRange(-1e9, 1e9)
        self.spin_vmax.setDecimals(4)
        clim_row = QHBoxLayout()
        clim_row.addWidget(self.chk_manual_clim)
        clim_row.addWidget(QLabel("vmin:"))
        clim_row.addWidget(self.spin_vmin)
        clim_row.addWidget(QLabel("vmax:"))
        clim_row.addWidget(self.spin_vmax)
        clim_row.addStretch(1)
        color_lay.addRow("Manual clim:", clim_row)

        self.spin_dyn_range_db = QDoubleSpinBox()
        self.spin_dyn_range_db.setRange(1.0, 200.0)
        self.spin_dyn_range_db.setValue(80.0)
        color_lay.addRow("dB dynamic range:", self.spin_dyn_range_db)

        outer.addWidget(color_box)

        wf_box = QFrame()
        wf_box.setFrameShape(QFrame.StyledPanel)
        wf_lay = QFormLayout(wf_box)

        self.chk_enable_waterfall = QCheckBox("Generate waterfall (RPM vs frequency)")
        wf_lay.addRow("", self.chk_enable_waterfall)

        self.btn_wf_setup = QPushButton("Configure waterfall channels...")
        self.btn_wf_setup.setEnabled(False)
        wf_lay.addRow("", self.btn_wf_setup)

        self.chk_wf_hq_zoom = QCheckBox("High-quality waterfall zoom (slower, nCode-like)")
        self.chk_wf_hq_zoom.setChecked(True)
        self.chk_wf_hq_zoom.setToolTip(
            "Enables monotone (PCHIP) interpolation and subpixel supersampling "
            "for smoother waterfall rendering when zooming in. Slightly slower."
        )
        wf_lay.addRow("", self.chk_wf_hq_zoom)

        outer.addWidget(wf_box)

        action_row = QHBoxLayout()
        self.btn_plot = QPushButton("Plot")
        self.btn_plot.setMinimumHeight(36)
        self.btn_export_tdms = QPushButton("Export cropped TDMS...")
        self.btn_export_tdms.setEnabled(False)
        action_row.addWidget(self.btn_plot)
        action_row.addWidget(self.btn_export_tdms)
        action_row.addStretch(1)
        outer.addLayout(action_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        outer.addWidget(self.progress)

        self._waterfall_setup_values = None

    def _build_plot_view_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Plot View")

        outer = QVBoxLayout(tab)

        mini_row = QHBoxLayout()

        self.mini_scroll_time = QScrollArea()
        self.mini_scroll_time.setWidgetResizable(True)
        self.mini_inner_time = QWidget()
        self.mini_inner_time_lay = QVBoxLayout(self.mini_inner_time)
        self.mini_inner_time_lay.addStretch(1)
        self.mini_scroll_time.setWidget(self.mini_inner_time)
        self.mini_scroll_time.setMinimumHeight(220)

        self.mini_scroll_avg = QScrollArea()
        self.mini_scroll_avg.setWidgetResizable(True)
        self.mini_inner_avg = QWidget()
        self.mini_inner_avg_lay = QVBoxLayout(self.mini_inner_avg)
        self.mini_inner_avg_lay.addStretch(1)
        self.mini_scroll_avg.setWidget(self.mini_inner_avg)
        self.mini_scroll_avg.setMinimumHeight(220)

        self.mini_scroll_spec = QScrollArea()
        self.mini_scroll_spec.setWidgetResizable(True)
        self.mini_inner_spec = QWidget()
        self.mini_inner_spec_lay = QVBoxLayout(self.mini_inner_spec)
        self.mini_inner_spec_lay.addStretch(1)
        self.mini_scroll_spec.setWidget(self.mini_inner_spec)
        self.mini_scroll_spec.setMinimumHeight(220)

        mini_row.addWidget(self._wrap_labelled(self.mini_scroll_time, "Time-domain"))
        mini_row.addWidget(self._wrap_labelled(self.mini_scroll_avg, "Averaged spectrum"))
        mini_row.addWidget(self._wrap_labelled(self.mini_scroll_spec, "Spectrograms"))

        outer.addLayout(mini_row, 1)

        self.detail = DetailPlotWidget()
        outer.addWidget(self.detail, 2)

    def _build_waterfall_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Waterfall")
        lay = QVBoxLayout(tab)

        self.waterfall_detail = WaterfallDetailWidget()
        lay.addWidget(self.waterfall_detail, 1)

        self.waterfall_detail.btn_export.clicked.connect(self._export_current_waterfall)
        self.detail.btn_export.clicked.connect(self._export_current_plot)

    def _wrap_labelled(self, widget, label_text):
        box = QFrame()
        box.setFrameShape(QFrame.StyledPanel)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(4, 4, 4, 4)
        lbl = QLabel(label_text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("font-weight: 700; color: #111;")
        lay.addWidget(lbl)
        lay.addWidget(widget, 1)
        return box

    def _wire_signals(self):
        self.btn_open.clicked.connect(self._on_open_file)
        self.cmb_container.currentTextChanged.connect(self._on_container_changed)
        self.btn_plot.clicked.connect(self._on_plot)
        self.chk_enable_waterfall.toggled.connect(self.btn_wf_setup.setEnabled)
        self.btn_wf_setup.clicked.connect(self._on_waterfall_setup)
        self.btn_export_tdms.clicked.connect(self._on_export_tdms)
        self.btn_edit_bands.clicked.connect(self._on_edit_bands)
        self.chk_manual_clim.toggled.connect(self._on_color_cfg_changed)
        self.spin_vmin.valueChanged.connect(self._on_color_cfg_changed)
        self.spin_vmax.valueChanged.connect(self._on_color_cfg_changed)
        self.cmb_cmap.currentTextChanged.connect(self._on_color_cfg_changed)
        self.chk_cmap_reverse.toggled.connect(self._on_color_cfg_changed)
        self.cmb_color_mode.currentTextChanged.connect(self._on_color_cfg_changed)
        self.spin_dyn_range_db.valueChanged.connect(self._on_color_cfg_changed)

    def _on_edit_bands(self):
        dlg = BandEditorDialog(self.color_cfg.get("custom_bands"), self)
        if dlg.exec_() == QDialog.Accepted:
            self.color_cfg["custom_bands"] = dlg.bands()
            self._on_color_cfg_changed()

    def _on_color_cfg_changed(self, *_):
        self.color_cfg["cmap"] = self.cmb_cmap.currentText()
        self.color_cfg["reverse"] = self.chk_cmap_reverse.isChecked()
        self.color_cfg["color_mode"] = self.cmb_color_mode.currentText()
        self.color_cfg["manual_clim"] = self.chk_manual_clim.isChecked()
        self.color_cfg["vmin"] = float(self.spin_vmin.value())
        self.color_cfg["vmax"] = float(self.spin_vmax.value())
        self.color_cfg["dyn_range_db"] = float(self.spin_dyn_range_db.value())

        # Live update the currently displayed plots
        self._refresh_detail_color_cfg()

    def _refresh_detail_color_cfg(self):
        try:
            state = self.detail.plot.plot_state
            if state is not None and state.get("kind") == "spectrogram":
                self.detail.set_state(state, color_cfg=self.color_cfg, preserve_view=True)
        except Exception:
            pass

        try:
            state = self.waterfall_detail.plot.state
            if state is not None:
                self.waterfall_detail.set_state(state, color_cfg=self.color_cfg, preserve_view=True)
        except Exception:
            pass

    def _on_open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open signal file",
            "",
            "All Supported (*.csv *.tdms *.xlsx *.xls);;CSV (*.csv);;TDMS (*.tdms);;Excel (*.xlsx *.xls);;All Files (*)"
        )
        if not path:
            return

        try:
            file_type = detect_file_type(path)
        except Exception as e:
            QMessageBox.critical(self, "Open file", str(e))
            return

        self.status.showMessage(f"Loading {os.path.basename(path)}...")
        self.progress.setVisible(True)

        self._open_thread = QThread()
        self._open_worker = FileOpenWorker(path, file_type)
        self._open_worker.moveToThread(self._open_thread)
        self._open_thread.started.connect(self._open_worker.run)
        self._open_worker.finished.connect(self._on_file_loaded)
        self._open_worker.failed.connect(self._on_worker_failed)
        self._open_worker.finished.connect(self._open_thread.quit)
        self._open_worker.failed.connect(self._open_thread.quit)
        self._open_thread.finished.connect(self._open_worker.deleteLater)
        self._open_thread.finished.connect(self._open_thread.deleteLater)
        self._open_thread.start()

    def _on_file_loaded(self, result: dict):
        self.progress.setVisible(False)
        self.current_file_path = result["path"]
        self.current_file_type = result["file_type"]
        self.available_containers = result["containers"]
        self.lbl_path.setText(os.path.basename(self.current_file_path))

        self.cmb_container.blockSignals(True)
        self.cmb_container.clear()
        self.cmb_container.addItems(self.available_containers)
        self.cmb_container.setEnabled(True)
        self.cmb_container.blockSignals(False)

        if self.available_containers:
            self._on_container_changed(self.available_containers[0])

        self.btn_export_tdms.setEnabled(True)
        self.status.showMessage(f"Loaded: {os.path.basename(self.current_file_path)}")

    def _on_container_changed(self, container: str):
        container = container.strip()
        if not container or not self.current_file_path:
            return

        self.current_container = container
        self.list_signals.clear()

        self.status.showMessage(f"Loading signals from '{container}'...")
        self.progress.setVisible(True)

        self._sheet_thread = QThread()
        self._sheet_worker = SheetLoadWorker(self.current_file_path, self.current_file_type, container)
        self._sheet_worker.moveToThread(self._sheet_thread)
        self._sheet_thread.started.connect(self._sheet_worker.run)
        self._sheet_worker.finished.connect(self._on_sheet_loaded)
        self._sheet_worker.failed.connect(self._on_worker_failed)
        self._sheet_worker.finished.connect(self._sheet_thread.quit)
        self._sheet_worker.failed.connect(self._sheet_thread.quit)
        self._sheet_thread.finished.connect(self._sheet_worker.deleteLater)
        self._sheet_thread.finished.connect(self._sheet_thread.deleteLater)
        self._sheet_thread.start()

    def _on_sheet_loaded(self, result: dict):
        self.progress.setVisible(False)
        self.available_signals = result["signals"]

        self.list_signals.clear()
        for name in self.available_signals:
            self.list_signals.addItem(QListWidgetItem(name))

        self.raw_signal_cache.clear()
        self.status.showMessage(f"Loaded {len(self.available_signals)} signals from '{self.current_container}'.")

    def _on_worker_failed(self, msg: str):
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Error", msg)
        self.status.showMessage("Error.")

    def _load_raw_signal(self, name: str) -> np.ndarray:
        key = (self.current_container, name)
        if key in self.raw_signal_cache:
            return self.raw_signal_cache[key]
        y = read_signal_from_source(
            self.current_file_path,
            self.current_file_type,
            self.current_container,
            name,
        )
        self.raw_signal_cache[key] = y
        return y

    def _on_waterfall_setup(self):
        if not self.available_signals:
            QMessageBox.warning(self, "Waterfall", "Load a file with at least two channels first.")
            return

        rpm_guess = ""
        acc_guess = ""
        for nm in self.available_signals:
            low = nm.lower()
            if not rpm_guess and ("rpm" in low or "speed" in low):
                rpm_guess = nm
            if not acc_guess and ("acc" in low or "vib" in low or "sensor" in low):
                acc_guess = nm

        dlg = WaterfallSetupDialog(
            self.available_signals, self,
            rpm_guess=rpm_guess, acc_guess=acc_guess,
            default_acc_fs=float(self.spin_fs.value()),
            default_rpm_fs=float(self.spin_fs.value()),
        )
        if dlg.exec_() == QDialog.Accepted:
            self._waterfall_setup_values = dlg.values()
            self.status.showMessage("Waterfall configured.")

    def _on_export_tdms(self):
        if not self.available_signals:
            QMessageBox.warning(self, "Export", "No signals loaded.")
            return

        dlg = ExportCroppedTdmsDialog(
            self.available_signals, self,
            default_fs=float(self.spin_fs.value()),
            skip_first_default=self.chk_skip_first.isChecked(),
        )
        if dlg.exec_() != QDialog.Accepted:
            return

        values = dlg.values()
        try:
            channels_data = {}
            for name in values["channel_names"]:
                channels_data[name] = self._load_raw_signal(name)

            info = export_cropped_tdms(
                values["output_path"],
                values["group_name"],
                channels_data,
                values["fs"],
                values["t0"],
                values["t1"],
                skip_first=values["skip_first"],
            )

            summary = "\n".join(
                f" - {d['name']}: {d['exported_samples']} samples ({d['duration_s']:.3f} s)"
                for d in info
            )
            QMessageBox.information(
                self, "Exported",
                f"Exported to:\n{values['output_path']}\n\n{summary}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    def _collect_plot_cfg(self):
        names = [it.text() for it in self.list_signals.selectedItems()]
        if not names:
            raise ValueError("Please select at least one signal.")

        signals_map = {n: self._load_raw_signal(n) for n in names}

        signal_kind = "vel" if self.cmb_signal_kind.currentText().startswith("Velocity") else "acc"

        cfg = {
            "signals_map": signals_map,
            "fs": float(self.spin_fs.value()),
            "nfft": int(self.spin_nfft.value()),
            "overlap": float(self.spin_overlap.value()),
            "output_type": self.cmb_output_type.currentText().strip(),
            "scale": self.cmb_scale.currentText().strip(),
            "amp_conv": self.cmb_amp_conv.currentText().strip(),
            "signal_kind": signal_kind,
            "skip_first": self.chk_skip_first.isChecked(),
            "color_cfg": dict(self.color_cfg),
        }

        hq = self.chk_wf_hq_zoom.isChecked()
        cfg["waterfall_high_quality_zoom"] = hq
        cfg["waterfall_hq_supersample"] = 2 if hq else 1
        cfg["waterfall_hq_monotone_interp"] = True
        cfg["waterfall_display_dense_factor"] = 5 if hq else 4
        cfg["waterfall_display_max_points"] = 12000 if hq else 8000
        cfg["waterfall_viewport_dense_factor"] = 6 if hq else 4
        cfg["waterfall_viewport_min_points"] = 900 if hq else 600
        cfg["waterfall_viewport_max_points"] = 4200 if hq else 3000
        cfg["waterfall_viewport_freq_dense_factor"] = 3.0 if hq else 2.0
        cfg["waterfall_viewport_min_cols"] = 1200 if hq else 900
        cfg["waterfall_viewport_max_cols"] = 6000 if hq else 4096
        cfg["waterfall_viewport_row_pixel_factor"] = 1.5 if hq else 1.25
        cfg["waterfall_viewport_col_pixel_factor"] = 1.5 if hq else 1.25

        if self.chk_enable_waterfall.isChecked():
            if self._waterfall_setup_values is None:
                raise ValueError("Configure the waterfall channels first.")
            wf = dict(self._waterfall_setup_values)
            wf["rpm_data"] = self._load_raw_signal(wf["rpm_name"])
            wf["acc_data"] = self._load_raw_signal(wf["acc_name"])
            cfg["waterfall_cfg"] = wf

        return cfg

    def _on_plot(self):
        try:
            cfg = self._collect_plot_cfg()
        except Exception as e:
            QMessageBox.warning(self, "Plot", str(e))
            return

        self.btn_plot.setEnabled(False)
        self.progress.setVisible(True)
        self.status.showMessage("Computing...")

        self._plot_thread = QThread()
        self._plot_worker = PlotWorker(cfg)
        self._plot_worker.moveToThread(self._plot_thread)
        self._plot_thread.started.connect(self._plot_worker.run)
        self._plot_worker.progress.connect(self.status.showMessage)
        self._plot_worker.finished.connect(self._on_plot_finished)
        self._plot_worker.failed.connect(self._on_plot_failed)
        self._plot_worker.finished.connect(self._plot_thread.quit)
        self._plot_worker.failed.connect(self._plot_thread.quit)
        self._plot_thread.finished.connect(self._plot_worker.deleteLater)
        self._plot_thread.finished.connect(self._plot_thread.deleteLater)
        self._plot_thread.start()

    def _on_plot_finished(self, states: dict):
        self.progress.setVisible(False)
        self.btn_plot.setEnabled(True)
        self.status.showMessage("Plot ready.")

        self.plot_states["time"] = states.get("time", {})
        self.plot_states["avg"] = states.get("avg", {})
        self.plot_states["spec"] = states.get("spec", {})
        self.plot_states["waterfall"] = states.get("waterfall", {})

        self._rebuild_mini_columns()

        # Show the first time-domain plot in the detail pane by default
        first_state = None
        for grp in (self.plot_states["time"], self.plot_states["avg"], self.plot_states["spec"]):
            if grp:
                first_state = next(iter(grp.values()))
                break
        if first_state is not None:
            self.detail.set_state(first_state, color_cfg=self.color_cfg)

        if self.plot_states["waterfall"]:
            wf_state = next(iter(self.plot_states["waterfall"].values()))
            self.waterfall_detail.set_state(wf_state, color_cfg=self.color_cfg)
            if states.get("waterfall_open_3d"):
                dlg = Waterfall3DWindow(wf_state, self)
                dlg.show()

        # Switch to plot view
        self.tabs.setCurrentIndex(1)

    def _on_plot_failed(self, msg: str):
        self.progress.setVisible(False)
        self.btn_plot.setEnabled(True)
        self.status.showMessage("Plot failed.")
        QMessageBox.critical(self, "Plot failed", msg)

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    def _rebuild_mini_columns(self):
        self._clear_layout(self.mini_inner_time_lay)
        self._clear_layout(self.mini_inner_avg_lay)
        self._clear_layout(self.mini_inner_spec_lay)

        for name, state in self.plot_states["time"].items():
            mini = PlotMiniWidget(state)
            mini.clicked.connect(self._on_mini_clicked)
            self.mini_inner_time_lay.addWidget(mini)
        self.mini_inner_time_lay.addStretch(1)

        for name, state in self.plot_states["avg"].items():
            mini = PlotMiniWidget(state)
            mini.clicked.connect(self._on_mini_clicked)
            self.mini_inner_avg_lay.addWidget(mini)
        self.mini_inner_avg_lay.addStretch(1)

        for name, state in self.plot_states["spec"].items():
            mini = PlotMiniWidget(state)
            mini.clicked.connect(self._on_mini_clicked)
            self.mini_inner_spec_lay.addWidget(mini)
        self.mini_inner_spec_lay.addStretch(1)

    def _on_mini_clicked(self, key: str):
        for grp in (self.plot_states["time"], self.plot_states["avg"], self.plot_states["spec"]):
            for name, state in grp.items():
                if state["key"] == key:
                    self.detail.set_state(state, color_cfg=self.color_cfg)
                    return

    def _export_current_plot(self):
        state = self.detail.plot.plot_state
        if state is None:
            QMessageBox.warning(self, "Export", "No active plot to export.")
            return

        name = state.get("name", "plot")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export plot as image",
            f"{name}_{state.get('kind', 'plot')}.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if not path:
            return

        try:
            fig = Figure(figsize=(12, 7), dpi=300, constrained_layout=True)
            ax = fig.add_subplot(111)

            if state["kind"] in ("time", "avg"):
                for s in state.get("series", []):
                    ax.plot(s["x"], s["y"], linewidth=1.0)
                vr = self.detail.plot.get_view_range()
                if vr is not None:
                    ax.set_xlim(vr[0][0], vr[0][1])
                    ax.set_ylim(vr[1][0], vr[1][1])
            else:
                times = np.asarray(state["times"])
                freqs = np.asarray(state["freqs"])
                S = np.asarray(state["S"])
                mpl_style = resolve_mpl_image_style(S, state.get("scale", "Linear"), self.color_cfg)
                if mpl_style["mode"] == "bands":
                    mesh = ax.imshow(
                        S, aspect="auto", origin="lower",
                        extent=[times[0], times[-1], freqs[0], freqs[-1]],
                        interpolation="bilinear",
                        cmap=mpl_style["cmap"],
                        norm=mpl_style["norm"],
                    )
                else:
                    mesh = ax.imshow(
                        S, aspect="auto", origin="lower",
                        extent=[times[0], times[-1], freqs[0], freqs[-1]],
                        interpolation="bilinear",
                        cmap=mpl_style["cmap"],
                        vmin=mpl_style["vmin"], vmax=mpl_style["vmax"],
                    )
                fig.colorbar(mesh, ax=ax, label=state.get("colorbar_label", ""))
                vr = self.detail.plot.get_view_range()
                if vr is not None:
                    ax.set_xlim(vr[0][0], vr[0][1])
                    ax.set_ylim(vr[1][0], vr[1][1])

            ax.set_xlabel(state.get("xlabel", ""))
            ax.set_ylabel(state.get("ylabel", ""))
            ax.set_title(state.get("title", ""))
            ax.grid(True, alpha=0.3)

            fig.savefig(path, dpi=300)
            QMessageBox.information(self, "Export", f"Saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    def _export_current_waterfall(self):
        state = self.waterfall_detail.plot.state
        if state is None:
            QMessageBox.warning(self, "Export", "No active waterfall to export.")
            return

        name = state.get("name", "waterfall")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export waterfall as image",
            f"{name}_waterfall.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if not path:
            return

        try:
            # Reuse the standalone window's figure logic via a transient instance
            # to avoid duplicating rendering code
            freqs = np.asarray(state.get("freqs_display", state["freqs"]))
            rpms = np.asarray(state.get("rpms_display", state["rpms"]))
            S = np.asarray(state.get("S_display", state["S"]))

            fig = Figure(figsize=(12, 7), dpi=300, constrained_layout=True)
            ax = fig.add_subplot(111)

            mpl_style = resolve_mpl_image_style(S, state.get("scale", "Linear"), self.color_cfg)
            if mpl_style["mode"] == "bands":
                mesh = ax.imshow(
                    S, aspect="auto", origin="lower",
                    extent=[freqs[0], freqs[-1], rpms[0], rpms[-1]],
                    interpolation="bilinear",
                    cmap=mpl_style["cmap"],
                    norm=mpl_style["norm"],
                )
            else:
                mesh = ax.imshow(
                    S, aspect="auto", origin="lower",
                    extent=[freqs[0], freqs[-1], rpms[0], rpms[-1]],
                    interpolation="bilinear",
                    cmap=mpl_style["cmap"],
                    vmin=mpl_style["vmin"], vmax=mpl_style["vmax"],
                )
            fig.colorbar(mesh, ax=ax, label=state.get("colorbar_label", ""))

            # Apply current zoom viewport
            try:
                xr, yr = self.waterfall_detail.plot.upper_item.getViewBox().viewRange()
                ax.set_xlim(float(xr[0]), float(xr[1]))
                ax.set_ylim(float(yr[0]), float(yr[1]))
            except Exception:
                pass

            # Overlay order cursors
            for c in self.waterfall_detail.plot._cursors:
                order_val = float(c.order)
                col = c.color
                if isinstance(col, (tuple, list)):
                    col = tuple(v / 255.0 for v in col[:3])
                fx = (order_val * rpms) / 60.0
                m = np.isfinite(fx) & (fx >= freqs[0]) & (fx <= freqs[-1])
                if np.any(m):
                    ax.plot(fx[m], rpms[m], color=col, linewidth=1.6,
                            label=f"Order {order_val:.2f}x")
            if self.waterfall_detail.plot._cursors:
                ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

            ax.set_xlabel(state.get("xlabel", "Frequency (Hz)"))
            ax.set_ylabel(state.get("ylabel", "RPM"))
            ax.set_title(state.get("title", "Waterfall"))
            ax.grid(True, alpha=0.25)

            fig.savefig(path, dpi=300)
            QMessageBox.information(self, "Export", f"Saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
