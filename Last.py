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




def _hann_window(N: int, kind: str = "periodic") -> np.ndarray:
    """Hann window of length N.

    kind="periodic"  -> 0.5 - 0.5*cos(2*pi*n / N).  Sum is exactly N/2.
                        Matches MATLAB hann(N,'periodic'), scipy hann(sym=False),
                        and nCode. Recommended for FFT / STFT / Welch.
    kind="symmetric" -> 0.5 - 0.5*cos(2*pi*n / (N-1)).  Identical to numpy.hanning(N)
                        and scipy hann(sym=True). Use for FIR design or to
                        reproduce legacy results.
    """
    N = int(N)
    if N <= 1:
        return np.ones(max(N, 0), dtype=np.float64)
    n = np.arange(N, dtype=np.float64)
    k = str(kind).strip().lower()
    if k.startswith("sym"):
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (N - 1))
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / N)


def compute_one_sided_amp_spectrum_from_segment(
    segment: np.ndarray,
    fs: float,
    nfft: int,
    amp_conv: str,
    signal_kind: str,
    output_type: str,
    *,
    window_kind: str = "periodic",
):
    """One valid segment -> one-sided spectrum.

    * Window length = nseg (actual segment); FFT length = nfft (zero-pad in rfft).
    * window_kind selects periodic (default, recommended) or symmetric Hann.
    * Amplitude output:  spec = A_peak * conv  (peak/RMS/p-p per amp_conv)
    * Power output:      spec = RMS^2          (independent of amp_conv)
    * DC and (when nfft even) Nyquist bins are NOT doubled.
    """
    segment = np.nan_to_num(np.asarray(segment, dtype=np.float64),
                            nan=0.0, posinf=0.0, neginf=0.0)
    fs = float(fs)
    nfft = max(2, int(nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    if segment.size == 0 or not np.isfinite(fs) or fs <= 0.0:
        return freqs, np.zeros(freqs.size, dtype=np.float64)

    nseg = int(segment.size)
    if nseg > nfft:
        segment = segment[:nfft]
        nseg = nfft
    if nseg <= 1:
        return freqs, np.zeros(freqs.size, dtype=np.float64)

    w = _hann_window(nseg, window_kind)
    w_sum = float(np.sum(w)) or float(nseg)

    X = np.fft.rfft(segment * w, n=nfft)
    mag = np.abs(X)

    A_peak = (2.0 * mag) / w_sum
    A_peak[0] *= 0.5
    if (nfft % 2 == 0) and (A_peak.size > 1):
        A_peak[-1] *= 0.5

    if str(output_type).strip() == "Amplitude":
        conv, _ = get_amp_conv_factor_label(amp_conv, signal_kind)
        spec = A_peak * conv
    else:
        # Power = RMS^2, regardless of amp_conv. DC and Nyquist are
        # deterministic for a real signal -> RMS == peak there.
        A_rms = A_peak / np.sqrt(2.0)
        A_rms[0] = A_peak[0]
        if (nfft % 2 == 0) and (A_rms.size > 1):
            A_rms[-1] = A_peak[-1]
        spec = A_rms * A_rms
    return freqs, spec

def compute_avg_spectrum_from_signal(
    y: np.ndarray,
    fs: float,
    nfft: int,
    overlap_pct: float,
    output_type: str,
    scale: str,
    amp_conv: str,
    signal_kind: str,
    *,
    window_kind: str = "periodic",
):
    """nCode-/Welch-style averaged spectrum.

    Averaging is performed in the POWER domain (RMS^2 frames), then
    converted to the requested output. Eliminates the Jensen-inequality
    bias of amplitude-domain averaging. Vectorized framing + batched rfft.
    """
    y = np.nan_to_num(np.asarray(y, dtype=np.float64),
                      nan=0.0, posinf=0.0, neginf=0.0)
    fs = float(fs)
    nfft = max(2, int(nfft))
    if y.size == 0 or not np.isfinite(fs) or fs <= 0.0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    nperseg = min(nfft, int(y.size))
    if nperseg <= 1:
        return (np.fft.rfftfreq(nfft, d=1.0 / fs),
                np.zeros(nfft // 2 + 1, dtype=np.float64))

    ovl = float(np.clip(float(overlap_pct) / 100.0, 0.0, 0.95))
    hop = max(1, int(round(nperseg * (1.0 - ovl))))
    n_frames = 1 if y.size <= nperseg else 1 + (y.size - nperseg) // hop

    w = _hann_window(nperseg, window_kind)
    w_sum = float(np.sum(w)) or float(nperseg)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    starts = np.arange(n_frames, dtype=np.int64) * hop
    starts = starts[starts + nperseg <= y.size]
    if starts.size == 0:
        return freqs, np.zeros(freqs.size, dtype=np.float64)

    idx = starts[:, None] + np.arange(nperseg, dtype=np.int64)[None, :]
    segs = y[idx] * w[None, :]
    X = np.fft.rfft(segs, n=nfft, axis=1)

    A_peak = (2.0 / w_sum) * np.abs(X)
    A_peak[:, 0] *= 0.5
    if (nfft % 2 == 0) and (A_peak.shape[1] > 1):
        A_peak[:, -1] *= 0.5

    A_rms = A_peak / np.sqrt(2.0)
    A_rms[:, 0] = A_peak[:, 0]
    if (nfft % 2 == 0) and (A_rms.shape[1] > 1):
        A_rms[:, -1] = A_peak[:, -1]
    P = A_rms * A_rms
    P_avg = P.mean(axis=0)

    if str(output_type).strip() == "Amplitude":
        conv, _ = get_amp_conv_factor_label(amp_conv, signal_kind)
        A_peak_avg = np.sqrt(np.maximum(P_avg, 0.0)) * np.sqrt(2.0)
        A_peak_avg[0] = np.sqrt(max(P_avg[0], 0.0))
        if (nfft % 2 == 0) and (A_peak_avg.size > 1):
            A_peak_avg[-1] = np.sqrt(max(P_avg[-1], 0.0))
        spec = A_peak_avg * conv
    else:
        spec = P_avg

    if str(scale).strip().lower() == "db":
        eps = 1e-30
        if str(output_type).strip() == "Amplitude":
            spec = 20.0 * np.log10(np.maximum(spec, eps))
        else:
            spec = 10.0 * np.log10(np.maximum(spec, eps))
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
        btn.setText("×")
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
    def __init__(self, channel_names, parent=None, rpm_guess="", acc_guess=""):
        super().__init__(parent)
        self.setWindowTitle("Waterfall setup")
        self.setModal(True)

        self.cmb_rpm = QComboBox()
        self.cmb_acc = QComboBox()
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

        # ── Channel selection ──
        self.list_channels = QListWidget()
        self.list_channels.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_channels.setMinimumHeight(120)
        for nm in channel_names:
            self.list_channels.addItem(QListWidgetItem(nm))
        # Select all by default
        self.list_channels.selectAll()

        self.btn_select_all = QPushButton("Select All")
        self.btn_clear_sel = QPushButton("Clear")
        self.btn_select_all.clicked.connect(self.list_channels.selectAll)
        self.btn_clear_sel.clicked.connect(self.list_channels.clearSelection)

        # ── Parameters ──
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

        # ── Layout ──
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

        # Live preview
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
    """Crop signal to [t0, t1] window and return the cropped array.
    
    The caller treats the result as starting at t=0 (zero-aligned).
    Metadata (wf_start_offset=0, wf_increment=1/fs) handles the alignment.
    """
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
    """Export cropped, zero-aligned signals to a .tdms file.
    
    Parameters
    ----------
    output_path : str
        Destination .tdms file path.
    group_name : str
        TDMS group name for the exported channels.
    channels : dict
        Mapping of channel_name -> raw numpy array (full-length signal).
    fs : float
        Sampling frequency in Hz.
    t0, t1 : float
        Crop window in seconds (original time axis).
    skip_first : bool
        If True, drop the first sample before cropping.
    source_properties : dict or None
        Optional dict of channel_name -> dict of extra TDMS properties
        to preserve from the original file.
    """
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

        # Merge any source properties (preserve unit_string etc.)
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
    """Standalone 2D waterfall colormap window with jet/turbo colormap.
    
    Renders only the upper waterfall heatmap (no order trace subplot)
    using a continuous jet-like colormap (blue -> green -> yellow -> red).
    Captures the current viewport zoom if provided.
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

        # Toolbar for colormap selection and export
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

        self.fig = Figure(figsize=(14, 7), dpi=110, constrained_layout=True)
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
        freqs = np.asarray(self.state["freqs"], dtype=np.float64)
        rpms = np.asarray(self.state["rpms"], dtype=np.float64)
        S = np.asarray(self.state["S"], dtype=np.float64)

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

        # Render with bilinear interpolation so zooming in the standalone
        # window stays visually smooth instead of showing flat-colored quads.
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
            aspect="auto",
            origin="lower",
            extent=[float(freqs[0]), float(freqs[-1]), float(rpms[0]), float(rpms[-1])],
            interpolation="bilinear",
            cmap=cmap,
            norm=norm_to_use,
            vmin=(None if norm_to_use is not None else vmin),
            vmax=(None if norm_to_use is not None else vmax),
        )

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
        freqs = np.asarray(self.state["freqs"], dtype=np.float64)
        rpms = np.asarray(self.state["rpms"], dtype=np.float64)
        S = np.asarray(self.state["S"], dtype=np.float64)

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
        self._display_S = np.empty((0, 0), dtype=np.float64)
        self._viewport_dense_factor = 6
        self._viewport_min_points = 1200
        self._viewport_max_points = 10000
        self._viewport_freq_dense_factor = 3.0
        self._viewport_min_cols = 1800
        self._viewport_max_cols = 12288
        self._viewport_row_pixel_factor = 2.25
        self._viewport_col_pixel_factor = 2.25
        self._last_viewport_signature = None
        self._suspend_viewport_reinterp = False
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
        # Rebuild the visible waterfall shortly after a zoom/pan.
        # A short delay keeps dragging responsive but avoids leaving the zoomed
        # image in a magnified low-resolution state for too long.
        self._visible_reinterp_timer.setInterval(35)
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

    def _debug_zoom_state(self, tag="zoom"):
        try:
            vb = self.upper_item.getViewBox()
            xr, yr = vb.viewRange()

            xw = float(xr[1] - xr[0])
            yw = float(yr[1] - yr[0])

            current_shape = tuple(self.S.shape) if isinstance(self.S, np.ndarray) else None
            full_shape = tuple(self.S_full_display.shape) if isinstance(self.S_full_display, np.ndarray) else None
            analysis_shape = tuple(self.S_analysis.shape) if isinstance(self.S_analysis, np.ndarray) else None

            disp = getattr(self, "_display_S", None)
            disp_shape = tuple(disp.shape) if isinstance(disp, np.ndarray) else None

            ncols = 1
            if disp_shape and len(disp_shape) >= 2:
                ncols = max(int(disp_shape[1]), 1)
            elif current_shape and len(current_shape) >= 2:
                ncols = max(int(current_shape[1]), 1)

            print(
                f"[{tag}] x=({xr[0]:.3f},{xr[1]:.3f}) w={xw:.3f} | "
                f"y=({yr[0]:.3f},{yr[1]:.3f}) h={yw:.3f} | "
                f"current_S={current_shape} | display_S={disp_shape} | "
                f"full_S={full_shape} | analysis_S={analysis_shape} | "
                f"px_per_freq_bin~{xw / float(ncols):.6f}"
            )
        except Exception as e:
            print(f"[{tag}] debug failed: {e}")

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

        self._upper_image_item.setImage(S, autoLevels=False)
        # The waterfall is already resampled to the current viewport resolution
        # in _render_waterfall_view_from_frames(). Extra GPU interpolation here
        # makes rectangle-zoomed ridges look soft/blurred.
        try:
            self._upper_image_item.setOpts(interpolate=False)
        except Exception:
            pass
        self._upper_image_item.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))

        self._display_S = np.asarray(S, dtype=np.float64)
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
        self._debug_zoom_state("range_changed")
        self._visible_reinterp_timer.start()

    def _render_waterfall_view_from_frames(self, view_f_lo: float, view_f_hi: float,
                                           view_r_lo: float, view_r_hi: float,
                                           pad_frac_f: float = 0.05,
                                           pad_frac_r: float = 0.08):
        rpms_u, spectra_u = self._aggregate_duplicate_rpms(self.frame_rpms, self.frame_spectra_linear)
        freqs_u = np.asarray(self.freqs_analysis, dtype=np.float64)

        if rpms_u.size < 2 or spectra_u.size == 0 or freqs_u.size < 2:
            return self.freqs_full_display, self.rpms_full_display, self.S_full_display

        full_f_lo = float(freqs_u[0])
        full_f_hi = float(freqs_u[-1])
        full_r_lo = float(rpms_u[0])
        full_r_hi = float(rpms_u[-1])

        vf0, vf1 = sorted((float(view_f_lo), float(view_f_hi)))
        vr0, vr1 = sorted((float(view_r_lo), float(view_r_hi)))
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

        near_full_f = vis_f_span >= 0.98 * full_f_span
        near_full_r = vis_r_span >= 0.98 * full_r_span

        if near_full_f:
            render_f_lo, render_f_hi = full_f_lo, full_f_hi
        else:
            f_pad = max(float(pad_frac_f) * vis_f_span, 0.001 * full_f_span)
            render_f_lo = max(view_f_lo - f_pad, full_f_lo)
            render_f_hi = min(view_f_hi + f_pad, full_f_hi)

        if near_full_r:
            render_r_lo, render_r_hi = full_r_lo, full_r_hi
        else:
            r_pad = max(float(pad_frac_r) * vis_r_span, 0.001 * full_r_span)
            render_r_lo = max(view_r_lo - r_pad, full_r_lo)
            render_r_hi = min(view_r_hi + r_pad, full_r_hi)

        rpm_sel = (rpms_u >= render_r_lo) & (rpms_u <= render_r_hi)
        if np.count_nonzero(rpm_sel) < 4:
            center = 0.5 * (render_r_lo + render_r_hi)
            idx = int(np.searchsorted(rpms_u, center, side='left'))
            i0 = max(0, idx - 12)
            i1 = min(rpms_u.size, idx + 12)
            local_rpms = rpms_u[i0:i1]
            local_spectra = spectra_u[i0:i1]
        else:
            local_rpms = rpms_u[rpm_sel]
            local_spectra = spectra_u[rpm_sel]

        if local_rpms.size < 2 or local_spectra.size == 0:
            return self.freqs_full_display, self.rpms_full_display, self.S_full_display

        freq_sel = (freqs_u >= render_f_lo) & (freqs_u <= render_f_hi)
        if np.count_nonzero(freq_sel) < 4:
            center = 0.5 * (render_f_lo + render_f_hi)
            idx = int(np.searchsorted(freqs_u, center, side='left'))
            j0 = max(0, idx - 32)
            j1 = min(freqs_u.size, idx + 32)
            local_freqs = freqs_u[j0:j1]
            local_spectra = local_spectra[:, j0:j1]
        else:
            local_freqs = freqs_u[freq_sel]
            local_spectra = local_spectra[:, freq_sel]

        if local_freqs.size < 2 or local_spectra.size == 0:
            return self.freqs_full_display, self.rpms_full_display, self.S_full_display

        pix_h = max(1, int(self.upper.viewport().height()))
        pix_w = max(1, int(self.upper.viewport().width()))

        target_rows = max(
            int(self._viewport_min_points),
            int(np.ceil(pix_h * float(self._viewport_row_pixel_factor))),
            int(np.ceil(local_rpms.size * max(1.0, float(self._viewport_dense_factor)))),
            int(np.ceil(pix_h * 2.0)),
        )
        target_rows = int(min(int(self._viewport_max_points), max(2, target_rows)))

        target_cols = max(
            int(self._viewport_min_cols),
            int(np.ceil(pix_w * float(self._viewport_col_pixel_factor))),
            int(np.ceil(local_freqs.size * max(1.0, float(self._viewport_freq_dense_factor)))),
            int(np.ceil(pix_w * 2.0)),
        )
        target_cols = int(min(int(self._viewport_max_cols), max(2, target_cols)))

        target_rpms = np.linspace(render_r_lo, render_r_hi, target_rows)
        target_freqs = np.linspace(render_f_lo, render_f_hi, target_cols)

        local_spectra = np.maximum(local_spectra, 0.0)

        S_rpm_linear = np.empty((target_rpms.size, local_spectra.shape[1]), dtype=np.float64)
        for j in range(local_spectra.shape[1]):
            S_rpm_linear[:, j] = np.interp(target_rpms, local_rpms, local_spectra[:, j])

        S_out_linear = np.empty((target_rpms.size, target_freqs.size), dtype=np.float64)
        for i in range(S_rpm_linear.shape[0]):
            S_out_linear[i, :] = np.interp(target_freqs, local_freqs, S_rpm_linear[i, :])

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

        # Viewport rebuild caching
        _pw = max(1, int(self.upper.viewport().width()))
        _ph = max(1, int(self.upper.viewport().height()))
        _sig = (
            round(view_f_lo, 3), round(view_f_hi, 3),
            round(view_r_lo, 3), round(view_r_hi, 3),
            _pw // 16, _ph // 16,
        )
        if _sig == self._last_viewport_signature:
            return
        self._last_viewport_signature = _sig

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
        self._debug_zoom_state("after_rebuild")
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
        btn.setText("×")
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
        btn.setText("×")
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
        self._suspend_viewport_reinterp = False
        self._last_viewport_signature = None
        self._visible_reinterp_timer.start()

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
        self._last_viewport_signature = None
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

        dlg = Waterfall2DStandaloneWindow(
            state=self.state,
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
        self._last_viewport_signature = None
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
            self._suspend_viewport_reinterp = False
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
        # For waterfall, disable ImageItem-level interpolation. The code below
        # rebuilds/resamples the visible viewport itself after zooming, so
        # enabling ImageItem interpolation causes an additional smoothing pass.
        try:
            img.setOpts(interpolate=False)
        except Exception:
            pass
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

        # IMPORTANT: set_state intentionally suspends viewport rebuilding while
        # the plot is being constructed and the initial ranges are assigned.
        # It must be re-enabled here; otherwise rectangle zoom only magnifies
        # the full-range ImageItem and the waterfall appears blurred.
        self._suspend_viewport_reinterp = False
        self._last_viewport_signature = None
        if not self.mini:
            self._visible_reinterp_timer.start()


class TimeDomainZoomFFTWidget(QWidget):
    clicked = pyqtSignal()
    color_levels_changed = pyqtSignal(float, float)
    color_levels_change_finished = pyqtSignal(float, float)

    def __init__(self, mini=False, parent=None):
        super().__init__(parent)
        self.mini = bool(mini)
        self.plot_state = None
        self.plot_key = None

        self.upper = InteractivePlotWidget(mini=mini)
        self.upper.clicked.connect(self.clicked.emit)

        self.lbl_zoom_fft = QLabel("Visible time window FFT: -")
        self.lbl_zoom_fft.setStyleSheet("color: #444;")
        self.lbl_zoom_fft.setWordWrap(True)

        self.lower = InteractivePlotWidget(mini=mini)
        self.lower.clicked.connect(self.clicked.emit)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.upper, 3)
        lay.addWidget(self.lbl_zoom_fft)
        lay.addWidget(self.lower, 2)

        self._range_timer = QTimer(self)
        self._range_timer.setSingleShot(True)
        self._range_timer.setInterval(120)
        self._range_timer.timeout.connect(self._update_zoom_fft_from_view)

        self._connected_upper_vb = None
        self._time_signal = np.array([], dtype=np.float64)
        self._time_axis = np.array([], dtype=np.float64)
        self._fft_cfg = {}
        self._lower_export_data = None

    def _connect_upper_viewbox(self):
        vb = self.upper.get_view_box()
        if vb is self._connected_upper_vb:
            return

        if self._connected_upper_vb is not None:
            try:
                self._connected_upper_vb.sigRangeChanged.disconnect(self._on_upper_range_changed)
            except Exception:
                pass

        self._connected_upper_vb = vb
        if vb is not None:
            try:
                vb.sigRangeChanged.connect(self._on_upper_range_changed)
            except Exception:
                pass

    def _on_upper_range_changed(self, *_):
        self._range_timer.start()

    def set_rect_zoom_active(self, active: bool):
        self.upper.set_rect_zoom_active(active)
        self.lower.set_rect_zoom_active(active)

    def zoom_axes(self, factor: float):
        self.upper.zoom_axes(factor)
        self.lower.zoom_axes(factor)

    def reset_view(self):
        self.upper.reset_view()
        self.lower.reset_view()
        self._update_zoom_fft_from_view()

    def clear_click_annotations(self):
        self.upper.clear_click_annotations()
        self.lower.clear_click_annotations()

    def get_export_annotations(self):
        return self.upper.get_export_annotations()

    def get_export_lower_annotations(self):
        return self.lower.get_export_annotations()

    def get_zoom_fft_export_data(self):
        return dict(self._lower_export_data) if isinstance(self._lower_export_data, dict) else None

    def set_state(self, state: dict, color_cfg=None, preserve_view=False, preserve_annotations=False):
        self.plot_state = state
        self.plot_key = state["key"]
        self._time_signal = np.asarray(state.get("time_signal", state["series"][0]["y"]), dtype=np.float64)
        self._time_axis = np.asarray(state.get("time_axis", state["series"][0]["x"]), dtype=np.float64)
        self._fft_cfg = dict(state.get("fft_zoom_cfg", {}))

        self.upper.set_state(
            state,
            color_cfg=color_cfg,
            preserve_view=preserve_view,
            preserve_annotations=preserve_annotations,
        )
        self._connect_upper_viewbox()
        self._update_zoom_fft_from_view()

    def _selected_time_window(self):
        if self._time_axis.size == 0 or self._time_signal.size == 0:
            return None

        xr = self.upper.get_view_range()
        if xr is None:
            x0 = float(self._time_axis[0])
            x1 = float(self._time_axis[-1])
        else:
            x0, x1 = xr[0]

        if x1 < x0:
            x0, x1 = x1, x0

        lo = float(self._time_axis[0])
        hi = float(self._time_axis[-1])
        x0 = float(np.clip(x0, lo, hi))
        x1 = float(np.clip(x1, lo, hi))

        if x1 <= x0:
            x0 = lo
            x1 = hi

        i0 = int(np.searchsorted(self._time_axis, x0, side="left"))
        i1 = int(np.searchsorted(self._time_axis, x1, side="right"))
        i0 = max(0, min(i0, self._time_signal.size - 1))
        i1 = max(i0 + 1, min(i1, self._time_signal.size))

        return x0, x1, i0, i1

    def _update_zoom_fft_from_view(self):
        self._lower_export_data = None

        if self._time_signal.size == 0 or self._time_axis.size == 0:
            self.lbl_zoom_fft.setText("Visible time window FFT: -")
            self.lower.clear_plot()
            return

        win = self._selected_time_window()
        if win is None:
            self.lbl_zoom_fft.setText("Visible time window FFT: -")
            self.lower.clear_plot()
            return

        x0, x1, i0, i1 = win
        y_sel = np.asarray(self._time_signal[i0:i1], dtype=np.float64)
        if y_sel.size < 2:
            self.lbl_zoom_fft.setText("Visible time window FFT: selected region is too short.")
            self.lower.clear_plot()
            return

        fs = float(self.plot_state.get("fs", 1.0))
        nfft = int(self._fft_cfg.get("nfft", 1024))
        overlap_pct = float(self._fft_cfg.get("overlap_pct", 67.0))
        out_type = str(self._fft_cfg.get("out_type", "Amplitude"))
        scale = str(self._fft_cfg.get("scale", "Linear"))
        amp_conv = str(self._fft_cfg.get("amp_conv", "RMS"))
        signal_kind = str(self._fft_cfg.get("signal_kind", "acc"))
        _, unit_lbl = get_amp_conv_factor_label(amp_conv, signal_kind)
        ylabel = build_spectrum_ylabel(out_type, scale, unit_lbl)

        freqs, spec = compute_avg_spectrum_from_signal(
            y=y_sel,
            fs=fs,
            nfft=nfft,
            overlap_pct=overlap_pct,
            output_type=out_type,
            scale=scale,
            amp_conv=amp_conv,
            signal_kind=signal_kind,
            window_kind=str(self._fft_cfg.get("window_kind", "periodic")),
        )

        if freqs.size == 0 or spec.size == 0:
            self.lbl_zoom_fft.setText("Visible time window FFT: selected region is empty.")
            self.lower.clear_plot()
            return

        dt = max(0.0, float(x1 - x0))
        title = (
            f"Visible time window FFT | t=[{x0:.6g}, {x1:.6g}] s | Δt={dt:.6g} s | "
            f"samples={y_sel.size} | NFFT={nfft} | Overlap={overlap_pct:.1f}%"
        )
        self.lbl_zoom_fft.setText(title)

        lower_state = {
            "key": f"{self.plot_key}|visible_fft",
            "kind": "avg",
            "name": f"{self.plot_state.get('name', 'signal')} | Visible FFT",
            "short_title": "Visible time window FFT",
            "title": title,
            "xlabel": "Frequency (Hz)",
            "ylabel": ylabel,
            "grid": True,
            "series": [{"x": freqs, "y": spec, "label": "Visible time window FFT"}],
            "stats_text": "",
        }
        self.lower.set_state(lower_state, preserve_view=False, preserve_annotations=False)

        self._lower_export_data = {
            "freqs": np.asarray(freqs, dtype=np.float64),
            "spec": np.asarray(spec, dtype=np.float64),
            "xlabel": "Frequency (Hz)",
            "ylabel": ylabel,
            "title": title,
            "window": (float(x0), float(x1)),
        }


class PlotMiniWidget(QFrame):
    request_open = pyqtSignal(str)
    request_close = pyqtSignal(str)
    activated = pyqtSignal(object, str)

    def __init__(self, state: dict):
        super().__init__()
        self.state = state
        self.plot_key = state["key"]

        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)

        self.lbl = QLabel(state.get("short_title", self.plot_key))
        self.lbl.setStyleSheet("font-weight: bold;")

        self.btn_open = QPushButton("Open Large")
        self.btn_close = QPushButton("Close")
        self.btn_open.setFixedWidth(95)
        self.btn_close.setFixedWidth(70)

        self.canvas = InteractivePlotWidget(mini=True)
        self.canvas.setMinimumHeight(220)

        top = QHBoxLayout()
        top.addWidget(self.lbl, 1)
        top.addWidget(self.btn_open)
        top.addWidget(self.btn_close)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addLayout(top)
        lay.addWidget(self.canvas)

        self.btn_open.clicked.connect(lambda: self.request_open.emit(self.plot_key))
        self.btn_close.clicked.connect(lambda: self.request_close.emit(self.plot_key))
        self.canvas.clicked.connect(self._on_canvas_clicked)

    def _on_canvas_clicked(self):
        self.activated.emit(self.canvas, self.plot_key)
        self.request_open.emit(self.plot_key)

    def set_selected(self, selected: bool):
        if selected:
            self.setStyleSheet("QFrame { border: 2px solid #2d89ef; }")
        else:
            self.setStyleSheet("")


class DetailPlotWidget(QWidget):
    activated = pyqtSignal(object, str)

    def __init__(self, state: dict):
        super().__init__()
        self.state = state
        self.plot_key = state["key"]

        self.title = QLabel(state.get("title", self.plot_key))
        self.title.setWordWrap(True)
        self.title.setStyleSheet("font-weight: bold;")

        if state.get("kind") == "time":
            self.canvas = TimeDomainZoomFFTWidget(mini=False)
            self.canvas.setMinimumHeight(680)
        else:
            self.canvas = InteractivePlotWidget(mini=False)
            self.canvas.setMinimumHeight(480)

        lay = QVBoxLayout(self)
        lay.addWidget(self.title)
        lay.addWidget(self.canvas, 1)

        self.canvas.clicked.connect(self._emit_active)

    def _emit_active(self):
        self.activated.emit(self.canvas, self.plot_key)


class WaterfallDetailWidget(QWidget):
    activated = pyqtSignal(object, str)

    def __init__(self, state: dict):
        super().__init__()
        self.state = state
        self.plot_key = state["key"]

        self.title = QLabel(state.get("title", self.plot_key))
        self.title.setWordWrap(True)
        self.title.setStyleSheet("font-weight: bold;")

        self.canvas = WaterfallPlotWidget(mini=False)
        self.canvas.setMinimumHeight(620)
        self.canvas.clicked.connect(self._emit_active)

        lay = QVBoxLayout(self)
        lay.addWidget(self.title)
        lay.addWidget(self.canvas, 1)

    def _emit_active(self):
        self.activated.emit(self.canvas, self.plot_key)


class FileOpenWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    @pyqtSlot()
    def run(self):
        try:
            file_type = detect_file_type(self.path)
            names = read_source_container_names(self.path, file_type)
            default_index = 1 if file_type == "excel" and len(names) > 1 else 0
            self.finished.emit({
                "file_type": file_type,
                "sheet_names": names,
                "default_index": default_index,
            })
        except Exception as e:
            self.error.emit(str(e))



class SheetLoadWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, file_path: str, file_type: str, name: str):
        super().__init__()
        self.file_path = file_path
        self.file_type = file_type
        self.name = name

    @pyqtSlot()
    def run(self):
        try:
            names = read_source_signal_names(self.file_path, self.file_type, self.name)
            prefer_index = 9 if self.file_type == "excel" else 0
            self.finished.emit({
                "kind": self.file_type,
                "name": self.name,
                "names": names,
                "prefer_index": prefer_index,
            })
        except Exception as e:
            self.error.emit(str(e))



class PlotWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    @pyqtSlot()
    def run(self):
        try:
            self.finished.emit(self._build_plot_states())
        except Exception as e:
            self.error.emit(str(e))

    def _signal_unit_short(self, signal_kind=None) -> str:
        sk = signal_kind or self.cfg["signal_kind"]
        return VEL_UNIT if sk == "vel" else "g"

    def _signal_ylabel(self, signal_kind=None) -> str:
        sk = signal_kind or self.cfg["signal_kind"]
        return f"Velocity ({VEL_UNIT})" if sk == "vel" else "Acceleration (g)"

    def _amp_conv_factor_label(self, signal_kind=None):
        sk = signal_kind or self.cfg["signal_kind"]
        return get_amp_conv_factor_label(self.cfg["amp_conv"], sk)

    def _make_time_axis(self, n: int, fs: float, start_offset_s: float = 0.0):
        return start_offset_s + (np.arange(n, dtype=np.float64) / fs)

    def _slice_by_time_window(self, y: np.ndarray, fs: float, t0: float, t1: float):
        n = int(y.size)
        i0 = int(np.floor(t0 * fs))
        i1 = int(np.ceil(t1 * fs))
        i0 = max(0, i0)
        i1 = min(n, i1)

        if i0 >= n:
            raise ValueError("Start time exceeds signal length.")
        if i1 <= i0:
            raise ValueError("End time must be greater than start time.")

        return y[i0:i1]

    def _prep_signal_with_fs(self, y: np.ndarray, fs: float) -> np.ndarray:
        crop = bool(self.cfg["crop"])
        t0 = float(self.cfg["t0"])
        t1 = float(self.cfg["t1"])
        skip_first = bool(self.cfg["skip_first"])

        y = np.asarray(y)

        if skip_first and y.size > 0:
            y = y[1:]

        if crop:
            y = self._slice_by_time_window(y, fs, t0, t1)

        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)

    def _prep_signal(self, y: np.ndarray) -> np.ndarray:
        return self._prep_signal_with_fs(y, float(self.cfg["fs"]))

    def _prep_paired_signals(self, a: np.ndarray, b: np.ndarray):
        fs = float(self.cfg["fs"])
        crop = bool(self.cfg["crop"])
        t0 = float(self.cfg["t0"])
        t1 = float(self.cfg["t1"])
        skip_first = bool(self.cfg["skip_first"])

        a = np.asarray(a)
        b = np.asarray(b)

        if skip_first:
            if a.size > 0:
                a = a[1:]
            if b.size > 0:
                b = b[1:]

        n = min(int(a.size), int(b.size))
        if n <= 0:
            raise RuntimeError("RPM and acceleration signals are empty after preprocessing.")
        a = a[:n]
        b = b[:n]

        if crop:
            i0 = max(0, int(np.floor(t0 * fs)))
            i1 = min(n, int(np.ceil(t1 * fs)))
            if i0 >= n:
                raise ValueError("Start time exceeds signal length.")
            if i1 <= i0:
                raise ValueError("End time must be greater than start time.")
            a = a[i0:i1]
            b = b[i0:i1]

        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
        return a, b

    def _time_stats_text(self, name: str, y: np.ndarray, signal_kind: str) -> str:
        unit = self._signal_unit_short(signal_kind)
        if y.size == 0:
            return f"{name}: empty"

        y = y[np.isfinite(y)]
        if y.size == 0:
            return f"{name}: non-finite"

        rms = float(np.sqrt(np.mean(y * y)))
        pk = float(np.max(np.abs(y)))
        p2p = float(np.max(y) - np.min(y))
        return f"{name}: RMS={rms:.5g} {unit}, Peak={pk:.5g} {unit}, Pk-Pk={p2p:.5g} {unit}"

    def _load_selected_signals(self):
        file_type = self.cfg["file_type"]
        file_path = self.cfg["file_path"]
        group_or_sheet = self.cfg["group_or_sheet"]
        names = self.cfg["names"]

        out = {}
        for nm in names:
            out[nm] = read_signal_from_source(file_path, file_type, group_or_sheet, nm)
        return out


    def _filter_params(self):
        ft = self.cfg["filter_type"]
        order = int(self.cfg["filter_order"])
        f1 = float(self.cfg["cut1"])
        f2 = float(self.cfg["cut2"])

        if ft == "Low-pass":
            return "lowpass", order, f1, None
        if ft == "High-pass":
            return "highpass", order, f1, None
        if ft == "Band-pass":
            return "bandpass", order, f1, f2
        return "bandstop", order, f1, f2

    def _apply_custom_butterworth_zero_phase(self, y: np.ndarray, fs: float, btype: str, order: int, f1: float, f2=None) -> np.ndarray:
        if not _HAVE_SCIPY:
            raise RuntimeError("SciPy is required for filtering. Install: pip install scipy")

        nyq = 0.5 * fs

        def _chk_fc(fc: float, name: str):
            if not np.isfinite(fc) or fc <= 0 or fc >= nyq:
                raise ValueError(f"{name} must be between 0 and Nyquist ({nyq:.6g} Hz).")

        _chk_fc(f1, "Cutoff 1")

        if btype in ("bandpass", "bandstop"):
            if f2 is None:
                raise ValueError("Band filters require two cutoff frequencies.")
            _chk_fc(f2, "Cutoff 2")
            if f2 <= f1:
                raise ValueError("For band filters, Cutoff 2 must be greater than Cutoff 1.")
            wn = [f1 / nyq, f2 / nyq]
        else:
            wn = f1 / nyq

        y0 = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
        sos = butter(order, wn, btype=btype, output="sos")
        return sosfiltfilt(sos, y0)

    def _apply_butterworth_zero_phase(self, y: np.ndarray, fs: float) -> np.ndarray:
        if not bool(self.cfg["filter_enabled"]):
            raise RuntimeError("Filtered data selected but filter is not enabled.")
        return self._apply_custom_butterworth_zero_phase(y, fs, *self._filter_params())

    def _maybe_apply_filter_target(self, y: np.ndarray, target_name: str, fs_value: float = None) -> np.ndarray:
        if not bool(self.cfg.get("filter_enabled", False)):
            return y

        filt_target = str(self.cfg.get("waterfall_filter_target", "None"))
        apply_it = (
            (filt_target == "Acceleration" and target_name == "acc") or
            (filt_target == "RPM" and target_name == "rpm") or
            (filt_target == "Both")
        )
        if not apply_it:
            return y

        fs_use = float(self.cfg["fs"] if fs_value is None else fs_value)
        return self._apply_custom_butterworth_zero_phase(y, fs_use, *self._filter_params())

    def _apply_velocity_highpass(self, acc_g: np.ndarray, fs: float) -> np.ndarray:
        return self._apply_custom_butterworth_zero_phase(acc_g, fs, "highpass", 4, 10.0, None)

    def _integrate_acc_to_velocity(self, acc_g: np.ndarray, fs: float) -> np.ndarray:
        if acc_g.size == 0:
            return np.array([], dtype=np.float64)

        dt = 1.0 / fs
        acc_si = acc_g * G0

        v = np.zeros_like(acc_si, dtype=np.float64)
        if acc_si.size > 1:
            v[1:] = np.cumsum(0.5 * (acc_si[:-1] + acc_si[1:]) * dt)

        return v * VEL_SCALE

    def _get_processed_signal(self, raw: np.ndarray) -> np.ndarray:
        fs = float(self.cfg["fs"])
        signal_kind = self.cfg["signal_kind"]
        use_filtered = bool(self.cfg["use_filtered"])

        y = self._prep_signal(raw)

        if signal_kind == "acc":
            return self._apply_butterworth_zero_phase(y, fs) if use_filtered else y

        if not _HAVE_SCIPY:
            raise RuntimeError("Velocity mode requires SciPy because automatic high-pass filtering is applied before integration.")

        acc_base = self._apply_butterworth_zero_phase(y, fs) if use_filtered else y
        acc_hp = self._apply_velocity_highpass(acc_base, fs)
        return self._integrate_acc_to_velocity(acc_hp, fs)

    def _compute_avg_spectrum(self, y: np.ndarray, fs: float, nfft: int, overlap_pct: float,
                              output_type: str, scale: str, signal_kind: str):
        return compute_avg_spectrum_from_signal(
            y=y,
            fs=fs,
            nfft=nfft,
            overlap_pct=overlap_pct,
            output_type=output_type,
            scale=scale,
            amp_conv=self.cfg["amp_conv"],
            signal_kind=signal_kind,
            window_kind=self.cfg.get("window_kind", "periodic"),
        )

    def _compute_stft(self, y: np.ndarray, fs: float, nfft: int, overlap_pct: float,
                      output_type: str, scale: str, signal_kind: str):
        """STFT/spectrogram. Center-of-frame timestamps. Selectable Hann kind."""
        y = np.nan_to_num(np.asarray(y, dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
        fs = float(fs)
        nfft = max(2, int(nfft))
        empty = (np.array([], dtype=np.float64),
                 np.array([], dtype=np.float64),
                 np.empty((0, 0), dtype=np.float64))
        if y.size == 0 or not np.isfinite(fs) or fs <= 0.0:
            return empty

        nperseg = min(nfft, int(y.size))
        if nperseg <= 1:
            return empty

        ovl = float(np.clip(float(overlap_pct) / 100.0, 0.0, 0.95))
        hop = max(1, int(round(nperseg * (1.0 - ovl))))
        n_frames = 1 if y.size <= nperseg else 1 + (y.size - nperseg) // hop

        window_kind = self.cfg.get("window_kind", "periodic")
        w = _hann_window(nperseg, window_kind)
        w_sum = float(np.sum(w)) or float(nperseg)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

        starts = np.arange(n_frames, dtype=np.int64) * hop
        starts = starts[starts + nperseg <= y.size]
        if starts.size == 0:
            return freqs, np.array([], dtype=np.float64), np.empty((freqs.size, 0))

        idx = starts[:, None] + np.arange(nperseg, dtype=np.int64)[None, :]
        segs = y[idx] * w[None, :]
        X = np.fft.rfft(segs, n=nfft, axis=1)

        A_peak = (2.0 / w_sum) * np.abs(X)
        A_peak[:, 0] *= 0.5
        if (nfft % 2 == 0) and (A_peak.shape[1] > 1):
            A_peak[:, -1] *= 0.5

        if str(output_type).strip() == "Amplitude":
            conv, _ = get_amp_conv_factor_label(self.cfg["amp_conv"], signal_kind)
            S = (A_peak * conv).T
        else:
            A_rms = A_peak / np.sqrt(2.0)
            A_rms[:, 0] = A_peak[:, 0]
            if (nfft % 2 == 0) and (A_rms.shape[1] > 1):
                A_rms[:, -1] = A_peak[:, -1]
            S = (A_rms * A_rms).T

        # Center-of-frame timestamps (matches MATLAB/nCode and the
        # window-weighted RPM labels used by _compute_waterfall_map).
        times = (starts + (nperseg - 1) / 2.0) / fs

        if str(scale).strip().lower() == "db":
            eps = 1e-30
            if str(output_type).strip() == "Amplitude":
                S = 20.0 * np.log10(np.maximum(S, eps))
            else:
                S = 10.0 * np.log10(np.maximum(S, eps))

        return freqs, times, S





    def _compute_frame_rpm_label(self, rpm_window: np.ndarray, window: np.ndarray, method: str) -> float:
        """Assign a single RPM to one FFT frame.

        Methods
        -------
        Median in window
            Robust against spikes, but not tied to FFT weighting.
        Center sample
            Uses the RPM at the frame center.
        Window-weighted (Hanning)
            Uses the same Hann/Hanning window as the FFT, so the RPM label
            matches the effective weighting of the samples inside the frame.
            This is generally the most physical choice for ramp/run-up data.
        """
        rpm_window = np.asarray(rpm_window, dtype=np.float64)
        window = np.asarray(window, dtype=np.float64)
        finite = np.isfinite(rpm_window)
        if not np.any(finite):
            return np.nan

        finite_r = rpm_window[finite]
        method = str(method).strip()

        if method == "Center sample":
            center_idx = int(np.clip(rpm_window.size // 2, 0, rpm_window.size - 1))
            center_value = float(rpm_window[center_idx])
            if np.isfinite(center_value):
                return center_value
            return float(np.nanmedian(finite_r))

        if method == "Window-weighted (Hanning)":
            w = window[finite]
            w_sum = float(np.sum(w))
            if np.isfinite(w_sum) and w_sum > 0.0:
                return float(np.sum(w * finite_r) / w_sum)
            return float(np.nanmedian(finite_r))

        return float(np.nanmedian(finite_r))

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

    def _build_dense_waterfall_display(self, frame_rpms: np.ndarray, spectra_linear: np.ndarray,
                                       output_type: str, scale: str,
                                       base_count: int, dense_factor: int = 6,
                                       max_points: int = 12000):
        frame_rpms = np.asarray(frame_rpms, dtype=np.float64)
        spectra_linear = np.asarray(spectra_linear, dtype=np.float64)

        frame_rpms_u, spectra_u = self._aggregate_duplicate_rpms(frame_rpms, spectra_linear)
        if frame_rpms_u.size == 0 or spectra_u.size == 0:
            return np.array([], dtype=np.float64), np.empty((0, 0), dtype=np.float64)

        if frame_rpms_u.size == 1:
            S_single = self._convert_waterfall_scale(spectra_u, output_type, scale)
            return frame_rpms_u.copy(), S_single.copy()

        dense_factor = int(max(1, dense_factor))
        dense_n = int(max(frame_rpms_u.size * dense_factor, base_count * dense_factor))
        dense_n = int(min(max_points, dense_n))
        dense_n = max(2, dense_n)

        display_rpms = np.linspace(float(frame_rpms_u[0]), float(frame_rpms_u[-1]), dense_n)
        S_display_linear = np.empty((display_rpms.size, spectra_u.shape[1]), dtype=np.float64)

        for k in range(spectra_u.shape[1]):
            S_display_linear[:, k] = np.interp(display_rpms, frame_rpms_u, spectra_u[:, k])

        S_display = self._convert_waterfall_scale(S_display_linear, output_type, scale)
        return display_rpms, S_display

    def _compute_waterfall_map(self, acc: np.ndarray, rpm: np.ndarray, fs: float,
                               nfft: int, overlap_pct: float,
                               output_type: str, scale: str,
                               num_spectra: int = 100, num_buffers: int = 1,
                               rpm_label_method: str = "Window-weighted (Hanning)"):
        """Waterfall FFT map. Vectorized framing/FFT, selectable Hann kind."""
        acc = np.nan_to_num(np.asarray(acc, dtype=np.float64),
                            nan=0.0, posinf=0.0, neginf=0.0)
        rpm = np.nan_to_num(np.asarray(rpm, dtype=np.float64),
                            nan=0.0, posinf=0.0, neginf=0.0)
        fs = float(fs)
        nfft = max(2, int(nfft))

        n_common = min(int(acc.size), int(rpm.size))
        if n_common <= 0 or not np.isfinite(fs) or fs <= 0.0:
            raise RuntimeError("Acceleration or RPM signal is empty.")
        acc = acc[:n_common]
        rpm = rpm[:n_common]

        nperseg = min(nfft, n_common)
        if nperseg <= 1:
            raise RuntimeError("Segment length too short for FFT.")

        ovl = float(np.clip(float(overlap_pct) / 100.0, 0.0, 0.95))
        hop = max(1, int(round(nperseg * (1.0 - ovl))))
        n_frames = 1 if n_common <= nperseg else 1 + (n_common - nperseg) // hop

        window_kind = self.cfg.get("window_kind", "periodic")
        w = _hann_window(nperseg, window_kind)
        w_sum = float(np.sum(w)) or float(nperseg)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

        starts = np.arange(n_frames, dtype=np.int64) * hop
        starts = starts[starts + nperseg <= n_common]
        if starts.size == 0:
            raise RuntimeError("Could not build waterfall frames.")

        idx_t = starts[:, None] + np.arange(nperseg, dtype=np.int64)[None, :]
        acc_segs = acc[idx_t] * w[None, :]
        X = np.fft.rfft(acc_segs, n=nfft, axis=1)

        A_peak = (2.0 / w_sum) * np.abs(X)
        A_peak[:, 0] *= 0.5
        if (nfft % 2 == 0) and (A_peak.shape[1] > 1):
            A_peak[:, -1] *= 0.5

        if str(output_type).strip() == "Amplitude":
            conv, _ = get_amp_conv_factor_label(self.cfg["amp_conv"], "acc")
            spectra_per_frame = A_peak * conv
        else:
            A_rms = A_peak / np.sqrt(2.0)
            A_rms[:, 0] = A_peak[:, 0]
            if (nfft % 2 == 0) and (A_rms.shape[1] > 1):
                A_rms[:, -1] = A_peak[:, -1]
            spectra_per_frame = A_rms * A_rms

        rpm_segs = rpm[idx_t]
        method = str(rpm_label_method).strip()
        if method == "Center sample":
            center = (nperseg - 1) // 2
            frame_rpms = rpm_segs[:, center]
        elif method == "Window-weighted (Hanning)":
            finite = np.isfinite(rpm_segs)
            rseg = np.where(finite, rpm_segs, 0.0)
            w_eff = np.where(finite, w[None, :], 0.0)
            denom = w_eff.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                frame_rpms = np.where(
                    denom > 0,
                    (w_eff * rseg).sum(axis=1) / denom,
                    np.nanmedian(rpm_segs, axis=1),
                )
        else:
            frame_rpms = np.nanmedian(rpm_segs, axis=1)

        m = np.isfinite(frame_rpms)
        spectra = spectra_per_frame[m]
        frame_rpms = frame_rpms[m]
        if frame_rpms.size < 2:
            raise RuntimeError("Not enough valid RPM frames for waterfall plotting.")

        order = np.argsort(frame_rpms)
        frame_rpms = frame_rpms[order]
        spectra = spectra[order]
        n_valid = int(frame_rpms.size)

        num_buffers = max(1, min(int(num_buffers), n_valid))
        if num_buffers > 1:
            half = num_buffers // 2
            csum = np.vstack([
                np.zeros((1, spectra.shape[1]), dtype=np.float64),
                np.cumsum(spectra, axis=0, dtype=np.float64),
            ])
            spectra_smoothed = np.empty_like(spectra)
            for i in range(n_valid):
                i0 = max(0, i - half)
                i1 = min(n_valid, i + half + 1)
                count = max(1, i1 - i0)
                spectra_smoothed[i, :] = (csum[i1, :] - csum[i0, :]) / float(count)
        else:
            spectra_smoothed = spectra

        min_rpm = float(frame_rpms[0])
        max_rpm = float(frame_rpms[-1])
        if min_rpm == max_rpm:
            target_rpms = np.array([min_rpm], dtype=np.float64)
            S_analysis_linear = spectra_smoothed[:1, :]
        else:
            target_count = max(int(num_spectra), n_valid)
            target_rpms = np.linspace(min_rpm, max_rpm, target_count, dtype=np.float64)

            if target_count == n_valid and np.all(np.diff(frame_rpms) > 0.0):
                S_analysis_linear = spectra_smoothed
            else:
                uniq_rpms, uniq_spectra = self._aggregate_duplicate_rpms(frame_rpms, spectra_smoothed)
                if uniq_rpms.size < 2:
                    S_analysis_linear = np.repeat(uniq_spectra[:1, :], target_count, axis=0)
                else:
                    S_analysis_linear = np.empty((target_count, uniq_spectra.shape[1]), dtype=np.float64)
                    for j in range(uniq_spectra.shape[1]):
                        S_analysis_linear[:, j] = np.interp(target_rpms, uniq_rpms, uniq_spectra[:, j])

        S_analysis_linear = np.maximum(np.asarray(S_analysis_linear, dtype=np.float64), 0.0)
        S_analysis = self._convert_waterfall_scale(S_analysis_linear, output_type, scale)

        dense_factor = int(max(1, self.cfg.get("waterfall_display_dense_factor", 6)))
        max_points = int(max(200, self.cfg.get("waterfall_display_max_points", 12000)))
        rpms_display, S_display = self._build_dense_waterfall_display(
            frame_rpms=frame_rpms,
            spectra_linear=spectra_smoothed,
            output_type=output_type,
            scale=scale,
            base_count=int(max(1, num_spectra)),
            dense_factor=dense_factor,
            max_points=max_points,
        )

        return {
            "freqs": freqs,
            "freqs_analysis": freqs.copy(),
            "freqs_display": freqs.copy(),
            "rpms_analysis": target_rpms,
            "S_analysis": S_analysis,
            "frame_rpms": frame_rpms,
            "frame_spectra_linear": spectra_smoothed,
            "rpms_display": rpms_display if rpms_display.size else target_rpms,
            "S_display": S_display if S_display.size else S_analysis,
        }

    def _processing_title(self, use_filtered: bool, signal_kind: str) -> str:
        if signal_kind == "vel":
            if use_filtered:
                btype, order, f1, f2 = self._filter_params()
                if btype in ("bandpass", "bandstop"):
                    return f" | AccBase=Filtered | Butterworth {btype} | order={order} | fc=[{f1:g}, {f2:g}] Hz | zero-phase | AutoVel: HP 10 Hz | order=4 | zero-phase | integrate"
                return f" | AccBase=Filtered | Butterworth {btype} | order={order} | fc={f1:g} Hz | zero-phase | AutoVel: HP 10 Hz | order=4 | zero-phase | integrate"
            return " | AccBase=Unfiltered | AutoVel: HP 10 Hz | order=4 | zero-phase | integrate"

        if use_filtered:
            btype, order, f1, f2 = self._filter_params()
            if btype in ("bandpass", "bandstop"):
                return f" | Source=Filtered | Butterworth {btype} | order={order} | fc=[{f1:g}, {f2:g}] Hz | zero-phase"
            return f" | Source=Filtered | Butterworth {btype} | order={order} | fc={f1:g} Hz | zero-phase"

        return " | Source=Unfiltered"

    def _build_waterfall_state(self):
        fs_acc = float(self.cfg.get("waterfall_acc_fs", self.cfg["fs"]))
        fs_rpm = float(self.cfg.get("waterfall_rpm_fs", self.cfg.get("waterfall_acc_fs", self.cfg["fs"])))
        nfft = int(self.cfg["nfft"])
        overlap_pct = float(self.cfg["overlap_pct"])
        out_type = self.cfg["out_type"]
        scale = self.cfg["scale"]
        filt_target = self.cfg.get("waterfall_filter_target", "None")
        resample_mode = str(self.cfg.get("waterfall_resample_mode", "Resample RPM -> Acc timebase"))
        interp_mode = str(self.cfg.get("waterfall_resample_interp", "Linear"))
        rpm_label_method = str(self.cfg.get("waterfall_rpm_label_method", "Window-weighted (Hanning)"))

        acc_src = self.cfg.get("waterfall_acc_source")
        rpm_src = self.cfg.get("waterfall_rpm_source")

        if acc_src and rpm_src:
            acc_name = acc_src["name"]
            rpm_name = rpm_src["name"]
            acc_group = acc_src.get("group_or_sheet", "CSV Data")
            rpm_group = rpm_src.get("group_or_sheet", "CSV Data")
            acc_file_base = os.path.splitext(os.path.basename(acc_src["file_path"]))[0]
            rpm_file_base = os.path.splitext(os.path.basename(rpm_src["file_path"]))[0]
            acc_raw = read_signal_from_source(acc_src["file_path"], acc_src["file_type"], acc_group, acc_name)
            rpm_raw = read_signal_from_source(rpm_src["file_path"], rpm_src["file_type"], rpm_group, rpm_name)
        else:
            raw_signals = self._load_selected_signals()
            acc_name = self.cfg["waterfall_acc_name"]
            rpm_name = self.cfg["waterfall_rpm_name"]
            acc_group = self.cfg["group_or_sheet"]
            rpm_group = self.cfg["group_or_sheet"]
            acc_file_base = os.path.splitext(os.path.basename(self.cfg["file_path"]))[0]
            rpm_file_base = acc_file_base

            if rpm_name not in raw_signals:
                raise KeyError(f"RPM channel not found: {rpm_name}")
            if acc_name not in raw_signals:
                raise KeyError(f"Acceleration channel not found: {acc_name}")

            acc_raw = raw_signals[acc_name]
            rpm_raw = raw_signals[rpm_name]

        acc = self._prep_signal_with_fs(acc_raw, fs_acc)
        rpm = self._prep_signal_with_fs(rpm_raw, fs_rpm)

        acc = self._maybe_apply_filter_target(acc, "acc", fs_value=fs_acc)
        rpm = self._maybe_apply_filter_target(rpm, "rpm", fs_value=fs_rpm)

        acc, rpm, wf_fs = self._align_waterfall_signals(acc, rpm, fs_acc, fs_rpm)

        num_spectra = max(1, int(self.cfg.get("waterfall_num_spectra", 100)))
        num_buffers = max(1, int(self.cfg.get("waterfall_num_buffers", 1)))

        wf_map = self._compute_waterfall_map(
            acc, rpm, wf_fs, nfft, overlap_pct, out_type, scale,
            num_spectra, num_buffers, rpm_label_method=rpm_label_method
        )
        freqs = wf_map["freqs"]
        freqs_display = wf_map.get("freqs_display", freqs)
        rpms = wf_map["rpms_display"] if wf_map["rpms_display"].size else wf_map["rpms_analysis"]
        S_map = wf_map["S_display"] if wf_map["S_display"].size else wf_map["S_analysis"]

        _, unit_lbl = self._amp_conv_factor_label("acc")
        colorbar_label = build_spectrum_ylabel(out_type, scale, unit_lbl)

        same_source = (
            acc_file_base == rpm_file_base and
            str(acc_group) == str(rpm_group) and
            acc_src is not None and rpm_src is not None
        ) or (acc_src is None and rpm_src is None)

        align_txt = f"{resample_mode} | Interp={interp_mode} | RPM label={rpm_label_method} | fs_acc={fs_acc:g} Hz | fs_rpm={fs_rpm:g} Hz | FFT fs={wf_fs:g} Hz"

        if same_source:
            title = (
                f"{acc_file_base} | {acc_group} | Waterfall | Acc={acc_name} | RPM={rpm_name}"
                f" | {align_txt} | FilterTarget={filt_target}"
                f" | NFFT={nfft}, Overlap={overlap_pct:.1f}% | Spectra={num_spectra} | Buffers/Spectrum={num_buffers} | {out_type} ({scale})"
            )
        else:
            title = (
                f"Waterfall | Acc={acc_name} @ {acc_file_base}/{acc_group} | "
                f"RPM={rpm_name} @ {rpm_file_base}/{rpm_group} | "
                f"{align_txt} | FilterTarget={filt_target} | "
                f"NFFT={nfft}, Overlap={overlap_pct:.1f}% | Spectra={num_spectra} | Buffers/Spectrum={num_buffers} | {out_type} ({scale})"
            )

        key = f"waterfall|{acc_file_base}|{acc_group}|{acc_name}|{rpm_file_base}|{rpm_group}|{rpm_name}"

        return {
            "key": key,
            "kind": "waterfall",
            "name": acc_name,
            "short_title": f"{acc_name} | Waterfall",
            "title": title,
            "xlabel": "Frequency (Hz)",
            "ylabel": "RPM",
            "freqs": freqs,
            "freqs_analysis": wf_map["freqs_analysis"],
            "freqs_display": freqs_display,
            "rpms": rpms,
            "S": S_map,
            "rpms_analysis": wf_map["rpms_analysis"],
            "S_analysis": wf_map["S_analysis"],
            "rpms_display": rpms,
            "S_display": S_map,
            "frame_rpms": wf_map["frame_rpms"],
            "frame_spectra_linear": wf_map["frame_spectra_linear"],
            "viewport_dense_factor": float(max(1.0, self.cfg.get("waterfall_viewport_dense_factor", 6))),
            "viewport_min_points": int(max(200, self.cfg.get("waterfall_viewport_min_points", 1200))),
            "viewport_max_points": int(max(1000, self.cfg.get("waterfall_viewport_max_points", 10000))),
            "viewport_freq_dense_factor": float(max(1.0, self.cfg.get("waterfall_viewport_freq_dense_factor", 3.0))),
            "viewport_min_cols": int(max(128, self.cfg.get("waterfall_viewport_min_cols", 1800))),
            "viewport_max_cols": int(max(512, self.cfg.get("waterfall_viewport_max_cols", 12288))),
            "viewport_row_pixel_factor": float(max(1.0, self.cfg.get("waterfall_viewport_row_pixel_factor", 2.25))),
            "viewport_col_pixel_factor": float(max(1.0, self.cfg.get("waterfall_viewport_col_pixel_factor", 2.25))),
            "scale": scale,
            "out_type": out_type,
            "colorbar_label": colorbar_label,
            "stats_text": (
                f"Waterfall built from Acc={acc_name} ({acc_file_base}/{acc_group}, fs={fs_acc:g} Hz) and "
                f"RPM={rpm_name} ({rpm_file_base}/{rpm_group}, fs={fs_rpm:g} Hz) | "
                f"{resample_mode} | Interp={interp_mode} | RPM label={rpm_label_method} | FFT fs={wf_fs:g} Hz | "
                f"Spectra={num_spectra} | Buffers/Spectrum={num_buffers}"
            ),
        }

    @staticmethod
    def _nearest_interp_1d(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(x_src, x_dst, side="left")
        idx = np.clip(idx, 0, x_src.size - 1)

        idx0 = np.clip(idx - 1, 0, x_src.size - 1)
        take_left = np.abs(x_dst - x_src[idx0]) <= np.abs(x_dst - x_src[idx])

        out = y_src[idx].copy()
        out[take_left] = y_src[idx0][take_left]
        return out

    def _resample_uniform_signal(self, y_src: np.ndarray, fs_src: float, t_dst: np.ndarray, method: str) -> np.ndarray:
        y_src = np.asarray(y_src, dtype=np.float64)
        if y_src.size == 0:
            return np.array([], dtype=np.float64)

        x_src = np.arange(y_src.size, dtype=np.float64) / float(fs_src)

        finite = np.isfinite(x_src) & np.isfinite(y_src)
        x_src = x_src[finite]
        y_src = y_src[finite]

        if x_src.size < 2:
            raise RuntimeError("Not enough finite source samples for resampling.")

        x_src, keep = np.unique(x_src, return_index=True)
        y_src = y_src[keep]
        t_dst = np.asarray(t_dst, dtype=np.float64)

        if str(method).strip() == "Nearest":
            return self._nearest_interp_1d(x_src, y_src, t_dst)

        return np.interp(t_dst, x_src, y_src)

    def _align_waterfall_signals(self, acc: np.ndarray, rpm: np.ndarray, fs_acc: float, fs_rpm: float):
        resample_mode = str(self.cfg.get("waterfall_resample_mode", "Resample RPM -> Acc timebase")).strip()
        interp_mode = str(self.cfg.get("waterfall_resample_interp", "Linear")).strip()

        if acc.size == 0 or rpm.size == 0:
            raise RuntimeError("Acceleration or RPM signal is empty after preprocessing.")

        if resample_mode == "Trim only (same fs)":
            tol = max(1e-12, 1e-9 * max(fs_acc, fs_rpm))
            if abs(fs_acc - fs_rpm) > tol:
                raise ValueError("Trim only (same fs) requires equal Acc fs and RPM fs.")
            n = min(acc.size, rpm.size)
            if n < 2:
                raise RuntimeError("Not enough common samples for waterfall plotting.")
            return acc[:n], rpm[:n], fs_acc

        t_acc = np.arange(acc.size, dtype=np.float64) / float(fs_acc)
        t_rpm = np.arange(rpm.size, dtype=np.float64) / float(fs_rpm)

        t_end = min(float(t_acc[-1]), float(t_rpm[-1]))
        if not np.isfinite(t_end) or t_end <= 0.0:
            raise RuntimeError("Common time span for Acc/RPM is too short.")

        if resample_mode == "Resample Acc -> RPM timebase":
            keep = t_rpm <= (t_end + 0.5 / max(fs_rpm, 1e-12))
            t_dst = t_rpm[keep]
            rpm_out = rpm[:t_dst.size]
            acc_out = self._resample_uniform_signal(acc, fs_acc, t_dst, interp_mode)
            return (
                np.nan_to_num(acc_out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False),
                np.nan_to_num(rpm_out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False),
                fs_rpm,
            )

        keep = t_acc <= (t_end + 0.5 / max(fs_acc, 1e-12))
        t_dst = t_acc[keep]
        acc_out = acc[:t_dst.size]
        rpm_out = self._resample_uniform_signal(rpm, fs_rpm, t_dst, interp_mode)
        return (
            np.nan_to_num(acc_out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False),
            np.nan_to_num(rpm_out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False),
            fs_acc,
        )

    def _build_plot_states(self):
        fs = float(self.cfg["fs"])
        crop = bool(self.cfg["crop"])
        t0 = float(self.cfg["t0"])
        mode = self.cfg["mode"]
        signal_kind = self.cfg["signal_kind"]
        signal_name = self.cfg["signal_name"]
        names = self.cfg["names"]
        group_or_sheet = self.cfg["group_or_sheet"]
        file_base = os.path.splitext(os.path.basename(self.cfg["file_path"]))[0]
        use_filtered = bool(self.cfg["use_filtered"])

        if mode == "Waterfall (RPM + Acceleration)":
            return [self._build_waterfall_state()]

        raw_signals = self._load_selected_signals()
        states = []

        for nm in names:
            y = self._get_processed_signal(np.asarray(raw_signals[nm]))
            short_title = f"{nm} | {signal_name}"
            proc_title = self._processing_title(use_filtered, signal_kind)
            key = f"{mode}|{signal_kind}|{group_or_sheet}|{nm}"

            if mode.startswith("Time Domain"):
                start_offset = t0 if crop else 0.0
                t = self._make_time_axis(y.size, fs, start_offset_s=start_offset)

                states.append({
                    "key": key,
                    "kind": "time",
                    "name": nm,
                    "short_title": short_title,
                    "title": f"{file_base} | {group_or_sheet} | {signal_name}{proc_title} | {nm}",
                    "xlabel": "Time (s)",
                    "ylabel": self._signal_ylabel(signal_kind),
                    "grid": True,
                    "series": [{"x": t, "y": y, "label": nm}],
                    "time_signal": y.copy(),
                    "time_axis": t.copy(),
                    "fs": fs,
                    "fft_zoom_cfg": {
                        "nfft": int(self.cfg["nfft"]),
                        "overlap_pct": float(self.cfg["overlap_pct"]),
                        "out_type": self.cfg["out_type"],
                        "scale": self.cfg["scale"],
                        "amp_conv": self.cfg["amp_conv"],
                        "signal_kind": signal_kind,
                        "window_kind": self.cfg.get("window_kind", "periodic"),
                    },
                    "stats_text": self._time_stats_text(nm, y, signal_kind),
                })
                continue

            nfft = int(self.cfg["nfft"])
            overlap_pct = float(self.cfg["overlap_pct"])
            out_type = self.cfg["out_type"]
            scale = self.cfg["scale"]
            method = self.cfg["method"]
            _, unit_lbl = self._amp_conv_factor_label(signal_kind)

            if method == "Averaged Spectrum":
                freqs, spec = self._compute_avg_spectrum(y, fs, nfft, overlap_pct, out_type, scale, signal_kind)
                ylabel = build_spectrum_ylabel(out_type, scale, unit_lbl)

                states.append({
                    "key": key,
                    "kind": "avg",
                    "name": nm,
                    "short_title": short_title,
                    "title": f"{file_base} | {group_or_sheet} | Averaged Spectrum ({signal_name}){proc_title} | {nm} | NFFT={nfft}, Overlap={overlap_pct:.1f}% | {out_type} ({scale}) | {unit_lbl}",
                    "xlabel": "Frequency (Hz)",
                    "ylabel": ylabel,
                    "grid": True,
                    "series": [{"x": freqs, "y": spec, "label": nm}],
                    "stats_text": "",
                })
                continue

            freqs, times, S = self._compute_stft(y, fs, nfft, overlap_pct, out_type, scale, signal_kind)
            if crop:
                times = times + t0

            colorbar_label = build_spectrum_ylabel(out_type, scale, unit_lbl)

            states.append({
                "key": key,
                "kind": "stft",
                "name": nm,
                "short_title": short_title,
                "title": f"{file_base} | {group_or_sheet} | STFT Spectrogram ({signal_name}){proc_title} | {nm} | NFFT={nfft}, Overlap={overlap_pct:.1f}% | {out_type} ({scale}) | {unit_lbl}",
                "xlabel": "Time (s)",
                "ylabel": "Frequency (Hz)",
                "freqs": freqs,
                "times": times,
                "S": S,
                "out_type": out_type,
                "scale": scale,
                "unit_lbl": unit_lbl,
                "colorbar_label": colorbar_label,
                "stats_text": "",
            })

        if not states:
            raise RuntimeError("No data left to plot after preprocessing.")

        return states

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Excel/TDMS/CSV Signal Plotter (PyQt5 + PyQtGraph)")

        self.file_path = None
        self.file_type = None
        self.sheet_names = []

        self.wf_acc_path = None
        self.wf_acc_type = None
        self.wf_acc_containers = []
        self.wf_rpm_path = None
        self.wf_rpm_type = None
        self.wf_rpm_containers = []

        self._file_thread = None
        self._file_worker = None
        self._sheet_thread = None
        self._sheet_worker = None
        self._plot_thread = None
        self._plot_worker = None

        self._plot_states = {}
        self._mini_widgets = {}
        self._detail_widgets = {}
        self._overview_order = []
        self._active_canvas = None
        self._active_plot_key = None
        self._pending_waterfall_open_3d = False
        self.custom_color_bands = default_custom_color_bands()

        self.btn_open = QPushButton("Open File (.xlsx / .tdms / .csv)")
        self.lbl_file = QLabel("Selected file: -")
        self.lbl_file.setWordWrap(True)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFixedHeight(10)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #444;")
        self.lbl_status.setWordWrap(True)

        self.cmb_sheet = QComboBox()
        self.cmb_sheet.setEnabled(False)

        self.list_channels = QListWidget()
        self.list_channels.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_channels.setEnabled(False)
        self.list_channels.setMinimumHeight(110)

        self.btn_select_all = QPushButton("Select All")
        self.btn_clear_sel = QPushButton("Clear Selection")
        self.btn_select_all.setEnabled(False)
        self.btn_clear_sel.setEnabled(False)

        self.fs_spin = QDoubleSpinBox()
        self.fs_spin.setRange(0.1, 1_000_000.0)
        self.fs_spin.setDecimals(2)
        self.fs_spin.setValue(10000.0)

        self.chk_skip_first = QCheckBox("Skip first sample (y[1:])")

        self.chk_crop = QCheckBox("Enable time window (crop)")
        self.t0_spin = QDoubleSpinBox()
        self.t0_spin.setRange(0.0, 1_000_000_000.0)
        self.t0_spin.setDecimals(3)
        self.t0_spin.setValue(10.0)
        self.t0_spin.setEnabled(False)

        self.t1_spin = QDoubleSpinBox()
        self.t1_spin.setRange(0.0, 1_000_000_000.0)
        self.t1_spin.setDecimals(3)
        self.t1_spin.setValue(30.0)
        self.t1_spin.setEnabled(False)

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems([
            "Time Domain (Acceleration)",
            "Time Domain (Velocity)",
            "Fourier Transform (Acceleration)",
            "Fourier Transform (Velocity)",
        ])

        self.cmb_fourier_method = QComboBox()
        self.cmb_fourier_method.addItems(["Averaged Spectrum", "STFT (Spectrogram)"])

        self.cmb_nfft = QComboBox()
        self.cmb_nfft.addItems([str(x) for x in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]])
        self.cmb_nfft.setCurrentText("8192")

        self.ovl_spin = QDoubleSpinBox()
        self.ovl_spin.setRange(0.0, 95.0)
        self.ovl_spin.setDecimals(1)
        self.ovl_spin.setSingleStep(5.0)
        self.ovl_spin.setValue(67.0)

        self.cmb_out = QComboBox()
        self.cmb_out.addItems(["Amplitude", "Power"])

        self.cmb_scale = QComboBox()
        self.cmb_scale.addItems(["Linear", "dB"])

        self.cmb_amp_conv = QComboBox()
        self.cmb_amp_conv.addItems(["Peak", "RMS", "Peak-to-Peak"])
        self.cmb_amp_conv.setCurrentText("RMS")

        self.cmb_window_kind = QComboBox()
        self.cmb_window_kind.addItems(["Periodic", "Symmetric"])
        self.cmb_window_kind.setCurrentText("Periodic")
        self.cmb_window_kind.setToolTip(
            "Hann window type.\n"
            "Periodic (recommended): matches MATLAB pwelch / scipy welch / nCode. "
            "Sum is exactly N/2.\n"
            "Symmetric: matches numpy.hanning() and legacy code."
        )

        self.cmb_data_source = QComboBox()
        self.cmb_data_source.addItems(["Unfiltered", "Filtered"])
        self.cmb_data_source.setCurrentText("Unfiltered")

        self.lbl_fft_info = QLabel("Frequency resolution: -")
        self.lbl_fft_info.setStyleSheet("color: #444;")

        self.lbl_stats = QLabel("")
        self.lbl_stats.setStyleSheet("color: #444;")
        self.lbl_stats.setWordWrap(True)

        self.chk_filter = QCheckBox("Enable Butterworth filter (zero-phase: forward+backward)")
        self.chk_filter.setChecked(False)
        if not _HAVE_SCIPY:
            self.chk_filter.setEnabled(False)
            self.chk_filter.setToolTip("SciPy not found. Install: pip install scipy")

        self.cmb_filter_type = QComboBox()
        self.cmb_filter_type.addItems(["Low-pass", "High-pass", "Band-pass", "Band-reject"])
        self.cmb_filter_type.setEnabled(False)

        self.cmb_filter_order = QComboBox()
        self.cmb_filter_order.addItems([str(i) for i in range(1, 9)])
        self.cmb_filter_order.setCurrentText("4")
        self.cmb_filter_order.setEnabled(False)

        self.cut1_spin = QDoubleSpinBox()
        self.cut1_spin.setRange(0.0, 1_000_000.0)
        self.cut1_spin.setDecimals(3)
        self.cut1_spin.setValue(100.0)
        self.cut1_spin.setEnabled(False)

        self.lbl_cut2 = QLabel("Cutoff 2 (Hz):")
        self.cut2_spin = QDoubleSpinBox()
        self.cut2_spin.setRange(0.0, 1_000_000.0)
        self.cut2_spin.setDecimals(3)
        self.cut2_spin.setValue(300.0)
        self.cut2_spin.setEnabled(False)

        self.btn_plot = QPushButton("Plot")
        self.btn_plot.setEnabled(False)

        self.btn_export_cropped_tdms = QPushButton("Export Cropped TDMS")
        self.btn_export_cropped_tdms.setEnabled(False)
        self.btn_export_cropped_tdms.setToolTip(
            "Crop selected channels to a time window, zero-align the time axis, "
            "and save as .tdms with metadata preserved."
        )


        self.lbl_wf_hint = QLabel("Dedicated waterfall workflow is available in the Waterfall tab.")
        self.lbl_wf_hint.setStyleSheet("color: #666;")
        self.lbl_wf_hint.setWordWrap(True)

        self.btn_zoom_in = QPushButton("Zoom In")
        self.btn_zoom_out = QPushButton("Zoom Out")
        self.btn_reset_view = QPushButton("Reset View")
        self.btn_rect_zoom = QPushButton("Rectangle Zoom")
        self.btn_rect_zoom.setCheckable(True)
        self.btn_apply_plot_view = QPushButton("Apply Plot View")
        self.btn_export_plot = QPushButton("Export Current Plot")

        self.edit_xlabel = QLineEdit()
        self.edit_ylabel = QLineEdit()
        self.edit_xlabel.setPlaceholderText("X axis label")
        self.edit_ylabel.setPlaceholderText("Y axis label")
        self.edit_xlabel.setEnabled(False)
        self.edit_ylabel.setEnabled(False)

        self.cmb_spec_render = QComboBox()
        self.cmb_spec_render.addItems(["ImageItem"])
        self.cmb_spec_render.setCurrentText("ImageItem")

        self.cmb_spec_cmap = QComboBox()
        for cmap_name in [
            "viridis", "plasma", "inferno", "magma", "cividis",
            "turbo", "jet", "hot", "cool", "coolwarm",
            "nipy_spectral", "gist_rainbow"
        ]:
            self.cmb_spec_cmap.addItem(make_colormap_preview_icon(cmap_name), cmap_name)
        self.cmb_spec_cmap.setCurrentText("viridis")

        self.chk_spec_cmap_reverse = QCheckBox("Reverse colors")

        self.lbl_dyn_range = QLabel("Dynamic Range (dB):")
        self.spec_levels_spin = QSpinBox()
        self.spec_levels_spin.setRange(4, 512)
        self.spec_levels_spin.setValue(80)
        self.spec_levels_spin.setToolTip("For dB plots in continuous mode: vmin = vmax - dynamic range")
        self.spec_levels_spin.setEnabled(False)

        self.cmb_color_mode = QComboBox()
        self.cmb_color_mode.addItems(["Continuous", "Custom Bands"])
        self.btn_edit_bands = QPushButton("Edit Bands...")

        self.chk_manual_clim = QCheckBox("Manual color range (vmin/vmax)")

        self.spec_vmin_spin = QDoubleSpinBox()
        self.spec_vmin_spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
        self.spec_vmin_spin.setDecimals(6)
        self.spec_vmin_spin.setValue(-80.0)
        self.spec_vmin_spin.setEnabled(False)

        self.spec_vmax_spin = QDoubleSpinBox()
        self.spec_vmax_spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
        self.spec_vmax_spin.setDecimals(6)
        self.spec_vmax_spin.setValue(0.0)
        self.spec_vmax_spin.setEnabled(False)

        self.plot_pages = QTabWidget()
        self.plot_pages.setTabsClosable(True)
        self.plot_pages.tabCloseRequested.connect(self._on_plot_page_close)
        self.plot_pages.currentChanged.connect(self._on_plot_page_changed)

        self.overview_page = QWidget()
        self.plot_pages.addTab(self.overview_page, "Overview")

        self.overview_scroll = QScrollArea()
        self.overview_scroll.setWidgetResizable(True)

        self.overview_inner = QWidget()
        self.overview_grid = QGridLayout(self.overview_inner)
        self.overview_grid.setContentsMargins(6, 6, 6, 6)
        self.overview_grid.setSpacing(8)

        self.overview_scroll.setWidget(self.overview_inner)

        self.lbl_overview_empty = QLabel("No plots yet.")
        self.lbl_overview_empty.setStyleSheet("color: #666;")

        ovlay = QVBoxLayout(self.overview_page)
        ovlay.addWidget(self.lbl_overview_empty)
        ovlay.addWidget(self.overview_scroll, 1)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self.tab_controls = QWidget()
        self.tab_waterfall = QWidget()
        self.tab_plot = QWidget()
        self.tabs.addTab(self.tab_controls, "Controls")
        self.tabs.addTab(self.tab_waterfall, "Waterfall")
        self.tabs.addTab(self.tab_plot, "Plot View")

        ctl = QVBoxLayout(self.tab_controls)
        ctl.addWidget(self.btn_open)
        ctl.addWidget(self.lbl_file)
        ctl.addWidget(self.progress)
        ctl.addWidget(self.lbl_status)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Sheet / Group:"))
        row1.addWidget(self.cmb_sheet, 2)
        row1.addSpacing(12)
        row1.addWidget(QLabel("Plot Mode:"))
        row1.addWidget(self.cmb_mode, 1)
        ctl.addLayout(row1)

        ctl.addWidget(self.lbl_wf_hint)
        ctl.addWidget(self.btn_export_cropped_tdms)
        ctl.addWidget(QLabel("Channel(s) / Column(s):"))
        ctl.addWidget(self.list_channels)

        row_ch = QHBoxLayout()
        row_ch.addWidget(self.btn_select_all)
        row_ch.addWidget(self.btn_clear_sel)
        row_ch.addStretch(1)
        ctl.addLayout(row_ch)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("fs (Hz):"))
        row2.addWidget(self.fs_spin)
        row2.addSpacing(14)
        row2.addWidget(self.chk_skip_first)
        row2.addSpacing(14)
        row2.addWidget(QLabel("Data Source:"))
        row2.addWidget(self.cmb_data_source)
        row2.addStretch(1)
        ctl.addLayout(row2)

        row_filt = QHBoxLayout()
        row_filt.addWidget(self.chk_filter)
        row_filt.addSpacing(12)
        row_filt.addWidget(QLabel("Type:"))
        row_filt.addWidget(self.cmb_filter_type)
        row_filt.addSpacing(8)
        row_filt.addWidget(QLabel("Order:"))
        row_filt.addWidget(self.cmb_filter_order)
        row_filt.addSpacing(8)
        row_filt.addWidget(QLabel("Cutoff 1 (Hz):"))
        row_filt.addWidget(self.cut1_spin)
        row_filt.addWidget(self.lbl_cut2)
        row_filt.addWidget(self.cut2_spin)
        row_filt.addStretch(1)
        ctl.addLayout(row_filt)

        row3 = QHBoxLayout()
        row3.addWidget(self.chk_crop)
        row3.addWidget(QLabel("Start (s):"))
        row3.addWidget(self.t0_spin)
        row3.addWidget(QLabel("End (s):"))
        row3.addWidget(self.t1_spin)
        row3.addStretch(1)
        row3.addWidget(self.btn_plot)
        ctl.addLayout(row3)

        row_fft = QHBoxLayout()
        row_fft.addWidget(QLabel("Fourier Method:"))
        row_fft.addWidget(self.cmb_fourier_method)
        row_fft.addSpacing(12)
        row_fft.addWidget(QLabel("Buffer Size (NFFT):"))
        row_fft.addWidget(self.cmb_nfft)
        row_fft.addWidget(QLabel("Overlap (%):"))
        row_fft.addWidget(self.ovl_spin)
        row_fft.addWidget(QLabel("Output:"))
        row_fft.addWidget(self.cmb_out)
        row_fft.addWidget(QLabel("Scale:"))
        row_fft.addWidget(self.cmb_scale)
        row_fft.addWidget(QLabel("Amplitude convention:"))
        row_fft.addWidget(self.cmb_amp_conv)
        row_fft.addWidget(QLabel("Window:"))
        row_fft.addWidget(self.cmb_window_kind)
        row_fft.addStretch(1)
        ctl.addLayout(row_fft)

        ctl.addWidget(self.lbl_fft_info)
        ctl.addWidget(self.lbl_stats)
        ctl.addStretch(1)

        self._build_waterfall_tab_ui()

        ptab = QVBoxLayout(self.tab_plot)

        row_nav = QHBoxLayout()
        row_nav.addWidget(self.btn_zoom_in)
        row_nav.addWidget(self.btn_zoom_out)
        row_nav.addWidget(self.btn_reset_view)
        row_nav.addWidget(self.btn_rect_zoom)
        row_nav.addSpacing(12)
        row_nav.addWidget(self.btn_apply_plot_view)
        row_nav.addWidget(self.btn_export_plot)
        row_nav.addStretch(1)
        ptab.addLayout(row_nav)

        row_axis = QHBoxLayout()
        row_axis.addWidget(QLabel("X Axis Name:"))
        row_axis.addWidget(self.edit_xlabel, 1)
        row_axis.addSpacing(10)
        row_axis.addWidget(QLabel("Y Axis Name:"))
        row_axis.addWidget(self.edit_ylabel, 1)
        ptab.addLayout(row_axis)

        row_spec_top = QHBoxLayout()
        row_spec_top.addWidget(QLabel("Spectrogram render:"))
        row_spec_top.addWidget(self.cmb_spec_render)
        row_spec_top.addSpacing(8)
        row_spec_top.addWidget(QLabel("Colormap:"))
        row_spec_top.addWidget(self.cmb_spec_cmap)
        row_spec_top.addWidget(self.chk_spec_cmap_reverse)
        row_spec_top.addSpacing(8)
        row_spec_top.addWidget(QLabel("Color Mode:"))
        row_spec_top.addWidget(self.cmb_color_mode)
        row_spec_top.addWidget(self.btn_edit_bands)
        row_spec_top.addStretch(1)
        ptab.addLayout(row_spec_top)

        row_spec_bottom = QHBoxLayout()
        row_spec_bottom.addWidget(self.lbl_dyn_range)
        row_spec_bottom.addWidget(self.spec_levels_spin)
        row_spec_bottom.addSpacing(8)
        row_spec_bottom.addWidget(self.chk_manual_clim)
        row_spec_bottom.addWidget(QLabel("vmin:"))
        row_spec_bottom.addWidget(self.spec_vmin_spin)
        row_spec_bottom.addWidget(QLabel("vmax:"))
        row_spec_bottom.addWidget(self.spec_vmax_spin)
        row_spec_bottom.addStretch(1)
        ptab.addLayout(row_spec_bottom)

        ptab.addWidget(self.plot_pages, 1)

        self.btn_open.clicked.connect(self.open_file)
        self.cmb_sheet.currentIndexChanged.connect(self.load_sheet_or_group)
        self.btn_plot.clicked.connect(self.plot)
        self.btn_export_cropped_tdms.clicked.connect(self._export_cropped_tdms)
        self.chk_crop.stateChanged.connect(self._on_crop_toggle)
        self.btn_select_all.clicked.connect(self._select_all_channels)
        self.btn_clear_sel.clicked.connect(self._clear_channel_selection)

        self.cmb_mode.currentIndexChanged.connect(self._on_mode_change)
        self.cmb_fourier_method.currentIndexChanged.connect(self._sync_spectrogram_controls)

        self.fs_spin.valueChanged.connect(self._update_fft_info)
        self.cmb_nfft.currentTextChanged.connect(self._update_fft_info)
        self.ovl_spin.valueChanged.connect(self._update_fft_info)
        self.cmb_amp_conv.currentIndexChanged.connect(self._update_fft_info)
        self.cmb_mode.currentIndexChanged.connect(self._update_fft_info)

        self.cmb_data_source.currentIndexChanged.connect(self._on_data_source_change)
        self.chk_filter.stateChanged.connect(self._on_filter_toggle)
        self.cmb_filter_type.currentIndexChanged.connect(self._on_filter_type_change)

        self.btn_zoom_in.clicked.connect(lambda: self._zoom_active(0.8))
        self.btn_zoom_out.clicked.connect(lambda: self._zoom_active(1.25))
        self.btn_reset_view.clicked.connect(self._reset_active_view)
        self.btn_rect_zoom.toggled.connect(self._on_rect_zoom_toggled)
        self.chk_manual_clim.stateChanged.connect(self._on_manual_clim_toggle)
        self.cmb_spec_cmap.currentIndexChanged.connect(self._on_color_controls_changed)
        self.chk_spec_cmap_reverse.stateChanged.connect(self._on_color_controls_changed)
        self.spec_levels_spin.valueChanged.connect(self._on_color_controls_changed)
        self.spec_vmin_spin.valueChanged.connect(self._on_color_controls_changed)
        self.spec_vmax_spin.valueChanged.connect(self._on_color_controls_changed)
        self.cmb_color_mode.currentIndexChanged.connect(self._on_color_mode_changed)
        self.btn_edit_bands.clicked.connect(self._edit_custom_bands)
        self.btn_apply_plot_view.clicked.connect(self._apply_plot_view)
        self.btn_export_plot.clicked.connect(self._export_current_plot)

        self._sync_data_source_items()
        self._on_mode_change()
        self._on_filter_type_change()
        self._sync_spectrogram_controls()
        self._update_fft_info()
        self._refresh_enabled_state()

    def _set_busy(self, active: bool, text: str = ""):
        if active:
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
        else:
            self.progress.setVisible(False)
            self.progress.setRange(0, 1)
            self.progress.setValue(0)

        self.lbl_status.setText(text)

    def _thread_running(self, thread_obj):
        try:
            return thread_obj is not None and thread_obj.isRunning()
        except Exception:
            return False

    def _refresh_enabled_state(self):
        busy = (
            self._thread_running(self._file_thread) or
            self._thread_running(self._sheet_thread) or
            self._thread_running(self._plot_thread)
        )

        self.btn_open.setEnabled(not busy)
        self.cmb_sheet.setEnabled(bool(self.sheet_names) and not busy)

        ready = (self.list_channels.count() > 0) and not busy
        self.list_channels.setEnabled(ready)
        self.btn_select_all.setEnabled(ready)
        self.btn_clear_sel.setEnabled(ready)
        self.btn_plot.setEnabled(ready)

        export_ready = (self.list_channels.count() > 0) and not busy and bool(self.file_path)
        self.btn_export_cropped_tdms.setEnabled(export_ready)

        if hasattr(self, "btn_wf_plot"):
            wf_ready = (
                bool(getattr(self, "wf_acc_path", None)) and
                bool(getattr(self, "wf_rpm_path", None)) and
                self.cmb_wf_acc_group.count() > 0 and
                self.cmb_wf_rpm_group.count() > 0 and
                self.cmb_wf_acc_channel.count() > 0 and
                self.cmb_wf_rpm_channel.count() > 0 and
                not busy
            )
            self.btn_wf_plot.setEnabled(wf_ready)
            self.btn_wf_acc_open.setEnabled(not busy)
            self.btn_wf_rpm_open.setEnabled(not busy)
            self.cmb_wf_acc_group.setEnabled(self.cmb_wf_acc_group.count() > 0 and not busy)
            self.cmb_wf_rpm_group.setEnabled(self.cmb_wf_rpm_group.count() > 0 and not busy)
            self.cmb_wf_acc_channel.setEnabled(self.cmb_wf_acc_channel.count() > 0 and not busy)
            self.cmb_wf_rpm_channel.setEnabled(self.cmb_wf_rpm_channel.count() > 0 and not busy)

    def _plot_color_cfg(self):
        return {
            "cmap": self.cmb_spec_cmap.currentText().strip(),
            "reverse": self.chk_spec_cmap_reverse.isChecked(),
            "manual_clim": self.chk_manual_clim.isChecked(),
            "vmin": float(self.spec_vmin_spin.value()),
            "vmax": float(self.spec_vmax_spin.value()),
            "dyn_range_db": float(self.spec_levels_spin.value()),
            "color_mode": self.cmb_color_mode.currentText().strip(),
            "custom_bands": [dict(b) for b in self.custom_color_bands],
        }

    def _connect_canvas_color_signals(self, canvas):
        if canvas is None:
            return
        try:
            canvas.color_levels_changed.connect(self._on_canvas_color_levels_changed)
        except Exception:
            pass
        try:
            canvas.color_levels_change_finished.connect(self._on_canvas_color_levels_change_finished)
        except Exception:
            pass

    def _has_active_color_plot(self):
        state = self._plot_states.get(self._active_plot_key) if self._active_plot_key else None
        return bool(state and state.get("kind") in ("stft", "waterfall"))

    def _on_canvas_color_levels_changed(self, vmin: float, vmax: float):
        if self._active_canvas is None or not self._has_active_color_plot():
            return
        self.chk_manual_clim.blockSignals(True)
        self.chk_manual_clim.setChecked(True)
        self.chk_manual_clim.blockSignals(False)
        self.spec_vmin_spin.blockSignals(True)
        self.spec_vmax_spin.blockSignals(True)
        self.spec_vmin_spin.setValue(float(vmin))
        self.spec_vmax_spin.setValue(float(vmax))
        self.spec_vmin_spin.blockSignals(False)
        self.spec_vmax_spin.blockSignals(False)
        self._sync_spectrogram_controls()

    def _on_canvas_color_levels_change_finished(self, vmin: float, vmax: float):
        self._on_canvas_color_levels_changed(vmin, vmax)
        if self._active_plot_key in self._plot_states:
            self._refresh_plot_views_for_key(
                self._active_plot_key,
                preserve_active_limits=True,
                preserve_active_annotations=True,
            )

    def _on_color_controls_changed(self, *_):
        if self._active_plot_key not in self._plot_states:
            return
        state = self._plot_states[self._active_plot_key]
        if state.get("kind") not in ("stft", "waterfall"):
            return
        self._sync_spectrogram_controls()
        self._refresh_plot_views_for_key(
            self._active_plot_key,
            preserve_active_limits=True,
            preserve_active_annotations=True,
        )

    def _on_color_mode_changed(self, *_):
        if self.cmb_color_mode.currentText() == "Custom Bands":
            self.chk_manual_clim.blockSignals(True)
            self.chk_manual_clim.setChecked(False)
            self.chk_manual_clim.blockSignals(False)
        self._sync_spectrogram_controls()
        self._on_color_controls_changed()

    def _edit_custom_bands(self):
        dlg = BandEditorDialog(self.custom_color_bands, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        try:
            self.custom_color_bands = dlg.bands()
        except Exception as e:
            self._err(str(e))
            return
        self._on_color_controls_changed()

    def _load_axis_editors_from_state(self, state=None):
        st = state or (self._plot_states.get(self._active_plot_key) if self._active_plot_key else None)

        self.edit_xlabel.blockSignals(True)
        self.edit_ylabel.blockSignals(True)

        if st is None:
            self.edit_xlabel.setText("")
            self.edit_ylabel.setText("")
        else:
            self.edit_xlabel.setText(str(st.get("xlabel", "")))
            self.edit_ylabel.setText(str(st.get("ylabel", "")))

        self.edit_xlabel.blockSignals(False)
        self.edit_ylabel.blockSignals(False)

    def _refresh_plot_views_for_key(self, key, preserve_active_limits=True, preserve_active_annotations=True):
        if key not in self._plot_states:
            return

        state = self._plot_states[key]

        canvases = []
        mini = self._mini_widgets.get(key)
        detail = self._detail_widgets.get(key)

        if mini is not None:
            canvases.append(mini.canvas)
        if detail is not None:
            canvases.append(detail.canvas)

        for canvas in canvases:
            is_active = (canvas is self._active_canvas and self._active_plot_key == key)
            canvas.set_state(
                state,
                color_cfg=self._plot_color_cfg(),
                preserve_view=(preserve_active_limits and is_active),
                preserve_annotations=(preserve_active_annotations and is_active),
            )

    def _clear_overview_widgets(self):
        for w in list(self._mini_widgets.values()):
            w.setParent(None)
            w.deleteLater()
        self._mini_widgets.clear()
        self._overview_order.clear()

        while self.overview_grid.count():
            item = self.overview_grid.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()

    def _clear_detail_tabs(self):
        for key, widget in list(self._detail_widgets.items()):
            idx = self.plot_pages.indexOf(widget)
            if idx >= 0:
                self.plot_pages.removeTab(idx)
            widget.deleteLater()
        self._detail_widgets.clear()

    def _reset_plot_workspace(self):
        self._plot_states.clear()
        self._clear_overview_widgets()
        self._clear_detail_tabs()

        self.lbl_overview_empty.setVisible(True)
        self._active_canvas = None
        self._active_plot_key = None
        self.plot_pages.setCurrentIndex(0)
        self._sync_spectrogram_controls()
        self._load_axis_editors_from_state(None)

    def _reset_loaded_state(self, keep_path=False):
        if not keep_path:
            self.file_path = None
            self.lbl_file.setText("Selected file: -")

        self.file_type = None
        self.sheet_names = []

        self.cmb_sheet.blockSignals(True)
        self.cmb_sheet.clear()
        self.cmb_sheet.blockSignals(False)

        self.list_channels.clear()
        self.lbl_stats.setText("")
        self._reset_plot_workspace()
        self._refresh_enabled_state()

    def _cleanup_file_thread(self):
        if self._file_worker is not None:
            self._file_worker.deleteLater()
            self._file_worker = None
        if self._file_thread is not None:
            self._file_thread.deleteLater()
            self._file_thread = None

        self._set_busy(False, "")
        self._refresh_enabled_state()

    def _cleanup_sheet_thread(self):
        if self._sheet_worker is not None:
            self._sheet_worker.deleteLater()
            self._sheet_worker = None
        if self._sheet_thread is not None:
            self._sheet_thread.deleteLater()
            self._sheet_thread = None

        self._set_busy(False, "")
        self._refresh_enabled_state()

    def _cleanup_plot_thread(self):
        if self._plot_worker is not None:
            self._plot_worker.deleteLater()
            self._plot_worker = None
        if self._plot_thread is not None:
            self._plot_thread.deleteLater()
            self._plot_thread = None

        self._set_busy(False, "")
        self._refresh_enabled_state()

    def _start_file_open_worker(self, path: str):
        if self._thread_running(self._file_thread) or self._thread_running(self._sheet_thread) or self._thread_running(self._plot_thread):
            self._err("Another operation is already running.")
            return

        self._set_busy(True, "Reading file metadata in background...")
        self._refresh_enabled_state()

        self._file_thread = QThread(self)
        self._file_worker = FileOpenWorker(path)
        self._file_worker.moveToThread(self._file_thread)

        self._file_thread.started.connect(self._file_worker.run)
        self._file_worker.finished.connect(self._on_file_open_finished)
        self._file_worker.error.connect(self._on_file_open_error)

        self._file_worker.finished.connect(self._file_thread.quit)
        self._file_worker.error.connect(self._file_thread.quit)
        self._file_thread.finished.connect(self._cleanup_file_thread)

        self._file_thread.start()

    def _start_sheet_load_worker(self, name: str):
        if not self.file_path or not self.file_type or not name:
            return
        if self._thread_running(self._sheet_thread) or self._thread_running(self._plot_thread):
            return

        self._set_busy(True, f"Loading '{name}' in background...")
        self.list_channels.clear()
        self.lbl_stats.setText("")
        self._refresh_enabled_state()

        self._sheet_thread = QThread(self)
        self._sheet_worker = SheetLoadWorker(self.file_path, self.file_type, name)
        self._sheet_worker.moveToThread(self._sheet_thread)

        self._sheet_thread.started.connect(self._sheet_worker.run)
        self._sheet_worker.finished.connect(self._on_sheet_load_finished)
        self._sheet_worker.error.connect(self._on_sheet_load_error)

        self._sheet_worker.finished.connect(self._sheet_thread.quit)
        self._sheet_worker.error.connect(self._sheet_thread.quit)
        self._sheet_thread.finished.connect(self._cleanup_sheet_thread)

        self._sheet_thread.start()

    def _on_file_open_finished(self, meta):
        self.file_type = meta["file_type"]
        self.sheet_names = list(meta["sheet_names"])

        self.cmb_sheet.blockSignals(True)
        self.cmb_sheet.clear()
        self.cmb_sheet.addItems(self.sheet_names)

        idx = max(0, min(int(meta.get("default_index", 0)), len(self.sheet_names) - 1))
        self.cmb_sheet.setCurrentIndex(idx)
        self.cmb_sheet.blockSignals(False)

        self._refresh_enabled_state()
        self.load_sheet_or_group()

    def _on_file_open_error(self, msg: str):
        self._err(f"Failed to open file:\n{msg}")

    def _on_sheet_load_finished(self, result):
        self._fill_channel_list(result["names"], result.get("prefer_index", 0))
        self._refresh_enabled_state()

    def _on_sheet_load_error(self, msg: str):
        self._err(f"Failed to load selected sheet/group:\n{msg}")

    def _on_crop_toggle(self, _state):
        enabled = self.chk_crop.isChecked()
        self.t0_spin.setEnabled(enabled)
        self.t1_spin.setEnabled(enabled)

    def _select_all_channels(self):
        self.list_channels.selectAll()

    def _clear_channel_selection(self):
        self.list_channels.clearSelection()

    def _fill_channel_list(self, names, prefer_index=0):
        self.list_channels.blockSignals(True)
        self.list_channels.clear()
        for nm in names:
            self.list_channels.addItem(QListWidgetItem(nm))
        self.list_channels.blockSignals(False)

        idx = prefer_index if 0 <= prefer_index < len(names) else 0
        if self.list_channels.count() > 0:
            self.list_channels.setCurrentRow(idx)
            self.list_channels.item(idx).setSelected(True)

        self._refresh_enabled_state()

    def _get_selected_names(self):
        items = self.list_channels.selectedItems()
        return [it.text() for it in items] if items else ([self.list_channels.currentItem().text()] if self.list_channels.currentItem() else [])

    def _set_combo_item_enabled(self, combo, index, enabled):
        try:
            item = combo.model().item(index)
            if item is not None:
                item.setEnabled(enabled)
        except Exception:
            pass

    def _is_fourier_mode(self):
        return self.cmb_mode.currentText().startswith("Fourier Transform")

    def _is_velocity_mode(self):
        return "Velocity" in self.cmb_mode.currentText()

    def _is_velocity_time_mode(self):
        return self.cmb_mode.currentText() == "Time Domain (Velocity)"

    def _is_waterfall_mode(self):
        return self.cmb_mode.currentText() == "Waterfall (RPM + Acceleration)"

    def _guess_rpm_channel(self):
        names = [self.list_channels.item(i).text() for i in range(self.list_channels.count())]
        if not names:
            return ""
        keys = ["rpm", "tach", "speed", "rev", "rot"]
        for nm in names:
            lo = nm.lower()
            if any(k in lo for k in keys):
                return nm
        return names[0]

    def _guess_acc_channel(self):
        names = [self.list_channels.item(i).text() for i in range(self.list_channels.count())]
        if not names:
            return ""
        keys = ["acc", "vib", "g", "z", "x", "y"]
        for nm in names:
            lo = nm.lower()
            if any(k in lo for k in keys) and ("rpm" not in lo) and ("tach" not in lo):
                return nm
        return names[min(1, len(names) - 1)]

    def _selected_signal_kind(self):
        return "vel" if self._is_velocity_mode() else "acc"

    def _signal_name(self):
        return "Velocity" if self._selected_signal_kind() == "vel" else "Acceleration"

    def _signal_unit_short(self):
        return VEL_UNIT if self._selected_signal_kind() == "vel" else "g"

    def _amp_conv_factor_label(self):
        unit = self._signal_unit_short()
        sel = self.cmb_amp_conv.currentText()

        if sel == "Peak":
            return 1.0, f"{unit} (peak)"
        if sel == "RMS":
            return 1.0 / np.sqrt(2.0), f"{unit} RMS"
        return 2.0, f"{unit} p-p"

    def _sync_data_source_items(self):
        velocity_mode = self._is_velocity_mode()

        self._set_combo_item_enabled(self.cmb_data_source, 0, True)
        self._set_combo_item_enabled(self.cmb_data_source, 1, bool(_HAVE_SCIPY) and not velocity_mode)

        if velocity_mode:
            self.cmb_data_source.blockSignals(True)
            self.cmb_data_source.setCurrentText("Unfiltered")
            self.cmb_data_source.blockSignals(False)
            self.cmb_data_source.setEnabled(False)
        else:
            self.cmb_data_source.setEnabled(True)
            if (not _HAVE_SCIPY) and self.cmb_data_source.currentText() == "Filtered":
                self.cmb_data_source.setCurrentText("Unfiltered")

    def _on_data_source_change(self, *_):
        self._sync_data_source_items()

        if self.cmb_data_source.currentText() == "Filtered":
            if not _HAVE_SCIPY:
                self._err("Filtered data requires SciPy.\nInstall with: pip install scipy")
                self.cmb_data_source.setCurrentText("Unfiltered")
                return
            if not self.chk_filter.isChecked():
                self.chk_filter.setChecked(True)

        self._sync_filter_enabled_state()
        self._update_fft_info()

    def _on_filter_type_change(self, *_):
        is_band = self.cmb_filter_type.currentText().startswith("Band-")
        self.lbl_cut2.setVisible(is_band)
        self.cut2_spin.setVisible(is_band)
        self._sync_filter_enabled_state()

    def _on_filter_toggle(self, *_):
        self._sync_filter_enabled_state()

    def _sync_filter_enabled_state(self):
        if self._is_velocity_mode():
            if self.chk_filter.isChecked():
                self.chk_filter.blockSignals(True)
                self.chk_filter.setChecked(False)
                self.chk_filter.blockSignals(False)

            self.chk_filter.setEnabled(False)
            self.cmb_filter_type.setEnabled(False)
            self.cmb_filter_order.setEnabled(False)
            self.cut1_spin.setEnabled(False)
            self.cut2_spin.setEnabled(False)
            return

        self.chk_filter.setEnabled(bool(_HAVE_SCIPY))
        enabled = self.chk_filter.isChecked() and _HAVE_SCIPY

        self.cmb_filter_type.setEnabled(enabled)
        self.cmb_filter_order.setEnabled(enabled)
        self.cut1_spin.setEnabled(enabled)
        self.cut2_spin.setEnabled(enabled and self.cmb_filter_type.currentText().startswith("Band-"))

    def _sync_spectrogram_controls(self):
        state = self._plot_states.get(self._active_plot_key) if self._active_plot_key else None
        is_color_plot = bool(state and state.get("kind") in ("stft", "waterfall"))
        is_db = bool(is_color_plot and str(state.get("scale", "")).strip().lower() == "db")
        custom_mode = self.cmb_color_mode.currentText().strip() == "Custom Bands"
        manual = self.chk_manual_clim.isChecked() and not custom_mode

        self.cmb_spec_render.setEnabled(is_color_plot)
        self.cmb_spec_cmap.setEnabled(is_color_plot and not custom_mode)
        self.chk_spec_cmap_reverse.setEnabled(is_color_plot and not custom_mode)
        self.cmb_color_mode.setEnabled(is_color_plot)
        self.btn_edit_bands.setEnabled(is_color_plot and custom_mode)
        self.lbl_dyn_range.setEnabled(is_color_plot and is_db and (not manual) and (not custom_mode))
        self.spec_levels_spin.setEnabled(is_color_plot and is_db and (not manual) and (not custom_mode))
        self.chk_manual_clim.setEnabled(is_color_plot and not custom_mode)
        self.spec_vmin_spin.setEnabled(is_color_plot and manual)
        self.spec_vmax_spin.setEnabled(is_color_plot and manual)
        self.btn_apply_plot_view.setEnabled(state is not None)
        self.btn_export_plot.setEnabled(state is not None)

        axis_enabled = state is not None
        self.edit_xlabel.setEnabled(axis_enabled)
        self.edit_ylabel.setEnabled(axis_enabled)

    def _on_manual_clim_toggle(self, *_):
        self._sync_spectrogram_controls()
        self._on_color_controls_changed()

    def _on_mode_change(self):
        has_fft_settings = self._is_fourier_mode() or self._is_waterfall_mode() or self.cmb_mode.currentText().startswith("Time Domain")

        self.cmb_fourier_method.setEnabled(not self._is_waterfall_mode() and self._is_fourier_mode())
        self.cmb_nfft.setEnabled(has_fft_settings)
        self.ovl_spin.setEnabled(has_fft_settings)
        self.cmb_out.setEnabled(has_fft_settings)
        self.cmb_scale.setEnabled(has_fft_settings)
        self.cmb_amp_conv.setEnabled(has_fft_settings)
        self.lbl_fft_info.setEnabled(has_fft_settings)

        self._sync_data_source_items()
        self._sync_filter_enabled_state()
        self._sync_spectrogram_controls()
        self._update_fft_info()

    def _get_nfft(self):
        nfft = int(self.cmb_nfft.currentText().strip())
        if nfft <= 0:
            raise ValueError("Invalid NFFT.")
        return nfft

    def _update_fft_info(self):
        try:
            fs = float(self.fs_spin.value())
            nfft = self._get_nfft()
            ovl = float(self.ovl_spin.value()) / 100.0
            df = fs / nfft
            win_t = nfft / fs
            hop = max(1, int(round(nfft * (1.0 - ovl))))
            hop_t = hop / fs
            _, unit_lbl = self._amp_conv_factor_label()

            base_info = (
                f"Velocity is auto-generated from {self.cmb_data_source.currentText().lower()} acceleration: HP 10 Hz, order 4, then integrate"
                if self._selected_signal_kind() == "vel"
                else f"Data source: {self.cmb_data_source.currentText()}"
            )

            zoom_note = "   |   Visible time window -> Averaged Spectrum" if self.cmb_mode.currentText().startswith("Time Domain") else ""
            self.lbl_fft_info.setText(
                f"Signal: {self._signal_name()}   |   {base_info}   |   Δf: {df:.6g} Hz   |   Window: {win_t:.6g} s   |   Hop: {hop} samples ({hop_t:.6g} s)   |   Amplitude convention: {unit_lbl}{zoom_note}"
            )
        except Exception:
            self.lbl_fft_info.setText("Frequency resolution: -")

    def _set_active_canvas(self, canvas, plot_key):
        if self._active_canvas is not None and self._active_canvas is not canvas:
            self._active_canvas.set_rect_zoom_active(False)

        self._active_canvas = canvas
        self._active_plot_key = plot_key

        for key, mini in self._mini_widgets.items():
            mini.set_selected(key == plot_key and mini.canvas is canvas)

        if self._active_canvas is not None:
            self._active_canvas.set_rect_zoom_active(self.btn_rect_zoom.isChecked())

        self._sync_spectrogram_controls()
        self._load_axis_editors_from_state(self._plot_states.get(plot_key))

    def _on_plot_page_changed(self, index):
        if index <= 0:
            return
        widget = self.plot_pages.widget(index)
        if hasattr(widget, "canvas") and hasattr(widget, "plot_key"):
            self._set_active_canvas(widget.canvas, widget.plot_key)

    def _on_plot_page_close(self, index):
        if index <= 0:
            return

        widget = self.plot_pages.widget(index)
        if hasattr(widget, "plot_key"):
            key = widget.plot_key
            self.plot_pages.removeTab(index)
            self._detail_widgets.pop(key, None)
            widget.deleteLater()

            if self._active_plot_key == key and self._mini_widgets:
                first_key = next(iter(self._mini_widgets))
                self._set_active_canvas(self._mini_widgets[first_key].canvas, first_key)
            elif not self._detail_widgets and self.plot_pages.currentIndex() != 0:
                self.plot_pages.setCurrentIndex(0)

        self._sync_spectrogram_controls()

    def _reflow_overview_grid(self):
        while self.overview_grid.count():
            item = self.overview_grid.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        for i, key in enumerate(self._overview_order):
            if key not in self._mini_widgets:
                continue
            row = i // 2
            col = i % 2
            self.overview_grid.addWidget(self._mini_widgets[key], row, col)

        self.lbl_overview_empty.setVisible(len(self._mini_widgets) == 0)

    def _close_plot_by_key(self, key):
        mini = self._mini_widgets.pop(key, None)
        if mini is not None:
            mini.setParent(None)
            mini.deleteLater()

        self._plot_states.pop(key, None)
        self._overview_order = [k for k in self._overview_order if k != key]

        detail = self._detail_widgets.pop(key, None)
        if detail is not None:
            idx = self.plot_pages.indexOf(detail)
            if idx >= 0:
                self.plot_pages.removeTab(idx)
            detail.deleteLater()

        self._reflow_overview_grid()

        if self._active_plot_key == key:
            self._active_canvas = None
            self._active_plot_key = None
            if self._mini_widgets:
                first_key = next(iter(self._mini_widgets))
                self._set_active_canvas(self._mini_widgets[first_key].canvas, first_key)
            else:
                self._load_axis_editors_from_state(None)

        self._sync_spectrogram_controls()

    def _draw_state_on_canvas(self, canvas, state: dict, preserve_limits=False):
        canvas.set_state(
            state,
            color_cfg=self._plot_color_cfg(),
            preserve_view=preserve_limits,
            preserve_annotations=True,
        )

    def _open_large_plot(self, key):
        if key not in self._plot_states:
            return

        if key in self._detail_widgets:
            idx = self.plot_pages.indexOf(self._detail_widgets[key])
            if idx >= 0:
                self.plot_pages.setCurrentIndex(idx)
                self._set_active_canvas(self._detail_widgets[key].canvas, key)
            return

        state = self._plot_states[key]
        detail = DetailPlotWidget(state)
        detail.activated.connect(self._set_active_canvas)

        self._connect_canvas_color_signals(detail.canvas)
        self._draw_state_on_canvas(detail.canvas, state, preserve_limits=False)

        self._detail_widgets[key] = detail
        idx = self.plot_pages.addTab(detail, f"Large: {state['name']}")
        self.plot_pages.setCurrentIndex(idx)
        self._set_active_canvas(detail.canvas, key)

    def _on_plot_worker_finished(self, states):
        self._plot_states = {st["key"]: st for st in states}
        self._clear_overview_widgets()
        self._clear_detail_tabs()

        if states and states[0].get("kind") == "waterfall":
            st = states[0]
            detail = WaterfallDetailWidget(st)
            detail.activated.connect(self._set_active_canvas)
            self._connect_canvas_color_signals(detail.canvas)
            self._draw_state_on_canvas(detail.canvas, st, preserve_limits=False)

            self._detail_widgets[st["key"]] = detail
            idx = self.plot_pages.addTab(detail, f"Waterfall: {st['name']}")
            self.plot_pages.setCurrentIndex(idx)
            self.tabs.setCurrentWidget(self.tab_plot)
            self._set_active_canvas(detail.canvas, st["key"])
            self.lbl_stats.setText(st.get("stats_text", ""))
            self.lbl_overview_empty.setVisible(True)

            if hasattr(self, "lbl_wf_status"):
                self.lbl_wf_status.setText(st.get("stats_text", ""))
            if self._pending_waterfall_open_3d:
                self._pending_waterfall_open_3d = False
                detail.canvas.open_3d_window()
            return

        all_stats = []
        for st in states:
            w = PlotMiniWidget(st)
            self._mini_widgets[st["key"]] = w
            self._overview_order.append(st["key"])

            w.request_open.connect(self._open_large_plot)
            w.request_close.connect(self._close_plot_by_key)
            w.activated.connect(self._set_active_canvas)

            self._connect_canvas_color_signals(w.canvas)
            self._draw_state_on_canvas(w.canvas, st, preserve_limits=False)
            if st.get("stats_text"):
                all_stats.append(st["stats_text"])

        self.lbl_stats.setText("\n".join(all_stats[:12]))
        self._reflow_overview_grid()
        self.tabs.setCurrentWidget(self.tab_plot)

        if states:
            first = states[0]
            self._set_active_canvas(self._mini_widgets[first["key"]].canvas, first["key"])

    def _on_plot_worker_error(self, msg: str):
        self.lbl_stats.setText("")
        if hasattr(self, "lbl_wf_status"):
            self.lbl_wf_status.setText("")
        self._err(f"Plot error:\n{msg}")

    def _start_plot_worker(self, cfg: dict):
        if self._thread_running(self._file_thread) or self._thread_running(self._sheet_thread) or self._thread_running(self._plot_thread):
            self._err("Another operation is already running.")
            return

        self._set_busy(True, "Preparing plot(s) in background...")
        self.lbl_stats.setText("Processing in background...")
        self._refresh_enabled_state()

        self._plot_thread = QThread(self)
        self._plot_worker = PlotWorker(cfg)
        self._plot_worker.moveToThread(self._plot_thread)

        self._plot_thread.started.connect(self._plot_worker.run)
        self._plot_worker.finished.connect(self._on_plot_worker_finished)
        self._plot_worker.error.connect(self._on_plot_worker_error)

        self._plot_worker.finished.connect(self._plot_thread.quit)
        self._plot_worker.error.connect(self._plot_thread.quit)
        self._plot_thread.finished.connect(self._cleanup_plot_thread)

        self._plot_thread.start()

    def _zoom_active(self, factor):
        if self._active_canvas is not None:
            self._active_canvas.zoom_axes(factor)

    def _reset_active_view(self):
        if self._active_canvas is not None:
            self._active_canvas.reset_view()

    def _on_rect_zoom_toggled(self, checked):
        if self._active_canvas is not None:
            self._active_canvas.set_rect_zoom_active(checked)

    def _apply_plot_view(self):
        if self._active_canvas is None or self._active_plot_key not in self._plot_states:
            self._err("There is no active plot to update yet.")
            return

        state = self._plot_states[self._active_plot_key]
        state["xlabel"] = self.edit_xlabel.text().strip()
        state["ylabel"] = self.edit_ylabel.text().strip()

        self._refresh_plot_views_for_key(
            self._active_plot_key,
            preserve_active_limits=True,
            preserve_active_annotations=True,
        )

    def _export_current_plot(self):
        if self._active_plot_key not in self._plot_states:
            self._err("There is no active plot to export.")
            return

        state = self._plot_states[self._active_plot_key]
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export current plot",
            f"{state['name']}.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if not path:
            return

        annotations = []
        if self._active_canvas is not None:
            try:
                annotations = self._active_canvas.get_export_annotations()
            except Exception:
                annotations = []

        lower_annotations = []
        if self._active_canvas is not None:
            try:
                if hasattr(self._active_canvas, "get_export_lower_annotations"):
                    lower_annotations = self._active_canvas.get_export_lower_annotations()
            except Exception:
                lower_annotations = []

        xlabel = state.get("xlabel", "")
        ylabel = state.get("ylabel", "")

        if self._active_plot_key == state["key"]:
            xlabel = self.edit_xlabel.text().strip()
            ylabel = self.edit_ylabel.text().strip()

        def _mpl_offset_points(i: int):
            offsets = [
                (8, 8),
                (8, -18),
                (20, 16),
                (-20, 10),
                (22, -22),
                (-22, -18),
                (32, 8),
                (-32, 8),
            ]
            return offsets[i % len(offsets)]

        try:
            if state["kind"] == "waterfall":
                fig = Figure(figsize=(12, 9), dpi=150, constrained_layout=True)
                gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.6])
                ax0 = fig.add_subplot(gs[0, 0])
                ax1 = fig.add_subplot(gs[1, 0])

                freqs = np.asarray(state["freqs"])
                rpms = np.asarray(state["rpms"])
                S = np.asarray(state["S"])
                color_cfg = self._plot_color_cfg()

                style = resolve_mpl_image_style(S, state.get("scale", "Linear"), color_cfg)

                if freqs.size > 1:
                    fx0 = float(freqs[0])
                    fx1 = float(freqs[-1])
                elif freqs.size == 1:
                    fx0 = float(freqs[0] - 0.5)
                    fx1 = float(freqs[0] + 0.5)
                else:
                    fx0, fx1 = 0.0, 1.0

                if rpms.size > 1:
                    ry0 = float(rpms[0])
                    ry1 = float(rpms[-1])
                elif rpms.size == 1:
                    ry0 = float(rpms[0] - 0.5)
                    ry1 = float(rpms[0] + 0.5)
                else:
                    ry0, ry1 = 0.0, 1.0

                if style["mode"] == "bands":
                    mesh = ax0.imshow(
                        S,
                        aspect="auto",
                        origin="lower",
                        extent=[fx0, fx1, ry0, ry1],
                        interpolation="bilinear",
                        cmap=style["cmap"],
                        norm=style["norm"],
                    )
                else:
                    mesh = ax0.imshow(
                        S,
                        aspect="auto",
                        origin="lower",
                        extent=[fx0, fx1, ry0, ry1],
                        interpolation="bilinear",
                        cmap=style["cmap"],
                        vmin=style["vmin"],
                        vmax=style["vmax"],
                    )
                cbar = fig.colorbar(mesh, ax=ax0)
                if style["mode"] == "bands":
                    cbar.set_ticks(style["boundaries"])
                cbar.set_label(state.get("colorbar_label", ""))
                ax0.set_xlabel(xlabel)
                ax0.set_ylabel(ylabel)
                ax0.set_title(state["title"])

                # ── Capture current zoom viewport from PyQtGraph ──
                if self._active_canvas is not None and hasattr(self._active_canvas, "upper_item"):
                    try:
                        vb = self._active_canvas.upper_item.getViewBox()
                        xr, yr = vb.viewRange()
                        _xr0, _xr1 = float(xr[0]), float(xr[1])
                        _yr0, _yr1 = float(yr[0]), float(yr[1])
                        if np.isfinite([_xr0, _xr1, _yr0, _yr1]).all() and _xr1 > _xr0 and _yr1 > _yr0:
                            ax0.set_xlim(_xr0, _xr1)
                            ax0.set_ylim(_yr0, _yr1)
                    except Exception:
                        pass

                lower_label = state.get("colorbar_label", "Amplitude")
                ax1.set_xlabel("RPM")
                ax1.set_ylabel(lower_label)
                ax1.grid(True)

                if hasattr(self._active_canvas, "export_order_cursor_specs") and hasattr(self._active_canvas, "_extract_order_trace"):
                    for spec in self._active_canvas.export_order_cursor_specs():
                        order_val = float(spec["order"])
                        color = np.array(spec["color"], dtype=np.float64) / 255.0
                        fx = (order_val * rpms) / 60.0
                        m = np.isfinite(fx) & (fx >= freqs[0]) & (fx <= freqs[-1])
                        ax0.plot(fx[m], rpms[m], color=color, linewidth=1.5)

                        tr = self._active_canvas._extract_order_trace(order_val)
                        mm = np.isfinite(tr) & np.isfinite(rpms)
                        ax1.plot(rpms[mm], tr[mm], color=color, linewidth=1.6, label=f"Order {order_val:.2f}x")

                    ax1.legend(loc="best")

                # ── Capture lower plot zoom viewport ──
                if self._active_canvas is not None and hasattr(self._active_canvas, "lower_item"):
                    try:
                        _lvb = self._active_canvas.lower_item.getViewBox()
                        _lxr, _lyr = _lvb.viewRange()
                        _lx0, _lx1 = float(_lxr[0]), float(_lxr[1])
                        _ly0, _ly1 = float(_lyr[0]), float(_lyr[1])
                        if np.isfinite([_lx0, _lx1, _ly0, _ly1]).all() and _lx1 > _lx0 and _ly1 > _ly0:
                            ax1.set_xlim(_lx0, _lx1)
                            ax1.set_ylim(_ly0, _ly1)
                    except Exception:
                        pass

                for i, ann in enumerate(annotations):
                    try:
                        if "x" in ann and "y" in ann:
                            dx, dy = _mpl_offset_points(i)
                            ax0.plot(
                                ann["x"], ann["y"],
                                linestyle="None",
                                marker="o",
                                markersize=5,
                                markerfacecolor="white",
                                markeredgecolor="red",
                                markeredgewidth=0.8,
                                zorder=6,
                            )
                            ax0.annotate(
                                ann.get("text", ""),
                                xy=(ann["x"], ann["y"]),
                                xytext=(dx, dy),
                                textcoords="offset points",
                                fontsize=8,
                                bbox=dict(
                                    facecolor="white",
                                    edgecolor="#666",
                                    alpha=0.9,
                                    boxstyle="round,pad=0.2"
                                ),
                                zorder=7,
                            )
                    except Exception:
                        pass

                for i, ann in enumerate(lower_annotations):
                    try:
                        if "x" in ann and "y" in ann:
                            dx, dy = _mpl_offset_points(i)
                            color = np.array(ann.get("color", (200, 40, 40)), dtype=np.float64) / 255.0
                            ax1.plot(
                                ann["x"], ann["y"],
                                linestyle="None",
                                marker="o",
                                markersize=5,
                                markerfacecolor="white",
                                markeredgecolor=color,
                                markeredgewidth=1.0,
                                zorder=6,
                            )
                            ax1.annotate(
                                ann.get("text", ""),
                                xy=(ann["x"], ann["y"]),
                                xytext=(dx, dy),
                                textcoords="offset points",
                                fontsize=8,
                                bbox=dict(
                                    facecolor="white",
                                    edgecolor="#666",
                                    alpha=0.9,
                                    boxstyle="round,pad=0.2"
                                ),
                                zorder=7,
                            )
                    except Exception:
                        pass

                fig.savefig(path, dpi=150)
                QMessageBox.information(self, "Export", f"Plot exported successfully:\n{path}")
                return

            zoom_fft_data = None
            if state["kind"] == "time" and self._active_canvas is not None and hasattr(self._active_canvas, "get_zoom_fft_export_data"):
                try:
                    zoom_fft_data = self._active_canvas.get_zoom_fft_export_data()
                except Exception:
                    zoom_fft_data = None

            if state["kind"] == "time" and zoom_fft_data:
                fig = Figure(figsize=(12, 9), dpi=150, constrained_layout=True)
                gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.7])
                ax = fig.add_subplot(gs[0, 0])
                ax_fft = fig.add_subplot(gs[1, 0])

                for series in state["series"]:
                    ax.plot(np.asarray(series["x"]), np.asarray(series["y"]), linewidth=0.9, label=series["label"])
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.grid(True)
                ax.margins(x=0)

                # ── Capture time domain upper plot zoom viewport ──
                if self._active_canvas is not None and hasattr(self._active_canvas, "upper"):
                    try:
                        _tvr = self._active_canvas.upper.get_view_range()
                        if _tvr is not None:
                            (_tx0, _tx1), (_ty0, _ty1) = _tvr
                            if np.isfinite([_tx0, _tx1, _ty0, _ty1]).all() and _tx1 > _tx0 and _ty1 > _ty0:
                                ax.set_xlim(_tx0, _tx1)
                                ax.set_ylim(_ty0, _ty1)
                    except Exception:
                        pass

                if zoom_fft_data.get("window") is not None:
                    x0, x1 = zoom_fft_data["window"]
                    ax.axvspan(x0, x1, alpha=0.12)

                fft_x = np.asarray(zoom_fft_data.get("freqs", []), dtype=np.float64)
                fft_y = np.asarray(zoom_fft_data.get("spec", []), dtype=np.float64)
                ax_fft.plot(fft_x, fft_y, linewidth=1.0)
                ax_fft.set_xlabel(zoom_fft_data.get("xlabel", "Frequency (Hz)"))
                ax_fft.set_ylabel(zoom_fft_data.get("ylabel", "Amplitude"))
                ax_fft.set_title(zoom_fft_data.get("title", "Visible time window FFT"))
                ax_fft.grid(True)
                ax_fft.margins(x=0)

                for i, ann in enumerate(lower_annotations):
                    try:
                        if "x" in ann and "y" in ann:
                            dx, dy = _mpl_offset_points(i)
                            ax_fft.plot(
                                ann["x"], ann["y"],
                                linestyle="None",
                                marker="o",
                                markersize=5,
                                markerfacecolor="white",
                                markeredgecolor="red",
                                markeredgewidth=0.8,
                                zorder=6,
                            )
                            ax_fft.annotate(
                                ann.get("text", ""),
                                xy=(ann["x"], ann["y"]),
                                xytext=(dx, dy),
                                textcoords="offset points",
                                fontsize=8,
                                bbox=dict(facecolor="white", edgecolor="#666", alpha=0.9, boxstyle="round,pad=0.2"),
                                zorder=7,
                            )
                    except Exception:
                        pass
            else:
                fig = Figure(figsize=(12, 6), dpi=150, constrained_layout=True)
                ax = fig.add_subplot(111)
                ax_fft = None

                if state["kind"] in ("time", "avg"):
                    for series in state["series"]:
                        ax.plot(np.asarray(series["x"]), np.asarray(series["y"]), linewidth=0.9, label=series["label"])
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
                    ax.grid(True)
                    ax.margins(x=0)
                else:
                    times = np.asarray(state["times"])
                    freqs = np.asarray(state["freqs"])
                    S = np.asarray(state["S"])
                    color_cfg = self._plot_color_cfg()

                    style = resolve_mpl_image_style(S, state.get("scale", "Linear"), color_cfg)

                    if style["mode"] == "bands":
                        mesh = ax.pcolormesh(times, freqs, S, shading="auto", cmap=style["cmap"], norm=style["norm"])
                    else:
                        mesh = ax.pcolormesh(
                            times, freqs, S, shading="auto",
                            cmap=style["cmap"], vmin=style["vmin"], vmax=style["vmax"]
                        )
                    cbar = fig.colorbar(mesh, ax=ax)
                    if style["mode"] == "bands":
                        cbar.set_ticks(style["boundaries"])
                    cbar.set_label(state.get("colorbar_label", ""))
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
                    ax.grid(False)

            for i, ann in enumerate(annotations):
                try:
                    if "x" in ann and "y" in ann:
                        dx, dy = _mpl_offset_points(i)
                        ax.plot(
                            ann["x"], ann["y"],
                            linestyle="None",
                            marker="o",
                            markersize=5,
                            markerfacecolor="white",
                            markeredgecolor="red",
                            markeredgewidth=0.8,
                            zorder=6,
                        )
                        ax.annotate(
                            ann.get("text", ""),
                            xy=(ann["x"], ann["y"]),
                            xytext=(dx, dy),
                            textcoords="offset points",
                            fontsize=8,
                            bbox=dict(facecolor="white", edgecolor="#666", alpha=0.9, boxstyle="round,pad=0.2"),
                            zorder=7,
                        )
                except Exception:
                    pass

            ax.set_title(state["title"])

            # ── Capture current zoom viewport from PyQtGraph ──
            if self._active_canvas is not None:
                _export_vr = None
                try:
                    if hasattr(self._active_canvas, "get_view_range"):
                        _export_vr = self._active_canvas.get_view_range()
                    elif hasattr(self._active_canvas, "upper") and hasattr(self._active_canvas.upper, "get_view_range"):
                        _export_vr = self._active_canvas.upper.get_view_range()
                except Exception:
                    pass

                if _export_vr is not None:
                    try:
                        (_evx0, _evx1), (_evy0, _evy1) = _export_vr
                        if np.isfinite([_evx0, _evx1, _evy0, _evy1]).all() and _evx1 > _evx0 and _evy1 > _evy0:
                            ax.set_xlim(_evx0, _evx1)
                            ax.set_ylim(_evy0, _evy1)
                    except Exception:
                        pass

            fig.savefig(path, dpi=150)
        except Exception as e:
            self._err(f"Export failed:\n{e}")
            return

        QMessageBox.information(self, "Export", f"Plot exported successfully:\n{path}")

    def _build_waterfall_tab_ui(self):
        self.btn_wf_acc_open = QPushButton("Open Acc file (.xlsx / .tdms / .csv)")
        self.lbl_wf_acc_file = QLabel("Acceleration source: -")
        self.lbl_wf_acc_file.setWordWrap(True)
        self.cmb_wf_acc_group = QComboBox()
        self.cmb_wf_acc_channel = QComboBox()

        self.btn_wf_rpm_open = QPushButton("Open RPM file (.xlsx / .tdms / .csv)")
        self.lbl_wf_rpm_file = QLabel("RPM source: -")
        self.lbl_wf_rpm_file.setWordWrap(True)
        self.cmb_wf_rpm_group = QComboBox()
        self.cmb_wf_rpm_channel = QComboBox()

        self.wf_fs_spin = QDoubleSpinBox()
        self.wf_fs_spin.setRange(0.1, 1_000_000.0)
        self.wf_fs_spin.setDecimals(2)
        self.wf_fs_spin.setValue(10000.0)

        self.wf_rpm_fs_spin = QDoubleSpinBox()
        self.wf_rpm_fs_spin.setRange(0.1, 1_000_000.0)
        self.wf_rpm_fs_spin.setDecimals(2)
        self.wf_rpm_fs_spin.setValue(10000.0)

        self.cmb_wf_resample_mode = QComboBox()
        self.cmb_wf_resample_mode.addItems([
            "Resample RPM -> Acc timebase",
            "Resample Acc -> RPM timebase",
            "Trim only (same fs)",
        ])
        self.cmb_wf_resample_mode.setCurrentText("Resample RPM -> Acc timebase")

        self.cmb_wf_interp = QComboBox()
        self.cmb_wf_interp.addItems(["Linear", "Nearest"])
        self.cmb_wf_interp.setCurrentText("Linear")

        self.cmb_wf_rpm_label_method = QComboBox()
        self.cmb_wf_rpm_label_method.addItems([
            "Median in window",
            "Center sample",
            "Window-weighted (Hanning)",
        ])
        self.cmb_wf_rpm_label_method.setCurrentText("Window-weighted (Hanning)")
        self.cmb_wf_rpm_label_method.setToolTip(
            "How a single RPM is assigned to each FFT frame.\n"
            "Window-weighted (Hanning) uses the same Hann window as the FFT and is usually the best choice for ramp/run-up data."
        )

        self.chk_wf_skip_first = QCheckBox("Skip first sample (y[1:])")
        self.chk_wf_crop = QCheckBox("Enable time window (crop)")
        self.wf_t0_spin = QDoubleSpinBox()
        self.wf_t0_spin.setRange(0.0, 1_000_000_000.0)
        self.wf_t0_spin.setDecimals(3)
        self.wf_t0_spin.setValue(10.0)
        self.wf_t0_spin.setEnabled(False)
        self.wf_t1_spin = QDoubleSpinBox()
        self.wf_t1_spin.setRange(0.0, 1_000_000_000.0)
        self.wf_t1_spin.setDecimals(3)
        self.wf_t1_spin.setValue(30.0)
        self.wf_t1_spin.setEnabled(False)

        self.wf_nfft = QComboBox()
        self.wf_nfft.addItems([str(x) for x in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]])
        self.wf_nfft.setCurrentText("8192")

        self.wf_ovl_spin = QDoubleSpinBox()
        self.wf_ovl_spin.setRange(0.0, 95.0)
        self.wf_ovl_spin.setDecimals(1)
        self.wf_ovl_spin.setSingleStep(5.0)
        self.wf_ovl_spin.setValue(67.0)

        self.wf_num_spectra_spin = QSpinBox()
        self.wf_num_spectra_spin.setRange(10, 5000)
        self.wf_num_spectra_spin.setValue(1000)
        self.wf_num_spectra_spin.setToolTip(
            "Number of Spectra: waterfall RPM eksenindeki minimum hedef satır sayısı. "
            "Kod artık available FFT frame sayısının altına yıkıcı biçimde düşmez."
        )

        self.wf_num_buffers_spin = QSpinBox()
        self.wf_num_buffers_spin.setRange(1, 500)
        self.wf_num_buffers_spin.setValue(1)
        self.wf_num_buffers_spin.setToolTip(
            "Number of Buffers: her RPM dilimine karşılık kaç raw FFT frame "
            "ortalaması alınsın (arttıkça smooth, azalır RPM çözünürlüğü)"
        )

        self.wf_out = QComboBox()
        self.wf_out.addItems(["Amplitude", "Power"])
        self.wf_scale = QComboBox()
        self.wf_scale.addItems(["Linear", "dB"])
        self.wf_amp_conv = QComboBox()
        self.wf_amp_conv.addItems(["Peak", "RMS", "Peak-to-Peak"])
        self.wf_amp_conv.setCurrentText("RMS")

        self.chk_wf_filter = QCheckBox("Enable Waterfall Butterworth filter (zero-phase)")
        self.chk_wf_filter.setChecked(False)
        if not _HAVE_SCIPY:
            self.chk_wf_filter.setEnabled(False)
            self.chk_wf_filter.setToolTip("SciPy not found. Install: pip install scipy")

        self.cmb_wf_filter_type = QComboBox()
        self.cmb_wf_filter_type.addItems(["Low-pass", "High-pass", "Band-pass", "Band-reject"])
        self.cmb_wf_filter_type.setEnabled(False)

        self.cmb_wf_filter_order = QComboBox()
        self.cmb_wf_filter_order.addItems([str(i) for i in range(1, 9)])
        self.cmb_wf_filter_order.setCurrentText("4")
        self.cmb_wf_filter_order.setEnabled(False)

        self.wf_cut1_spin = QDoubleSpinBox()
        self.wf_cut1_spin.setRange(0.0, 1_000_000.0)
        self.wf_cut1_spin.setDecimals(3)
        self.wf_cut1_spin.setValue(100.0)
        self.wf_cut1_spin.setEnabled(False)
        self.lbl_wf_cut2 = QLabel("Cutoff 2 (Hz):")
        self.wf_cut2_spin = QDoubleSpinBox()
        self.wf_cut2_spin.setRange(0.0, 1_000_000.0)
        self.wf_cut2_spin.setDecimals(3)
        self.wf_cut2_spin.setValue(300.0)
        self.wf_cut2_spin.setEnabled(False)
        self.lbl_wf_cut2.setVisible(False)
        self.wf_cut2_spin.setVisible(False)

        self.cmb_wf_filter_target = QComboBox()
        self.cmb_wf_filter_target.addItems(["None", "Acceleration", "RPM", "Both"])
        self.chk_wf_open_3d = QCheckBox("Open 3D waterfall in separate window after plotting")
        self.chk_wf_open_3d.setChecked(False)
        self.lbl_wf_info = QLabel(
    "Use this tab when acceleration and RPM come from the same or different TDMS/Excel/CSV files. "
    "Recommended default: Resample RPM -> Acc timebase. RPM is usually slow-varying, so Linear interpolation is generally sufficient. "
    "For run-up/coast-down data, prefer RPM label = Window-weighted (Hanning) so each FFT frame gets an RPM consistent with the same Hann window used in the FFT. "
    "FFT frequency axis follows the selected target timebase.")

        self.lbl_wf_info.setStyleSheet("color: #666;")
        self.lbl_wf_info.setWordWrap(True)
        self.lbl_wf_status = QLabel("")
        self.lbl_wf_status.setStyleSheet("color: #444;")
        self.lbl_wf_status.setWordWrap(True)
        self.btn_wf_plot = QPushButton("Plot Waterfall")
        self.btn_wf_plot.setEnabled(False)

        lay = QVBoxLayout(self.tab_waterfall)
        lay.addWidget(self.lbl_wf_info)
        lay.addSpacing(6)

        lay.addWidget(self.btn_wf_acc_open)
        lay.addWidget(self.lbl_wf_acc_file)
        row_acc = QHBoxLayout()
        row_acc.addWidget(QLabel("Acc sheet/group:"))
        row_acc.addWidget(self.cmb_wf_acc_group, 1)
        row_acc.addSpacing(10)
        row_acc.addWidget(QLabel("Acc channel:"))
        row_acc.addWidget(self.cmb_wf_acc_channel, 1)
        lay.addLayout(row_acc)

        lay.addSpacing(8)
        lay.addWidget(self.btn_wf_rpm_open)
        lay.addWidget(self.lbl_wf_rpm_file)
        row_rpm = QHBoxLayout()
        row_rpm.addWidget(QLabel("RPM sheet/group:"))
        row_rpm.addWidget(self.cmb_wf_rpm_group, 1)
        row_rpm.addSpacing(10)
        row_rpm.addWidget(QLabel("RPM channel:"))
        row_rpm.addWidget(self.cmb_wf_rpm_channel, 1)
        lay.addLayout(row_rpm)

        lay.addSpacing(10)
        row_wf_1 = QHBoxLayout()
        row_wf_1.addWidget(QLabel("Acc fs (Hz):"))
        row_wf_1.addWidget(self.wf_fs_spin)
        row_wf_1.addSpacing(14)
        row_wf_1.addWidget(QLabel("RPM fs (Hz):"))
        row_wf_1.addWidget(self.wf_rpm_fs_spin)
        row_wf_1.addSpacing(14)
        row_wf_1.addWidget(self.chk_wf_skip_first)
        row_wf_1.addStretch(1)
        lay.addLayout(row_wf_1)

        row_wf_2 = QHBoxLayout()
        row_wf_2.addWidget(self.chk_wf_crop)
        row_wf_2.addWidget(QLabel("Start (s):"))
        row_wf_2.addWidget(self.wf_t0_spin)
        row_wf_2.addWidget(QLabel("End (s):"))
        row_wf_2.addWidget(self.wf_t1_spin)
        row_wf_2.addStretch(1)
        lay.addLayout(row_wf_2)

        row_wf_align = QHBoxLayout()
        row_wf_align.addWidget(QLabel("Resample mode:"))
        row_wf_align.addWidget(self.cmb_wf_resample_mode)
        row_wf_align.addSpacing(12)
        row_wf_align.addWidget(QLabel("Interpolation:"))
        row_wf_align.addWidget(self.cmb_wf_interp)
        row_wf_align.addSpacing(12)
        row_wf_align.addWidget(QLabel("RPM label:"))
        row_wf_align.addWidget(self.cmb_wf_rpm_label_method)
        row_wf_align.addStretch(1)
        lay.addLayout(row_wf_align)

        row_wf_3 = QHBoxLayout()
        row_wf_3.addWidget(QLabel("Buffer Size (NFFT):"))
        row_wf_3.addWidget(self.wf_nfft)
        row_wf_3.addWidget(QLabel("Overlap (%):"))
        row_wf_3.addWidget(self.wf_ovl_spin)
        row_wf_3.addWidget(QLabel("Output:"))
        row_wf_3.addWidget(self.wf_out)
        row_wf_3.addWidget(QLabel("Scale:"))
        row_wf_3.addWidget(self.wf_scale)
        row_wf_3.addWidget(QLabel("Amplitude convention:"))
        row_wf_3.addWidget(self.wf_amp_conv)
        row_wf_3.addWidget(QLabel("Num Spectra:"))
        row_wf_3.addWidget(self.wf_num_spectra_spin)
        row_wf_3.addWidget(QLabel("Num Buffers:"))
        row_wf_3.addWidget(self.wf_num_buffers_spin)
        row_wf_3.addStretch(1)
        lay.addLayout(row_wf_3)

        row_wf_4 = QHBoxLayout()
        row_wf_4.addWidget(self.chk_wf_filter)
        row_wf_4.addSpacing(12)
        row_wf_4.addWidget(QLabel("Type:"))
        row_wf_4.addWidget(self.cmb_wf_filter_type)
        row_wf_4.addSpacing(8)
        row_wf_4.addWidget(QLabel("Order:"))
        row_wf_4.addWidget(self.cmb_wf_filter_order)
        row_wf_4.addSpacing(8)
        row_wf_4.addWidget(QLabel("Cutoff 1 (Hz):"))
        row_wf_4.addWidget(self.wf_cut1_spin)
        row_wf_4.addWidget(self.lbl_wf_cut2)
        row_wf_4.addWidget(self.wf_cut2_spin)
        row_wf_4.addStretch(1)
        lay.addLayout(row_wf_4)

        row_wf_5 = QHBoxLayout()
        row_wf_5.addWidget(QLabel("Apply filter to:"))
        row_wf_5.addWidget(self.cmb_wf_filter_target)
        row_wf_5.addSpacing(12)
        row_wf_5.addWidget(self.chk_wf_open_3d)
        row_wf_5.addStretch(1)
        row_wf_5.addWidget(self.btn_wf_plot)
        lay.addLayout(row_wf_5)

        lay.addWidget(self.lbl_wf_status)
        lay.addStretch(1)

        self.btn_wf_acc_open.clicked.connect(lambda: self._open_waterfall_source("acc"))
        self.btn_wf_rpm_open.clicked.connect(lambda: self._open_waterfall_source("rpm"))
        self.cmb_wf_acc_group.currentIndexChanged.connect(lambda *_: self._reload_waterfall_channels("acc"))
        self.cmb_wf_rpm_group.currentIndexChanged.connect(lambda *_: self._reload_waterfall_channels("rpm"))
        self.chk_wf_crop.stateChanged.connect(self._on_wf_crop_toggle)
        self.chk_wf_filter.stateChanged.connect(self._sync_wf_filter_enabled_state)
        self.cmb_wf_filter_type.currentIndexChanged.connect(self._on_wf_filter_type_change)
        self.btn_wf_plot.clicked.connect(self.plot_waterfall)

    def _on_wf_crop_toggle(self, _state):
        enabled = self.chk_wf_crop.isChecked()
        self.wf_t0_spin.setEnabled(enabled)
        self.wf_t1_spin.setEnabled(enabled)

    def _on_wf_filter_type_change(self, *_):
        is_band = self.cmb_wf_filter_type.currentText().startswith("Band-")
        self.lbl_wf_cut2.setVisible(is_band)
        self.wf_cut2_spin.setVisible(is_band)
        self._sync_wf_filter_enabled_state()

    def _sync_wf_filter_enabled_state(self, *_):
        enabled = self.chk_wf_filter.isChecked() and _HAVE_SCIPY
        self.cmb_wf_filter_type.setEnabled(enabled)
        self.cmb_wf_filter_order.setEnabled(enabled)
        self.wf_cut1_spin.setEnabled(enabled)
        self.wf_cut2_spin.setEnabled(enabled and self.cmb_wf_filter_type.currentText().startswith("Band-"))

    def _open_waterfall_source(self, which: str):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open waterfall source file",
            "",
            "Data Files (*.xlsx *.xls *.tdms *.csv);;Excel Files (*.xlsx *.xls);;TDMS Files (*.tdms);;CSV Files (*.csv)"
        )
        if not path:
            return

        try:
            file_type = detect_file_type(path)
            containers = read_source_container_names(path, file_type)
        except Exception as e:
            self._err(f"Failed to open waterfall source:\n{e}")
            return

        if which == "acc":
            self.wf_acc_path = path
            self.wf_acc_type = file_type
            self.wf_acc_containers = containers
            self.lbl_wf_acc_file.setText(f"Acceleration source: {path}")
            combo = self.cmb_wf_acc_group
        else:
            self.wf_rpm_path = path
            self.wf_rpm_type = file_type
            self.wf_rpm_containers = containers
            self.lbl_wf_rpm_file.setText(f"RPM source: {path}")
            combo = self.cmb_wf_rpm_group

        combo.blockSignals(True)
        combo.clear()
        combo.addItems(containers)
        combo.setCurrentIndex(0)
        combo.blockSignals(False)
        self._reload_waterfall_channels(which)
        self._refresh_enabled_state()

    def _reload_waterfall_channels(self, which: str):
        if which == "acc":
            path = self.wf_acc_path
            file_type = self.wf_acc_type
            container = self.cmb_wf_acc_group.currentText().strip()
            combo = self.cmb_wf_acc_channel
            guess_fn = self._guess_acc_from_names
        else:
            path = self.wf_rpm_path
            file_type = self.wf_rpm_type
            container = self.cmb_wf_rpm_group.currentText().strip()
            combo = self.cmb_wf_rpm_channel
            guess_fn = self._guess_rpm_from_names

        combo.blockSignals(True)
        combo.clear()
        combo.blockSignals(False)

        if not path or not file_type or not container:
            self._refresh_enabled_state()
            return

        try:
            names = read_source_signal_names(path, file_type, container)
        except Exception as e:
            self._err(f"Failed to load waterfall channels:\n{e}")
            self._refresh_enabled_state()
            return

        combo.blockSignals(True)
        combo.clear()
        combo.addItems(names)
        guess = guess_fn(names)
        if guess in names:
            combo.setCurrentText(guess)
        combo.blockSignals(False)
        self._refresh_enabled_state()

    def _guess_rpm_from_names(self, names):
        keys = ["rpm", "tach", "speed", "rev", "rot"]
        for nm in names:
            lo = str(nm).lower()
            if any(k in lo for k in keys):
                return nm
        return names[0] if names else ""

    def _guess_acc_from_names(self, names):
        keys = ["acc", "vib", "g", "z", "x", "y"]
        for nm in names:
            lo = str(nm).lower()
            if any(k in lo for k in keys) and ("rpm" not in lo) and ("tach" not in lo):
                return nm
        return names[min(1, len(names) - 1)] if names else ""

    def plot_waterfall(self):
        if not self.wf_acc_path or not self.wf_rpm_path:
            self._err("Please select both acceleration and RPM source files in the Waterfall tab.")
            return

        acc_group = self.cmb_wf_acc_group.currentText().strip()
        rpm_group = self.cmb_wf_rpm_group.currentText().strip()
        acc_name = self.cmb_wf_acc_channel.currentText().strip()
        rpm_name = self.cmb_wf_rpm_channel.currentText().strip()

        if not acc_group or not rpm_group or not acc_name or not rpm_name:
            self._err("Please select both source containers and channels for the waterfall plot.")
            return
        if acc_name == rpm_name and self.wf_acc_path == self.wf_rpm_path and acc_group == rpm_group:
            self._err("RPM and acceleration channels must be different when they come from the same source container.")
            return

        fs_acc = float(self.wf_fs_spin.value())
        fs_rpm = float(self.wf_rpm_fs_spin.value())
        crop = self.chk_wf_crop.isChecked()
        t0 = float(self.wf_t0_spin.value())
        t1 = float(self.wf_t1_spin.value())
        if crop and t1 <= t0:
            self._err("End time must be greater than start time.")
            return

        self._pending_waterfall_open_3d = self.chk_wf_open_3d.isChecked()
        self.lbl_wf_status.setText("Preparing waterfall in background...")

        cfg = {
            "file_path": self.wf_acc_path,
            "file_type": self.wf_acc_type,
            "group_or_sheet": acc_group,
            "names": [acc_name],
            "fs": fs_acc,
            "waterfall_acc_fs": fs_acc,
            "waterfall_rpm_fs": fs_rpm,
            "waterfall_resample_mode": self.cmb_wf_resample_mode.currentText(),
            "waterfall_resample_interp": self.cmb_wf_interp.currentText(),
            "waterfall_rpm_label_method": self.cmb_wf_rpm_label_method.currentText(),
            "crop": crop,
            "t0": t0,
            "t1": t1,
            "skip_first": self.chk_wf_skip_first.isChecked(),
            "mode": "Waterfall (RPM + Acceleration)",
            "signal_kind": "acc",
            "signal_name": "Acceleration",
            "use_filtered": False,
            "filter_enabled": self.chk_wf_filter.isChecked(),
            "filter_type": self.cmb_wf_filter_type.currentText(),
            "filter_order": int(self.cmb_wf_filter_order.currentText()),
            "cut1": float(self.wf_cut1_spin.value()),
            "cut2": float(self.wf_cut2_spin.value()),
            "nfft": int(self.wf_nfft.currentText()),
            "overlap_pct": float(self.wf_ovl_spin.value()),
            "out_type": self.wf_out.currentText(),
            "scale": self.wf_scale.currentText(),
            "method": "Averaged Spectrum",
            "amp_conv": self.wf_amp_conv.currentText(),
            "waterfall_filter_target": self.cmb_wf_filter_target.currentText(),
            "waterfall_num_spectra": int(self.wf_num_spectra_spin.value()),
            "waterfall_num_buffers": int(self.wf_num_buffers_spin.value()),
            "waterfall_acc_name": acc_name,
            "waterfall_rpm_name": rpm_name,
            "waterfall_acc_source": {
                "file_path": self.wf_acc_path,
                "file_type": self.wf_acc_type,
                "group_or_sheet": acc_group,
                "name": acc_name,
            },
            "waterfall_rpm_source": {
                "file_path": self.wf_rpm_path,
                "file_type": self.wf_rpm_type,
                "group_or_sheet": rpm_group,
                "name": rpm_name,
            },
        }

        self._start_plot_worker(cfg)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open data file",
            "",
            "Data Files (*.xlsx *.xls *.tdms *.csv);;Excel Files (*.xlsx *.xls);;TDMS Files (*.tdms);;CSV Files (*.csv)"
        )
        if not path:
            return

        self.file_path = path
        self.lbl_file.setText(f"Selected file: {path}")
        self._reset_loaded_state(keep_path=True)
        self._start_file_open_worker(path)

    def load_sheet_or_group(self):
        if not self.file_path or not self.file_type:
            return

        name = self.cmb_sheet.currentText()
        if not name:
            return

        self._start_sheet_load_worker(name)

    def plot(self):
        fs = float(self.fs_spin.value())
        crop = self.chk_crop.isChecked()
        t0 = float(self.t0_spin.value())
        t1 = float(self.t1_spin.value())

        if crop and t1 <= t0:
            self._err("End time must be greater than start time.")
            return

        signal_kind = self._selected_signal_kind()
        if signal_kind == "vel" and not _HAVE_SCIPY:
            self._err("Velocity mode requires SciPy because automatic high-pass filtering is applied before integration.")
            return

        names = self._get_selected_names()
        wf_cfg = None

        if self._is_waterfall_mode():
            all_names = [self.list_channels.item(i).text() for i in range(self.list_channels.count())]
            if len(all_names) < 2:
                self._err("At least two channels are needed for waterfall mode: one RPM and one acceleration.")
                return

            dlg = WaterfallSetupDialog(
                all_names,
                self,
                rpm_guess=self._guess_rpm_channel(),
                acc_guess=self._guess_acc_channel(),
            )
            if dlg.exec_() != QDialog.Accepted:
                return
            wf_cfg = dlg.values()
            self._pending_waterfall_open_3d = bool(wf_cfg.get("open_3d", False))
            names = [wf_cfg["rpm_name"], wf_cfg["acc_name"]]
        else:
            if not names:
                self._err("No channel/column selected.")
                return
            self._pending_waterfall_open_3d = False

        cfg = {
            "file_path": self.file_path,
            "file_type": self.file_type,
            "group_or_sheet": self.cmb_sheet.currentText(),
            "names": names,
            "fs": fs,
            "crop": crop,
            "t0": t0,
            "t1": t1,
            "skip_first": self.chk_skip_first.isChecked(),

            "mode": self.cmb_mode.currentText(),
            "signal_kind": signal_kind,
            "signal_name": self._signal_name(),

           "use_filtered": (
                (self.cmb_data_source.currentText() == "Filtered")
                if (not self._is_waterfall_mode() and not self._is_velocity_mode())
                else False
            ),
            "filter_enabled": self.chk_filter.isChecked(),
            "filter_type": self.cmb_filter_type.currentText(),
            "filter_order": int(self.cmb_filter_order.currentText()),
            "cut1": float(self.cut1_spin.value()),
            "cut2": float(self.cut2_spin.value()),

            "nfft": self._get_nfft(),
            "overlap_pct": float(self.ovl_spin.value()),
            "out_type": self.cmb_out.currentText(),
            "scale": self.cmb_scale.currentText(),
            "method": self.cmb_fourier_method.currentText(),
            "amp_conv": self.cmb_amp_conv.currentText(),
            "window_kind": self.cmb_window_kind.currentText().lower(),
        }

        if wf_cfg is not None:
            cfg["waterfall_rpm_name"] = wf_cfg["rpm_name"]
            cfg["waterfall_acc_name"] = wf_cfg["acc_name"]
            cfg["waterfall_filter_target"] = wf_cfg["filter_target"]

        self._start_plot_worker(cfg)


    def _export_cropped_tdms(self):
        """Open dialog and export time-cropped, zero-aligned signals to .tdms."""
        if not _HAVE_TDMS_WRITE:
            self._err(
                "TDMS writing requires the nptdms package with write support.\n"
                "Install with: pip install nptdms"
            )
            return

        if not self.file_path or not self.file_type:
            self._err("No source file loaded. Please open a file first.")
            return

        group_or_sheet = self.cmb_sheet.currentText().strip()
        if not group_or_sheet:
            self._err("No sheet/group selected.")
            return

        channel_names = [
            self.list_channels.item(i).text()
            for i in range(self.list_channels.count())
        ]
        if not channel_names:
            self._err("No channels available. Please load a file first.")
            return

        default_group = group_or_sheet.replace(" ", "_")

        dlg = ExportCroppedTdmsDialog(
            channel_names=channel_names,
            parent=self,
            default_fs=float(self.fs_spin.value()),
            default_t0=float(self.t0_spin.value()),
            default_t1=float(self.t1_spin.value()),
            default_group=default_group,
            skip_first_default=self.chk_skip_first.isChecked(),
        )

        if dlg.exec_() != QDialog.Accepted:
            return

        vals = dlg.values()
        selected_names = vals["channel_names"]
        fs = vals["fs"]
        t0 = vals["t0"]
        t1 = vals["t1"]
        skip_first = vals["skip_first"]
        tdms_group = vals["group_name"]
        output_path = vals["output_path"]

        # ── Read raw signals and collect source metadata ──
        try:
            raw_channels = {}
            source_properties = {}

            for nm in selected_names:
                raw_channels[nm] = read_signal_from_source(
                    self.file_path, self.file_type, group_or_sheet, nm
                )

                # Try to read original TDMS properties if source is TDMS
                if self.file_type == "tdms" and _HAVE_TDMS:
                    try:
                        tdms = TdmsFile.read(self.file_path)
                        grp = tdms[group_or_sheet]
                        if nm in grp:
                            ch = grp[nm]
                            source_properties[nm] = dict(ch.properties)
                    except Exception:
                        pass

            info = export_cropped_tdms(
                output_path=output_path,
                group_name=tdms_group,
                channels=raw_channels,
                fs=fs,
                t0=t0,
                t1=t1,
                skip_first=skip_first,
                source_properties=source_properties,
            )

            dt = t1 - t0
            summary_lines = [
                f"Cropped TDMS exported successfully!",
                f"",
                f"File: {output_path}",
                f"Group: {tdms_group}",
                f"Original window: [{t0:.6g}, {t1:.6g}] s",
                f"Zero-aligned window: [0, {dt:.6g}] s",
                f"Sampling rate: {fs:.6g} Hz  |  wf_increment: {1.0/fs:.8g} s",
                f"",
            ]
            for ch_info in info:
                summary_lines.append(
                    f"  {ch_info['name']}: {ch_info['exported_samples']} samples "
                    f"({ch_info['duration_s']:.6g} s)"
                )

            QMessageBox.information(self, "Export Cropped TDMS", "\n".join(summary_lines))

        except Exception as e:
            self._err(f"TDMS export failed:\n{e}")


    def _err(self, msg: str):
        QMessageBox.critical(self, "Error", msg)


def main():
    app = QApplication(sys.argv)

    if not _HAVE_PG:
        QMessageBox.critical(None, "Missing dependency", "pyqtgraph is required. Install with: pip install pyqtgraph")
        sys.exit(1)

    w = MainWindow()
    w.resize(1600, 900)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
