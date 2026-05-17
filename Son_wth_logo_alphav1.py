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
    QScrollArea, QGridLayout, QFrame, QGroupBox, QToolButton, QLineEdit,
    QDialog, QDialogButtonBox, QFormLayout, QColorDialog,
    QTableWidget, QTableWidgetItem, QHeaderView,QSplitter, QStyle, QStyledItemDelegate
)
from PyQt5.QtGui import QPixmap, QPainter, QColor, QIcon

try:
    from PyQt5 import sip as _qt_sip
except Exception:
    try:
        import sip as _qt_sip
    except Exception:
        _qt_sip = None


def _qt_alive(obj) -> bool:
    """Return False when a PyQt wrapper already lost its C++ QObject."""
    if obj is None:
        return False
    if _qt_sip is not None:
        try:
            if _qt_sip.isdeleted(obj):
                return False
        except Exception:
            pass
    try:
        obj.objectName()
        return True
    except RuntimeError:
        return False
    except Exception:
        return True


def resource_path(relative_path: str) -> str:
    """Return an asset path that works both normally and inside PyInstaller builds."""
    if not relative_path:
        return ""
    if os.path.isabs(relative_path):
        return relative_path

    candidates = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(os.path.join(meipass, relative_path))
    try:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), relative_path))
    except Exception:
        pass
    try:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path))
    except Exception:
        pass
    candidates.append(os.path.join(os.path.abspath("."), relative_path))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    # Fall back to the first PyInstaller-style path when bundled, otherwise cwd.
    return candidates[0] if candidates else relative_path

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
# External branding wrapper removed: the built-in _build_logo_header() is used instead.

# --- Envelope / kurtogram backend + widget ---
try:
    from dsp_core.envelope_kurtogram import (
        compute_envelope_spectrum,
        compute_fast_kurtogram,
        extract_bearing_fault_frequencies,
    )
    _HAVE_ENVELOPE_BACKEND = True
except Exception:
    compute_envelope_spectrum = None
    compute_fast_kurtogram = None
    extract_bearing_fault_frequencies = None
    _HAVE_ENVELOPE_BACKEND = False

try:
    from gui_widgets.envelope_widget import EnvelopeKurtogramWidget
except Exception:
    EnvelopeKurtogramWidget = None

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
    from scipy.signal import butter, sosfiltfilt, sosfilt, detrend as scipy_detrend
    _HAVE_SCIPY = True
except Exception:
    butter = None
    sosfiltfilt = None
    sosfilt = None
    scipy_detrend = None
    _HAVE_SCIPY = False

if _HAVE_PG:
    pg.setConfigOptions(antialias=False, imageAxisOrder="row-major")

# Set DEBUG = True to enable verbose console output from TRDSP internals.
# Keep False in production to avoid flooding the terminal.
DEBUG: bool = False

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
    out = normalize_output_type(output_type) if "normalize_output_type" in globals() else (str(output_type).strip() or "Amplitude")
    base_unit = "mm/s" if "mm/s" in str(unit_lbl) else "g"
    if out == "Power":
        label_unit = f"({base_unit})²"
    elif out == "PSD":
        label_unit = f"({base_unit})²/Hz"
    elif out == "ASD":
        label_unit = f"{base_unit}/√Hz"
    else:
        label_unit = unit_lbl
    scl = str(scale).strip().lower()
    if scl == "db":
        factor = "20log10" if out in ("Amplitude", "ASD") else "10log10"
        return f"{out} ({label_unit}) [dB re 1, {factor}]"
    return f"{out} ({label_unit})"





def normalize_output_type(output_type: str) -> str:
    """Return canonical spectrum output type: Amplitude, Power, PSD, or ASD."""
    out = str(output_type or "Amplitude").strip().upper()
    if out in ("PSD", "POWER SPECTRAL DENSITY"):
        return "PSD"
    if out in ("ASD", "AMPLITUDE SPECTRAL DENSITY"):
        return "ASD"
    if out.startswith("POW"):
        return "Power"
    return "Amplitude"


def window_coherent_gain(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64)
    return float(np.sum(w) / max(w.size, 1)) if w.size else 0.0


def window_power(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64)
    return float(np.sum(w * w))


def window_enbw_hz(w: np.ndarray, fs: float) -> float:
    w = np.asarray(w, dtype=np.float64)
    s1 = float(np.sum(w))
    if w.size == 0 or s1 == 0.0:
        return np.nan
    return float(fs) * float(np.sum(w * w)) / (s1 * s1)


def one_sided_correction(nfft: int) -> np.ndarray:
    """One-sided power correction for rFFT bins (DC/Nyquist not doubled)."""
    nfft = max(2, int(nfft))
    c = np.ones(nfft // 2 + 1, dtype=np.float64)
    if c.size > 2:
        if nfft % 2 == 0:
            c[1:-1] = 2.0
        else:
            c[1:] = 2.0
    elif c.size == 2 and nfft % 2 != 0:
        c[1] = 2.0
    return c


def spectrum_db_factor(output_type: str) -> float:
    out = normalize_output_type(output_type)
    return 20.0 if out in ("Amplitude", "ASD") else 10.0


def convert_linear_to_db(values: np.ndarray, output_type: str, ref: float = 1.0) -> np.ndarray:
    eps = 1e-30
    ref = abs(float(ref)) or 1.0
    return spectrum_db_factor(output_type) * np.log10(np.maximum(np.asarray(values, dtype=np.float64), eps) / ref)


def _amplitude_to_power(A: np.ndarray, nfft: int) -> np.ndarray:
    """Convert one-sided peak amplitude bins to RMS^2 line power."""
    A = np.asarray(A, dtype=np.float64)
    P = (A / np.sqrt(2.0)) ** 2
    if P.size:
        P[0] = A[0] * A[0]
        if nfft % 2 == 0 and P.size > 1:
            P[-1] = A[-1] * A[-1]
    return P


def _power_to_peak_amplitude(P: np.ndarray, nfft: int) -> np.ndarray:
    P = np.maximum(np.asarray(P, dtype=np.float64), 0.0)
    A = np.sqrt(P) * np.sqrt(2.0)
    if A.size:
        A[0] = np.sqrt(P[0])
        if nfft % 2 == 0 and A.size > 1:
            A[-1] = np.sqrt(P[-1])
    return A


def compute_spectrum_frame(
    segment: np.ndarray,
    fs: float,
    nfft: int,
    amp_conv: str = "Peak",
    signal_kind: str = "acc",
    output_type: str = "Amplitude",
    *,
    window_kind: str = "periodic",
):
    """Shared one-sided FFT kernel with Amplitude/Power/PSD/ASD semantics."""
    segment = np.nan_to_num(np.asarray(segment, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    fs = float(fs)
    nfft = max(2, int(nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs) if np.isfinite(fs) and fs > 0 else np.arange(nfft // 2 + 1)
    if segment.size == 0 or not np.isfinite(fs) or fs <= 0.0:
        return freqs, np.zeros(freqs.size, dtype=np.float64)
    nseg = min(int(segment.size), nfft)
    if nseg <= 1:
        return freqs, np.zeros(freqs.size, dtype=np.float64)
    x = segment[:nseg]
    w = _hann_window(nseg, window_kind)
    X = np.fft.rfft(x * w, n=nfft)
    out = normalize_output_type(output_type)
    if out in ("PSD", "ASD"):
        U = window_power(w) or float(nseg)
        psd = (np.abs(X) ** 2) / (fs * U)
        psd *= one_sided_correction(nfft)
        spec = np.sqrt(np.maximum(psd, 0.0)) if out == "ASD" else psd
    else:
        w_sum = float(np.sum(w)) or float(nseg)
        A_peak = (2.0 * np.abs(X)) / w_sum
        A_peak[0] *= 0.5
        if nfft % 2 == 0 and A_peak.size > 1:
            A_peak[-1] *= 0.5
        if out == "Amplitude":
            conv, _ = get_amp_conv_factor_label(amp_conv, signal_kind)
            spec = A_peak * conv
        else:
            spec = _amplitude_to_power(A_peak, nfft)
    return freqs, np.maximum(np.asarray(spec, dtype=np.float64), 0.0)



def compute_spectrum_frames_vectorized(
    signal: np.ndarray,
    starts: np.ndarray,
    nperseg: int,
    fs: float,
    nfft: int,
    amp_conv: str = "Peak",
    signal_kind: str = "acc",
    output_type: str = "Amplitude",
    *,
    window_kind: str = "periodic",
    max_chunk_frames: int = None,
) -> np.ndarray:
    """Compute one-sided spectra for many STFT frames using chunked NumPy FFTs.

    This preserves ``compute_spectrum_frame`` scaling/window conventions while
    replacing per-frame Python FFT calls with batched ``np.fft.rfft(..., axis=1)``.
    Chunking limits temporary frame/FFT arrays for large TDMS files and retries
    with smaller chunks if a MemoryError is raised.
    """
    y = np.nan_to_num(np.asarray(signal, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    starts = np.asarray(starts, dtype=np.int64)
    fs = float(fs)
    nfft = max(2, int(nfft))
    nperseg = max(0, int(nperseg))
    n_freq = nfft // 2 + 1
    if starts.size == 0 or nperseg <= 1 or y.size == 0 or not np.isfinite(fs) or fs <= 0.0:
        return np.empty((0, n_freq), dtype=np.float64)

    w = _hann_window(nperseg, window_kind).astype(np.float64, copy=False)
    out_type = normalize_output_type(output_type)
    spectra = np.empty((starts.size, n_freq), dtype=np.float64)

    if max_chunk_frames is None:
        # Approximate temporary bytes per frame: indexed samples + complex FFT + output magnitude.
        bytes_per_frame = max(1, nperseg * 8 + n_freq * (16 + 8))
        max_chunk_frames = int(max(1, (256 * 1024 * 1024) // bytes_per_frame))
    chunk_frames = int(max(1, min(starts.size, max_chunk_frames)))
    sample_offsets = np.arange(nperseg, dtype=np.int64)

    if out_type in ("PSD", "ASD"):
        U = window_power(w) or float(nperseg)
        one_sided = one_sided_correction(nfft)
    else:
        w_sum = float(np.sum(w)) or float(nperseg)
        conv = 1.0
        if out_type == "Amplitude":
            conv, _ = get_amp_conv_factor_label(amp_conv, signal_kind)

    pos = 0
    while pos < starts.size:
        stop = min(starts.size, pos + chunk_frames)
        try:
            idx = starts[pos:stop, None] + sample_offsets[None, :]
            frames = y[idx] * w[None, :]
            X = np.fft.rfft(frames, n=nfft, axis=1)
            if out_type in ("PSD", "ASD"):
                spec = (np.abs(X) ** 2) / (fs * U)
                spec *= one_sided[None, :]
                if out_type == "ASD":
                    spec = np.sqrt(np.maximum(spec, 0.0))
            else:
                A_peak = (2.0 * np.abs(X)) / w_sum
                A_peak[:, 0] *= 0.5
                if nfft % 2 == 0 and A_peak.shape[1] > 1:
                    A_peak[:, -1] *= 0.5
                if out_type == "Amplitude":
                    spec = A_peak * conv
                else:
                    spec = (A_peak / np.sqrt(2.0)) ** 2
                    spec[:, 0] = A_peak[:, 0] * A_peak[:, 0]
                    if nfft % 2 == 0 and spec.shape[1] > 1:
                        spec[:, -1] = A_peak[:, -1] * A_peak[:, -1]
            spectra[pos:stop, :] = np.maximum(np.asarray(spec, dtype=np.float64), 0.0)
            pos = stop
        except MemoryError:
            if chunk_frames <= 1:
                raise
            chunk_frames = max(1, chunk_frames // 2)

    return spectra

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


def clean_rpm_signal(
    rpm: np.ndarray,
    mode: str = "interpolate",
    spike_mad_thresh: float = 5.0,
    smooth: bool = False,
    smooth_kernel: int = 5,
) -> np.ndarray:
    """Validate and clean an RPM signal.

    Parameters
    ----------
    rpm : array-like
        Raw RPM samples (same length as acceleration).
    mode : {"interpolate", "mask"}
        "interpolate" – fill short NaN/<=0/Inf gaps by linear interpolation from
        valid neighbors; isolated leading/trailing invalids become NaN.
        "mask" – replace invalid samples with NaN (no gap-filling).
    spike_mad_thresh : float
        Spike rejection threshold in median-absolute-deviation units on the
        first derivative.  Set to 0 to disable.
    smooth : bool
        If True, apply a centered running-median smoother of width smooth_kernel
        *after* cleaning (does not affect frequency resolution).
    smooth_kernel : int
        Running-median half-width (samples).

    Returns
    -------
    np.ndarray, dtype float64
        Cleaned RPM with NaN where invalid.  The caller is responsible for
        deciding what to do with remaining NaN frames.
    """
    rpm = np.asarray(rpm, dtype=np.float64).copy()

    # Mark fundamentally invalid values
    invalid = ~np.isfinite(rpm) | (rpm <= 0.0)

    # Spike rejection using MAD on derivative
    if spike_mad_thresh > 0.0 and rpm.size > 2:
        drpm = np.diff(rpm)
        med_d = float(np.nanmedian(drpm))
        mad_d = float(np.nanmedian(np.abs(drpm - med_d))) or 1e-12
        spikes = np.abs(drpm - med_d) > spike_mad_thresh * mad_d * 1.4826
        # Mark both samples surrounding a derivative spike
        spike_idx = np.where(spikes)[0]
        invalid[spike_idx] = True
        invalid[np.clip(spike_idx + 1, 0, rpm.size - 1)] = True

    rpm[invalid] = np.nan

    if mode == "interpolate":
        valid_idx = np.where(np.isfinite(rpm))[0]
        if valid_idx.size >= 2:
            all_idx = np.arange(rpm.size)
            rpm = np.interp(all_idx, valid_idx, rpm[valid_idx],
                            left=np.nan, right=np.nan)
        # Re-mark any remaining out-of-range interpolated values
        rpm[rpm <= 0.0] = np.nan

    if smooth and rpm.size > 2:
        k = max(1, int(smooth_kernel))
        smoothed = rpm.copy()
        for i in range(rpm.size):
            lo = max(0, i - k)
            hi = min(rpm.size, i + k + 1)
            patch = rpm[lo:hi]
            fin = patch[np.isfinite(patch)]
            if fin.size:
                smoothed[i] = float(np.median(fin))
        rpm = smoothed

    return rpm


def apply_order_tracking_filter(
    freqs: np.ndarray,
    frame_rpms: np.ndarray,
    frame_spectra_linear: np.ndarray,
    target_order: float,
    half_bw: float,
    bw_unit: str = "Order",
    filter_type: str = "Band-pass",
    filter_order: int = 4,
    output_type: str = "Amplitude",
) -> tuple:
    """Frequency-domain order mask (frame-based, RPM-adaptive).

    For each FFT frame, computes the center frequency
        f_center = target_order * RPM_frame / 60
    and applies a Butterworth-shaped frequency-domain mask centered on
    f_center with half-bandwidth ``half_bw`` (in Hz or Order units).

    Band-pass keeps only the band; Band-stop notches it.  The returned map
    has the same shape as ``frame_spectra_linear`` and represents the spectra
    after the filter is applied.  A 1-D trace of band-energy vs frame is also
    returned so the GUI can display the extracted order.

    Notes
    -----
    This is not a true nCode time-domain adaptive order filter. It is a
    display/FFT-domain mask used to isolate order content in waterfall frames.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    frame_rpms = np.asarray(frame_rpms, dtype=np.float64)
    S = np.asarray(frame_spectra_linear, dtype=np.float64)

    if S.ndim != 2 or freqs.size != S.shape[1]:
        return S.copy(), np.zeros(S.shape[0], dtype=np.float64)

    k = float(target_order)
    bw = max(0.0, float(half_bw))
    order_unit = str(bw_unit).strip().lower().startswith("order")
    n = int(max(1, filter_order))
    is_bandpass = str(filter_type).strip().lower().startswith("band-pass")
    is_amplitude = normalize_output_type(output_type) in ("Amplitude", "ASD")

    out = np.zeros_like(S)
    trace = np.zeros(S.shape[0], dtype=np.float64)

    # Butterworth power response:
    #   |H(f)|^2 = 1 / (1 + ((f - fc)/bw)^(2n))   (band-pass)
    # Band-stop power response = 1 - |H(f)|^2.
    #
    # LTI gain convention:
    #   amplitude spectra     -> multiply by |H|   = sqrt(power response)
    #   power / RMS^2 spectra -> multiply by |H|^2 = power response
    #
    # This keeps Amplitude and Power outputs consistent:
    #   Power_after = Amplitude_after^2
    for i in range(S.shape[0]):
        rpm = frame_rpms[i]
        if not np.isfinite(rpm) or rpm <= 0.0:
            # Without a valid RPM we cannot place the center frequency.
            # Leave the row at zero (everything suppressed for that frame).
            continue

        f_center = k * rpm / 60.0
        if order_unit:
            half_bw_hz = max(1e-9, bw * rpm / 60.0)
        else:
            half_bw_hz = max(1e-9, bw)

        denom = ((freqs - f_center) / half_bw_hz) ** (2 * n)
        bp_power = 1.0 / (1.0 + denom)
        power_gain = bp_power if is_bandpass else (1.0 - bp_power)

        if is_amplitude:
            out[i, :] = S[i, :] * np.sqrt(power_gain)
        else:
            out[i, :] = S[i, :] * power_gain

        # Extracted trace (sum of band energy, sqrt for amplitude semantics).
        # We use the band-pass response regardless of selected type so that
        # the trace always tracks the chosen order's content.
        bp = 1.0 / (1.0 + denom)
        trace[i] = float(np.sqrt(np.sum((S[i, :] ** 2) * bp)))

    return out, trace



def apply_time_domain_order_tracking_filter(
    signal: np.ndarray,
    rpm: np.ndarray,
    fs: float,
    target_order: float,
    half_bw: float,
    bw_unit: str = "Order",
    filter_type: str = "Band-pass",
    filter_order: int = 4,
    method: str = "blockwise_sosfiltfilt",
    block_revs=None,
    block_seconds=None,
    crossfade: bool = True,
) -> tuple:
    """Approximate nCode-like time-domain adaptive order filter.

    The instantaneous center frequency is k*RPM/60. Because scipy's Butterworth
    design is LTI, this function applies SOS filters blockwise using the median
    RPM in each block and overlap-add crossfading. Metadata records clamping,
    rejected blocks, cutoff ranges, and warnings.
    """
    meta = {
        "mode": "Time-domain adaptive order filter",
        "enabled": True,
        "target_order": float(target_order),
        "half_bw": float(half_bw),
        "bw_unit": bw_unit,
        "filter_type": filter_type,
        "filter_order": int(filter_order),
        "method": method,
        "block_count": 0,
        "rejected_blocks": 0,
        "cutoff_low_min": np.nan,
        "cutoff_low_max": np.nan,
        "cutoff_high_min": np.nan,
        "cutoff_high_max": np.nan,
        "rpm_min": np.nan,
        "rpm_max": np.nan,
        "warnings": [],
    }
    x = np.nan_to_num(np.asarray(signal, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    r_raw = np.asarray(rpm, dtype=np.float64)
    n = min(x.size, r_raw.size)
    if n == 0 or not np.isfinite(fs) or fs <= 0.0:
        meta["warnings"].append("Empty signal/RPM or invalid sampling rate; filter bypassed.")
        return x.copy(), meta
    x = x[:n]
    r_clean = clean_rpm_signal(r_raw[:n], mode="interpolate")
    finite_r = r_clean[np.isfinite(r_clean) & (r_clean > 0)]
    if finite_r.size < max(8, n // 100):
        meta["warnings"].append("Insufficient finite positive RPM samples; filter bypassed.")
        return x.copy(), meta
    meta["rpm_min"] = float(np.min(finite_r)); meta["rpm_max"] = float(np.max(finite_r))
    # Fill leading/trailing RPM from nearest valid for block labelling.
    bad = ~np.isfinite(r_clean) | (r_clean <= 0)
    if np.any(bad):
        idx = np.arange(n)
        good = np.where(~bad)[0]
        r_clean = np.interp(idx, good, r_clean[good])
        meta["warnings"].append("RPM invalid spans interpolated/fill-applied before adaptive filtering.")

    nyq = 0.5 * float(fs)
    target_order = float(target_order)
    half_bw = max(float(half_bw), 1e-9)
    order_bw = str(bw_unit).strip().lower().startswith("order")
    btype = "bandpass" if str(filter_type).strip().lower().startswith("band-pass") else "bandstop"
    filt_order = int(max(1, filter_order))

    if block_seconds is None:
        if block_revs is not None and np.nanmedian(r_clean) > 0:
            block_seconds = max(0.05, float(block_revs) / (float(np.nanmedian(r_clean)) / 60.0))
        else:
            block_seconds = 0.5
    block_len = int(max(32, round(float(block_seconds) * float(fs))))
    block_len = min(max(block_len, 32), n)
    overlap = int(0.25 * block_len) if crossfade and block_len > 64 else 0
    step = max(1, block_len - overlap)

    y_sum = np.zeros(n, dtype=np.float64)
    w_sum = np.zeros(n, dtype=np.float64)
    lows = []; highs = []
    starts = list(range(0, n, step))
    for start in starts:
        stop = min(n, start + block_len)
        if stop - start < max(16, 3 * filt_order * 2):
            meta["rejected_blocks"] += 1
            continue
        rpm_rep = float(np.nanmedian(r_clean[start:stop]))
        if not np.isfinite(rpm_rep) or rpm_rep <= 0:
            meta["rejected_blocks"] += 1
            continue
        fc = target_order * rpm_rep / 60.0
        hb = half_bw * rpm_rep / 60.0 if order_bw else half_bw
        flo = fc - hb
        fhi = fc + hb
        orig = (flo, fhi)
        flo = max(flo, max(1e-6, nyq * 1e-6))
        fhi = min(fhi, nyq * (1.0 - 1e-6))
        if not (0.0 < flo < fhi < nyq):
            meta["rejected_blocks"] += 1
            meta["warnings"].append(f"Rejected block {start}:{stop}; invalid cutoffs {orig} Hz.")
            continue
        if orig != (flo, fhi):
            meta["warnings"].append("One or more adaptive filter cutoffs were clamped to valid Nyquist range.")
        try:
            sos = butter(filt_order, [flo / nyq, fhi / nyq], btype=btype, output="sos")
            seg = x[start:stop]
            padlen = min(seg.size - 1, 3 * (2 * sos.shape[0] + 1))
            if sosfiltfilt is not None and seg.size > padlen + 1:
                yb = sosfiltfilt(sos, seg, padlen=padlen)
            else:
                yb = sosfilt(sos, seg) if sosfilt is not None else seg.copy()
            ww = np.ones(stop - start, dtype=np.float64)
            if crossfade and overlap > 0 and ww.size > 2 * overlap:
                ramp = np.linspace(0.0, 1.0, overlap, endpoint=False)
                ww[:overlap] = ramp
                ww[-overlap:] = ramp[::-1]
            y_sum[start:stop] += yb * ww
            w_sum[start:stop] += ww
            lows.append(flo); highs.append(fhi); meta["block_count"] += 1
        except Exception as e:
            meta["rejected_blocks"] += 1
            meta["warnings"].append(f"Adaptive filter block {start}:{stop} failed: {e}")
    valid = w_sum > 0
    y = x.copy()
    y[valid] = y_sum[valid] / w_sum[valid]
    if lows:
        meta["cutoff_low_min"] = float(np.min(lows)); meta["cutoff_low_max"] = float(np.max(lows))
        meta["cutoff_high_min"] = float(np.min(highs)); meta["cutoff_high_max"] = float(np.max(highs))
    if meta["block_count"] == 0:
        meta["warnings"].append("No adaptive filter blocks accepted; signal bypassed.")
    return y, meta

def compute_order_map(
    freqs: np.ndarray,
    frame_rpms: np.ndarray,
    frame_spectra_linear: np.ndarray,
    order_max: float = 10.0,
    n_orders: int = 512,
    output_type: str = "Amplitude",
    scale: str = "Linear",
) -> tuple:
    """Convert per-frame frequency spectra to an order-domain waterfall map.

    Each frame's frequency axis is converted to order axis:
        order = f / (RPM / 60)
    Frames are then interpolated onto a uniform order grid.

    Parameters
    ----------
    freqs : (Nf,) array
        Frequency axis in Hz.
    frame_rpms : (M,) array
        RPM label for each frame (must be > 0 and finite).
    frame_spectra_linear : (M, Nf) array
        Linear amplitude or power spectra — one row per frame.
    order_max : float
        Maximum order to include in the output grid.
    n_orders : int
        Number of order bins in the output grid.
    output_type : "Amplitude" or "Power"
        Selects 20·log10 or 10·log10 when scale=="dB".
    scale : "Linear" or "dB"
        Output scale.

    Returns
    -------
    order_axis : (n_orders,) array
        Uniform order grid from 0 to order_max.
    sorted_rpms : (M_valid,) array
        RPM values of valid frames, sorted ascending.
    S_order_out : (M_valid, n_orders) array
        Order-domain spectra in the requested output/scale.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    frame_rpms = np.asarray(frame_rpms, dtype=np.float64)
    frame_spectra_linear = np.asarray(frame_spectra_linear, dtype=np.float64)

    valid = np.isfinite(frame_rpms) & (frame_rpms > 0.0)
    if not np.any(valid):
        raise RuntimeError("No valid (>0, finite) RPM frames for order-map computation.")

    rpms_v = frame_rpms[valid]
    spec_v = frame_spectra_linear[valid]
    order_axis = np.linspace(0.0, float(order_max), int(n_orders))

    M = int(rpms_v.size)
    S_order_linear = np.empty((M, n_orders), dtype=np.float64)

    for i in range(M):
        rpm_i = float(rpms_v[i])
        freq_for_order = order_axis * (rpm_i / 60.0)   # Hz value of each order bin
        # Interpolate linearly; bins beyond Nyquist become 0
        S_order_linear[i, :] = np.interp(
            freq_for_order, freqs, spec_v[i, :], left=0.0, right=0.0
        )

    S_order_linear = np.maximum(S_order_linear, 0.0)

    if str(scale).strip().lower() == "db":
        eps = 1e-30
        S_order_out = convert_linear_to_db(S_order_linear, output_type)
    else:
        S_order_out = S_order_linear

    # Sort by RPM ascending (caller may pass unsorted frames)
    sort_idx = np.argsort(rpms_v)
    return order_axis, rpms_v[sort_idx], S_order_out[sort_idx]


def compute_order_tracking_spectrum(
    acc: np.ndarray,
    rpm: np.ndarray,
    fs: float,
    samples_per_rev: int = 512,
    order_max: float = 20.0,
    amp_conv: str = "Peak",
    signal_kind: str = "acc",
    *,
    detrend: str = "None",
    window_kind: str = "periodic",
    return_metadata: bool = False,
) -> tuple:
    """Computed Order Tracking using true RPM-driven angle resampling."""
    meta = {"rpm_min": np.nan, "rpm_max": np.nan, "total_revs": 0.0,
            "samples_per_rev": int(samples_per_rev), "order_max": float(order_max),
            "rpm_interpolated": False, "warnings": []}
    acc = np.asarray(acc, dtype=np.float64)
    rpm_raw = np.asarray(rpm, dtype=np.float64)
    fs = float(fs)
    n = min(acc.size, rpm_raw.size)
    if n <= 1 or not np.isfinite(fs) or fs <= 0.0:
        raise RuntimeError("Computed Order Tracking requires same-length nonempty acc/RPM and valid fs.")
    acc = np.nan_to_num(acc[:n], nan=0.0, posinf=0.0, neginf=0.0)
    rpm_raw = rpm_raw[:n]
    valid_raw = np.isfinite(rpm_raw) & (rpm_raw > 0.0)
    if np.count_nonzero(valid_raw) < max(4, min(n, 16)):
        raise RuntimeError("Computed Order Tracking: insufficient finite positive RPM samples.")
    rpm_clean = clean_rpm_signal(rpm_raw, mode="interpolate")
    bad = ~np.isfinite(rpm_clean) | (rpm_clean <= 0.0)
    if np.any(~valid_raw) or np.any(bad):
        meta["rpm_interpolated"] = True
    if np.any(bad):
        good = np.where(~bad)[0]
        if good.size < 2:
            raise RuntimeError("Computed Order Tracking: RPM cleaning left too few usable samples.")
        rpm_clean = np.interp(np.arange(n), good, rpm_clean[good])
        meta["warnings"].append("RPM leading/trailing invalid spans filled from nearest valid samples.")
    finite = rpm_clean[np.isfinite(rpm_clean) & (rpm_clean > 0.0)]
    if finite.size < 4:
        raise RuntimeError("Computed Order Tracking: insufficient cleaned RPM samples.")
    meta["rpm_min"] = float(np.min(finite)); meta["rpm_max"] = float(np.max(finite))
    dsel = str(detrend).strip().lower()
    if dsel in ("remove mean", "mean"):
        acc = acc - float(np.mean(acc))
    elif dsel.startswith("linear") and _HAVE_SCIPY:
        acc = scipy_detrend(acc, type="linear")
    dt = 1.0 / fs
    omega = 2.0 * np.pi * rpm_clean / 60.0
    theta = np.empty(n, dtype=np.float64)
    theta[0] = 0.0
    if n > 1:
        theta[1:] = np.cumsum(0.5 * (omega[:-1] + omega[1:]) * dt)
    keep = np.isfinite(theta) & np.isfinite(acc)
    keep &= np.r_[True, np.diff(theta) > 1e-12]
    theta_u = theta[keep]; acc_u = acc[keep]
    if theta_u.size < 4:
        raise RuntimeError("Computed Order Tracking: shaft angle is not sufficiently increasing for interpolation.")
    total_revs = float((theta_u[-1] - theta_u[0]) / (2.0 * np.pi))
    meta["total_revs"] = total_revs
    if total_revs < 0.5:
        raise RuntimeError(f"Computed Order Tracking: total shaft rotation is only {total_revs:.2f} revolutions.")
    samples_per_rev = int(max(8, samples_per_rev))
    n_angle_samples = int(max(8, round(total_revs * samples_per_rev)))
    theta_uniform = np.linspace(float(theta_u[0]), float(theta_u[-1]), n_angle_samples)
    acc_angle = np.interp(theta_uniform, theta_u, acc_u)
    orders_full, spec = compute_spectrum_frame(
        acc_angle, fs=float(samples_per_rev), nfft=n_angle_samples,
        amp_conv=amp_conv, signal_kind=signal_kind, output_type="Amplitude",
        window_kind=window_kind,
    )
    keep_ord = orders_full <= float(order_max)
    orders = orders_full[keep_ord]
    amp_spectrum = spec[keep_ord]
    rev_edges = np.arange(float(theta_u[0]), float(theta_u[-1]) + 2.0 * np.pi, 2.0 * np.pi)
    rev_rpms = []; rev_amps = []
    for k in range(len(rev_edges) - 1):
        m = (theta >= rev_edges[k]) & (theta < rev_edges[k + 1])
        if np.any(m):
            rev_amps.append(float(np.sqrt(np.mean(acc[m] ** 2))))
            rev_rpms.append(float(np.mean(rpm_clean[m])))
    amp_vs_rpm = {"rpms": np.asarray(rev_rpms, dtype=np.float64), "amp": np.asarray(rev_amps, dtype=np.float64)}
    if return_metadata:
        return orders, amp_spectrum, amp_vs_rpm, meta
    return orders, amp_spectrum, amp_vs_rpm

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
    """Backward-compatible wrapper for the shared one-sided spectrum kernel.

    Supports Amplitude, Power (RMS²/line), PSD (unit²/Hz), and ASD
    (unit/√Hz).  Kept under its historical name so existing call sites and
    plug-ins continue to work.
    """
    return compute_spectrum_frame(segment, fs, nfft, amp_conv, signal_kind, output_type, window_kind=window_kind)

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
    """Welch-style averaged spectrum with explicit output semantics.

    Amplitude/Power frames are averaged in RMS² line-power domain and converted
    back as requested. PSD/ASD use window-power normalization and are averaged
    as PSD before ASD square-root conversion.
    """
    y = np.nan_to_num(np.asarray(y, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    fs = float(fs); nfft = max(2, int(nfft))
    if y.size == 0 or not np.isfinite(fs) or fs <= 0.0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    nperseg = min(nfft, int(y.size))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    if nperseg <= 1:
        return freqs, np.zeros(freqs.size, dtype=np.float64)
    ovl = float(np.clip(float(overlap_pct) / 100.0, 0.0, 0.95))
    hop = max(1, int(round(nperseg * (1.0 - ovl))))
    n_frames = 1 if y.size <= nperseg else 1 + (y.size - nperseg) // hop
    starts = np.arange(n_frames, dtype=np.int64) * hop
    starts = starts[starts + nperseg <= y.size]
    if starts.size == 0:
        return freqs, np.zeros(freqs.size, dtype=np.float64)
    w = _hann_window(nperseg, window_kind)
    idx = starts[:, None] + np.arange(nperseg, dtype=np.int64)[None, :]
    X = np.fft.rfft(y[idx] * w[None, :], n=nfft, axis=1)
    out = normalize_output_type(output_type)
    if out in ("PSD", "ASD"):
        U = window_power(w) or float(nperseg)
        Pxx = (np.abs(X) ** 2) / (fs * U)
        Pxx *= one_sided_correction(nfft)[None, :]
        P_avg = np.mean(Pxx, axis=0)
        spec = np.sqrt(np.maximum(P_avg, 0.0)) if out == "ASD" else P_avg
    else:
        w_sum = float(np.sum(w)) or float(nperseg)
        A_peak = (2.0 / w_sum) * np.abs(X)
        A_peak[:, 0] *= 0.5
        if nfft % 2 == 0 and A_peak.shape[1] > 1:
            A_peak[:, -1] *= 0.5
        P = np.vstack([_amplitude_to_power(row.copy(), nfft) for row in A_peak])
        P_avg = P.mean(axis=0)
        if out == "Amplitude":
            conv, _ = get_amp_conv_factor_label(amp_conv, signal_kind)
            spec = _power_to_peak_amplitude(P_avg, nfft) * conv
        else:
            spec = P_avg
    if str(scale).strip().lower() == "db":
        spec = convert_linear_to_db(spec, out)
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
        if self._view_box is None:
            return
        if not _qt_alive(self._view_box):
            self._view_box = None
            return
        try:
            self._view_box.setMouseMode(pg.ViewBox.RectMode if active else pg.ViewBox.PanMode)
        except RuntimeError:
            self._view_box = None

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

        kind = str(state.get("kind", "")).lower()
        if kind == "waterfall":
            raise RuntimeError(
                "Waterfall states must be rendered with WaterfallPlotWidget, "
                "not InteractivePlotWidget. "
                "Use WaterfallMiniWidget / WaterfallDetailWidget for waterfall plots."
            )

        if kind in ("time", "avg"):
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
        missing = [k for k in ("times", "freqs", "S") if k not in state]
        if missing:
            raise RuntimeError(
                f"InteractivePlotWidget._draw_image_state requires keys "
                f"{missing} in the plot state — got keys {list(state.keys())}. "
                f"Hint: waterfall states (kind='waterfall') must be rendered "
                f"with WaterfallPlotWidget instead."
            )
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
    """Standalone 2D waterfall colormap window.

    Renders only the upper waterfall heatmap (no order trace subplot) and
    mirrors the live PyQtGraph color mapping, including manual levels and
    custom color bands when supplied by the parent waterfall widget.
    """

    def __init__(self, state: dict, parent=None,
                 view_range=None, cmap_name="jet",
                 order_cursor_specs=None,
                 color_limits=None,
                 color_cfg=None):
        super().__init__(parent)
        self.state = state
        self.view_range = view_range
        self.cmap_name = cmap_name
        self.order_cursor_specs = order_cursor_specs or []
        self.color_limits = color_limits
        self.color_cfg = dict(color_cfg or {})
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
        base_cmap = str(cmap_name or "jet")
        if base_cmap.endswith("_r"):
            base_cmap = base_cmap[:-2]
        if self.cmb_cmap.findText(base_cmap) < 0:
            self.cmb_cmap.addItem(base_cmap)
        self.cmb_cmap.setCurrentText(base_cmap)
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

    def _effective_color_cfg(self):
        cfg = dict(self.color_cfg or {})
        cmap_name = self.cmb_cmap.currentText().strip() or str(cfg.get("cmap") or "jet")
        if cmap_name.endswith("_r"):
            cmap_name = cmap_name[:-2]
            cfg["reverse"] = True
        cfg["cmap"] = cmap_name
        if self.color_limits is not None and str(cfg.get("color_mode", "Continuous")).strip() != "Custom Bands":
            try:
                cfg["manual_clim"] = True
                cfg["vmin"] = float(self.color_limits.get("vmin", cfg.get("vmin", 0.0)))
                cfg["vmax"] = float(self.color_limits.get("vmax", cfg.get("vmax", 1.0)))
            except Exception:
                pass
        return cfg

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

        use_log = self.chk_log.isChecked()
        is_db = str(self.state.get("scale", "")).strip().lower() == "db"
        color_cfg = self._effective_color_cfg()
        style = resolve_mpl_image_style(S, self.state.get("scale", "Linear"), color_cfg)

        S_to_display = S
        norm_to_use = style.get("norm")
        vmin = style.get("vmin")
        vmax = style.get("vmax")
        if style.get("mode") != "bands" and use_log and not is_db:
            vmin_plot = max(float(vmin), 1e-20)
            S_to_display = np.clip(S, vmin_plot, None)
            norm_to_use = mcolors.LogNorm(vmin=vmin_plot, vmax=float(vmax))
            vmin = None
            vmax = None
        elif norm_to_use is not None:
            vmin = None
            vmax = None

        mesh = ax.imshow(
            S_to_display,
            aspect="auto",
            origin="lower",
            extent=[float(freqs[0]), float(freqs[-1]), float(rpms[0]), float(rpms[-1])],
            interpolation="bilinear",
            cmap=style["cmap"],
            norm=norm_to_use,
            vmin=vmin,
            vmax=vmax,
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
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if not self.mini:
            self.setMinimumHeight(500)
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
        self.show_order_trace_panel = True
        self._last_waterfall_splitter_sizes = None
        self._waterfall_advanced_dialog = None
        self._embedded_toolbar_mode = False
        self.waterfall_display_settings = {
            "x_axis_name": "Frequency (Hz)",
            "y_axis_name": "RPM",
            "render_mode": "ImageItem",
            "colormap": "viridis",
            "reverse_colors": False,
            "color_mode": "Continuous",
            "custom_color_bands": default_custom_color_bands(),
            "dynamic_range_db": 80.0,
            "manual_color_range": False,
            "vmin": None,
            "vmax": None,
            "show_order_trace_panel": True,
        }
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
        self.btn_open_3d.setToolTip(
            "Open a 3D waterfall view after the 2D waterfall has been computed. "
            "Useful for presentation/shape inspection; use 2D heatmap for precise reading."
        )
        self.btn_open_3d_help = QToolButton()
        self.btn_open_3d_help.setText("?")
        self.btn_open_3d_help.setFixedWidth(22)
        self.btn_open_3d_help.setToolTip("Click for explanation of Open 3D waterfall")
        self.btn_open_3d_help.clicked.connect(lambda: QMessageBox.information(
            self, "Open 3D Waterfall — Explanation",
            "<b>Open 3D waterfall</b><br><br>Opens a standalone 3D view of the computed waterfall. "
            "It is helpful for presentation and visualizing surface shape, but the 2D heatmap is usually better for exact cursor and color readings.<br><br>"
            "<b>Tip:</b> use after parameters are finalized; repeated 3D windows can slow iterative tuning."
        ))
        self.btn_open_2d_standalone = QPushButton("Open 2D Standalone")
        self.btn_open_2d_help = QToolButton()
        self.btn_open_2d_help.setText("?")
        self.btn_open_2d_help.setFixedWidth(22)
        self.btn_open_2d_help.setToolTip("Click for explanation of Open 2D Standalone")
        self.btn_open_2d_help.clicked.connect(lambda: QMessageBox.information(
            self, "Open 2D Standalone — Explanation",
            "<b>Open 2D Standalone</b><br><br>Opens the waterfall heatmap in a separate Matplotlib window. "
            "The standalone view inherits the current colormap, color limits, zoom viewport, and order cursors when available.<br><br>"
            "<b>Use for:</b> publication-style export, comparing views, or keeping a waterfall visible while changing the main GUI."
        ))
        self.btn_open_2d_standalone.setToolTip(
            "Open waterfall heatmap in a standalone matplotlib window. "
            "Inherits current colormap, color range (vmin/vmax), "
            "zoom viewport, and order cursors from the main view."
        )
        self.edit_custom_order = QLineEdit()
        self.edit_custom_order.setPlaceholderText("Order(s): 0.5, 1/3, 2/3, 1.5")
        self.edit_custom_order.setFixedWidth(210)
        self.edit_custom_order.setToolTip(
            "Enter one or more visual order cursors. Separate multiple orders with "
            "commas, semicolons, newlines, tabs, or spaces; examples: 0.5x, 1/3, ord 0.5."
        )
        self.btn_add_custom_order = QPushButton("Add order")
        self.btn_add_custom_order.setToolTip(
            "Add custom order cursor(s), including subharmonics such as 0.5x, 1/3x, 2/3x."
        )
        self.btn_add_subharmonics = QPushButton("Add subharmonics")
        self.btn_add_subharmonics.setToolTip(
            "Add common subharmonic cursors: 1/4x, 1/3x, 1/2x, 2/3x, 3/2x."
        )
        self.chk_show_order_trace = QCheckBox("Order trace")
        self.chk_show_order_trace.setChecked(True)
        self.chk_show_order_trace.setToolTip("Show or hide the lower order trace panel.")
        self.btn_waterfall_advanced_settings = QPushButton("Advanced Settings")
        self.btn_waterfall_advanced_settings.setToolTip(
            "Open advanced display settings for the waterfall plot."
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
        topbar.addSpacing(10)
        topbar.addWidget(QLabel("Order:"))
        topbar.addWidget(self.edit_custom_order)
        topbar.addWidget(self.btn_add_custom_order)
        topbar.addWidget(self.btn_add_subharmonics)
        topbar.addWidget(self.chk_show_order_trace)
        topbar.addSpacing(10)
        topbar.addWidget(self.btn_open_3d)
        topbar.addWidget(self.btn_open_3d_help)
        topbar.addWidget(self.btn_open_2d_standalone)
        topbar.addWidget(self.btn_open_2d_help)
        topbar.addWidget(self.btn_waterfall_advanced_settings)
        topbar.addSpacing(10)
        topbar.addWidget(self.lbl_order_info, 1)
        topbar.addStretch(1)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if not self.mini:
            lay.addLayout(topbar)
            self.waterfall_splitter = QSplitter(Qt.Vertical)
            self.waterfall_splitter.addWidget(self.upper)
            self.waterfall_splitter.addWidget(self.lower)
            self.waterfall_splitter.setStretchFactor(0, 7)
            self.waterfall_splitter.setStretchFactor(1, 3)
            self.waterfall_splitter.setSizes([700, 300])
            lay.addWidget(self.waterfall_splitter, 1)
        else:
            self.waterfall_splitter = None
            lay.addWidget(self.upper, 1)
            self.lower.hide()

        self.btn_add_cursor.clicked.connect(self._add_cursor_default)
        self.btn_clear_cursors.clicked.connect(self.clear_cursors)
        self.btn_add_custom_order.clicked.connect(self._add_custom_orders_from_input)
        self.edit_custom_order.returnPressed.connect(self._add_custom_orders_from_input)
        self.btn_add_subharmonics.clicked.connect(self._add_common_subharmonics)
        self.chk_show_order_trace.toggled.connect(self.set_order_trace_visible)
        self.btn_waterfall_advanced_settings.clicked.connect(self._open_waterfall_advanced_settings)
        self.btn_open_3d.clicked.connect(self.open_3d_window)
        self.btn_open_2d_standalone.clicked.connect(self.open_2d_standalone_window)

        if self.mini:
            self.btn_add_cursor.hide()
            self.btn_clear_cursors.hide()
            self.edit_custom_order.hide()
            self.btn_add_custom_order.hide()
            self.btn_add_subharmonics.hide()
            self.chk_show_order_trace.hide()
            self.btn_waterfall_advanced_settings.hide()
            self.btn_open_3d.hide()
            self.btn_open_3d_help.hide()
            self.btn_open_2d_standalone.hide()
            self.btn_open_2d_help.hide()
            self.lbl_order_info.hide()
            self.lower.hide()

    def _get_order_trace_widget(self):
        """Return the lower order-trace widget without assuming a future name."""
        for name in ("lower", "lower_plot", "plot_lower"):
            widget = getattr(self, name, None)
            if widget is not None:
                return widget
        item = getattr(self, "lower_item", None)
        if item is not None:
            try:
                return item.getViewWidget()
            except Exception:
                return None
        return None

    def _sync_order_trace_controls(self):
        checked = bool(getattr(self, "show_order_trace_panel", True))
        for widget in (
            getattr(self, "chk_show_order_trace", None),
            getattr(getattr(self, "_waterfall_advanced_dialog", None), "chk_show_order_trace_panel", None),
        ):
            if widget is None:
                continue
            try:
                old = widget.blockSignals(True)
                widget.setChecked(checked)
                widget.blockSignals(old)
            except Exception:
                pass
        if hasattr(self, "waterfall_display_settings"):
            self.waterfall_display_settings["show_order_trace_panel"] = checked

    def set_order_trace_visible(self, visible: bool):
        visible = bool(visible)
        lower_widget = self._get_order_trace_widget()
        splitter = getattr(self, "waterfall_splitter", None)

        if not visible and splitter is not None:
            try:
                sizes = splitter.sizes()
                if sizes and len(sizes) >= 2 and sizes[-1] > 0:
                    self._last_waterfall_splitter_sizes = list(sizes)
            except Exception:
                pass

        self.show_order_trace_panel = visible
        if lower_widget is not None:
            lower_widget.setVisible(visible and not self.mini)

        if splitter is not None and not self.mini:
            try:
                if visible:
                    sizes = self._last_waterfall_splitter_sizes
                    if not sizes or len(sizes) < 2 or sizes[-1] <= 0:
                        total = max(2, sum(splitter.sizes()) or splitter.height() or 1000)
                        sizes = [int(total * 0.7), max(1, int(total * 0.3))]
                    splitter.setSizes(sizes)
                else:
                    sizes = splitter.sizes()
                    upper_size = max(1, sum(sizes) if sizes else splitter.height() or 1000)
                    splitter.setSizes([upper_size, 0])
            except Exception:
                pass

        self._sync_order_trace_controls()

    def set_embedded_toolbar_mode(self, embedded: bool):
        self._embedded_toolbar_mode = bool(embedded)
        for widget in (
            getattr(self, "btn_open_3d", None),
            getattr(self, "btn_open_3d_help", None),
            getattr(self, "btn_open_2d_standalone", None),
            getattr(self, "btn_open_2d_help", None),
            getattr(self, "btn_waterfall_advanced_settings", None),
        ):
            if widget is not None:
                widget.setVisible((not self._embedded_toolbar_mode) and (not self.mini))

    def _open_waterfall_advanced_settings(self):
        dlg = getattr(self, "_waterfall_advanced_dialog", None)
        if dlg is None or not _qt_alive(dlg):
            dlg = WaterfallAdvancedSettingsDialog(self)
            self._waterfall_advanced_dialog = dlg
        dlg.refresh_from_widget()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _waterfall_color_data(self):
        for arr in (getattr(self, "_display_S", None), getattr(self, "S", None), getattr(self, "S_full_display", None)):
            if arr is None:
                continue
            data = np.asarray(arr, dtype=np.float64)
            if data.size:
                valid = data[np.isfinite(data)]
                if valid.size:
                    return valid
        return np.array([], dtype=np.float64)

    def apply_waterfall_display_settings(self, settings: dict):
        settings = dict(settings or {})
        merged = dict(getattr(self, "waterfall_display_settings", {}))
        merged.update(settings)
        merged["show_order_trace_panel"] = bool(merged.get("show_order_trace_panel", True))
        merged["reverse_colors"] = bool(merged.get("reverse_colors", False))
        color_mode = str(merged.get("color_mode", "Continuous") or "Continuous").strip()
        if color_mode not in ("Continuous", "Custom Bands"):
            color_mode = "Custom Bands" if "Band" in color_mode else "Continuous"
        merged["color_mode"] = color_mode
        if color_mode == "Custom Bands":
            merged["manual_color_range"] = False
            try:
                merged["custom_color_bands"] = normalize_custom_bands(
                    merged.get("custom_color_bands", default_custom_color_bands())
                )
            except ValueError as exc:
                QMessageBox.warning(self, "Custom Bands", str(exc))
                return False
        else:
            merged["manual_color_range"] = bool(merged.get("manual_color_range", False))
        try:
            merged["dynamic_range_db"] = float(merged.get("dynamic_range_db", 80.0))
        except Exception:
            merged["dynamic_range_db"] = 80.0

        cmap_name = str(merged.get("colormap") or "viridis")
        reverse = bool(merged.get("reverse_colors", False))
        color_cfg = {
            "cmap": cmap_name,
            "reverse": reverse,
            "color_mode": color_mode,
            "manual_clim": bool(merged.get("manual_color_range", False)) and color_mode == "Continuous",
            "vmin": merged.get("vmin", 0.0),
            "vmax": merged.get("vmax", 1.0),
            "dyn_range_db": float(merged.get("dynamic_range_db", 80.0)),
            "custom_bands": merged.get("custom_color_bands", default_custom_color_bands()),
        }

        data = getattr(self, "S", None)
        if not (isinstance(data, np.ndarray) and data.size):
            data = getattr(self, "S_full_display", None)
        if not (isinstance(data, np.ndarray) and data.size):
            data = np.asarray([0.0, 1.0], dtype=np.float64)

        try:
            style = resolve_image_style(data, (self.state or {}).get("scale", "Linear"), color_cfg)
        except ValueError as exc:
            QMessageBox.warning(self, "Custom Bands", str(exc))
            return False

        merged["vmin"] = float(style["vmin"])
        merged["vmax"] = float(style["vmax"])
        self.waterfall_display_settings = merged
        self.show_order_trace_panel = bool(merged.get("show_order_trace_panel", True))

        x_label = str(merged.get("x_axis_name") or "Frequency (Hz)")
        y_label = str(merged.get("y_axis_name") or "RPM")
        try:
            self.upper_item.setLabel("bottom", x_label)
            self.upper_item.setLabel("left", y_label)
        except Exception:
            pass
        if self.state is not None:
            self.state["xlabel"] = x_label
            self.state["ylabel"] = y_label

        if self._upper_image_item is not None:
            try:
                self._upper_image_item.setColorMap(style["cm"])
            except Exception:
                pass
            try:
                self._upper_image_item.setLevels((style["vmin"], style["vmax"]))
            except Exception:
                pass
        if self._upper_color_bar is not None:
            self._suspend_colorbar_sync = True
            try:
                if hasattr(self._upper_color_bar, "setColorMap"):
                    self._upper_color_bar.setColorMap(style["cm"])
                self._upper_color_bar.setLevels((style["vmin"], style["vmax"]))
            except Exception:
                pass
            finally:
                self._suspend_colorbar_sync = False

        self._last_color_cfg.update({
            "cmap": cmap_name,
            "reverse": reverse,
            "color_mode": color_mode,
            "manual_clim": bool(merged.get("manual_color_range", False)) and color_mode == "Continuous",
            "vmin": float(style["vmin"]),
            "vmax": float(style["vmax"]),
            "dyn_range_db": float(merged.get("dynamic_range_db", 80.0)),
            "custom_bands": list(merged.get("custom_color_bands", default_custom_color_bands())),
        })
        self.set_order_trace_visible(self.show_order_trace_panel)
        return True

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
        return convert_linear_to_db(S_linear, output_type)

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

            if DEBUG:
                print(
                    f"[{tag}] x=({xr[0]:.3f},{xr[1]:.3f}) w={xw:.3f} | "
                    f"y=({yr[0]:.3f},{yr[1]:.3f}) h={yw:.3f} | "
                    f"current_S={current_shape} | display_S={disp_shape} | "
                    f"full_S={full_shape} | analysis_S={analysis_shape} | "
                    f"px_per_freq_bin~{xw / float(ncols):.6f}"
                )
        except Exception as e:
            if DEBUG:
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

            rpms_axis = self.rpms_trace if self.rpms_trace.size else self.rpms
            m = np.isfinite(tr) & np.isfinite(rpms_axis)
            if np.count_nonzero(m) < 2:
                continue

            xr = rpms_axis[m]
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
        for item in (getattr(self, "upper_item", None), None if self.mini else getattr(self, "lower_item", None)):
            if item is None:
                continue
            try:
                vb = item.getViewBox()
            except RuntimeError:
                continue
            except Exception:
                continue
            if not _qt_alive(vb):
                continue
            try:
                vb.setMouseMode(mode)
            except RuntimeError:
                continue

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
        try:
            order_value = float(order_value)
        except Exception:
            return
        if not np.isfinite(order_value) or order_value <= 0.0:
            return
        color = self._cursor_colors[len(self._cursors) % len(self._cursor_colors)]
        cursor = WaterfallOrderCursor(self, order_value, color)
        cursor.changed.connect(self._update_lower_plot)
        self._cursors.append(cursor)
        self._update_lower_plot()

    def _parse_order_token(self, token: str):
        token = str(token).strip().lower()
        if not token:
            return None
        token = token.replace("order", "").replace("ord", "")
        token = token.replace("×", "").replace("x", "").strip()
        if not token:
            return None
        try:
            if token.count("/") == 1:
                num_txt, den_txt = token.split("/", 1)
                num = float(num_txt.strip())
                den = float(den_txt.strip())
                if den == 0.0:
                    return None
                value = num / den
            elif "/" in token:
                return None
            else:
                value = float(token)
        except Exception:
            return None
        if not np.isfinite(value) or value <= 0.0:
            return None
        return float(value)

    def _parse_custom_orders(self, text: str):
        cleaned = str(text)
        for sep in (",", ";", "\n", "\t"):
            cleaned = cleaned.replace(sep, " ")
        orders = []
        for token in cleaned.split():
            value = self._parse_order_token(token)
            if value is None:
                continue
            if not any(abs(value - existing) <= 1e-9 for existing in orders):
                orders.append(value)
        return orders

    def _order_already_exists(self, order_value: float, tol: float = 1e-6):
        try:
            order_value = float(order_value)
        except Exception:
            return False
        return any(abs(float(c.order) - order_value) <= float(tol) for c in self._cursors)

    def _order_is_visible_somewhere(self, order_value: float):
        try:
            order_value = float(order_value)
        except Exception:
            return False
        if not np.isfinite(order_value) or order_value <= 0.0:
            return False
        rpms = np.asarray(self.rpms, dtype=np.float64)
        freqs = np.asarray(self.freqs, dtype=np.float64)
        rpms = rpms[np.isfinite(rpms)]
        freqs = freqs[np.isfinite(freqs)]
        if rpms.size == 0 or freqs.size == 0:
            return False
        order_freqs = order_value * rpms / 60.0
        order_freqs = order_freqs[np.isfinite(order_freqs)]
        if order_freqs.size == 0:
            return False
        f_min = float(np.nanmin(freqs))
        f_max = float(np.nanmax(freqs))
        o_min = float(np.nanmin(order_freqs))
        o_max = float(np.nanmax(order_freqs))
        return bool(o_max >= f_min and o_min <= f_max)

    def _add_custom_orders_from_input(self):
        text = self.edit_custom_order.text()
        orders = self._parse_custom_orders(text)
        if not orders:
            QMessageBox.warning(
                self, "Add order",
                "Enter one or more positive orders, for example: 0.5, 0.5x, 1/3, 2/3, 1.5, order 0.5."
            )
            return

        skipped = []
        added = 0
        for order in orders:
            label = f"{order:.6g}x"
            if self._order_already_exists(order):
                skipped.append(f"{label}: already active")
                continue
            if not self._order_is_visible_somewhere(order):
                skipped.append(f"{label}: outside current waterfall frequency/RPM range")
                continue
            before = len(self._cursors)
            self.add_cursor(order)
            if len(self._cursors) > before:
                added += 1

        self.edit_custom_order.clear()
        if skipped:
            QMessageBox.information(
                self, "Add order",
                "Some order cursors were skipped:\n" + "\n".join(skipped)
            )

    def _add_common_subharmonics(self):
        presets = [0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.5]
        added = 0
        for order in presets:
            if self._order_already_exists(order) or not self._order_is_visible_somewhere(order):
                continue
            before = len(self._cursors)
            self.add_cursor(order)
            if len(self._cursors) > before:
                added += 1
        if added == 0:
            QMessageBox.information(
                self, "Add subharmonics",
                "No new subharmonic cursor was added. They may already exist or be outside the frequency range."
            )

    def export_order_cursor_specs(self):
        out = []
        for c in self._cursors:
            out.append({"order": float(c.order), "color": c.color})
        return out

    def _extract_order_trace(self, order_value: float):
        """Extract order trace using configured interpolation/peak/RMS-band mode.

        Quantitative extraction uses frame_spectra_linear on freqs_analysis when
        available; display-grid data is only a fallback. Linear spectra are used
        first and converted to dB after extraction when needed.
        """
        state = self.state or {}
        target_rpms = self.rpms_trace if self.rpms_trace.size else self.rpms
        if target_rpms.size == 0:
            return np.array([], dtype=np.float64)
        use_raw = (
            self.frame_rpms.size > 0 and self.frame_spectra_linear.ndim == 2
            and self.frame_spectra_linear.shape[0] == self.frame_rpms.size
            and self.freqs_analysis.size > 0
        )
        if use_raw:
            rpms_src = self.frame_rpms
            freqs_src = self.freqs_analysis
            S_lin = self.frame_spectra_linear
        else:
            rpms_src = self.rpms_trace if self.rpms_trace.size else self.rpms
            freqs_src = self.freqs_trace if self.freqs_trace.size else self.freqs
            S_src = self.S_trace if self.S_trace.size else self.S
            scale0 = str(state.get("scale", "Linear")).lower()
            out0 = normalize_output_type(state.get("out_type", "Amplitude"))
            if scale0 == "db":
                fac = spectrum_db_factor(out0)
                S_lin = np.power(10.0, np.asarray(S_src, dtype=np.float64) / fac)
            else:
                S_lin = np.asarray(S_src, dtype=np.float64)
        if S_lin.size == 0 or rpms_src.size == 0 or freqs_src.size == 0:
            return np.full(target_rpms.shape, np.nan, dtype=np.float64)
        mode = str(state.get("order_extract_mode", state.get("waterfall_order_extract_mode", "Interpolated"))).strip()
        bw = float(state.get("order_bw", state.get("waterfall_order_bw", 0.0)) or 0.0)
        bw_unit = str(state.get("order_bw_unit", state.get("waterfall_order_bw_unit", "Hz"))).strip()
        out_type = normalize_output_type(state.get("out_type", "Amplitude"))
        values = np.full(rpms_src.shape, np.nan, dtype=np.float64)
        fmin = float(freqs_src[0]); fmax = float(freqs_src[-1])
        df = float(np.nanmedian(np.diff(freqs_src))) if freqs_src.size > 1 else 1.0
        for i, rpm_i in enumerate(rpms_src):
            fc = float(order_value) * float(rpm_i) / 60.0
            if not np.isfinite(fc) or fc < fmin or fc > fmax:
                continue
            if mode == "Peak in band" and bw > 0.0:
                hb = bw * rpm_i / 60.0 if bw_unit.lower().startswith("order") else bw
                m = (freqs_src >= fc - hb) & (freqs_src <= fc + hb)
                if np.any(m):
                    values[i] = float(np.nanmax(S_lin[i, m]))
            elif mode == "RMS band power" and bw > 0.0:
                hb = bw * rpm_i / 60.0 if bw_unit.lower().startswith("order") else bw
                m = (freqs_src >= fc - hb) & (freqs_src <= fc + hb)
                if np.any(m):
                    band = np.maximum(S_lin[i, m], 0.0)
                    if out_type == "PSD":
                        values[i] = float(np.sum(band) * df)
                    elif out_type == "ASD":
                        values[i] = float(np.sqrt(np.sum(band * band) * df))
                    elif out_type == "Power":
                        values[i] = float(np.sum(band))
                    else:
                        # Amplitude bins are converted to RMS² approximately,
                        # summed, then converted back to peak amplitude.
                        values[i] = float(np.sqrt(np.sum((band / np.sqrt(2.0)) ** 2)) * np.sqrt(2.0))
            else:
                values[i] = float(np.interp(fc, freqs_src, S_lin[i, :], left=np.nan, right=np.nan))
        if str(state.get("scale", "Linear")).strip().lower() == "db":
            values = convert_linear_to_db(values, out_type)
        if rpms_src.size != target_rpms.size or not np.array_equal(rpms_src, target_rpms):
            finite = np.isfinite(values) & np.isfinite(rpms_src)
            if np.count_nonzero(finite) < 2:
                return np.full(target_rpms.shape, np.nan, dtype=np.float64)
            x_in = rpms_src[finite]; y_in = values[finite]
            sort_idx = np.argsort(x_in); x_in = x_in[sort_idx]; y_in = y_in[sort_idx]
            if np.any(np.diff(x_in) == 0):
                uniq_x, inv = np.unique(x_in, return_inverse=True)
                sums = np.bincount(inv, weights=y_in, minlength=uniq_x.size)
                counts = np.bincount(inv, minlength=uniq_x.size).astype(np.float64); counts[counts == 0] = 1.0
                x_in = uniq_x; y_in = sums / counts
            return np.interp(target_rpms, x_in, y_in, left=np.nan, right=np.nan)
        return values


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
        if self.state:
            mode = self.state.get("order_extract_mode", "Interpolated")
            bw = float(self.state.get("order_bw", 0.0) or 0.0)
            unit = self.state.get("order_bw_unit", "Hz")
            base_label = self.state.get("colorbar_label", "Amplitude")
            if mode == "Peak in band" and bw > 0:
                lower_label = f"Peak in ±{bw:g} {unit} band — {base_label}"
            elif mode == "RMS band power" and bw > 0:
                lower_label = f"RMS band power ±{bw:g} {unit} — {base_label}"
            else:
                lower_label = f"Interpolated order trace — {base_label}"
        else:
            lower_label = "Amplitude"
        self.lower_item.setLabel("left", lower_label)

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
            color_cfg=dict(self._last_color_cfg or {}),
        )
        dlg.exec_()

    def set_state(self, state: dict, color_cfg=None, preserve_view=False, preserve_annotations=False):
        color_cfg = color_cfg or {}
        self._last_color_cfg = dict(color_cfg)
        if hasattr(self, "waterfall_display_settings"):
            self.waterfall_display_settings.update({
                "x_axis_name": state.get("xlabel", self.waterfall_display_settings.get("x_axis_name", "Frequency (Hz)")),
                "y_axis_name": state.get("ylabel", self.waterfall_display_settings.get("y_axis_name", "RPM")),
                "colormap": color_cfg.get("cmap", self.waterfall_display_settings.get("colormap", "viridis")),
                "reverse_colors": bool(color_cfg.get("reverse", self.waterfall_display_settings.get("reverse_colors", False))),
                "color_mode": color_cfg.get("color_mode", self.waterfall_display_settings.get("color_mode", "Continuous")),
                "custom_color_bands": color_cfg.get("custom_bands", self.waterfall_display_settings.get("custom_color_bands", default_custom_color_bands())),
                "dynamic_range_db": float(color_cfg.get("dyn_range_db", self.waterfall_display_settings.get("dynamic_range_db", 80.0))),
                "manual_color_range": bool(color_cfg.get("manual_clim", self.waterfall_display_settings.get("manual_color_range", False))),
                "vmin": color_cfg.get("vmin", self.waterfall_display_settings.get("vmin")),
                "vmax": color_cfg.get("vmax", self.waterfall_display_settings.get("vmax")),
                "show_order_trace_panel": bool(getattr(self, "show_order_trace_panel", True)),
            })

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
        self.apply_waterfall_display_settings(self.waterfall_display_settings)

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
        for canvas in (getattr(self, "upper", None), getattr(self, "lower", None)):
            if canvas is None or not _qt_alive(canvas):
                continue
            try:
                canvas.set_rect_zoom_active(active)
            except RuntimeError:
                continue

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
    request_large = pyqtSignal(str)
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

        if str(state.get("kind", "")).lower() in ("kurtogram", "envelope_spectrum") and EnvelopeKurtogramWidget is not None:
            self.canvas = EnvelopeKurtogramWidget(mini=True)
        else:
            self.canvas = InteractivePlotWidget(mini=True)
        self.canvas.setMinimumHeight(160)

        top = QHBoxLayout()
        top.addWidget(self.lbl, 1)
        top.addWidget(self.btn_open)
        top.addWidget(self.btn_close)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addLayout(top)
        lay.addWidget(self.canvas)

        self.btn_open.clicked.connect(lambda: self.request_large.emit(self.plot_key))
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

        # Use a plain InteractivePlotWidget for normal non-waterfall kinds so
        # that the time-domain detail shows ONLY the time-domain trace.
        # Envelope/Kurtogram states use their dedicated modular widget.
        if str(state.get("kind", "")).lower() in ("kurtogram", "envelope_spectrum") and EnvelopeKurtogramWidget is not None:
            self.canvas = EnvelopeKurtogramWidget(mini=False)
        else:
            self.canvas = InteractivePlotWidget(mini=False)
        self.canvas.setMinimumHeight(360)

        lay = QVBoxLayout(self)
        lay.addWidget(self.title)
        lay.addWidget(self.canvas, 1)

        self.canvas.clicked.connect(self._emit_active)

    def _emit_active(self):
        self.activated.emit(self.canvas, self.plot_key)


class WaterfallAdvancedSettingsDialog(QDialog):
    """Advanced display-only controls for the Waterfall View."""

    _COLORMAPS = ("viridis", "turbo", "jet", "plasma", "inferno", "magma", "cividis", "gray")

    def __init__(self, waterfall_widget, parent=None):
        super().__init__(parent or waterfall_widget)
        self.waterfall_widget = waterfall_widget
        self.setWindowTitle("Waterfall View — Advanced Settings")
        self.setModal(False)
        self.setMinimumWidth(460)

        self.edit_x_axis_name = QLineEdit()
        self.edit_y_axis_name = QLineEdit()

        self.cmb_render_mode = QComboBox()
        self.cmb_render_mode.addItems(["ImageItem", "PColorMesh"])
        pcolor_idx = self.cmb_render_mode.findText("PColorMesh")
        if pcolor_idx >= 0:
            self.cmb_render_mode.model().item(pcolor_idx).setEnabled(False)
        self.cmb_render_mode.setToolTip("PColorMesh is reserved for a future renderer; ImageItem remains active.")

        self.cmb_colormap = QComboBox()
        for cmap_name in self._COLORMAPS:
            try:
                self.cmb_colormap.addItem(make_colormap_preview_icon(cmap_name), cmap_name)
            except Exception:
                self.cmb_colormap.addItem(cmap_name)

        self.chk_reverse_colors = QCheckBox("Reverse colors")

        self.cmb_color_mode = QComboBox()
        self.cmb_color_mode.addItems(["Continuous", "Custom Bands"])
        self.btn_edit_bands = QPushButton("Edit Bands...")

        self.spin_dynamic_range_db = QDoubleSpinBox()
        self.spin_dynamic_range_db.setRange(1.0, 300.0)
        self.spin_dynamic_range_db.setDecimals(1)
        self.spin_dynamic_range_db.setSingleStep(1.0)

        self.chk_manual_color_range = QCheckBox("Manual color range (vmin/vmax)")
        self.spin_vmin = QDoubleSpinBox()
        self.spin_vmin.setRange(-1.0e12, 1.0e12)
        self.spin_vmin.setDecimals(6)
        self.spin_vmax = QDoubleSpinBox()
        self.spin_vmax.setRange(-1.0e12, 1.0e12)
        self.spin_vmax.setDecimals(6)

        self.chk_show_order_trace_panel = QCheckBox("Show order trace panel")
        self.chk_show_order_trace_panel.setToolTip("Show or hide the lower order trace panel.")

        self.btn_apply_plot_view = QPushButton("Apply Plot View")
        self.btn_close = QPushButton("Close")

        band_row = QHBoxLayout()
        band_row.addWidget(self.cmb_color_mode, 1)
        band_row.addWidget(self.btn_edit_bands)

        form = QFormLayout()
        form.addRow("X Axis Name", self.edit_x_axis_name)
        form.addRow("Y Axis Name", self.edit_y_axis_name)
        form.addRow("Spectrogram render", self.cmb_render_mode)
        form.addRow("Colormap", self.cmb_colormap)
        form.addRow("", self.chk_reverse_colors)
        form.addRow("Color Mode", band_row)
        form.addRow("Dynamic Range (dB):", self.spin_dynamic_range_db)
        form.addRow("", self.chk_manual_color_range)
        form.addRow("vmin", self.spin_vmin)
        form.addRow("vmax", self.spin_vmax)
        form.addRow("", self.chk_show_order_trace_panel)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.btn_apply_plot_view)
        buttons.addWidget(self.btn_close)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addLayout(buttons)

        self.chk_manual_color_range.toggled.connect(self._sync_manual_enabled)
        self.cmb_color_mode.currentTextChanged.connect(self._on_color_mode_changed)
        self.btn_edit_bands.clicked.connect(self._edit_bands)
        self.chk_show_order_trace_panel.toggled.connect(self._on_order_trace_toggled)
        self.btn_apply_plot_view.clicked.connect(self.apply_to_plot)
        self.btn_close.clicked.connect(self.close)
        self.refresh_from_widget()

    def _sync_manual_enabled(self):
        custom_mode = self.cmb_color_mode.currentText().strip() == "Custom Bands"
        if custom_mode and self.chk_manual_color_range.isChecked():
            self.chk_manual_color_range.blockSignals(True)
            self.chk_manual_color_range.setChecked(False)
            self.chk_manual_color_range.blockSignals(False)
        manual = self.chk_manual_color_range.isChecked() and not custom_mode
        self.chk_manual_color_range.setEnabled(not custom_mode)
        self.spin_vmin.setEnabled(manual)
        self.spin_vmax.setEnabled(manual)
        self.spin_dynamic_range_db.setEnabled((not manual) and (not custom_mode))
        self.cmb_colormap.setEnabled(not custom_mode)
        self.chk_reverse_colors.setEnabled(not custom_mode)
        self.btn_edit_bands.setEnabled(custom_mode)

    def _on_color_mode_changed(self, *_):
        self._sync_manual_enabled()

    def _edit_bands(self):
        wf = self.waterfall_widget
        current = getattr(wf, "waterfall_display_settings", {}).get("custom_color_bands", default_custom_color_bands())
        dlg = BandEditorDialog(current, self)
        if dlg.exec_() == QDialog.Accepted:
            settings = dict(getattr(wf, "waterfall_display_settings", {}) or {})
            settings["custom_color_bands"] = dlg.bands()
            wf.waterfall_display_settings.update(settings)
            self.apply_to_plot()

    def refresh_from_widget(self):
        wf = self.waterfall_widget
        settings = dict(getattr(wf, "waterfall_display_settings", {}))
        if getattr(wf, "state", None) is not None:
            settings.setdefault("x_axis_name", wf.state.get("xlabel", "Frequency (Hz)"))
            settings.setdefault("y_axis_name", wf.state.get("ylabel", "RPM"))
        settings.setdefault("x_axis_name", "Frequency (Hz)")
        settings.setdefault("y_axis_name", "RPM")
        settings.setdefault("render_mode", "ImageItem")
        settings.setdefault("colormap", "viridis")
        settings.setdefault("reverse_colors", False)
        settings.setdefault("color_mode", "Continuous")
        settings.setdefault("custom_color_bands", default_custom_color_bands())
        settings.setdefault("dynamic_range_db", 80.0)
        settings.setdefault("manual_color_range", False)
        settings.setdefault("vmin", 0.0)
        settings.setdefault("vmax", 1.0)
        settings.setdefault("show_order_trace_panel", getattr(wf, "show_order_trace_panel", True))

        widgets = [
            self.edit_x_axis_name, self.edit_y_axis_name, self.cmb_render_mode,
            self.cmb_colormap, self.chk_reverse_colors, self.cmb_color_mode,
            self.spin_dynamic_range_db, self.chk_manual_color_range, self.spin_vmin,
            self.spin_vmax, self.chk_show_order_trace_panel,
        ]
        old = []
        for widget in widgets:
            try:
                old.append((widget, widget.blockSignals(True)))
            except Exception:
                pass
        try:
            self.edit_x_axis_name.setText(str(settings.get("x_axis_name") or "Frequency (Hz)"))
            self.edit_y_axis_name.setText(str(settings.get("y_axis_name") or "RPM"))
            self.cmb_render_mode.setCurrentText("ImageItem")
            self.cmb_colormap.setCurrentText(str(settings.get("colormap") or "viridis"))
            self.chk_reverse_colors.setChecked(bool(settings.get("reverse_colors", False)))
            self.cmb_color_mode.setCurrentText("Custom Bands" if str(settings.get("color_mode", "Continuous")).strip() == "Custom Bands" else "Continuous")
            self.spin_dynamic_range_db.setValue(float(settings.get("dynamic_range_db", 80.0)))
            self.chk_manual_color_range.setChecked(bool(settings.get("manual_color_range", False)))
            vmin = settings.get("vmin", 0.0)
            vmax = settings.get("vmax", 1.0)
            self.spin_vmin.setValue(float(vmin) if vmin is not None and np.isfinite(float(vmin)) else 0.0)
            self.spin_vmax.setValue(float(vmax) if vmax is not None and np.isfinite(float(vmax)) else 1.0)
            self.chk_show_order_trace_panel.setChecked(bool(settings.get("show_order_trace_panel", True)))
        finally:
            for widget, state in old:
                try:
                    widget.blockSignals(state)
                except Exception:
                    pass
        self._sync_manual_enabled()

    def _collect_settings(self):
        wf_settings = dict(getattr(self.waterfall_widget, "waterfall_display_settings", {}) or {})
        color_mode = self.cmb_color_mode.currentText().strip() or "Continuous"
        return {
            "x_axis_name": self.edit_x_axis_name.text().strip() or "Frequency (Hz)",
            "y_axis_name": self.edit_y_axis_name.text().strip() or "RPM",
            "render_mode": "ImageItem",
            "colormap": self.cmb_colormap.currentText() or "viridis",
            "reverse_colors": self.chk_reverse_colors.isChecked(),
            "color_mode": color_mode,
            "custom_color_bands": wf_settings.get("custom_color_bands", default_custom_color_bands()),
            "dynamic_range_db": float(self.spin_dynamic_range_db.value()),
            "manual_color_range": self.chk_manual_color_range.isChecked() and color_mode == "Continuous",
            "vmin": float(self.spin_vmin.value()),
            "vmax": float(self.spin_vmax.value()),
            "show_order_trace_panel": self.chk_show_order_trace_panel.isChecked(),
        }

    def _on_order_trace_toggled(self, checked):
        wf = self.waterfall_widget
        if hasattr(wf, "set_order_trace_visible"):
            wf.set_order_trace_visible(bool(checked))

    def apply_to_plot(self):
        wf = self.waterfall_widget
        if not hasattr(wf, "apply_waterfall_display_settings"):
            return
        ok = wf.apply_waterfall_display_settings(self._collect_settings())
        if ok:
            self.refresh_from_widget()


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
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(500)
        self.canvas.clicked.connect(self._emit_active)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        lay = QVBoxLayout(self)
        lay.addWidget(self.title)
        lay.addWidget(self.canvas, 1)

    def _emit_active(self):
        self.activated.emit(self.canvas, self.plot_key)


class WaterfallMiniWidget(QFrame):
    """Mini preview card for waterfall states.

    Mirrors the public API of PlotMiniWidget (same signals, same buttons) but
    its canvas is a WaterfallPlotWidget(mini=True) so that waterfall state
    dicts — which carry rpms/frame_rpms instead of times — are rendered with
    the correct widget.  Required because feeding a waterfall state into
    InteractivePlotWidget.set_state crashes with KeyError: 'times'.
    """
    request_open = pyqtSignal(str)
    request_large = pyqtSignal(str)
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

        self.canvas = WaterfallPlotWidget(mini=True)
        self.canvas.setMinimumHeight(160)

        top = QHBoxLayout()
        top.addWidget(self.lbl, 1)
        top.addWidget(self.btn_open)
        top.addWidget(self.btn_close)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addLayout(top)
        lay.addWidget(self.canvas)

        self.btn_open.clicked.connect(lambda: self.request_large.emit(self.plot_key))
        self.btn_close.clicked.connect(lambda: self.request_close.emit(self.plot_key))
        self.canvas.clicked.connect(self._on_canvas_clicked)

    def _on_canvas_clicked(self):
        self.activated.emit(self.canvas, self.plot_key)
        self.request_open.emit(self.plot_key)

    def set_selected(self, selected: bool):
        if selected:
            self.setStyleSheet("QFrame { border: 2px solid #D71920; }")
        else:
            self.setStyleSheet("")


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
        hp = float(self.cfg.get("velocity_highpass_hz", 10.0))
        if hp <= 0.0:
            return np.asarray(acc_g, dtype=np.float64)
        return self._apply_custom_butterworth_zero_phase(acc_g, fs, "highpass", 4, hp, None)

    def _velocity_detrend(self, y: np.ndarray) -> np.ndarray:
        mode = str(self.cfg.get("velocity_detrend", "Remove mean")).strip()
        y = np.asarray(y, dtype=np.float64)
        if mode == "None":
            return y
        if mode == "Linear detrend" and _HAVE_SCIPY:
            return scipy_detrend(y, type="linear")
        return y - float(np.mean(y)) if y.size else y

    def _integrate_acc_to_velocity(self, acc_g: np.ndarray, fs: float) -> np.ndarray:
        """Acceleration(g) -> velocity(mm/s) with selectable integration method."""
        acc_g = np.asarray(acc_g, dtype=np.float64)
        if acc_g.size == 0:
            return np.array([], dtype=np.float64)
        fs = float(fs)
        method = str(self.cfg.get("velocity_integration_method", "Time-domain trapezoidal"))
        hp = float(self.cfg.get("velocity_highpass_hz", 10.0))
        acc_si = self._velocity_detrend(np.nan_to_num(acc_g, nan=0.0, posinf=0.0, neginf=0.0)) * G0
        if method == "Frequency-domain integration":
            n = acc_si.size
            A = np.fft.rfft(acc_si)
            freqs = np.fft.rfftfreq(n, d=1.0 / fs)
            V = np.zeros_like(A, dtype=np.complex128)
            valid = freqs > max(hp, 0.0)
            V[valid] = A[valid] / (1j * 2.0 * np.pi * freqs[valid])
            v = np.fft.irfft(V, n=n)
            return v * VEL_SCALE
        # Backward compatible path: high-pass externally, then trapezoidal.
        dt = 1.0 / fs
        v = np.zeros_like(acc_si, dtype=np.float64)
        if acc_si.size > 1:
            v[1:] = np.cumsum(0.5 * (acc_si[:-1] + acc_si[1:]) * dt)
        v = self._velocity_detrend(v)
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
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

        starts = np.arange(n_frames, dtype=np.int64) * hop
        starts = starts[starts + nperseg <= y.size]
        if starts.size == 0:
            return freqs, np.array([], dtype=np.float64), np.empty((freqs.size, 0))

        spectra_rows = compute_spectrum_frames_vectorized(
            y, starts, nperseg, fs, nfft,
            amp_conv=self.cfg.get("amp_conv", "Peak"),
            signal_kind=signal_kind, output_type=output_type,
            window_kind=window_kind,
        )
        S = np.asarray(spectra_rows, dtype=np.float64).T

        # Center-of-frame timestamps (matches MATLAB/nCode and the
        # window-weighted RPM labels used by _compute_waterfall_map).
        times = (starts + (nperseg - 1) / 2.0) / fs

        if str(scale).strip().lower() == "db":
            S = convert_linear_to_db(S, output_type)

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
        return convert_linear_to_db(S_linear, output_type)

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
        """Waterfall FFT map. Vectorized framing/FFT, selectable Hann kind.

        RPM NaN/<=0/Inf values are treated as invalid frames and excluded
        from the output (they are NOT converted to 0 RPM).  Acceleration NaN
        is zero-filled so it does not corrupt the FFT arithmetic.
        """
        acc = np.nan_to_num(np.asarray(acc, dtype=np.float64),
                            nan=0.0, posinf=0.0, neginf=0.0)
        # Keep RPM as-is; invalid samples are filtered at frame-labelling stage.
        rpm = np.asarray(rpm, dtype=np.float64)
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
        spectra_per_frame = compute_spectrum_frames_vectorized(
            acc, starts, nperseg, fs, nfft,
            amp_conv=self.cfg.get("amp_conv", "Peak"), signal_kind="acc",
            output_type=output_type, window_kind=window_kind,
        )

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

        # Preserve the raw STFT analysis frames in original time order for future
        # physical/event tracking. The plotted waterfall intentionally uses the
        # existing RPM-domain order below for run-up/down visualization.
        raw_frame_rpms_time = frame_rpms.copy()
        raw_frame_spectra_linear_time = spectra.copy()

        order = np.argsort(frame_rpms)
        frame_rpms = frame_rpms[order]
        spectra = spectra[order]
        n_valid = int(frame_rpms.size)

        num_buffers = max(1, min(int(num_buffers), n_valid))
        if num_buffers > 1:
            # Number of buffers is RPM-neighbor averaging on the sorted raw STFT
            # frame grid. It smooths adjacent physical spectra in RPM space; it
            # does not change df, nfft, or the effective FFT window length.
            half = num_buffers // 2
            csum = np.vstack([
                np.zeros((1, spectra.shape[1]), dtype=np.float64),
                np.cumsum(spectra, axis=0, dtype=np.float64),
            ])
            i0 = np.maximum(0, np.arange(n_valid) - half)
            i1 = np.minimum(n_valid, np.arange(n_valid) + half + 1)
            counts = np.maximum(1, i1 - i0).astype(np.float64)
            spectra_smoothed = (csum[i1, :] - csum[i0, :]) / counts[:, None]
        else:
            spectra_smoothed = spectra

        # Number of spectra controls the RPM analysis/display slice count.
        # It resamples already-computed FFT frames and therefore cannot improve
        # true frequency resolution; df remains fs/nfft (or the effective FFT
        # window length). Higher overlap increases frame density, not independent
        # frequency-bin resolution.
        min_rpm = float(frame_rpms[0])
        max_rpm = float(frame_rpms[-1])
        if min_rpm == max_rpm:
            target_rpms = np.array([min_rpm], dtype=np.float64)
            S_analysis_linear = spectra_smoothed[:1, :]
        else:
            target_count = max(1, int(num_spectra))
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
            "raw_frame_rpms_time": raw_frame_rpms_time,
            "raw_frame_spectra_linear_time": raw_frame_spectra_linear_time,
            "rpms_display": rpms_display if rpms_display.size else target_rpms,
            "S_display": S_display if S_display.size else S_analysis,
            "raw_frame_count": n_valid,
            "analysis_spectrum_count": int(target_rpms.size),
            "buffers_per_spectrum": int(num_buffers),
            "analysis_interpolated": bool(target_rpms.size != n_valid or not np.all(np.diff(frame_rpms) > 0.0)),
        }

    def _processing_title(self, use_filtered: bool, signal_kind: str) -> str:
        if signal_kind == "vel":
            if use_filtered:
                btype, order, f1, f2 = self._filter_params()
                if btype in ("bandpass", "bandstop"):
                    return f" | AccBase=Filtered | Butterworth {btype} | order={order} | fc=[{f1:g}, {f2:g}] Hz | zero-phase | AutoVel: configurable HP | order=4 | zero-phase | integrate"
                return f" | AccBase=Filtered | Butterworth {btype} | order={order} | fc={f1:g} Hz | zero-phase | AutoVel: configurable HP | order=4 | zero-phase | integrate"
            return " | AccBase=Unfiltered | AutoVel: configurable HP | order=4 | zero-phase | integrate"

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

        otf_time_info = None
        otf_mode = str(self.cfg.get("waterfall_otf_mode", "Frequency-domain order mask" if self.cfg.get("waterfall_otf_enabled", False) else "Off"))
        if otf_mode.startswith("Time-domain") and bool(self.cfg.get("waterfall_otf_enabled", False)):
            acc, otf_time_info = apply_time_domain_order_tracking_filter(
                acc, rpm, wf_fs,
                target_order=float(self.cfg.get("waterfall_otf_order", 1.0)),
                half_bw=float(self.cfg.get("waterfall_otf_bw", 0.25)),
                bw_unit=str(self.cfg.get("waterfall_otf_bw_unit", "Order")),
                filter_type=str(self.cfg.get("waterfall_otf_type", "Band-pass")),
                filter_order=int(self.cfg.get("waterfall_otf_filter_order", 4)),
                block_seconds=float(self.cfg.get("waterfall_otf_block_seconds", 0.5)),
                crossfade=True,
            )

        num_spectra = max(1, int(self.cfg.get("waterfall_num_spectra", 100)))
        num_buffers = max(1, int(self.cfg.get("waterfall_num_buffers", 1)))
        wf_domain = str(self.cfg.get("waterfall_domain", "Frequency Waterfall")).strip()

        wf_map = self._compute_waterfall_map(
            acc, rpm, wf_fs, nfft, overlap_pct, out_type, scale,
            num_spectra, num_buffers, rpm_label_method=rpm_label_method
        )

        # ── Order filter / mask ──────────────────────────
        # Mutates wf_map in place so the downstream frequency-domain or
        # order-domain waterfall reflects only the isolated order.
        otf_info = self._maybe_apply_order_tracking_filter(wf_map, out_type, scale)
        if otf_time_info is not None:
            otf_info = otf_time_info

        _, unit_lbl = self._amp_conv_factor_label("acc")
        colorbar_label = build_spectrum_ylabel(out_type, scale, unit_lbl)

        # ── Order-domain waterfall path ─────────────────────────────────
        if wf_domain == "Order Waterfall":
            order_max = float(self.cfg.get("waterfall_order_max", 20.0))
            n_orders = int(self.cfg.get("waterfall_order_bins", 512))
            order_axis, sorted_rpms, S_order = compute_order_map(
                wf_map["freqs_analysis"],
                wf_map["frame_rpms"],
                wf_map["frame_spectra_linear"],
                order_max=order_max,
                n_orders=n_orders,
                output_type=out_type,
                scale=scale,
            )
            key = (
                f"waterfall_order|{acc_file_base}|{acc_group}|{acc_name}"
                f"|{rpm_file_base}|{rpm_group}|{rpm_name}"
            )
            otf_tag = ""
            if otf_info:
                otf_tag = (
                    f" | OTF={otf_info['target_order']:.2f}x"
                    f" ±{otf_info['half_bw']:g}{otf_info['bw_unit']}"
                    f" ({otf_info['filter_type']})"
                )
            return {
                "key": key,
                "kind": "waterfall",
                "name": acc_name,
                "short_title": f"{acc_name} | Order Waterfall" + (otf_tag if otf_info else ""),
                "title": (
                    f"{acc_file_base} | Order Waterfall | Acc={acc_name} | RPM={rpm_name}"
                    f" | NFFT={nfft}, Overlap={overlap_pct:.1f}% | OrderMax={order_max:.1f}"
                    f" | Spectra={num_spectra} | {out_type} ({scale})" + otf_tag
                ),
                "otf_info": otf_info,
                "xlabel": "Order",
                "ylabel": "RPM",
                # Frequency axis is now the order axis
                "freqs": order_axis,
                "freqs_analysis": order_axis.copy(),
                "freqs_display": order_axis.copy(),
                "rpms": sorted_rpms,
                "S": S_order,
                "rpms_analysis": sorted_rpms,
                "S_analysis": S_order,
                "rpms_display": sorted_rpms,
                "S_display": S_order,
                # Keep raw frame data for order-cursor extraction
                "frame_rpms": wf_map["frame_rpms"],
                "frame_spectra_linear": wf_map["frame_spectra_linear"],
                "scale": scale,
                "out_type": out_type,
                "colorbar_label": colorbar_label,
                "order_extract_mode": self.cfg.get("waterfall_order_extract_mode", "Interpolated"),
                "order_bw": float(self.cfg.get("waterfall_order_bw", 0.0)),
                "order_bw_unit": self.cfg.get("waterfall_order_bw_unit", "Hz"),
                "raw_frame_count": wf_map.get("raw_frame_count"),
                "analysis_spectrum_count": wf_map.get("analysis_spectrum_count"),
                "buffers_per_spectrum": wf_map.get("buffers_per_spectrum"),
                "analysis_interpolated": wf_map.get("analysis_interpolated"),
                "stats_text": (
                    f"Order Waterfall | Acc={acc_name} (fs={fs_acc:g} Hz) | "
                    f"RPM={rpm_name} (fs={fs_rpm:g} Hz) | FFT fs={wf_fs:g} Hz | "
                    f"NFFT={nfft} | Spectra={num_spectra} | OrderMax={order_max:.1f}"
                ),
            }

        # ── Frequency-domain waterfall path (original behavior) ─────────
        freqs = wf_map["freqs"]
        freqs_display = wf_map.get("freqs_display", freqs)
        rpms = wf_map["rpms_display"] if wf_map["rpms_display"].size else wf_map["rpms_analysis"]
        S_map = wf_map["S_display"] if wf_map["S_display"].size else wf_map["S_analysis"]

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

        if otf_info:
            otf_tag = (
                f" | OTF={otf_info['target_order']:.2f}x"
                f" ±{otf_info['half_bw']:g}{otf_info['bw_unit']}"
                f" ({otf_info['filter_type']})"
            )
            title = title + otf_tag
            short = f"{acc_name} | Waterfall" + otf_tag
        else:
            short = f"{acc_name} | Waterfall"

        return {
            "key": key,
            "kind": "waterfall",
            "name": acc_name,
            "short_title": short,
            "title": title,
            "otf_info": otf_info,
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
            "order_extract_mode": self.cfg.get("waterfall_order_extract_mode", "Interpolated"),
            "order_bw": float(self.cfg.get("waterfall_order_bw", 0.0)),
            "order_bw_unit": self.cfg.get("waterfall_order_bw_unit", "Hz"),
            "raw_frame_count": wf_map.get("raw_frame_count"),
            "analysis_spectrum_count": wf_map.get("analysis_spectrum_count"),
            "buffers_per_spectrum": wf_map.get("buffers_per_spectrum"),
            "analysis_interpolated": wf_map.get("analysis_interpolated"),
            "stats_text": (
                f"Waterfall built from Acc={acc_name} ({acc_file_base}/{acc_group}, fs={fs_acc:g} Hz) and "
                f"RPM={rpm_name} ({rpm_file_base}/{rpm_group}, fs={fs_rpm:g} Hz) | "
                f"{resample_mode} | Interp={interp_mode} | RPM label={rpm_label_method} | FFT fs={wf_fs:g} Hz | "
                f"Spectra={num_spectra} | Buffers/Spectrum={num_buffers}"
            ),
            # COT: computed when checkbox is enabled. Stored for downstream use.
            "cot_results": self._maybe_compute_cot(acc, rpm, wf_fs),
        }

    def _maybe_apply_order_tracking_filter(self, wf_map: dict, output_type: str, scale: str):
        """Apply the legacy frequency-domain order mask to ``wf_map`` in place.

        Returns a small dict describing what was applied (or ``None`` when the
        filter is disabled), so the caller can record it in the plot state.
        """
        mode = str(self.cfg.get("waterfall_otf_mode", "Frequency-domain order mask" if self.cfg.get("waterfall_otf_enabled", False) else "Off"))
        if (not bool(self.cfg.get("waterfall_otf_enabled", False))) or mode.startswith("Time-domain"):
            return None

        try:
            target = float(self.cfg.get("waterfall_otf_order", 1.0))
            half_bw = float(self.cfg.get("waterfall_otf_bw", 0.25))
            bw_unit = str(self.cfg.get("waterfall_otf_bw_unit", "Order"))
            ftype = str(self.cfg.get("waterfall_otf_type", "Band-pass"))
            forder = int(self.cfg.get("waterfall_otf_filter_order", 4))

            freqs = np.asarray(wf_map.get("freqs", []), dtype=np.float64)
            frame_rpms = np.asarray(wf_map.get("frame_rpms", []), dtype=np.float64)
            frame_spec = np.asarray(wf_map.get("frame_spectra_linear", []), dtype=np.float64)

            if freqs.size == 0 or frame_rpms.size == 0 or frame_spec.size == 0:
                return None

            filtered_frames, trace = apply_order_tracking_filter(
                freqs, frame_rpms, frame_spec,
                target_order=target,
                half_bw=half_bw,
                bw_unit=bw_unit,
                filter_type=ftype,
                filter_order=forder,
                output_type=output_type,
            )
            wf_map["frame_spectra_linear"] = filtered_frames

            # Rebuild S_analysis / S_display by re-interpolating the filtered
            # frames onto the existing RPM grids so the downstream display
            # matches the new content.
            rpms_analysis = np.asarray(wf_map.get("rpms_analysis", []), dtype=np.float64)
            rpms_display = np.asarray(wf_map.get("rpms_display", []), dtype=np.float64)

            def _interp_to(target_rpms):
                if target_rpms.size == 0 or filtered_frames.size == 0:
                    return np.zeros((target_rpms.size, freqs.size), dtype=np.float64)
                out = np.empty((target_rpms.size, freqs.size), dtype=np.float64)
                src = frame_rpms
                # Sort source RPMs for np.interp
                order = np.argsort(src)
                src_sorted = src[order]
                spec_sorted = filtered_frames[order]
                for j in range(freqs.size):
                    out[:, j] = np.interp(target_rpms, src_sorted, spec_sorted[:, j],
                                          left=spec_sorted[0, j], right=spec_sorted[-1, j])
                return np.maximum(out, 0.0)

            S_an_linear = _interp_to(rpms_analysis)
            S_dp_linear = _interp_to(rpms_display) if rpms_display.size else S_an_linear

            wf_map["S_analysis"] = self._convert_waterfall_scale(S_an_linear, output_type, scale)
            wf_map["S_display"] = self._convert_waterfall_scale(S_dp_linear, output_type, scale)

            return {
                "enabled": True,
                "mode": "Frequency-domain order mask",
                "target_order": target,
                "half_bw": half_bw,
                "bw_unit": bw_unit,
                "filter_type": ftype,
                "filter_order": forder,
                "trace_rpms": frame_rpms.copy(),
                "trace_amp": trace,
                "show_trace": bool(self.cfg.get("waterfall_otf_show_trace", True)),
            }
        except Exception as e:
            if DEBUG:
                print(f"[TRDSP] Order tracking filter failed: {e}")
            return None

    def _maybe_compute_cot(self, acc: np.ndarray, rpm_aligned: np.ndarray, fs: float):
        """Compute COT from the full aligned RPM signal; returns result dict or None."""
        if not bool(self.cfg.get("waterfall_cot_enabled", False)):
            return None
        try:
            acc = np.asarray(acc, dtype=np.float64)
            rpm_aligned = np.asarray(rpm_aligned, dtype=np.float64)
            if acc.size == 0 or rpm_aligned.size == 0 or acc.size != rpm_aligned.size:
                return {"warning": "COT skipped: acceleration and aligned RPM must have same nonzero length."}
            finite_pos = np.isfinite(rpm_aligned) & (rpm_aligned > 0.0)
            if np.count_nonzero(finite_pos) < max(16, acc.size // 100):
                return {"warning": "COT skipped: insufficient finite positive RPM samples."}
            spr = int(self.cfg.get("waterfall_cot_spr", 512))
            order_max = float(self.cfg.get("waterfall_order_max", 20.0))
            orders, amp_spectrum, amp_vs_rpm, meta = compute_order_tracking_spectrum(
                acc, rpm_aligned, fs, spr, order_max,
                amp_conv=self.cfg.get("amp_conv", "Peak"),
                signal_kind="acc", return_metadata=True,
            )
            return {
                "orders": orders,
                "amp_spectrum": amp_spectrum,
                "amp_vs_rpm": amp_vs_rpm,
                "metadata": meta,
                "note": "COT used the full aligned RPM time history for angle-domain resampling.",
            }
        except Exception as e:
            if DEBUG:
                print(f"[TRDSP] COT computation failed: {e}")
            return {"warning": f"COT failed: {e}"}


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
            # acc: zero-fill NaN (FFT-safe); rpm: preserve NaN for invalid-frame exclusion
            acc_clean = np.nan_to_num(acc_out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
            rpm_clean = clean_rpm_signal(rpm_out, mode="interpolate").astype(np.float64)
            n_invalid = int(np.sum(~np.isfinite(rpm_clean)))
            if n_invalid and DEBUG:
                print(f"[TRDSP] _align_waterfall_signals: {n_invalid} RPM samples remain NaN after cleaning.")
            return acc_clean, rpm_clean, fs_rpm

        keep = t_acc <= (t_end + 0.5 / max(fs_acc, 1e-12))
        t_dst = t_acc[keep]
        acc_out = acc[:t_dst.size]
        rpm_out = self._resample_uniform_signal(rpm, fs_rpm, t_dst, interp_mode)
        acc_clean = np.nan_to_num(acc_out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
        rpm_clean = clean_rpm_signal(rpm_out, mode="interpolate").astype(np.float64)
        n_invalid = int(np.sum(~np.isfinite(rpm_clean)))
        if n_invalid:
            print(f"[TRDSP] _align_waterfall_signals: {n_invalid} RPM samples remain NaN after cleaning.")
        return acc_clean, rpm_clean, fs_acc

    def _build_envelope_analysis_state(self):
        """Build Envelope/Kurtogram states in the worker thread.

        Audit note: this localized dispatch calls the existing modular
        dsp_core.envelope_kurtogram backend and returns normal TRDSP state
        dictionaries; FFT/STFT/Waterfall/Order Tracking code paths are left
        untouched.
        """
        if not _HAVE_ENVELOPE_BACKEND or compute_envelope_spectrum is None:
            raise RuntimeError("Envelope/Kurtogram backend is not available.")

        fs = float(self.cfg["fs"])
        names = list(self.cfg.get("names", []))
        if not names:
            raise RuntimeError("No channel selected for Envelope/Kurtogram analysis.")
        name = names[0]
        group_or_sheet = self.cfg["group_or_sheet"]
        file_base = os.path.splitext(os.path.basename(self.cfg["file_path"]))[0]
        raw = self._load_selected_signals()[name]
        y = self._prep_signal_with_fs(np.asarray(raw), fs)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
        if y.size < 16:
            raise RuntimeError("Envelope/Kurtogram analysis needs at least 16 samples after preprocessing.")

        band = self.cfg.get("envelope_band")
        if band is not None:
            band = (float(band[0]), float(band[1]))

        bearing_freqs = self.cfg.get("bearing_freqs") or None
        diagnostic_kind = str(self.cfg.get("diagnostic_kind", "envelope_spectrum"))

        if diagnostic_kind == "kurtogram":
            if compute_fast_kurtogram is None:
                raise RuntimeError("Kurtogram backend is not available.")
            n_levels = int(self.cfg.get("kurtogram_levels", 5))
            kg = compute_fast_kurtogram(y, fs, n_levels=n_levels)
            best = dict(kg["best_band"])
            env = compute_envelope_spectrum(y, fs, band=(best["f_low"], best["f_high"]))
            return [{
                "key": f"kurtogram|{group_or_sheet}|{name}",
                "kind": "kurtogram",
                "name": name,
                "short_title": f"Envelope / Kurtogram: {name}",
                "title": (
                    f"{file_base} | {group_or_sheet} | Envelope / Kurtogram | {name} | "
                    f"levels={n_levels} | best band={best['f_low']:.6g}-{best['f_high']:.6g} Hz | "
                    f"K={best['kurtosis']:.3g}"
                ),
                "xlabel": "Frequency [Hz]",
                "ylabel": "Envelope amplitude [peak]",
                "kurtosis_grid": kg["kurtosis_grid"],
                "center_freq_grid": kg["center_freq_grid"],
                "bandwidth_grid": kg["bandwidth_grid"],
                "best_band": best,
                "bands": kg.get("bands", []),
                "envelope_freqs": env["freqs"],
                "envelope_spectrum": env["spectrum"],
                "fs": fs,
                "n_levels": n_levels,
                "band": env.get("band"),
                "bearing_freqs": bearing_freqs,
                "colorbar_label": "Excess kurtosis",
                "scale": "Linear",
                "stats_text": (
                    f"Envelope/Kurtogram | {name} | fs={fs:g} Hz | samples={y.size} | "
                    f"best={best['f_low']:.3g}-{best['f_high']:.3g} Hz | K={best['kurtosis']:.3g}"
                ),
            }]

        env = compute_envelope_spectrum(y, fs, band=band)
        band_txt = "none" if env.get("band") is None else f"{env['band'][0]:.6g}-{env['band'][1]:.6g} Hz"
        return [{
            "key": f"envelope_spectrum|{group_or_sheet}|{name}",
            "kind": "envelope_spectrum",
            "name": name,
            "short_title": f"Envelope spectrum: {name}",
            "title": f"{file_base} | {group_or_sheet} | Envelope Spectrum | {name} | band={band_txt}",
            "xlabel": "Frequency [Hz]",
            "ylabel": "Envelope amplitude [peak]",
            "freqs": env["freqs"],
            "spectrum": env["spectrum"],
            "fs": fs,
            "band": env.get("band"),
            "scale": "Linear",
            "bearing_freqs": bearing_freqs,
            "stats_text": f"Envelope spectrum | {name} | fs={fs:g} Hz | samples={y.size} | band={band_txt}",
        }]

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

        if mode == "Envelope / Kurtogram":
            return self._build_envelope_analysis_state()

        raw_signals = self._load_selected_signals()
        states = []

        # Resolve independent AVG / STFT settings once (fall back to globals).
        avg_override = bool(self.cfg.get("avg_override", False))
        stft_override = bool(self.cfg.get("stft_override", False))
        avg_out = self.cfg.get("avg_out_type", self.cfg["out_type"]) if avg_override else self.cfg["out_type"]
        avg_scale = self.cfg.get("avg_scale", self.cfg["scale"]) if avg_override else self.cfg["scale"]
        avg_amp_conv = self.cfg.get("avg_amp_conv", self.cfg["amp_conv"]) if avg_override else self.cfg["amp_conv"]
        stft_out = self.cfg.get("stft_out_type", self.cfg["out_type"]) if stft_override else self.cfg["out_type"]
        stft_scale = self.cfg.get("stft_scale", self.cfg["scale"]) if stft_override else self.cfg["scale"]
        stft_amp_conv = self.cfg.get("stft_amp_conv", self.cfg["amp_conv"]) if stft_override else self.cfg["amp_conv"]

        # Normal Plot View always creates the complete plot family for the
        # selected physical signal: time-domain + averaged spectrum + STFT.
        # The legacy Fourier Method combo remains hidden for compatibility,
        # but it no longer gates which states are generated.
        wants_avg = True
        wants_stft = True

        nfft = int(self.cfg["nfft"])
        overlap_pct = float(self.cfg["overlap_pct"])

        for nm in names:
            y = self._get_processed_signal(np.asarray(raw_signals[nm]))
            short_title = f"{nm} | {signal_name}"
            proc_title = self._processing_title(use_filtered, signal_kind)
            base_key = f"{signal_kind}|{group_or_sheet}|{nm}"

            # ── 1) ALWAYS emit a time-domain state ───────────────────────
            start_offset = t0 if crop else 0.0
            t = self._make_time_axis(y.size, fs, start_offset_s=start_offset)
            states.append({
                "key": f"time|{base_key}",
                "kind": "time",
                "name": nm,
                "short_title": f"Time-domain: {nm} / {signal_name}",
                "title": f"{file_base} | {group_or_sheet} | Time-domain ({signal_name}){proc_title} | {nm}",
                "xlabel": "Time (s)",
                "ylabel": self._signal_ylabel(signal_kind),
                "grid": True,
                "series": [{"x": t, "y": y, "label": nm}],
                "time_signal": y.copy(),
                "time_axis": t.copy(),
                "fs": fs,
                "signal_kind": signal_kind,
                "fft_zoom_cfg": {
                    "nfft": nfft,
                    "overlap_pct": overlap_pct,
                    "out_type": avg_out,
                    "scale": avg_scale,
                    "amp_conv": avg_amp_conv,
                    "signal_kind": signal_kind,
                    "window_kind": self.cfg.get("window_kind", "periodic"),
                },
                "stats_text": self._time_stats_text(nm, y, signal_kind),
            })

            # If neither avg nor stft were requested we still produced a
            # time-domain state, which is the minimum.
            key = base_key
            if wants_avg:
                # Temporarily swap amp_conv into cfg so existing helpers pick up
                # the AVG-specific convention; restore after.
                _saved_amp = self.cfg.get("amp_conv")
                self.cfg["amp_conv"] = avg_amp_conv
                try:
                    freqs_a, spec_a = self._compute_avg_spectrum(
                        y, fs, nfft, overlap_pct, avg_out, avg_scale, signal_kind
                    )
                finally:
                    self.cfg["amp_conv"] = _saved_amp
                _, unit_lbl_a = get_amp_conv_factor_label(avg_amp_conv, signal_kind)
                ylabel_a = build_spectrum_ylabel(avg_out, avg_scale, unit_lbl_a)
                states.append({
                    "key": f"avg|{key}",
                    "kind": "avg",
                    "name": nm,
                    "signal_kind": signal_kind,
                    "short_title": f"Averaged spectrum: {nm} / {signal_name}",
                    "title": (
                        f"{file_base} | {group_or_sheet} | Averaged Spectrum "
                        f"({signal_name}){proc_title} | {nm} | NFFT={nfft}, "
                        f"Overlap={overlap_pct:.1f}% | {avg_out} ({avg_scale}) | {unit_lbl_a}"
                    ),
                    "xlabel": "Frequency (Hz)",
                    "ylabel": ylabel_a,
                    "grid": True,
                    "series": [{"x": freqs_a, "y": spec_a, "label": nm}],
                    "stats_text": "",
                })

            if wants_stft:
                _saved_amp = self.cfg.get("amp_conv")
                self.cfg["amp_conv"] = stft_amp_conv
                try:
                    freqs_s, times_s, S = self._compute_stft(
                        y, fs, nfft, overlap_pct, stft_out, stft_scale, signal_kind
                    )
                finally:
                    self.cfg["amp_conv"] = _saved_amp
                if crop:
                    times_s = times_s + t0
                _, unit_lbl_s = get_amp_conv_factor_label(stft_amp_conv, signal_kind)
                colorbar_label = build_spectrum_ylabel(stft_out, stft_scale, unit_lbl_s)
                states.append({
                    "key": f"stft|{key}",
                    "kind": "stft",
                    "name": nm,
                    "signal_kind": signal_kind,
                    "short_title": f"Spectrogram: {nm} / {signal_name}",
                    "title": (
                        f"{file_base} | {group_or_sheet} | STFT Spectrogram "
                        f"({signal_name}){proc_title} | {nm} | NFFT={nfft}, "
                        f"Overlap={overlap_pct:.1f}% | {stft_out} ({stft_scale}) | {unit_lbl_s}"
                    ),
                    "xlabel": "Time (s)",
                    "ylabel": "Frequency (Hz)",
                    "freqs": freqs_s,
                    "times": times_s,
                    "S": S,
                    "out_type": stft_out,
                    "scale": stft_scale,
                    "unit_lbl": unit_lbl_s,
                    "colorbar_label": colorbar_label,
                    "stats_text": "",
                })

        if not states:
            raise RuntimeError("No data left to plot after preprocessing.")

        return states

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TRDSP — TRMOTOR Digital Signal Processor")
        try:
            self.setWindowIcon(QIcon(resource_path("logo.png")))
        except Exception:
            pass

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
        self._waterfall_view_widget = None
        self._waterfall_result_widget = None
        self._waterfall_result_key = None
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

        # ── Signal Type selector (visible) ────────────────────────────────
        # Replaces the old "Plot Mode" dropdown.  The user now only picks
        # WHAT signal they are plotting; the worker always produces
        # time + averaged-spectrum + STFT states for that signal type.
        self.cmb_signal_type = QComboBox()
        self.cmb_signal_type.addItems(["Acceleration", "Velocity"])
        self.cmb_signal_type.setToolTip(
            "Physical signal to plot.\n"
            "One Plot click produces: Time-domain, Averaged spectrum, and STFT/Spectrogram\n"
            "for every selected channel — separately for each panel of Plot View."
        )

        # ── Legacy "Plot Mode" combo (kept hidden) ────────────────────────
        # A lot of internal helpers (_is_fourier_mode, _is_velocity_mode, …)
        # still read this combo, so we keep it as the single source of truth
        # for downstream code and drive it from cmb_signal_type instead of
        # exposing it in the UI.
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems([
            "Time Domain (Acceleration)",
            "Time Domain (Velocity)",
            "Fourier Transform (Acceleration)",
            "Fourier Transform (Velocity)",
        ])
        # Default to acceleration FFT — the worker always emits the time
        # state too, so "Fourier Transform (…)" effectively means "all three".
        self.cmb_mode.setCurrentText("Fourier Transform (Acceleration)")
        self.cmb_mode.setVisible(False)

        def _sync_mode_from_signal_type(*_):
            sig = self.cmb_signal_type.currentText().strip()
            target = (
                "Fourier Transform (Velocity)"
                if sig == "Velocity"
                else "Fourier Transform (Acceleration)"
            )
            if self.cmb_mode.currentText() != target:
                self.cmb_mode.setCurrentText(target)
        self.cmb_signal_type.currentIndexChanged.connect(_sync_mode_from_signal_type)

        self.cmb_fourier_method = QComboBox()
        self.cmb_fourier_method.addItems([
            "Averaged Spectrum",
            "STFT (Spectrogram)",
            "Both (Averaged + STFT)",
        ])
        # Default: emit both averaged and STFT alongside the always-emitted
        # time-domain state.  Users can narrow it down to one method here.
        self.cmb_fourier_method.setCurrentText("Both (Averaged + STFT)")

        # Independent AVG settings — used when the worker builds an averaged
        # spectrum state.  These default to the global Output/Scale/Amplitude
        # combos but can be overridden via the "AVG overrides" group below.
        self.chk_avg_override = QCheckBox("AVG overrides")
        self.chk_avg_override.setToolTip(
            "When checked, the Averaged Spectrum uses its own Output/Scale/Convention\n"
            "instead of the global FFT settings. Lets you have Avg=Linear while STFT=dB."
        )
        self.cmb_avg_out = QComboBox()
        self.cmb_avg_out.addItems(["Amplitude", "Power", "PSD", "ASD"])
        self.cmb_avg_scale = QComboBox()
        self.cmb_avg_scale.addItems(["Linear", "dB"])
        self.cmb_avg_amp_conv = QComboBox()
        self.cmb_avg_amp_conv.addItems(["Peak", "RMS", "Peak-to-Peak"])
        self.cmb_avg_amp_conv.setCurrentText("RMS")
        for _w in (self.cmb_avg_out, self.cmb_avg_scale, self.cmb_avg_amp_conv):
            _w.setEnabled(False)
        self.chk_avg_override.toggled.connect(
            lambda on: [w.setEnabled(on) for w in
                        (self.cmb_avg_out, self.cmb_avg_scale, self.cmb_avg_amp_conv)]
        )

        # Independent STFT settings.
        self.chk_stft_override = QCheckBox("STFT overrides")
        self.chk_stft_override.setToolTip(
            "When checked, the STFT/Spectrogram uses its own Output/Scale/Convention\n"
            "instead of the global FFT settings. Lets you have STFT=dB while Avg=Linear."
        )
        self.cmb_stft_out = QComboBox()
        self.cmb_stft_out.addItems(["Amplitude", "Power", "PSD", "ASD"])
        self.cmb_stft_scale = QComboBox()
        self.cmb_stft_scale.addItems(["Linear", "dB"])
        self.cmb_stft_scale.setCurrentText("dB")
        self.cmb_stft_amp_conv = QComboBox()
        self.cmb_stft_amp_conv.addItems(["Peak", "RMS", "Peak-to-Peak"])
        self.cmb_stft_amp_conv.setCurrentText("RMS")
        for _w in (self.cmb_stft_out, self.cmb_stft_scale, self.cmb_stft_amp_conv):
            _w.setEnabled(False)
        self.chk_stft_override.toggled.connect(
            lambda on: [w.setEnabled(on) for w in
                        (self.cmb_stft_out, self.cmb_stft_scale, self.cmb_stft_amp_conv)]
        )

        self.cmb_nfft = QComboBox()
        self.cmb_nfft.addItems([str(x) for x in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]])
        self.cmb_nfft.setCurrentText("8192")

        self.ovl_spin = QDoubleSpinBox()
        self.ovl_spin.setRange(0.0, 95.0)
        self.ovl_spin.setDecimals(1)
        self.ovl_spin.setSingleStep(5.0)
        self.ovl_spin.setValue(67.0)

        self.cmb_out = QComboBox()
        self.cmb_out.addItems(["Amplitude", "Power", "PSD", "ASD"])

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

        # Plot View tab layout (Get_Plot_Side-style):
        #   ┌── three preview sections (Time / Avg Spectrum / Spectrograms) ──┐
        #   │  scrollable column of PlotMiniWidget thumbnails per kind         │
        #   ├──────────────────────────────────────────────────────────────────┤
        #   │  shared larger detail plot area (DetailPlotWidget)               │
        #   └──────────────────────────────────────────────────────────────────┘
        # plot_pages is retained internally as a flat container for the current
        # inline detail widget so that legacy callbacks keep working, but only
        # one detail is shown at a time and the tab bar is hidden.
        self.plot_pages = QTabWidget()
        self.plot_pages.setTabBarAutoHide(True)
        self.plot_pages.setDocumentMode(True)
        self.plot_pages.tabBar().setVisible(False)
        self.plot_pages.setTabsClosable(False)
        self.plot_pages.currentChanged.connect(self._on_plot_page_changed)

        self.overview_page = QWidget()
        self.plot_pages.addTab(self.overview_page, "Overview")

        # --- Three preview sections (top half) -----------------------------
        def _make_mini_column(title: str):
            box = QGroupBox(title)
            box.setStyleSheet(
                "QGroupBox { font-weight: bold; border: 1px solid #999;"
                " border-radius: 4px; margin-top: 8px; padding: 4px; }"
                "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
            )
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            inner = QWidget()
            inner_lay = QVBoxLayout(inner)
            inner_lay.setContentsMargins(4, 4, 4, 4)
            inner_lay.setSpacing(6)
            inner_lay.addStretch(1)
            scroll.setWidget(inner)
            v = QVBoxLayout(box)
            v.setContentsMargins(4, 14, 4, 4)
            v.addWidget(scroll, 1)
            return box, inner, inner_lay

        (self.mini_col_time, self.mini_inner_time,
         self.mini_inner_time_lay) = _make_mini_column("Time-domain")
        (self.mini_col_avg, self.mini_inner_avg,
         self.mini_inner_avg_lay) = _make_mini_column("Averaged Spectrum")
        (self.mini_col_spec, self.mini_inner_spec,
         self.mini_inner_spec_lay) = _make_mini_column("Spectrograms")

        mini_row_widget = QWidget()
        mini_row_lay = QHBoxLayout(mini_row_widget)
        mini_row_lay.setContentsMargins(0, 0, 0, 0)
        mini_row_lay.setSpacing(6)
        mini_row_lay.addWidget(self.mini_col_time, 1)
        mini_row_lay.addWidget(self.mini_col_avg, 1)
        mini_row_lay.addWidget(self.mini_col_spec, 1)

        # --- Shared inline detail area --------------------------------------
        # Stays on the "Selected" tab; updates when a mini plot is clicked.
        self.inline_detail_holder = QWidget()
        inline_lay = QVBoxLayout(self.inline_detail_holder)
        inline_lay.setContentsMargins(0, 4, 0, 0)
        inline_lay.setSpacing(2)
        self.lbl_inline_detail_empty = QLabel(
            "Click a mini plot above to open it in the detail view."
        )
        self.lbl_inline_detail_empty.setStyleSheet("color: #666;")
        self.lbl_inline_detail_empty.setAlignment(Qt.AlignCenter)
        inline_lay.addWidget(self.lbl_inline_detail_empty, 1)
        self._inline_detail_lay = inline_lay
        self._inline_detail_widget = None

        # --- Large-tabs area (Open Large opens new closable tab here) -------
        self.large_tabs = QTabWidget()
        self.large_tabs.setTabsClosable(True)
        self.large_tabs.setMovable(True)
        self.large_tabs.setDocumentMode(True)
        self.large_tabs.tabCloseRequested.connect(self._on_large_tab_close)
        self.large_tabs.currentChanged.connect(self._on_large_tab_changed)
        # Until the user opens at least one Large tab, keep the tab widget
        # collapsed via setVisible(False).  The first addTab will show it.
        self.large_tabs.setVisible(False)
        self._large_tab_widgets = {}

        # The Plot View tab body uses a vertical splitter so the user can
        # adjust how much room minis vs. detail vs. large-tabs get.
        from PyQt5.QtWidgets import QSplitter
        self.plot_view_splitter = QSplitter(Qt.Vertical)
        self.plot_view_splitter.addWidget(mini_row_widget)
        self.plot_view_splitter.addWidget(self.inline_detail_holder)
        self.plot_view_splitter.addWidget(self.large_tabs)
        self.plot_view_splitter.setStretchFactor(0, 1)
        self.plot_view_splitter.setStretchFactor(1, 2)
        self.plot_view_splitter.setStretchFactor(2, 2)

        self.lbl_overview_empty = QLabel("No plots yet.")
        self.lbl_overview_empty.setStyleSheet("color: #666;")

        ovlay = QVBoxLayout(self.overview_page)
        ovlay.setContentsMargins(0, 0, 0, 0)
        ovlay.addWidget(self.lbl_overview_empty)
        ovlay.addWidget(self.plot_view_splitter, 1)

        # Bookkeeping: which kind a mini was placed in (for cleanup).
        self._mini_kinds = {}

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # Fixed-height header at the top; the main tab area gets all remaining space.
        root.addWidget(self._build_logo_header(), 0)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        self.tab_controls = QWidget()
        self.tab_waterfall = QWidget()
        self.tab_envelope = QWidget()
        self.tab_waterfall_view = QWidget()
        self.tab_plot = QWidget()
        self.tab_info = QWidget()
        self.tabs.addTab(self.tab_controls, self._make_tab_icon("controls"), "Controls")
        self.tabs.addTab(self.tab_waterfall, self._make_tab_icon("waterfall"), "Waterfall")
        self.tabs.addTab(self.tab_envelope, self._make_tab_icon("plot"), "Envelope / Kurtogram")
        self.tabs.addTab(self.tab_waterfall_view, self._make_tab_icon("waterfall"), "Waterfall View")
        self.tabs.addTab(self.tab_plot, self._make_tab_icon("plot"), "Plot View")
        self.tabs.addTab(self.tab_info, self._make_tab_icon("help"), "Help / Info")

        # TRMOTOR palette for the main tab bar — dark charcoal with a red
        # underline on the selected tab.
        self.tabs.setStyleSheet(
            "QTabWidget::pane { border-top: 1px solid #3A3A3A; background: #FFFFFF; }"
            "QTabBar::tab { background: #222222; color: #F5F5F5;"
            " padding: 6px 14px; margin-right: 2px;"
            " border: 1px solid #3A3A3A; border-bottom: none;"
            " border-top-left-radius: 4px; border-top-right-radius: 4px; }"
            "QTabBar::tab:selected { background: #050505; color: #FFFFFF;"
            " border-bottom: 3px solid #D71920; }"
            "QTabBar::tab:hover { background: #3A3A3A; color: #FFFFFF; }"
        )

        ctl = QVBoxLayout(self.tab_controls)
        ctl.addWidget(self.btn_open)
        ctl.addWidget(self.lbl_file)
        ctl.addWidget(self.progress)
        ctl.addWidget(self.lbl_status)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Sheet / Group:"))
        row1.addWidget(self.cmb_sheet, 2)
        row1.addSpacing(12)
        row1.addWidget(QLabel("Signal Type:"))
        row1.addWidget(self.cmb_signal_type, 1)
        row1.addWidget(self._make_help_button("Signal Type — Explanation", self._help_html("signal_type"), "Signal Type"))
        ctl.addLayout(row1)

        ctl.addWidget(self.lbl_wf_hint)
        row_export = QHBoxLayout()
        row_export.addWidget(self.btn_export_cropped_tdms)
        row_export.addWidget(self._make_help_button("Export Cropped TDMS — Explanation", self._help_html("export_tdms"), "Export Cropped TDMS"))
        row_export.addStretch(1)
        ctl.addLayout(row_export)
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
        row2.addWidget(self._make_help_button("Sampling Frequency — Explanation", self._help_html("fs"), "fs"))
        row2.addSpacing(14)
        row2.addWidget(self.chk_skip_first)
        row2.addSpacing(14)
        row2.addWidget(QLabel("Data Source:"))
        row2.addWidget(self.cmb_data_source)
        row2.addWidget(self._make_help_button("Data Source — Explanation", self._help_html("data_source"), "Data Source"))
        row2.addStretch(1)
        ctl.addLayout(row2)

        row_filt = QHBoxLayout()
        row_filt.addWidget(self.chk_filter)
        row_filt.addWidget(self._make_help_button("Butterworth Filter — Explanation", self._help_html("filter_enable"), "Butterworth filter"))
        row_filt.addSpacing(12)
        row_filt.addWidget(QLabel("Type:"))
        row_filt.addWidget(self.cmb_filter_type)
        row_filt.addWidget(self._make_help_button("Filter Type — Explanation", self._help_html("filter_type"), "Filter Type"))
        row_filt.addSpacing(8)
        row_filt.addWidget(QLabel("Order:"))
        row_filt.addWidget(self.cmb_filter_order)
        row_filt.addWidget(self._make_help_button("Filter Order — Explanation", self._help_html("filter_order"), "Filter Order"))
        row_filt.addSpacing(8)
        row_filt.addWidget(QLabel("Cutoff 1 (Hz):"))
        row_filt.addWidget(self.cut1_spin)
        row_filt.addWidget(self.lbl_cut2)
        row_filt.addWidget(self.cut2_spin)
        row_filt.addWidget(self._make_help_button("Filter Cutoffs — Explanation", self._help_html("cutoffs"), "Cutoff frequencies"))
        row_filt.addStretch(1)
        ctl.addLayout(row_filt)

        row3 = QHBoxLayout()
        row3.addWidget(self.chk_crop)
        row3.addWidget(self._make_help_button("Crop Time Window — Explanation", self._help_html("crop"), "Crop time window"))
        row3.addWidget(QLabel("Start (s):"))
        row3.addWidget(self.t0_spin)
        row3.addWidget(QLabel("End (s):"))
        row3.addWidget(self.t1_spin)
        row3.addStretch(1)
        row3.addWidget(self.btn_plot)
        ctl.addLayout(row3)

        row_fft = QHBoxLayout()
        # The old Fourier Method selector is hidden from the main workflow:
        # one Plot action now always creates time + avg + STFT.
        self.cmb_fourier_method.setVisible(False)
        row_fft.addWidget(QLabel("Buffer Size (NFFT):"))
        row_fft.addWidget(self.cmb_nfft)
        row_fft.addWidget(self._make_help_button("Buffer Size / NFFT — Explanation", self._help_html("nfft"), "Buffer Size / NFFT"))
        row_fft.addWidget(QLabel("Overlap (%):"))
        row_fft.addWidget(self.ovl_spin)
        row_fft.addWidget(self._make_help_button("Overlap — Explanation", self._help_html("overlap"), "Overlap"))
        row_fft.addWidget(QLabel("Output:"))
        row_fft.addWidget(self.cmb_out)
        row_fft.addWidget(self._make_help_button("Output — Explanation", self._help_html("output"), "Output"))
        row_fft.addWidget(QLabel("Scale:"))
        row_fft.addWidget(self.cmb_scale)
        row_fft.addWidget(self._make_help_button("Scale — Explanation", self._help_html("scale"), "Scale"))
        row_fft.addWidget(QLabel("Amplitude convention:"))
        row_fft.addWidget(self.cmb_amp_conv)
        row_fft.addWidget(self._make_help_button("Amplitude Conversion — Explanation", self._help_html("amp"), "Amplitude conversion"))
        row_fft.addWidget(QLabel("Window:"))
        row_fft.addWidget(self.cmb_window_kind)
        row_fft.addWidget(self._make_help_button("Window Type — Explanation", self._help_html("window"), "Window type"))
        row_fft.addStretch(1)
        ctl.addLayout(row_fft)

        # Independent AVG / STFT overrides — let the user set, e.g.,
        # Avg = Linear and STFT = dB at the same time.
        row_avg_over = QHBoxLayout()
        row_avg_over.addWidget(self.chk_avg_override)
        row_avg_over.addSpacing(6)
        row_avg_over.addWidget(QLabel("AVG Output:"))
        row_avg_over.addWidget(self.cmb_avg_out)
        row_avg_over.addWidget(QLabel("Scale:"))
        row_avg_over.addWidget(self.cmb_avg_scale)
        row_avg_over.addWidget(QLabel("Convention:"))
        row_avg_over.addWidget(self.cmb_avg_amp_conv)
        row_avg_over.addStretch(1)
        ctl.addLayout(row_avg_over)

        row_stft_over = QHBoxLayout()
        row_stft_over.addWidget(self.chk_stft_override)
        row_stft_over.addSpacing(6)
        row_stft_over.addWidget(QLabel("STFT Output:"))
        row_stft_over.addWidget(self.cmb_stft_out)
        row_stft_over.addWidget(QLabel("Scale:"))
        row_stft_over.addWidget(self.cmb_stft_scale)
        row_stft_over.addWidget(QLabel("Convention:"))
        row_stft_over.addWidget(self.cmb_stft_amp_conv)
        row_stft_over.addStretch(1)
        ctl.addLayout(row_stft_over)

        ctl.addWidget(self.lbl_fft_info)
        ctl.addWidget(self.lbl_stats)
        ctl.addStretch(1)

        self._build_waterfall_tab_ui()
        self._build_envelope_tab_ui()
        self._build_waterfall_view_tab()
        self._build_help_tab_ui()

        ptab = QVBoxLayout(self.tab_plot)
        ptab.setContentsMargins(6, 6, 6, 6)
        ptab.setSpacing(4)

        # Clean detail-area toolbar — only zoom / reset / export, as requested.
        # Axis labels, spectrogram colormap, vmin/vmax, "Apply Plot View" are
        # moved into the Advanced Settings dialog (btn_plot_advanced below).
        row_nav = QHBoxLayout()
        row_nav.addWidget(self.btn_rect_zoom)
        row_nav.addWidget(self.btn_zoom_in)
        row_nav.addWidget(self.btn_zoom_out)
        row_nav.addWidget(self.btn_reset_view)
        row_nav.addSpacing(12)
        row_nav.addWidget(self.btn_export_plot)
        row_nav.addStretch(1)
        self.btn_plot_advanced = QPushButton("Advanced Settings…")
        self.btn_plot_advanced.setToolTip(
            "Open advanced display settings: axis labels, spectrogram colormap, "
            "color range, dynamic range."
        )
        row_nav.addWidget(self.btn_plot_advanced)
        ptab.addLayout(row_nav)

        # The previously top-mounted axis / spectrogram / color-range rows live
        # in a hidden auxiliary widget so signals/handlers still work; they are
        # also packed inside the Advanced Settings dialog.
        self._plot_view_aux_widget = QWidget()
        self._plot_view_aux_widget.setVisible(False)
        aux_lay = QVBoxLayout(self._plot_view_aux_widget)
        aux_lay.setContentsMargins(0, 0, 0, 0)

        row_axis = QHBoxLayout()
        row_axis.addWidget(QLabel("X Axis Name:"))
        row_axis.addWidget(self.edit_xlabel, 1)
        row_axis.addSpacing(10)
        row_axis.addWidget(QLabel("Y Axis Name:"))
        row_axis.addWidget(self.edit_ylabel, 1)
        row_axis.addWidget(self.btn_apply_plot_view)
        aux_lay.addLayout(row_axis)

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
        aux_lay.addLayout(row_spec_top)

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
        aux_lay.addLayout(row_spec_bottom)

        ptab.addWidget(self._plot_view_aux_widget)
        ptab.addWidget(self.plot_pages, 1)

        self.btn_plot_advanced.clicked.connect(self._open_plot_view_advanced_dialog)

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

        # Branding wrapper intentionally disabled.
        # The stable header is created with self._build_logo_header() above.

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

        if hasattr(self, "btn_env_compute"):
            env_ready = (self.list_channels.count() > 0) and bool(self.file_path) and not busy
            self.btn_env_compute.setEnabled(env_ready)
            self.btn_env_kurtogram.setEnabled(env_ready)
            self.btn_env_clear.setEnabled(not busy)
            self.btn_env_export.setEnabled((self._active_plot_key in self._plot_states) and not busy)
            self.cmb_env_channel.setEnabled(self.cmb_env_channel.count() > 0 and not busy)

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
        return bool(state and state.get("kind") in ("stft", "waterfall", "kurtogram"))

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
        if state.get("kind") not in ("stft", "waterfall", "kurtogram"):
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

    def _is_qobject_alive(self, obj) -> bool:
        return _qt_alive(obj)

    def _safe_delete_widget(self, widget):
        if widget is None or not self._is_qobject_alive(widget):
            return
        try:
            widget.setParent(None)
        except RuntimeError:
            return
        except Exception:
            pass
        try:
            widget.deleteLater()
        except RuntimeError:
            pass
        except Exception:
            pass

    def _clear_overview_widgets(self):
        for w in list(self._mini_widgets.values()):
            if self._active_canvas is getattr(w, "canvas", None):
                self._active_canvas = None
                self._active_plot_key = None
            self._safe_delete_widget(w)
        self._mini_widgets.clear()
        self._mini_kinds.clear()
        self._overview_order.clear()

        for lay in (self.mini_inner_time_lay, self.mini_inner_avg_lay, self.mini_inner_spec_lay):
            self._strip_mini_layout(lay, delete_widgets=False)

    def _strip_mini_layout(self, lay, delete_widgets=False):
        # Remove layout items without double-deleting widgets that are merely
        # being reflowed between mini columns.  Widget deletion is explicit.
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                try:
                    w.setParent(None)
                except RuntimeError:
                    continue
                except Exception:
                    pass
                if delete_widgets:
                    self._safe_delete_widget(w)
        lay.addStretch(1)

    def _section_layout_for_kind(self, kind: str):
        k = str(kind).strip().lower()
        if k == "time":
            return self.mini_inner_time_lay
        if k == "avg":
            return self.mini_inner_avg_lay
        if k in ("stft", "spectrogram", "spec", "image"):
            return self.mini_inner_spec_lay
        if k == "waterfall":
            raise RuntimeError("Waterfall states are routed to the Waterfall View tab, not Plot View mini sections.")
        return self.mini_inner_spec_lay

    def _clear_detail_tabs(self):
        for key, widget in list(self._detail_widgets.items()):
            if widget is getattr(self, "_waterfall_result_widget", None):
                continue
            idx = self.plot_pages.indexOf(widget)
            if idx >= 0:
                self.plot_pages.removeTab(idx)
            if widget is self._inline_detail_widget:
                self._inline_detail_widget = None
            self._safe_delete_widget(widget)
            self._detail_widgets.pop(key, None)
        self._clear_large_tabs()
        self._clear_inline_detail()

    def _clear_inline_detail(self):
        detail = self._inline_detail_widget
        if detail is not None:
            for k, w in list(self._detail_widgets.items()):
                if w is detail:
                    self._detail_widgets.pop(k, None)
            canvas = getattr(detail, "canvas", None)
            if self._active_canvas is canvas:
                self._active_canvas = None
                self._active_plot_key = None
            self._safe_delete_widget(detail)
            self._inline_detail_widget = None
        self.lbl_inline_detail_empty.setVisible(True)
        self.inline_detail_holder.setVisible(False)

    def _clear_large_tabs(self):
        if not hasattr(self, "large_tabs"):
            return
        active_cleared = False
        self.large_tabs.blockSignals(True)
        try:
            for key, widget in list(self._large_tab_widgets.items()):
                canvas = getattr(widget, "canvas", None)
                if self._active_canvas is canvas:
                    active_cleared = True
                idx = self.large_tabs.indexOf(widget)
                if idx >= 0:
                    self.large_tabs.removeTab(idx)
                self._safe_delete_widget(widget)
            while self.large_tabs.count():
                widget = self.large_tabs.widget(0)
                self.large_tabs.removeTab(0)
                self._safe_delete_widget(widget)
        finally:
            self.large_tabs.blockSignals(False)
        self._large_tab_widgets.clear()
        self.large_tabs.setVisible(False)
        if active_cleared:
            self._active_canvas = None
            self._active_plot_key = None

    def _show_inline_detail(self, key: str):
        """Replace the shared inline detail area with a fresh widget for `key`.

        Inline detail and Open Large are mutually exclusive in the single-large
        Plot View: selecting a mini clears any large tab before showing detail.
        """
        if key not in self._plot_states:
            return

        self._clear_large_tabs()
        self._clear_inline_detail()
        self.inline_detail_holder.setVisible(True)
        self.lbl_inline_detail_empty.setVisible(False)

        state = self._plot_states[key]
        detail = self._make_detail_for_state(state)

        detail.activated.connect(self._set_active_canvas)
        self._connect_canvas_color_signals(detail.canvas)
        self._draw_state_on_canvas(detail.canvas, state, preserve_limits=False)

        self._inline_detail_widget = detail
        self._inline_detail_lay.addWidget(detail, 1)
        self._detail_widgets[key] = detail
        self._set_active_canvas(detail.canvas, key)

    def _reset_plot_workspace(self):
        self._plot_states.clear()
        self._clear_overview_widgets()
        self._clear_detail_tabs()
        self._clear_waterfall_result()

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
        self._sync_envelope_channel_combo()
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
        self._sync_envelope_channel_combo()
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

        self._sync_envelope_channel_combo()
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
        if hasattr(self, "cmb_signal_type"):
            return self.cmb_signal_type.currentText().strip() == "Velocity"
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
        # The visible Signal Type selector is the main user-flow source of
        # truth.  The hidden legacy cmb_mode is only synchronized for older
        # helpers and must not decide which plot kinds are generated.
        if hasattr(self, "cmb_signal_type"):
            return "vel" if self.cmb_signal_type.currentText().strip() == "Velocity" else "acc"
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
        is_color_plot = bool(state and state.get("kind") in ("stft", "waterfall", "kurtogram"))
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
        has_fft_settings = True

        # Hidden legacy control: normal Plot View always emits Avg + STFT.
        self.cmb_fourier_method.setEnabled(False)
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

            self.lbl_fft_info.setText(
                f"Signal: {self._signal_name()}   |   {base_info}   |   Δf: {df:.6g} Hz   |   Window: {win_t:.6g} s   |   Hop: {hop} samples ({hop_t:.6g} s)   |   Amplitude convention: {unit_lbl}"
            )
        except Exception:
            self.lbl_fft_info.setText("Frequency resolution: -")

    def _set_active_canvas(self, canvas, plot_key):
        old_canvas = self._active_canvas
        if old_canvas is not None and old_canvas is not canvas:
            if self._is_qobject_alive(old_canvas):
                try:
                    old_canvas.set_rect_zoom_active(False)
                except RuntimeError:
                    pass
                except Exception:
                    pass
            else:
                old_canvas = None

        if canvas is not None and not self._is_qobject_alive(canvas):
            self._active_canvas = None
            self._active_plot_key = None
            self._load_axis_editors_from_state(None)
            return

        self._active_canvas = canvas
        self._active_plot_key = plot_key if canvas is not None else None

        for key, mini in list(self._mini_widgets.items()):
            if not self._is_qobject_alive(mini):
                continue
            try:
                mini.set_selected(key == plot_key and getattr(mini, "canvas", None) is canvas)
            except RuntimeError:
                continue

        if self._active_canvas is not None:
            try:
                self._active_canvas.set_rect_zoom_active(self.btn_rect_zoom.isChecked())
            except RuntimeError:
                self._active_canvas = None
                self._active_plot_key = None
            except Exception:
                pass

        self._sync_spectrogram_controls()
        self._load_axis_editors_from_state(self._plot_states.get(self._active_plot_key))

    def _on_plot_page_changed(self, index):
        # The Plot View no longer uses tabs for detail pages; this hook is kept
        # for the Overview-only QTabWidget so any future internal tab changes
        # still wire the active canvas correctly.
        if index < 0:
            return
        widget = self.plot_pages.widget(index)
        if hasattr(widget, "canvas") and hasattr(widget, "plot_key"):
            self._set_active_canvas(widget.canvas, widget.plot_key)

    def _reflow_overview_grid(self):
        # Re-distribute normal mini widgets into the three section columns
        # according to plot kind (time / avg / stft). Waterfall is explicitly
        # excluded because it owns a dedicated Waterfall-tab result area.
        for lay in (self.mini_inner_time_lay, self.mini_inner_avg_lay, self.mini_inner_spec_lay):
            self._strip_mini_layout(lay)

        for key in self._overview_order:
            mini = self._mini_widgets.get(key)
            if mini is None:
                continue
            state = self._plot_states.get(key, {})
            kind = state.get("kind") or self._mini_kinds.get(key, "")
            if str(kind).strip().lower() == "waterfall":
                continue
            self._mini_kinds[key] = kind
            lay = self._section_layout_for_kind(kind)
            # Insert before the trailing stretch (last item).
            insert_at = max(0, lay.count() - 1)
            lay.insertWidget(insert_at, mini)

        self.lbl_overview_empty.setVisible(len(self._mini_widgets) == 0)

    def _select_fallback_active_canvas(self):
        if self.large_tabs.isVisible() and self.large_tabs.count() > 0:
            widget = self.large_tabs.currentWidget()
            if widget is not None and self._is_qobject_alive(widget):
                canvas = getattr(widget, "canvas", None)
                if self._is_qobject_alive(canvas):
                    self._set_active_canvas(canvas, getattr(widget, "plot_key", None))
                    return
        detail = self._inline_detail_widget
        if detail is not None and self._is_qobject_alive(detail):
            canvas = getattr(detail, "canvas", None)
            if self._is_qobject_alive(canvas):
                self._set_active_canvas(canvas, getattr(detail, "plot_key", None))
                return
        for fallback_key in list(self._overview_order):
            mini = self._mini_widgets.get(fallback_key)
            canvas = getattr(mini, "canvas", None) if mini is not None else None
            if self._is_qobject_alive(mini) and self._is_qobject_alive(canvas):
                self._set_active_canvas(canvas, fallback_key)
                return
        self._active_canvas = None
        self._active_plot_key = None
        self._load_axis_editors_from_state(None)

    def _close_plot_by_key(self, key):
        mini = self._mini_widgets.pop(key, None)
        if mini is not None:
            try:
                mini.request_open.disconnect()
            except Exception:
                pass
            try:
                mini.request_large.disconnect()
            except Exception:
                pass
            try:
                mini.request_close.disconnect()
            except Exception:
                pass
            try:
                mini.activated.disconnect()
            except Exception:
                pass
            if self._active_canvas is getattr(mini, "canvas", None):
                self._active_canvas = None
                self._active_plot_key = None
            self._safe_delete_widget(mini)

        self._mini_kinds.pop(key, None)
        self._plot_states.pop(key, None)
        self._overview_order = [k for k in self._overview_order if k != key]

        detail = self._inline_detail_widget
        if detail is not None and getattr(detail, "plot_key", None) == key:
            self._clear_inline_detail()
        else:
            removed_detail = self._detail_widgets.pop(key, None)
            if removed_detail is not None and removed_detail is not getattr(self, "_waterfall_result_widget", None):
                if self._active_canvas is getattr(removed_detail, "canvas", None):
                    self._active_canvas = None
                    self._active_plot_key = None
                self._safe_delete_widget(removed_detail)

        large = self._large_tab_widgets.get(key)
        if large is not None:
            idx = self.large_tabs.indexOf(large)
            if idx >= 0:
                self._on_large_tab_close(idx)
            else:
                self._large_tab_widgets.pop(key, None)
                if self._active_canvas is getattr(large, "canvas", None):
                    self._active_canvas = None
                    self._active_plot_key = None
                self._safe_delete_widget(large)

        self._reflow_overview_grid()
        if self._active_plot_key == key or self._active_canvas is None or not self._is_qobject_alive(self._active_canvas):
            self._select_fallback_active_canvas()

        self._sync_spectrogram_controls()

    def _draw_state_on_canvas(self, canvas, state: dict, preserve_limits=False):
        kind = str(state.get("kind", "")).lower()
        # Defensive routing: waterfall states must land on a WaterfallPlotWidget,
        # everything else on an InteractivePlotWidget / TimeDomainZoomFFTWidget.
        if kind == "waterfall" and not isinstance(canvas, WaterfallPlotWidget):
            raise RuntimeError(
                f"Refusing to draw a waterfall state on {type(canvas).__name__}; "
                "waterfall plots require WaterfallPlotWidget."
            )
        if kind in ("kurtogram", "envelope_spectrum"):
            if EnvelopeKurtogramWidget is None or not isinstance(canvas, EnvelopeKurtogramWidget):
                raise RuntimeError(
                    f"Refusing to draw an Envelope/Kurtogram state on {type(canvas).__name__}; "
                    "these states require EnvelopeKurtogramWidget."
                )
        elif isinstance(canvas, WaterfallPlotWidget):
            raise RuntimeError(
                f"Refusing to draw a non-waterfall state (kind={kind!r}) "
                "on a WaterfallPlotWidget."
            )
        canvas.set_state(
            state,
            color_cfg=self._plot_color_cfg(),
            preserve_view=preserve_limits,
            preserve_annotations=True,
        )

    def _make_detail_for_state(self, state):
        """Factory: pick WaterfallDetailWidget vs DetailPlotWidget."""
        kind = str(state.get("kind", "")).lower()
        if kind == "waterfall":
            return WaterfallDetailWidget(state)
        return DetailPlotWidget(state)

    def _open_large_plot(self, key):
        """Open only ``key`` in the single-selected Large Plot View area."""
        if key not in self._plot_states:
            return

        self._clear_inline_detail()
        self._clear_large_tabs()

        state = self._plot_states[key]
        detail = self._make_detail_for_state(state)
        detail.activated.connect(self._set_active_canvas)
        self._connect_canvas_color_signals(detail.canvas)
        self._draw_state_on_canvas(detail.canvas, state, preserve_limits=False)

        tab_label = state.get("short_title", state.get("name", key))
        if len(tab_label) > 32:
            tab_label = tab_label[:29] + "…"
        self._large_tab_widgets[key] = detail
        self.large_tabs.blockSignals(True)
        try:
            idx = self.large_tabs.addTab(detail, tab_label)
            self.large_tabs.setCurrentIndex(idx)
        finally:
            self.large_tabs.blockSignals(False)
        self.large_tabs.setVisible(True)
        self._set_active_canvas(detail.canvas, key)

    def _on_large_tab_close(self, index: int):
        if index < 0 or index >= self.large_tabs.count():
            return
        widget = self.large_tabs.widget(index)
        if widget is None:
            return

        dead_key = None
        for k, w in list(self._large_tab_widgets.items()):
            if w is widget:
                dead_key = k
                break

        if self._active_canvas is getattr(widget, "canvas", None):
            self._active_canvas = None
            self._active_plot_key = None

        self.large_tabs.blockSignals(True)
        try:
            self.large_tabs.removeTab(index)
        finally:
            self.large_tabs.blockSignals(False)

        if dead_key is not None:
            self._large_tab_widgets.pop(dead_key, None)
        self._safe_delete_widget(widget)

        if self.large_tabs.count() == 0:
            self.large_tabs.setVisible(False)
        self._select_fallback_active_canvas()

    def _on_large_tab_changed(self, index: int):
        if index < 0 or not self.large_tabs.isVisible():
            return
        widget = self.large_tabs.widget(index)
        if widget is None or not self._is_qobject_alive(widget):
            return
        canvas = getattr(widget, "canvas", None)
        if not self._is_qobject_alive(canvas):
            return
        self._set_active_canvas(canvas, getattr(widget, "plot_key", None))

    def _make_mini_for_state(self, st):
        """Factory for normal Plot View mini cards. Waterfall is excluded."""
        kind = str(st.get("kind", "")).lower()
        if kind == "waterfall":
            raise RuntimeError("Waterfall states must not create Plot View mini widgets.")
        return PlotMiniWidget(st)

    def _show_waterfall_result(self, state: dict):
        """Compatibility wrapper: waterfall states now render in Waterfall View."""
        self._show_waterfall_view_result(state)

    def _show_waterfall_view_result(self, state: dict):
        """Render a waterfall state in the dedicated large Waterfall View tab.

        This keeps the Waterfall tab control-only and keeps waterfall states out
        of normal Plot View mini/detail panels, while still using the full
        WaterfallDetailWidget/WaterfallPlotWidget stack (upper heatmap + lower
        extracted order traces).
        """
        if state.get("kind") != "waterfall":
            raise RuntimeError("_show_waterfall_view_result requires a waterfall state.")

        key = state["key"]
        self._plot_states[key] = state
        self._waterfall_result_key = key

        old = getattr(self, "_waterfall_view_widget", None)
        if old is not None:
            try:
                self._waterfall_view_lay.removeWidget(old)
                old.setParent(None)
                old.deleteLater()
            except Exception:
                pass
            self._detail_widgets.pop(getattr(old, "plot_key", key), None)
            self._waterfall_view_widget = None
            self._waterfall_result_widget = None

        if hasattr(self, "lbl_waterfall_view_empty"):
            self.lbl_waterfall_view_empty.setVisible(False)
            try:
                self._waterfall_view_lay.removeWidget(self.lbl_waterfall_view_empty)
            except Exception:
                pass

        detail = WaterfallDetailWidget(state)
        if hasattr(detail.canvas, "set_embedded_toolbar_mode"):
            detail.canvas.set_embedded_toolbar_mode(True)
        detail.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        detail.activated.connect(self._set_active_canvas)
        self._connect_canvas_color_signals(detail.canvas)
        self._draw_state_on_canvas(detail.canvas, state, preserve_limits=False)

        self._waterfall_view_widget = detail
        # Retain the old attribute name only as an alias for legacy cleanup paths;
        # the widget is no longer embedded in the Waterfall controls tab.
        self._waterfall_result_widget = detail
        self._waterfall_view_lay.addWidget(detail, 1)
        self._detail_widgets[key] = detail
        self._set_active_canvas(detail.canvas, key)

        stats = state.get("stats_text", "")
        self.lbl_stats.setText(stats)
        if hasattr(self, "lbl_waterfall_view_meta"):
            self.lbl_waterfall_view_meta.setText(stats or state.get("title", "Waterfall result ready."))
        if hasattr(self, "lbl_wf_status"):
            self.lbl_wf_status.setText(stats or "Waterfall result ready in Waterfall View.")
        if hasattr(self, "lbl_wf_result_empty"):
            self.lbl_wf_result_empty.setText("Last result available in Waterfall View.")

        self.tabs.setCurrentWidget(self.tab_waterfall_view)

        if self._pending_waterfall_open_3d:
            self._pending_waterfall_open_3d = False
            if hasattr(detail.canvas, "open_3d_window"):
                detail.canvas.open_3d_window()

    def _clear_waterfall_result(self):
        widget = getattr(self, "_waterfall_view_widget", None)
        if widget is not None:
            try:
                self._waterfall_view_lay.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()
            except Exception:
                pass
        key = getattr(self, "_waterfall_result_key", None)
        if key is not None:
            self._plot_states.pop(key, None)
            self._detail_widgets.pop(key, None)
        self._waterfall_view_widget = None
        self._waterfall_result_widget = None
        self._waterfall_result_key = None
        if hasattr(self, "lbl_waterfall_view_empty"):
            self.lbl_waterfall_view_empty.setVisible(True)
            if self._waterfall_view_lay.indexOf(self.lbl_waterfall_view_empty) < 0:
                self._waterfall_view_lay.addWidget(self.lbl_waterfall_view_empty, 1)
        if hasattr(self, "lbl_waterfall_view_meta"):
            self.lbl_waterfall_view_meta.setText("")
        if hasattr(self, "lbl_wf_result_empty"):
            self.lbl_wf_result_empty.setText(
                "Waterfall result opens in the Waterfall View tab. Configure acceleration/RPM sources above, then click Plot Waterfall."
            )

    def _on_mini_request_open(self, key: str):
        """Dispatch a mini's request_open: clicking the canvas updates the
        inline detail; pressing the explicit Open Large button opens a Large
        tab.  We tell them apart by checking the QObject sender's class:
        QPushButton emits via clicked, the InteractivePlotWidget emits via
        canvas.clicked → mini._on_canvas_clicked."""
        sender = self.sender()
        # Mini widgets emit request_open from two places:
        #   1) btn_open.clicked → sender() is a QPushButton
        #   2) canvas.clicked   → sender() is the PlotMiniWidget / WaterfallMiniWidget itself
        try:
            from PyQt5.QtWidgets import QPushButton as _QPB
        except Exception:
            _QPB = None
        if _QPB is not None and isinstance(sender, _QPB):
            self._open_large_plot(key)
        else:
            self._show_inline_detail(key)

    def _on_plot_worker_finished(self, states):
        if states and all(st.get("kind") == "waterfall" for st in states):
            self._show_waterfall_view_result(states[0])
            return

        normal_states = [st for st in states if st.get("kind") != "waterfall"]
        if len(normal_states) != len(states):
            self._err("Waterfall results were returned with normal plots; waterfall was kept out of Plot View.")
        preserved_waterfall = {}
        if getattr(self, "_waterfall_result_key", None) in self._plot_states:
            preserved_waterfall[self._waterfall_result_key] = self._plot_states[self._waterfall_result_key]
        self._plot_states = {**preserved_waterfall, **{st["key"]: st for st in normal_states}}
        self._clear_overview_widgets()
        self._clear_detail_tabs()

        all_stats = []
        for st in normal_states:
            w = self._make_mini_for_state(st)
            self._mini_widgets[st["key"]] = w
            self._overview_order.append(st["key"])
            self._mini_kinds[st["key"]] = st.get("kind", "")

            w.request_open.connect(self._show_inline_detail)
            if hasattr(w, "request_large"):
                w.request_large.connect(self._open_large_plot)
            else:
                w.request_open.connect(self._on_mini_request_open)
            w.request_close.connect(self._close_plot_by_key)
            w.activated.connect(self._set_active_canvas)

            self._connect_canvas_color_signals(w.canvas)
            self._draw_state_on_canvas(w.canvas, st, preserve_limits=False)
            if st.get("stats_text"):
                all_stats.append(st["stats_text"])

        self.lbl_stats.setText("\n".join(all_stats[:12]))
        if any(st.get("kind") in ("envelope_spectrum", "kurtogram") for st in normal_states) and hasattr(self, "lbl_env_status"):
            self.lbl_env_status.setText("Envelope/Kurtogram result ready in Plot View.")
        self._reflow_overview_grid()
        self.tabs.setCurrentWidget(self.tab_plot)

        if normal_states:
            first = normal_states[0]
            # Auto-populate the shared detail area with the first new plot.
            self._show_inline_detail(first["key"])

    def _on_plot_worker_error(self, msg: str):
        self.lbl_stats.setText("")
        if hasattr(self, "lbl_wf_status"):
            self.lbl_wf_status.setText("")
        if hasattr(self, "lbl_env_status"):
            self.lbl_env_status.setText("")
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
        if self._active_canvas is not None and self._is_qobject_alive(self._active_canvas):
            try:
                self._active_canvas.zoom_axes(factor)
            except RuntimeError:
                self._active_canvas = None
                self._active_plot_key = None

    def _reset_active_view(self):
        if self._active_canvas is not None and self._is_qobject_alive(self._active_canvas):
            try:
                self._active_canvas.reset_view()
            except RuntimeError:
                self._active_canvas = None
                self._active_plot_key = None

    def _on_rect_zoom_toggled(self, checked):
        if self._active_canvas is not None and self._is_qobject_alive(self._active_canvas):
            try:
                self._active_canvas.set_rect_zoom_active(checked)
            except RuntimeError:
                self._active_canvas = None
                self._active_plot_key = None
        if hasattr(self, "btn_wfv_rect_zoom") and self.btn_wfv_rect_zoom.isChecked() != checked:
            self.btn_wfv_rect_zoom.blockSignals(True)
            self.btn_wfv_rect_zoom.setChecked(checked)
            self.btn_wfv_rect_zoom.blockSignals(False)

    def _open_plot_view_advanced_dialog(self):
        """Modal that exposes axis labels, colormap, color mode, and dB range.

        The widgets themselves still live in the hidden ``_plot_view_aux_widget``;
        we just temporarily reparent that widget into a QDialog while it is
        being edited.  This avoids duplicating widget state.
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("Plot View — Advanced Settings")
        dlg_lay = QVBoxLayout(dlg)
        dlg_lay.setContentsMargins(8, 8, 8, 8)

        old_parent = self._plot_view_aux_widget.parentWidget()
        self._plot_view_aux_widget.setVisible(True)
        dlg_lay.addWidget(self._plot_view_aux_widget, 1)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(dlg.reject)
        btns.accepted.connect(dlg.accept)
        dlg_lay.addWidget(btns)

        dlg.resize(720, 220)
        dlg.exec_()

        # Restore the aux widget to its hidden home in the Plot View tab.
        self._plot_view_aux_widget.setParent(old_parent)
        self._plot_view_aux_widget.setVisible(False)
        # Make sure it is re-inserted into the layout that owned it.
        if old_parent is not None:
            lay = old_parent.layout()
            if lay is not None and lay.indexOf(self._plot_view_aux_widget) < 0:
                # Re-add to layout (it should already be there; this is a safety net).
                lay.insertWidget(lay.count() - 1, self._plot_view_aux_widget)

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
            if state["kind"] in ("envelope_spectrum", "kurtogram"):
                if state["kind"] == "kurtogram":
                    fig = Figure(figsize=(12, 8), dpi=150, constrained_layout=True)
                    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 2.0])
                    ax = fig.add_subplot(gs[0, 0])
                    ax_env = fig.add_subplot(gs[1, 0])
                    K = np.asarray(state.get("kurtosis_grid", []), dtype=np.float64)
                    centers = np.asarray(state.get("center_freq_grid", []), dtype=np.float64)
                    finite_centers = centers[np.isfinite(centers)]
                    if finite_centers.size:
                        x0, x1 = float(np.min(finite_centers)), float(np.max(finite_centers))
                    else:
                        x0, x1 = 0.0, float(state.get("fs", 2.0)) * 0.5
                    mesh = ax.imshow(
                        K, aspect="auto", origin="lower", interpolation="nearest",
                        extent=[x0, x1, -0.5, max(0.5, K.shape[0] - 0.5)],
                        cmap=self.cmb_spec_cmap.currentText().strip() or "viridis",
                    )
                    cbar = fig.colorbar(mesh, ax=ax)
                    cbar.set_label(state.get("colorbar_label", "Excess kurtosis"))
                    best = state.get("best_band") or {}
                    if best:
                        ax.axvspan(float(best["f_low"]), float(best["f_high"]), color="red", alpha=0.15)
                        ax.axvline(float(best["center_freq"]), color="red", lw=1.0)
                    ax.set_xlabel("Band center frequency [Hz]")
                    ax.set_ylabel("Kurtogram level")
                    ax.set_title(state["title"])
                    env_x = np.asarray(state.get("envelope_freqs", []), dtype=np.float64)
                    env_y = np.asarray(state.get("envelope_spectrum", []), dtype=np.float64)
                else:
                    fig = Figure(figsize=(12, 6), dpi=150, constrained_layout=True)
                    ax = fig.add_subplot(111)
                    ax_env = ax
                    env_x = np.asarray(state.get("freqs", []), dtype=np.float64)
                    env_y = np.asarray(state.get("spectrum", []), dtype=np.float64)

                ax_env.plot(env_x, env_y, linewidth=0.9)
                ax_env.set_xlabel(state.get("xlabel", "Frequency [Hz]"))
                ax_env.set_ylabel(state.get("ylabel", "Envelope amplitude [peak]"))
                ax_env.grid(True, alpha=0.3)
                if state["kind"] != "kurtogram":
                    ax_env.set_title(state["title"])
                bearing = state.get("bearing_freqs") or {}
                for label, freq in bearing.items():
                    if label == "shaft_hz":
                        continue
                    try:
                        ax_env.axvline(float(freq), color="red", ls="--", lw=0.8)
                        ax_env.text(float(freq), ax_env.get_ylim()[1], label, rotation=90, va="top", color="red", fontsize=8)
                    except Exception:
                        pass
            elif state["kind"] == "waterfall":
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
                        rpms_axis = getattr(self._active_canvas, "rpms_trace", np.array([]))
                        if rpms_axis.size == 0 or rpms_axis.size != np.asarray(tr).size:
                            rpms_axis = rpms
                        mm = np.isfinite(tr) & np.isfinite(rpms_axis)
                        ax1.plot(rpms_axis[mm], tr[mm], color=color, linewidth=1.6, label=f"Order {order_val:.2f}x")

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

    # ------------------------------------------------------------------ #
    #  TRDSP Logo Header                                                   #
    # ------------------------------------------------------------------ #
    # Replace the asset path constants below with real image file paths
    # to swap placeholders for actual logos.  Supported formats: PNG, JPG.
    # If a path does not exist the widget will fall back to a styled text label.
    _LOGO_APP_PATH   = "logo.png"                    # application / TRDSP logo
    _LOGO_INST1_PATH = "logo.png"                    # right-side TRMOTOR logo
    _LOGO_INST2_PATH = ""                            # optional second right-side logo
    _LOGO_HEIGHT_PX  = 96                              # uniform scaled height for all logos

    def _make_logo_label(self, path: str, fallback_text: str) -> QLabel:
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
        h = self._LOGO_HEIGHT_PX
        resolved_path = resource_path(path) if path else ""
        if resolved_path and os.path.isfile(resolved_path):
            pix = QPixmap(resolved_path)
            if not pix.isNull():
                pix = pix.scaled(
                    pix.width() * h // max(pix.height(), 1), h,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
                lbl.setPixmap(pix)
                lbl.setFixedSize(pix.width(), h)
                return lbl
        # Fallback: styled text when image not found
        lbl.setText(fallback_text)
        lbl.setStyleSheet(
            "font-weight: bold; font-size: 11pt; color: #2c4a7c; letter-spacing: 1px;"
        )
        return lbl

    def _build_logo_header(self) -> QFrame:
        # TRMOTOR-inspired palette: deep black background, red accent, light text.
        frame = QFrame()
        frame.setFrameShape(QFrame.NoFrame)
        frame.setFixedHeight(self._LOGO_HEIGHT_PX + 10)
        frame.setStyleSheet(
            "QFrame { background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            " stop:0 #050505, stop:0.6 #1a1a1a, stop:1 #222222);"
            " border-bottom: 2px solid #D71920; }"
        )

        lay = QHBoxLayout(frame)
        lay.setContentsMargins(12, 4, 12, 4)
        lay.setSpacing(10)

        # Left: app logo + TRDSP name + extended subtitle
        app_logo = self._make_logo_label(self._LOGO_APP_PATH, "")
        lay.addWidget(app_logo)

        title_lbl = QLabel("TRDSP")
        title_lbl.setStyleSheet(
            "color: #FFFFFF; font-size: 18pt; font-weight: bold;"
            " letter-spacing: 4px;"
        )
        title_lbl.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        lay.addWidget(title_lbl)

        # Vertical red separator between TRDSP and the subtitle.
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Plain)
        sep.setStyleSheet("color: #D71920; background: #D71920;")
        sep.setFixedWidth(2)
        sep.setFixedHeight(self._LOGO_HEIGHT_PX - 8)
        lay.addWidget(sep)

        sub_lbl = QLabel("TRMOTOR Digital Signal Processor")
        sub_lbl.setStyleSheet(
            "color: #F5F5F5; font-size: 11pt; font-weight: 500;"
            " letter-spacing: 1px;"
        )
        sub_lbl.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        lay.addWidget(sub_lbl)

        lay.addStretch(1)

        # Right: company / institution logos
        inst1 = self._make_logo_label(self._LOGO_INST1_PATH, "TRMOTOR")
        inst1.setStyleSheet(
            "font-weight: bold; font-size: 12pt; color: #D71920; letter-spacing: 2px;"
        )
        lay.addWidget(inst1)

        if self._LOGO_INST2_PATH:
            inst2 = self._make_logo_label(self._LOGO_INST2_PATH, "")
            inst2.setStyleSheet(inst2.styleSheet() + " color: #F5F5F5;")
            lay.addWidget(inst2)

        return frame

    # ------------------------------------------------------------------ #
    # Tab icons — small, drawn programmatically to avoid bundling assets. #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_tab_icon(kind: str) -> QIcon:
        pix = QPixmap(20, 20)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing, True)

        red = QColor("#D71920")
        dark = QColor("#222222")
        white = QColor("#FFFFFF")

        kind = kind.lower()
        if kind == "controls":
            # Sliders icon: two horizontal bars with knobs.
            p.setPen(dark)
            p.setBrush(dark)
            p.drawRect(3, 6, 14, 2)
            p.drawRect(3, 13, 14, 2)
            p.setBrush(red)
            p.setPen(Qt.NoPen)
            p.drawEllipse(11, 4, 6, 6)
            p.drawEllipse(5, 11, 6, 6)
        elif kind == "waterfall":
            # Stacked horizontal bars suggesting a waterfall heatmap.
            colors = [QColor("#3A3A3A"), QColor("#D71920"),
                      QColor("#F5F5F5"), QColor("#D71920"),
                      QColor("#3A3A3A")]
            p.setPen(Qt.NoPen)
            for i, c in enumerate(colors):
                p.setBrush(c)
                p.drawRect(3, 3 + i * 3, 14, 3)
        elif kind == "plot":
            # Small line-chart with red trace.
            p.setPen(dark)
            p.drawLine(3, 17, 17, 17)
            p.drawLine(3, 3, 3, 17)
            pen = p.pen()
            pen.setWidth(2)
            pen.setColor(red)
            p.setPen(pen)
            from PyQt5.QtCore import QPoint
            pts = [QPoint(4, 14), QPoint(8, 9), QPoint(12, 12), QPoint(16, 5)]
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
        elif kind == "help":
            # Red circle with white "?".
            p.setPen(Qt.NoPen)
            p.setBrush(red)
            p.drawEllipse(2, 2, 16, 16)
            p.setPen(white)
            f = p.font()
            f.setBold(True)
            f.setPointSize(11)
            p.setFont(f)
            p.drawText(pix.rect(), Qt.AlignCenter, "?")
        else:
            p.setPen(dark)
            p.setBrush(dark)
            p.drawRect(4, 4, 12, 12)

        p.end()
        return QIcon(pix)

    # ------------------------------------------------------------------ #
    #  Help / Info tab                                                     #
    # ------------------------------------------------------------------ #
    def _build_help_tab_ui(self):
        outer = QVBoxLayout(self.tab_info)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # --- Search box -------------------------------------------------
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Search:"))
        self._help_search = QLineEdit()
        self._help_search.setPlaceholderText("Type a keyword to filter sections…")
        self._help_search.setClearButtonEnabled(True)
        search_row.addWidget(self._help_search)
        outer.addLayout(search_row)

        # --- Scroll area ------------------------------------------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll, 1)

        inner = QWidget()
        scroll.setWidget(inner)
        vlay = QVBoxLayout(inner)
        vlay.setContentsMargins(4, 4, 4, 4)
        vlay.setSpacing(10)

        self._help_sections = []   # list of (QGroupBox, full_text_lower)

        def _section(title: str, html: str) -> QGroupBox:
            gb = QGroupBox(title)
            gb.setStyleSheet(
                "QGroupBox { font-weight: bold; font-size: 10pt; border: 1px solid #aaa;"
                " border-radius: 4px; margin-top: 6px; padding: 6px; }"
                "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
            )
            lbl = QLabel(html)
            lbl.setWordWrap(True)
            lbl.setTextFormat(Qt.RichText)
            lbl.setOpenExternalLinks(False)
            lbl.setStyleSheet("font-size: 9pt; font-weight: normal;")
            glay = QVBoxLayout(gb)
            glay.addWidget(lbl)
            vlay.addWidget(gb)
            self._help_sections.append((gb, (title + " " + html).lower()))
            return gb

        # ── 1. Quick Start ─────────────────────────────────────────────
        _section("1 · Quick Start / How to Use", """
<ol>
  <li><b>Open a file</b> — click <i>Open File</i> and select an Excel, TDMS, or CSV file.</li>
  <li><b>Select sheet / group</b> — choose the sheet or TDMS group from the drop-down.</li>
  <li><b>Select channel(s)</b> — pick one or more columns from the channel list.</li>
  <li><b>Set fs</b> — enter the correct <i>sampling frequency</i> in Hz. This is the most critical parameter.</li>
  <li><b>Choose signal type</b> — Acceleration or Velocity. This selects the physical signal only.</li>
  <li><b>Configure FFT settings</b> — Buffer Size (NFFT), Overlap, Output, Scale, Amplitude convention.</li>
  <li><b>Click Plot</b> — the Plot View creates Time-domain, Averaged Spectrum, and Spectrogram panels together.</li>
  <li><b>Inspect</b> — zoom, pan, click to annotate. Use the Plot View tab for axis/color controls.</li>
  <li><b>Export</b> — use the Export button for PNG/PDF/TDMS output.</li>
</ol>
<p><b>Normal Plot View:</b> used only for Time-domain, Averaged Spectrum, and Spectrogram panels.
One normal Plot request creates those three plot families for the selected channels.</p>
<p><b>Waterfall workflow:</b> Switch to the <i>Waterfall</i> tab, open your acceleration and RPM files
(can be the same file or different files), configure NFFT / Overlap / Spectra / Buffers, then click
<i>Plot Waterfall</i>. Waterfall opens in the dedicated <i>Waterfall View</i> tab and
<b>does not populate</b> the normal Time-domain / Averaged Spectrum / Spectrogram mini panels.</p>""")

        # ── 2. Signal & FFT Parameters ─────────────────────────────────
        _section("2 · Signal &amp; FFT Parameters", """
<table width="100%" cellspacing="4">
<tr><td width="28%"><b>fs — Sampling frequency (Hz)</b></td>
    <td>The number of samples recorded per second. <i>Wrong fs = wrong frequency axis.</i>
    Every frequency result is scaled by fs. Check your hardware spec or file metadata.</td></tr>
<tr><td><b>NFFT / Buffer Size</b></td>
    <td>Length of each FFT window in samples. Larger NFFT → finer frequency resolution
    (Δf = fs / NFFT) but longer time per frame. Powers of 2 are required (256 … 32768).</td></tr>
<tr><td><b>Overlap (%)</b></td>
    <td>How much consecutive FFT windows overlap. 50 % overlap halves the hop.
    67 % overlap is a common default for run-up data. More overlap = more frames = smoother
    spectrogram, but does not add spectral resolution.</td></tr>
<tr><td><b>Window — Periodic vs Symmetric Hann</b></td>
    <td><i>Periodic</i> (default): w[n] = 0.5 − 0.5·cos(2π·n/N). Recommended for FFT/STFT/Welch.
    Matches MATLAB hann(N,'periodic') and nCode.<br>
    <i>Symmetric</i>: w[n] = 0.5 − 0.5·cos(2π·n/(N−1)). Use only to match legacy symmetric Hann results.</td></tr>
<tr><td><b>Output: Amplitude vs Power</b></td>
    <td><i>Amplitude</i>: one-sided peak amplitude in physical units (g, mm/s …) × your convention factor.<br>
    <i>Power</i>: RMS² per bin. Averaged in the power domain to remove Jensen-inequality bias.</td></tr>
<tr><td><b>Scale: Linear vs dB</b></td>
    <td><i>Linear</i>: raw amplitude or power values.<br>
    <i>dB (Amplitude)</i>: 20·log₁₀(value). <i>dB (Power)</i>: 10·log₁₀(value).
    Never mix the two conventions.</td></tr>
<tr><td><b>Amplitude convention</b></td>
    <td><i>Peak</i>: amplitude of the sinusoidal component (A).<br>
    <i>RMS</i>: A / √2 (for a pure sine).<br>
    <i>Peak-to-Peak</i>: 2·A.</td></tr>
</table>""")

        # ── 3. Waterfall & Order Tracking ──────────────────────────────
        _section("3 · Waterfall &amp; Order Tracking", """
<p><b>Waterfall View display:</b> waterfall configuration stays in the Waterfall tab, while computed results open in the dedicated
Waterfall View tab, separate from normal Plot View. The upper plot is a <b>Frequency vs RPM</b> heatmap with colorbar and order cursors;
the lower plot is an extracted <b>Amplitude vs RPM</b> order-trace display for active orders such as 1.00x and 2.00x.
This is different from an STFT spectrogram, which is <b>Frequency vs Time</b>.</p>
<p><b>Order Tracking Filter:</b> not a cursor. It changes the waterfall data by isolating or rejecting
a selected-order band around <i>f = order × RPM / 60</i>. An order cursor is only a visual guide drawn
on top of the result.</p>
<table width="100%" cellspacing="4">
<tr><td width="28%"><b>Number of Spectra</b></td>
    <td><b>Display density</b> — the minimum number of RPM rows shown in the waterfall after
    RPM-axis interpolation.<br>
    A larger value makes the waterfall look denser and smoother, but
    <i>does not create new physical information</i>.
    The actual resolution is determined by how many FFT frames were computed from the raw signal.
    If you set Num Spectra much larger than the number of raw frames, the extra rows are
    interpolated from the existing frames.<br>
    <b>Tip:</b> start with Num Spectra ≈ number of raw FFT frames (shown in the stats label after plotting).</td></tr>
<tr><td><b>Number of Buffers</b></td>
    <td><b>Smoothing kernel width</b> — how many neighboring raw FFT frames are averaged together
    for each RPM position using a sliding-window mean.<br>
    Num Buffers = 1 means no smoothing (each frame is its own spectrum).
    Larger values smooth noisy spectra and reduce variance, but can blur narrow spectral peaks
    and hide rapid RPM-speed transitions.<br>
    <b>Rule of thumb:</b> keep Num Buffers ≤ 3–5 unless your data is very noisy.</td></tr>
<tr><td><b>Display density vs true resolution</b></td>
    <td>Num Spectra controls how finely the RPM axis is sampled for display.
    Num Buffers controls spectral smoothing in the raw data.
    Increasing Num Spectra beyond the raw frame count only adds interpolated rows.
    True RPM resolution depends on the hop size: ΔRPM ≈ (RPM range) / (number of raw frames).</td></tr>
<tr><td><b>RPM label method</b></td>
    <td>How a single RPM value is assigned to each FFT frame:<br>
    <i>Window-weighted (Hanning)</i>: weighted by the same Hann window as the FFT — recommended for run-up.<br>
    <i>Center sample</i>: RPM at the middle time sample of the frame.<br>
    <i>Median in window</i>: robust median of the RPM samples in the frame.</td></tr>
<tr><td><b>Resample mode</b></td>
    <td>How to put acceleration and RPM on a common time grid when their sampling rates differ:<br>
    <i>Resample RPM → Acc timebase</i> (recommended): RPM is resampled to the acceleration grid. FFT uses fs_acc.<br>
    <i>Resample Acc → RPM timebase</i>: acceleration is resampled to the RPM grid. FFT uses fs_rpm.<br>
    <i>Trim only (same fs)</i>: signals must have identical fs; only lengths are trimmed to match.</td></tr>
<tr><td><b>Frequency Waterfall</b></td>
    <td>Classic waterfall: X-axis = frequency (Hz), Y-axis = RPM, color = amplitude/power.
    Constant-frequency features appear as horizontal lines.
    Order-related features (which scale with RPM) appear as diagonal lines.</td></tr>
<tr><td><b>Order Tracking Filter</b> (primary)</td>
    <td>nCode-like adaptive band-pass / band-stop centered on f<sub>c</sub>(t) = k · RPM(t) / 60.
    For each FFT frame a Butterworth-shaped frequency mask is applied so that only the
    chosen order k (1x, 2x, …) is <i>extracted/isolated</i> in the displayed waterfall —
    every other order ridge is suppressed.  This is the recommended way to look at a
    single order.  Set target order in the Waterfall tab.</td></tr>
<tr><td><b>Order Cursor</b> (visual only)</td>
    <td><i>Not a filter.</i> A diagonal line drawn at f = order · RPM/60 over the existing
    waterfall, used as a visual guide.  Amplitude is read out by interpolation along the
    line — the underlying spectrum is unchanged.  Order cursors are an advanced/optional
    overlay; if needed they can still be added via the waterfall detail toolbar.</td></tr>
<tr><td><b>Computed Order Tracking (COT)</b> (advanced)</td>
    <td>Different concept from the Order Tracking Filter: acceleration is resampled from
    uniform time to uniform shaft-angle (θ = ∫ω dt) and an order-domain FFT is computed
    in the angle domain.  Hidden by default — kept for advanced users.  Requires a clean,
    monotonic RPM signal.  <b>Do not confuse with the Order Tracking Filter.</b></td></tr>
<tr><td><b>Order Waterfall</b> (advanced)</td>
    <td>Converts each frame from Hz to order axis. Hidden by default — the standard Frequency
    Waterfall plus the Order Tracking Filter usually does what you want.</td></tr>
</table>""")

        # ── 4. Amplitude & Scale Conventions ───────────────────────────
        _section("4 · Amplitude &amp; Scale Conventions", """
<ul>
  <li><b>Peak amplitude</b> of a pure sine A·sin(2πft): the value A itself.</li>
  <li><b>RMS</b> of a pure sine: A / √2 ≈ 0.707·A.</li>
  <li><b>Peak-to-Peak</b> of a pure sine: 2·A.</li>
  <li><b>Power (spectral)</b>: stored as RMS², averaged across frames. Independent of Peak/RMS/P-P selection.</li>
  <li><b>dB (Amplitude)</b>: 20·log₁₀(amplitude). A factor-of-2 amplitude change = +6 dB.</li>
  <li><b>dB (Power)</b>: 10·log₁₀(power). A factor-of-2 power change = +3 dB.</li>
  <li><b>DC bin</b> (0 Hz) and <b>Nyquist bin</b> (fs/2 for even NFFT): not doubled when computing one-sided amplitude.</li>
</ul>
<p><b>Velocity integration:</b> acceleration is high-pass filtered at 10 Hz (4th-order Butterworth, zero-phase)
before integration to suppress DC drift, then converted to mm/s.</p>""")

        # ── 5. Best Practices & Warnings ───────────────────────────────
        _section("5 · Best Practices &amp; Warnings", """
<ul>
  <li><b>Wrong fs</b> shifts the entire frequency axis — always verify fs matches your hardware setting.</li>
  <li><b>Large Num Spectra</b> controls display interpolation density only; it makes the plot look smoother but adds no real information beyond the raw frame count.</li>
  <li><b>Large Num Buffers</b> smooths neighboring frames and may hide sharp spectral peaks — use sparingly.</li>
  <li><b>STFT vs Waterfall:</b> STFT is Frequency vs Time; Waterfall is Frequency vs RPM and appears only in the dedicated Waterfall View tab.</li>
  <li><b>Too much overlap</b> (e.g. 95 %) creates very many frames but does not improve frequency resolution.</li>
  <li><b>RPM = 0 or NaN in data</b> is treated as an invalid frame and excluded from the waterfall.</li>
  <li><b>RPM spikes</b> can corrupt order analysis — use the spike-rejection option (MAD-based).</li>
  <li><b>1X</b> typically relates to shaft rotational fundamental. In a waterfall it appears as a diagonal line with slope 1/60.</li>
  <li><b>2X</b> may indicate misalignment, looseness, or harmonic distortion — interpret in machine context.</li>
  <li><b>Order Waterfall</b> is most useful for run-up / coast-down data where RPM changes significantly.</li>
  <li><b>Computed Order Tracking</b> requires a clean, monotonic RPM signal — large gaps or spikes degrade results.</li>
  <li><b>dB dynamic range</b>: the color bar clamps to (peak − dyn_range_dB). Increase dyn_range_dB to see weaker features.</li>
</ul>""")

        vlay.addStretch(1)

        # Wire search box
        self._help_search.textChanged.connect(self._on_help_search)

    def _on_help_search(self, text: str):
        kw = text.strip().lower()
        for gb, full_text in self._help_sections:
            gb.setVisible(not kw or kw in full_text)

    def _make_help_button(self, title: str, html: str, topic: str = None) -> QToolButton:
        btn = QToolButton()
        btn.setText("?")
        btn.setFixedWidth(22)
        btn.setToolTip(f"Click for explanation of {topic or title}")
        btn.clicked.connect(lambda _=False, t=title, h=html: QMessageBox.information(self, t, h))
        return btn

    def _help_html(self, key: str) -> str:
        help_map = {
            "fs": "<b>Sampling frequency fs</b><br><br>Samples per second in Hz. It sets the frequency axis for FFT, averaged spectra, STFT, and velocity integration. If fs is too high or too low, every frequency feature shifts by the same ratio.<br><br><b>Start with:</b> the acquisition hardware sample rate or file metadata. Verify before interpreting orders or resonances.",
            "nfft": "<b>Buffer Size / NFFT</b><br><br>Number of samples in each FFT window. Frequency spacing is Δf = fs / NFFT. Larger NFFT gives finer frequency bins but longer time/RPM averaging per frame; smaller NFFT follows transients better but has coarser frequency resolution.<br><br><b>Start with:</b> 4096 or 8192 for many vibration run-up datasets, then adjust from the displayed Δf.",
            "overlap": "<b>Overlap %</b><br><br>How much adjacent FFT windows overlap. More overlap creates more frames and a smoother-looking STFT/waterfall, but it does not improve true frequency resolution.<br><br><b>Too high:</b> slower processing and highly correlated frames. <b>Start with:</b> 50–67%; use 75–90% only when you need denser display during rapid changes.",
            "output": "<b>Output: Amplitude vs Power</b><br><br>Amplitude shows one-sided spectral magnitude in physical units using the selected Peak/RMS/P-P convention. Power stores RMS² per bin and averages in the power domain, which is often better for energy comparisons.<br><br><b>Use Amplitude</b> for peak readouts; <b>use Power</b> for energy/noise comparisons.",
            "scale": "<b>Scale: Linear vs dB</b><br><br>Linear keeps physical values. dB compresses the dynamic range so small components near large peaks are visible. Amplitude dB uses 20·log10(value); Power dB uses 10·log10(value).<br><br><b>Warning:</b> do not compare amplitude-dB and power-dB as if they use the same reference.",
            "amp": "<b>Amplitude conversion</b><br><br>Peak is the sinusoid amplitude A, RMS is A/√2 for a pure sine, and Peak-to-Peak is 2A. This affects amplitude spectra and waterfall levels when Output is Amplitude.<br><br><b>Start with:</b> RMS for vibration severity comparisons; Peak for component amplitude readouts.",
            "window": "<b>Window type: Periodic vs Symmetric Hann</b><br><br>Periodic Hann is recommended for FFT/STFT/Welch-style spectral analysis and matches MATLAB hann(N,'periodic') / many analyzer conventions. Symmetric Hann matches numpy.hanning() and some legacy scripts.<br><br><b>Start with:</b> Periodic unless you must reproduce older symmetric-window results.",
            "data_source": "<b>Data source: Unfiltered vs Filtered</b><br><br>Selects whether normal plotting uses the raw channel or the Butterworth-filtered signal. Filtered is useful for focusing on a band or removing low-frequency drift before spectra.<br><br><b>Warning:</b> filtering changes amplitudes and phase characteristics; document it when exporting plots.",
            "filter_enable": "<b>Butterworth filter enable</b><br><br>Applies a zero-phase forward/backward Butterworth filter before plotting. Zero-phase avoids time shift but doubles the effective filtering action.<br><br><b>Use when:</b> removing drift, isolating a band, or rejecting unwanted content. Avoid filtering unless it supports the analysis question.",
            "filter_type": "<b>Filter type</b><br><br>Low-pass keeps content below Cutoff 1; High-pass keeps content above Cutoff 1; Band-pass keeps between Cutoff 1 and Cutoff 2; Band-reject removes that band.<br><br><b>Check:</b> cutoff values must be below Nyquist (fs/2) and ordered correctly for band filters.",
            "filter_order": "<b>Filter order</b><br><br>Higher order makes a steeper transition around the cutoff, but can increase ringing and edge artifacts. Because the app uses zero-phase filtering, the effective slope is stronger than a one-pass filter.<br><br><b>Start with:</b> 2–4; increase only when the transition band must be sharper.",
            "cutoffs": "<b>Cutoff 1 / Cutoff 2</b><br><br>Frequency limits in Hz for the Butterworth filter. Single-cutoff filters use Cutoff 1. Band-pass and Band-reject use Cutoff 1 as the lower edge and Cutoff 2 as the upper edge.<br><br><b>Warning:</b> cutoffs at or above Nyquist (fs/2) are invalid and very low cutoffs can require long signals to behave well.",
            "crop": "<b>Crop time window</b><br><br>Restricts analysis to Start–End seconds before plotting/export. This is useful for steady operating regions or a chosen run-up segment.<br><br><b>Too narrow:</b> may leave too few samples for NFFT or produce unstable low-frequency results. Start with a window that contains several FFT buffers.",
            "signal_type": "<b>Signal type: Acceleration vs Velocity</b><br><br>Acceleration plots measured vibration in g. Velocity integrates acceleration after high-pass conditioning and displays velocity in mm/s.<br><br><b>Use Acceleration</b> for high-frequency bearing/gear content; <b>use Velocity</b> for many machine-vibration severity standards and lower-frequency motion.",
            "export_tdms": "<b>Export cropped TDMS</b><br><br>Saves selected channels after the current crop settings, zero-aligning the time axis and preserving useful metadata where possible.<br><br><b>Check before export:</b> fs, selected channels, crop start/end, and whether Skip first sample is intended.",
            "wf_acc_fs": "<b>Waterfall Acc fs</b><br><br>Sample rate of the acceleration signal. It defines the FFT frequency axis unless you choose to resample acceleration to the RPM timebase.<br><br><b>Start with:</b> the actual accelerometer acquisition rate; wrong values shift all waterfall frequencies and order-filter locations.",
            "wf_rpm_fs": "<b>Waterfall RPM fs</b><br><br>Sample rate of the RPM/tach signal. The app aligns RPM with acceleration before assigning an RPM to each FFT frame.<br><br><b>Start with:</b> the tach acquisition rate. If RPM is sparse/noisy, use linear interpolation and inspect the stats/status text.",
            "wf_resample": "<b>Resample mode</b><br><br>Controls which signal is interpolated so acceleration and RPM share a timebase. Resample RPM → Acc timebase usually preserves the vibration waveform and is recommended when acceleration is sampled faster.<br><br><b>Use Trim only</b> only when both files already share fs and timing.",
            "wf_interp": "<b>Interpolation mode</b><br><br>Linear interpolation is smooth and usually best for RPM because speed changes continuously. Nearest preserves original samples but can create stair-step RPM labels.<br><br><b>Start with:</b> Linear for run-up/coast-down data.",
            "wf_rpm_label": "<b>RPM label method</b><br><br>Assigns one RPM value to each FFT frame. Center sample is simple, Median rejects spikes, and Window-weighted Hann matches the FFT window weighting.<br><br><b>Start with:</b> Window-weighted (Hanning) for run-up/coast-down data.",
            "wf_spectra": "<b>Num Spectra</b><br><br>Controls display interpolation density: the minimum number of RPM rows in the waterfall image. It makes the display smoother but does not increase physical resolution beyond the raw FFT frame count.<br><br><b>Start with:</b> near the raw frame count shown in the stats text.",
            "wf_buffers": "<b>Num Buffers</b><br><br>Sliding average width across neighboring FFT frames before display. Larger values reduce random noise but blur narrow peaks and fast speed transitions.<br><br><b>Start with:</b> 1 (no smoothing), then 3–5 only if the waterfall is very noisy.",
            "wf_otf_enable": "<b>Order Tracking Filter enable</b><br><br>Enables a true filter around the selected order trajectory f = order × RPM / 60. This changes the displayed waterfall data; it is not merely a cursor overlay.<br><br><b>Use when:</b> isolating one order family or rejecting it with band-stop mode.",
            "wf_target_order": "<b>Target order k</b><br><br>The shaft order to isolate or reject. For example 1.0x follows f = RPM/60, 2.0x follows twice running speed.<br><br><b>Start with:</b> 1.0 for running speed, then known gear/blade/bearing orders as needed.",
            "wf_half_bw": "<b>Half-bandwidth</b><br><br>Width on either side of the target order trajectory. Too narrow can miss smeared or speed-varying energy; too wide includes neighboring orders/noise.<br><br><b>Start with:</b> a small order bandwidth (such as 0.05–0.10 order) or a Hz width matched to expected speed variation.",
            "wf_bw_unit": "<b>Bandwidth unit: Order vs Hz</b><br><br>Order bandwidth scales with RPM, keeping a proportional band around k×RPM/60. Hz bandwidth stays fixed in frequency regardless of RPM.<br><br><b>Use Order</b> for rotating-order phenomena; <b>use Hz</b> for fixed-frequency resonances.",
            "wf_otf_type": "<b>Band-pass vs Band-stop</b><br><br>Band-pass keeps only the selected-order neighborhood. Band-stop removes that order neighborhood while leaving other content visible.<br><br><b>Warning:</b> either mode changes the data, unlike visual order cursors.",
            "wf_trace": "<b>Show extracted trace vs RPM</b><br><br>Displays the selected-order amplitude trend in the lower waterfall panel. This helps quantify how an order grows or decays with speed.<br><br><b>Interpretation:</b> the trace follows the configured order/band settings and should be compared with the heatmap context.",
            "wf_open_3d": "<b>Open 3D waterfall</b><br><br>Automatically opens a 3D standalone waterfall view after computation. Useful for presentations and shape inspection, but the 2D heatmap is usually better for precise cursor reading.<br><br><b>Tip:</b> leave off during iterative parameter tuning.",
            "wf_open_2d": "<b>Open 2D Standalone</b><br><br>Opens the waterfall heatmap in a separate Matplotlib window using the current colormap, color limits, zoom viewport, and order cursors when available.<br><br><b>Use for:</b> publication-style export or comparing a standalone view with the main GUI.",
        }
        return help_map[key]

    def _show_num_spectra_help(self):
        QMessageBox.information(
            self, "Number of Spectra — Explanation",
            "<b>Number of Spectra</b><br><br>"
            "Controls how many RPM rows appear in the waterfall analysis/display grid after interpolation.<br><br>"
            "<b>Frequency resolution:</b> unchanged by this setting; Δf = fs / NFFT, and only NFFT/effective window length changes true frequency-bin spacing.<br><br>"
            "<b>What it does:</b> the raw FFT computation produces one spectrum per hop "
            "(frame), typically a few hundred to a few thousand frames. "
            "After sorting by RPM, the code interpolates those frames onto a uniform RPM grid "
            "with this many analysis rows; denser image rows are display-only interpolation.<br><br>"
            "<b>Important:</b> increasing Num Spectra beyond the number of raw frames only "
            "adds <i>interpolated</i> rows — it does not create new physical information. "
            "The apparent smoothness of the waterfall increases, but the underlying data "
            "resolution does not.<br><br>"
            "<b>Rule of thumb:</b> set Num Spectra ≈ number of raw FFT frames for a 1-to-1 "
            "display. Check the stats label after plotting to see how many raw frames were used.",
        )

    def _show_num_buffers_help(self):
        QMessageBox.information(
            self, "Number of Buffers — Explanation",
            "<b>Number of Buffers</b><br><br>"
            "Controls how many neighboring raw FFT frames are averaged together for each "
            "RPM position using a sliding-window mean.<br><br>"
            "<b>What it does:</b><br>"
            "  Num Buffers = 1 → no smoothing; each frame is its own spectrum.<br>"
            "  Num Buffers = N → each RPM slice is the average of ±N/2 surrounding frames.<br><br>"
            "<b>Effect:</b> smooths noisy spectra and reduces random variance. "
            "However, large values can blur sharp spectral peaks and hide rapid RPM-speed "
            "transitions that are physically real.<br><br>"
            "<b>Rule of thumb:</b> keep Num Buffers ≤ 3–5 for typical run-up / coast-down data. "
            "Only increase it if your data is very noisy and you have confirmed that the "
            "features of interest are broad.",
        )

    def _build_waterfall_view_tab(self):
        """Build the dedicated large waterfall result tab.

        The Waterfall tab remains a control/configuration surface; computed
        waterfall states are rendered here with the full WaterfallDetailWidget
        so the upper heatmap and lower order-trace plot both have room.
        """
        lay = QVBoxLayout(self.tab_waterfall_view)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        toolbar = QHBoxLayout()
        self.btn_wfv_rect_zoom = QPushButton("Rectangle Zoom")
        self.btn_wfv_rect_zoom.setCheckable(True)
        self.btn_wfv_zoom_in = QPushButton("Zoom In")
        self.btn_wfv_zoom_out = QPushButton("Zoom Out")
        self.btn_wfv_reset_view = QPushButton("Reset View")
        self.btn_wfv_export = QPushButton("Export Current Plot")
        self.btn_wfv_open_3d = QPushButton("Open 3D")
        self.btn_wfv_open_2d = QPushButton("Open 2D Standalone")
        self.btn_waterfall_advanced_settings = QPushButton("Advanced Settings")
        self.btn_waterfall_advanced_settings.setToolTip(
            "Open advanced display settings for the waterfall plot."
        )
        self.btn_wfv_clear = QPushButton("Clear annotations/cursors")

        for btn in (
            self.btn_wfv_rect_zoom,
            self.btn_wfv_zoom_in,
            self.btn_wfv_zoom_out,
            self.btn_wfv_reset_view,
            self.btn_wfv_export,
            self.btn_wfv_open_3d,
            self.btn_wfv_open_2d,
            self.btn_waterfall_advanced_settings,
            self.btn_wfv_clear,
        ):
            toolbar.addWidget(btn)
        toolbar.addStretch(1)
        lay.addLayout(toolbar)

        self.lbl_waterfall_view_meta = QLabel("")
        self.lbl_waterfall_view_meta.setWordWrap(True)
        self.lbl_waterfall_view_meta.setStyleSheet("color: #444;")
        lay.addWidget(self.lbl_waterfall_view_meta)

        self._waterfall_view_host = QWidget()
        self._waterfall_view_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._waterfall_view_lay = QVBoxLayout(self._waterfall_view_host)
        self._waterfall_view_lay.setContentsMargins(0, 0, 0, 0)
        self._waterfall_view_lay.setSpacing(0)

        self.lbl_waterfall_view_empty = QLabel(
            "No waterfall result yet. Configure acceleration/RPM sources in the Waterfall tab, "
            "then click Plot Waterfall."
        )
        self.lbl_waterfall_view_empty.setAlignment(Qt.AlignCenter)
        self.lbl_waterfall_view_empty.setStyleSheet("color: #666; font-size: 14px;")
        self.lbl_waterfall_view_empty.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._waterfall_view_lay.addWidget(self.lbl_waterfall_view_empty, 1)
        lay.addWidget(self._waterfall_view_host, 1)

        self._waterfall_view_widget = None
        self._waterfall_result_widget = None
        self._waterfall_result_key = None

        self.btn_wfv_rect_zoom.toggled.connect(self._on_waterfall_view_rect_zoom_toggled)
        self.btn_wfv_zoom_in.clicked.connect(lambda: self._zoom_active(0.8))
        self.btn_wfv_zoom_out.clicked.connect(lambda: self._zoom_active(1.25))
        self.btn_wfv_reset_view.clicked.connect(self._reset_active_view)
        self.btn_wfv_export.clicked.connect(self._export_current_plot)
        self.btn_wfv_open_3d.clicked.connect(self._open_active_waterfall_3d)
        self.btn_wfv_open_2d.clicked.connect(self._open_active_waterfall_2d)
        self.btn_waterfall_advanced_settings.clicked.connect(self._open_waterfall_advanced_settings)
        self.btn_wfv_clear.clicked.connect(self._clear_active_waterfall_annotations)

    def _on_waterfall_view_rect_zoom_toggled(self, checked):
        self._on_rect_zoom_toggled(checked)
        if hasattr(self, "btn_rect_zoom") and self.btn_rect_zoom.isChecked() != checked:
            self.btn_rect_zoom.blockSignals(True)
            self.btn_rect_zoom.setChecked(checked)
            self.btn_rect_zoom.blockSignals(False)

    def _go_to_waterfall_view(self):
        if hasattr(self, "tab_waterfall_view"):
            self.tabs.setCurrentWidget(self.tab_waterfall_view)
            widget = getattr(self, "_waterfall_view_widget", None)
            if widget is not None:
                self._set_active_canvas(widget.canvas, widget.plot_key)

    def _open_waterfall_advanced_settings(self):
        canvas = self._active_canvas
        if canvas is None or not hasattr(canvas, "_open_waterfall_advanced_settings"):
            canvas = getattr(getattr(self, "_waterfall_view_widget", None), "canvas", None)
        if canvas is None or not hasattr(canvas, "_open_waterfall_advanced_settings"):
            self._err("There is no active waterfall plot for advanced settings.")
            return
        canvas._open_waterfall_advanced_settings()

    def _open_active_waterfall_3d(self):
        canvas = self._active_canvas
        if canvas is None or not hasattr(canvas, "open_3d_window"):
            self._err("There is no active waterfall plot to open in 3D.")
            return
        canvas.open_3d_window()

    def _open_active_waterfall_2d(self):
        canvas = self._active_canvas
        if canvas is None or not hasattr(canvas, "open_2d_standalone_window"):
            self._err("There is no active waterfall plot to open standalone.")
            return
        canvas.open_2d_standalone_window()

    def _clear_active_waterfall_annotations(self):
        canvas = self._active_canvas
        if canvas is None:
            return
        if hasattr(canvas, "clear_click_annotations"):
            canvas.clear_click_annotations()
        if hasattr(canvas, "clear_cursors"):
            canvas.clear_cursors()

    def _build_envelope_tab_ui(self):
        """Build the modular Envelope/Kurtogram diagnostic controls."""
        lay = QVBoxLayout(self.tab_envelope)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        self.lbl_env_status = QLabel("Load a data file, choose a vibration channel, then compute an envelope spectrum or kurtogram.")
        self.lbl_env_status.setWordWrap(True)
        self.lbl_env_status.setStyleSheet("color: #444;")
        lay.addWidget(self.lbl_env_status)

        grp_signal = QGroupBox("Signal")
        sig = QGridLayout(grp_signal)
        self.cmb_env_channel = QComboBox()
        self.env_fs_spin = QDoubleSpinBox()
        self.env_fs_spin.setRange(1e-9, 1_000_000_000.0)
        self.env_fs_spin.setDecimals(6)
        self.env_fs_spin.setValue(float(self.fs_spin.value()))
        self.env_fs_spin.setToolTip("Sampling frequency used for this envelope/kurtogram computation.")
        self.chk_env_crop = QCheckBox("Use crop window")
        self.chk_env_crop.setChecked(False)
        self.env_t0_spin = QDoubleSpinBox()
        self.env_t0_spin.setRange(0.0, 1_000_000_000.0)
        self.env_t0_spin.setDecimals(6)
        self.env_t0_spin.setValue(0.0)
        self.env_t1_spin = QDoubleSpinBox()
        self.env_t1_spin.setRange(0.0, 1_000_000_000.0)
        self.env_t1_spin.setDecimals(6)
        self.env_t1_spin.setValue(1.0)
        sig.addWidget(QLabel("Acceleration channel:"), 0, 0)
        sig.addWidget(self.cmb_env_channel, 0, 1, 1, 3)
        sig.addWidget(QLabel("Sampling frequency (Hz):"), 1, 0)
        sig.addWidget(self.env_fs_spin, 1, 1)
        sig.addWidget(self.chk_env_crop, 1, 2)
        sig.addWidget(QLabel("Start (s):"), 2, 0)
        sig.addWidget(self.env_t0_spin, 2, 1)
        sig.addWidget(QLabel("End (s):"), 2, 2)
        sig.addWidget(self.env_t1_spin, 2, 3)
        lay.addWidget(grp_signal)

        grp_analysis = QGroupBox("Envelope / Kurtogram settings")
        ana = QGridLayout(grp_analysis)
        self.env_band_low_spin = QDoubleSpinBox()
        self.env_band_low_spin.setRange(0.0, 1_000_000_000.0)
        self.env_band_low_spin.setDecimals(6)
        self.env_band_low_spin.setSpecialValueText("None")
        self.env_band_low_spin.setValue(0.0)
        self.env_band_high_spin = QDoubleSpinBox()
        self.env_band_high_spin.setRange(0.0, 1_000_000_000.0)
        self.env_band_high_spin.setDecimals(6)
        self.env_band_high_spin.setSpecialValueText("None")
        self.env_band_high_spin.setValue(0.0)
        self.env_levels_spin = QSpinBox()
        self.env_levels_spin.setRange(1, 9)
        self.env_levels_spin.setValue(5)
        self.btn_env_compute = QPushButton("Compute Envelope Spectrum")
        self.btn_env_kurtogram = QPushButton("Compute Kurtogram + Auto Band")
        self.btn_env_clear = QPushButton("Clear")
        self.btn_env_export = QPushButton("Export Figure")
        ana.addWidget(QLabel("Bandpass low (Hz):"), 0, 0)
        ana.addWidget(self.env_band_low_spin, 0, 1)
        ana.addWidget(QLabel("Bandpass high (Hz):"), 0, 2)
        ana.addWidget(self.env_band_high_spin, 0, 3)
        ana.addWidget(QLabel("Kurtogram levels:"), 1, 0)
        ana.addWidget(self.env_levels_spin, 1, 1)
        ana.addWidget(self.btn_env_compute, 2, 0)
        ana.addWidget(self.btn_env_kurtogram, 2, 1)
        ana.addWidget(self.btn_env_clear, 2, 2)
        ana.addWidget(self.btn_env_export, 2, 3)
        lay.addWidget(grp_analysis)

        grp_bearing = QGroupBox("Optional bearing fault frequencies")
        brg = QGridLayout(grp_bearing)
        self.env_rpm_spin = QDoubleSpinBox(); self.env_rpm_spin.setRange(0.0, 1_000_000_000.0); self.env_rpm_spin.setDecimals(6)
        self.env_balls_spin = QSpinBox(); self.env_balls_spin.setRange(1, 1000); self.env_balls_spin.setValue(8)
        self.env_ball_d_spin = QDoubleSpinBox(); self.env_ball_d_spin.setRange(0.0, 1_000_000_000.0); self.env_ball_d_spin.setDecimals(6); self.env_ball_d_spin.setValue(10.0)
        self.env_pitch_d_spin = QDoubleSpinBox(); self.env_pitch_d_spin.setRange(0.0, 1_000_000_000.0); self.env_pitch_d_spin.setDecimals(6); self.env_pitch_d_spin.setValue(50.0)
        self.env_angle_spin = QDoubleSpinBox(); self.env_angle_spin.setRange(-89.999, 89.999); self.env_angle_spin.setDecimals(6); self.env_angle_spin.setValue(0.0)
        self.btn_env_bearing = QPushButton("Calculate BPFO/BPFI/BSF/FTF")
        self.lbl_env_bearing = QLabel("Bearing frequencies: not calculated")
        self.lbl_env_bearing.setWordWrap(True)
        self._env_bearing_freqs = None
        brg.addWidget(QLabel("RPM:"), 0, 0); brg.addWidget(self.env_rpm_spin, 0, 1)
        brg.addWidget(QLabel("Number of balls:"), 0, 2); brg.addWidget(self.env_balls_spin, 0, 3)
        brg.addWidget(QLabel("Ball diameter:"), 1, 0); brg.addWidget(self.env_ball_d_spin, 1, 1)
        brg.addWidget(QLabel("Pitch diameter:"), 1, 2); brg.addWidget(self.env_pitch_d_spin, 1, 3)
        brg.addWidget(QLabel("Contact angle (deg):"), 2, 0); brg.addWidget(self.env_angle_spin, 2, 1)
        brg.addWidget(self.btn_env_bearing, 2, 2, 1, 2)
        brg.addWidget(self.lbl_env_bearing, 3, 0, 1, 4)
        lay.addWidget(grp_bearing)
        lay.addStretch(1)

        self.btn_env_compute.clicked.connect(lambda: self._start_envelope_diagnostic("envelope_spectrum"))
        self.btn_env_kurtogram.clicked.connect(lambda: self._start_envelope_diagnostic("kurtogram"))
        self.btn_env_clear.clicked.connect(self._clear_envelope_result)
        self.btn_env_export.clicked.connect(self._export_current_plot)
        self.btn_env_bearing.clicked.connect(self._calculate_env_bearing_freqs)
        self.fs_spin.valueChanged.connect(lambda v: self.env_fs_spin.setValue(float(v)))
        self._sync_envelope_channel_combo()

    def _sync_envelope_channel_combo(self):
        if not hasattr(self, "cmb_env_channel"):
            return
        current = self.cmb_env_channel.currentText()
        self.cmb_env_channel.blockSignals(True)
        self.cmb_env_channel.clear()
        for i in range(self.list_channels.count()):
            self.cmb_env_channel.addItem(self.list_channels.item(i).text())
        if current:
            idx = self.cmb_env_channel.findText(current)
            if idx >= 0:
                self.cmb_env_channel.setCurrentIndex(idx)
        if self.cmb_env_channel.count() == 0 and self.list_channels.currentItem() is not None:
            self.cmb_env_channel.addItem(self.list_channels.currentItem().text())
        self.cmb_env_channel.blockSignals(False)

    def _clear_envelope_result(self):
        keys = [k for k, st in self._plot_states.items() if st.get("kind") in ("envelope_spectrum", "kurtogram")]
        for key in keys:
            self._close_plot_by_key(key)
        if hasattr(self, "lbl_env_status"):
            self.lbl_env_status.setText("Envelope/Kurtogram result cleared.")

    def _calculate_env_bearing_freqs(self):
        if extract_bearing_fault_frequencies is None:
            self._err("Bearing frequency backend is not available.")
            return
        try:
            freqs = extract_bearing_fault_frequencies(
                rpm=float(self.env_rpm_spin.value()),
                n_balls=int(self.env_balls_spin.value()),
                d_ball=float(self.env_ball_d_spin.value()),
                d_pitch=float(self.env_pitch_d_spin.value()),
                contact_angle=float(self.env_angle_spin.value()),
            )
        except Exception as e:
            self._err(f"Bearing frequency calculation failed:\n{e}")
            return
        self._env_bearing_freqs = freqs
        self.lbl_env_bearing.setText(
            "Bearing frequencies: " + ", ".join(f"{k}={v:.6g} Hz" for k, v in freqs.items())
        )

    def _envelope_band_from_ui(self):
        low = float(self.env_band_low_spin.value())
        high = float(self.env_band_high_spin.value())
        if low <= 0.0 and high <= 0.0:
            return None
        if high <= low:
            raise ValueError("Envelope bandpass high frequency must be greater than low frequency.")
        return (low, high)

    def _start_envelope_diagnostic(self, diagnostic_kind: str):
        if not _HAVE_ENVELOPE_BACKEND:
            self._err("Envelope/Kurtogram backend is not available.")
            return
        self._sync_envelope_channel_combo()
        channel = self.cmb_env_channel.currentText().strip() if hasattr(self, "cmb_env_channel") else ""
        if not channel:
            self._err("No acceleration channel selected for Envelope/Kurtogram analysis.")
            return
        if not self.file_path or not self.file_type:
            self._err("No data file is loaded.")
            return
        fs = float(self.env_fs_spin.value())
        crop = self.chk_env_crop.isChecked()
        t0 = float(self.env_t0_spin.value())
        t1 = float(self.env_t1_spin.value())
        if crop and t1 <= t0:
            self._err("Envelope crop end time must be greater than start time.")
            return
        try:
            band = self._envelope_band_from_ui()
        except ValueError as e:
            self._err(str(e))
            return
        cfg = {
            "file_path": self.file_path,
            "file_type": self.file_type,
            "group_or_sheet": self.cmb_sheet.currentText(),
            "names": [channel],
            "fs": fs,
            "crop": crop,
            "t0": t0,
            "t1": t1,
            "skip_first": False,
            "mode": "Envelope / Kurtogram",
            "signal_kind": "acc",
            "signal_name": "Acceleration",
            "use_filtered": False,
            "diagnostic_kind": str(diagnostic_kind),
            "envelope_band": band,
            "kurtogram_levels": int(self.env_levels_spin.value()),
            "bearing_freqs": dict(self._env_bearing_freqs) if isinstance(self._env_bearing_freqs, dict) else None,
            # Defaults required by shared worker helpers, not used in this diagnostic path.
            "filter_enabled": False,
            "filter_type": "Band-pass",
            "filter_order": 4,
            "cut1": 0.0,
            "cut2": 0.0,
            "nfft": self._get_nfft(),
            "overlap_pct": float(self.ovl_spin.value()),
            "out_type": self.cmb_out.currentText(),
            "scale": self.cmb_scale.currentText(),
            "amp_conv": self.cmb_amp_conv.currentText(),
            "window_kind": self.cmb_window_kind.currentText().lower(),
        }
        self.lbl_env_status.setText("Envelope/Kurtogram processing in background...")
        self._start_plot_worker(cfg)

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
            "Number of Spectra — RPM analysis/display rows after interpolation.\n"
            "Larger = denser time/RPM sampling view, NOT finer frequency bins.\n"
            "Frequency resolution remains Δf = fs / NFFT; overlap only increases frame density.\n"
            "Click the ? button for a full explanation."
        )
        self.btn_wf_num_spectra_help = self._make_help_button(
            "Number of Spectra — Explanation", self._help_html("wf_spectra"), "Number of Spectra"
        )

        self.wf_num_buffers_spin = QSpinBox()
        self.wf_num_buffers_spin.setRange(1, 500)
        self.wf_num_buffers_spin.setValue(1)
        self.wf_num_buffers_spin.setToolTip(
            "Number of Buffers — RPM-neighbor sliding average width across sorted raw FFT frames.\n"
            "1 = no smoothing. Larger = smoother spectra, but blurs narrow peaks and\n"
            "RPM-speed transitions. Keep ≤ 3–5 for typical run-up data.\n"
            "Click the ? button for a full explanation."
        )
        self.btn_wf_num_buffers_help = self._make_help_button(
            "Number of Buffers — Explanation", self._help_html("wf_buffers"), "Number of Buffers"
        )

        self.wf_out = QComboBox()
        self.wf_out.addItems(["Amplitude", "Power", "PSD", "ASD"])
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

        # ── Order-domain waterfall controls ────────────────────────────
        # "Frequency Waterfall" keeps existing behavior unchanged.
        # "Order Waterfall" converts each frame from Hz to order axis.
        self.cmb_wf_domain = QComboBox()
        self.cmb_wf_domain.addItems(["Frequency Waterfall", "Order Waterfall"])
        self.cmb_wf_domain.setCurrentText("Frequency Waterfall")
        self.cmb_wf_domain.setToolTip(
            "Frequency Waterfall: classic X=Hz, Y=RPM.\n"
            "Order Waterfall: X=order (f/(RPM/60)), Y=RPM.\n"
            "1X, 2X … appear as nearly-vertical lines in a run-up."
        )

        self.wf_order_max_spin = QDoubleSpinBox()
        self.wf_order_max_spin.setRange(0.5, 200.0)
        self.wf_order_max_spin.setDecimals(1)
        self.wf_order_max_spin.setValue(20.0)
        self.wf_order_max_spin.setToolTip("Maximum order shown on the X-axis of the Order Waterfall.")
        self.wf_order_max_spin.setEnabled(False)

        self.wf_order_bins_spin = QSpinBox()
        self.wf_order_bins_spin.setRange(64, 4096)
        self.wf_order_bins_spin.setValue(512)
        self.wf_order_bins_spin.setToolTip(
            "Number of order bins in the order-domain grid.\n"
            "More bins = finer order resolution but more memory."
        )
        self.wf_order_bins_spin.setEnabled(False)

        self.cmb_wf_domain.currentIndexChanged.connect(self._on_wf_domain_changed)

        # ── Order extraction bandwidth controls ────────────────────────
        self.wf_order_bw_spin = QDoubleSpinBox()
        self.wf_order_bw_spin.setRange(0.0, 1000.0)
        self.wf_order_bw_spin.setDecimals(3)
        self.wf_order_bw_spin.setValue(0.0)
        self.wf_order_bw_spin.setToolTip(
            "Order-trace extraction bandwidth.\n"
            "0 = single interpolated bin (current behavior).\n"
            ">0 = use peak-in-band or RMS-band-power within ±BW around the order line."
        )

        self.cmb_wf_order_bw_unit = QComboBox()
        self.cmb_wf_order_bw_unit.addItems(["Hz", "Order"])
        self.cmb_wf_order_bw_unit.setToolTip("Unit for the extraction bandwidth.")

        self.cmb_wf_order_extract_mode = QComboBox()
        self.cmb_wf_order_extract_mode.addItems(["Interpolated", "Peak in band", "RMS band power"])
        self.cmb_wf_order_extract_mode.setToolTip(
            "Interpolated: single-bin linear interpolation at f=order*RPM/60.\n"
            "Peak in band: maximum amplitude within the bandwidth window.\n"
            "RMS band power: square-root of mean power within the bandwidth window."
        )

        # ── Order filter / mask ─────────────────────────
        # Adaptive band-pass / band-stop around f_center(t) = k * RPM(t) / 60.
        # Implemented frame-by-frame so the displayed waterfall isolates the
        # selected order; remaining bins are masked.  Distinct from
        # "Order Cursor" (visual only) and "Computed Order Tracking" (angle
        # resampling).
        self.chk_wf_otf = QCheckBox("Order Tracking Filter (isolate selected order)")
        self.chk_wf_otf.setChecked(False)
        self.chk_wf_otf.setToolTip(
            "When enabled, choose either an FFT-frame frequency-domain order mask or a time-domain adaptive order filter."
        )

        self.cmb_wf_otf_mode = QComboBox()
        self.cmb_wf_otf_mode.addItems(["Frequency-domain order mask", "Time-domain adaptive order filter"])
        self.cmb_wf_otf_mode.setToolTip(
            "Frequency-domain order mask: masks FFT waterfall bins only (legacy display isolation).\n"
            "Time-domain adaptive order filter: blockwise SOS filter before FFT; closer to nCode but still approximate."
        )
        self.cmb_wf_otf_mode.setEnabled(False)

        self.wf_otf_order_spin = QDoubleSpinBox()
        self.wf_otf_order_spin.setRange(0.1, 200.0)
        self.wf_otf_order_spin.setDecimals(3)
        self.wf_otf_order_spin.setSingleStep(0.5)
        self.wf_otf_order_spin.setValue(1.0)
        self.wf_otf_order_spin.setToolTip("Target order k (e.g. 1.0, 2.0, 0.5, ...)")
        self.wf_otf_order_spin.setEnabled(False)

        self.wf_otf_bw_spin = QDoubleSpinBox()
        self.wf_otf_bw_spin.setRange(0.001, 1000.0)
        self.wf_otf_bw_spin.setDecimals(3)
        self.wf_otf_bw_spin.setValue(0.25)
        self.wf_otf_bw_spin.setToolTip("Half-bandwidth around the target order.")
        self.wf_otf_bw_spin.setEnabled(False)

        self.cmb_wf_otf_bw_unit = QComboBox()
        self.cmb_wf_otf_bw_unit.addItems(["Order", "Hz"])
        self.cmb_wf_otf_bw_unit.setCurrentText("Order")
        self.cmb_wf_otf_bw_unit.setToolTip("Unit of the half-bandwidth value.")
        self.cmb_wf_otf_bw_unit.setEnabled(False)

        self.cmb_wf_otf_type = QComboBox()
        self.cmb_wf_otf_type.addItems(["Band-pass", "Band-stop"])
        self.cmb_wf_otf_type.setToolTip(
            "Band-pass keeps only the band around k*RPM/60.\n"
            "Band-stop notches it (keeps everything else)."
        )
        self.cmb_wf_otf_type.setEnabled(False)

        self.wf_otf_filter_order_spin = QSpinBox()
        self.wf_otf_filter_order_spin.setRange(1, 12)
        self.wf_otf_filter_order_spin.setValue(4)
        self.wf_otf_filter_order_spin.setToolTip(
            "Effective filter order. Controls steepness of the band edges\n"
            "in the frame-based implementation (higher = sharper edges)."
        )
        self.wf_otf_filter_order_spin.setEnabled(False)

        self.chk_wf_otf_trace = QCheckBox("Show extracted trace vs RPM")
        self.chk_wf_otf_trace.setChecked(True)
        self.chk_wf_otf_trace.setEnabled(False)

        def _sync_otf_enabled(*_):
            on = self.chk_wf_otf.isChecked()
            self.cmb_wf_otf_mode.setEnabled(on)
            self.wf_otf_order_spin.setEnabled(on)
            self.wf_otf_bw_spin.setEnabled(on)
            self.cmb_wf_otf_bw_unit.setEnabled(on)
            self.cmb_wf_otf_type.setEnabled(on)
            self.wf_otf_filter_order_spin.setEnabled(on)
            self.chk_wf_otf_trace.setEnabled(on)
        self.chk_wf_otf.stateChanged.connect(_sync_otf_enabled)

        # ── Computed Order Tracking controls ───────────────────────────
        self.chk_wf_cot = QCheckBox("Computed Order Tracking (COT) — angle-domain resampling")
        self.chk_wf_cot.setToolTip(
            "Resample acceleration to uniform shaft-angle grid, then compute\n"
            "order-domain FFT.  1X, 2X … appear as sharp peaks regardless of\n"
            "speed variation.  Requires a clean, monotonic RPM signal."
        )
        self.wf_cot_spr_spin = QSpinBox()
        self.wf_cot_spr_spin.setRange(64, 4096)
        self.wf_cot_spr_spin.setValue(512)
        self.wf_cot_spr_spin.setToolTip(
            "Samples Per Revolution (COT): resampling density in the angle domain.\n"
            "512 → max detectable order ≈ 256.  Higher = finer order resolution."
        )
        self.wf_cot_spr_spin.setEnabled(False)
        self.chk_wf_cot.stateChanged.connect(
            lambda s: self.wf_cot_spr_spin.setEnabled(bool(s))
        )

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
        row_wf_1.addWidget(self._make_help_button("Waterfall Acc fs — Explanation", self._help_html("wf_acc_fs"), "Acc fs"))
        row_wf_1.addSpacing(14)
        row_wf_1.addWidget(QLabel("RPM fs (Hz):"))
        row_wf_1.addWidget(self.wf_rpm_fs_spin)
        row_wf_1.addWidget(self._make_help_button("Waterfall RPM fs — Explanation", self._help_html("wf_rpm_fs"), "RPM fs"))
        row_wf_1.addSpacing(14)
        row_wf_1.addWidget(self.chk_wf_skip_first)
        row_wf_1.addStretch(1)
        lay.addLayout(row_wf_1)

        row_wf_2 = QHBoxLayout()
        row_wf_2.addWidget(self.chk_wf_crop)
        row_wf_2.addWidget(self._make_help_button("Waterfall Crop Time Window — Explanation", self._help_html("crop"), "Waterfall crop time window"))
        row_wf_2.addWidget(QLabel("Start (s):"))
        row_wf_2.addWidget(self.wf_t0_spin)
        row_wf_2.addWidget(QLabel("End (s):"))
        row_wf_2.addWidget(self.wf_t1_spin)
        row_wf_2.addStretch(1)
        lay.addLayout(row_wf_2)

        row_wf_align = QHBoxLayout()
        row_wf_align.addWidget(QLabel("Resample mode:"))
        row_wf_align.addWidget(self.cmb_wf_resample_mode)
        row_wf_align.addWidget(self._make_help_button("Resample Mode — Explanation", self._help_html("wf_resample"), "Resample mode"))
        row_wf_align.addSpacing(12)
        row_wf_align.addWidget(QLabel("Interpolation:"))
        row_wf_align.addWidget(self.cmb_wf_interp)
        row_wf_align.addWidget(self._make_help_button("Interpolation Mode — Explanation", self._help_html("wf_interp"), "Interpolation mode"))
        row_wf_align.addSpacing(12)
        row_wf_align.addWidget(QLabel("RPM label:"))
        row_wf_align.addWidget(self.cmb_wf_rpm_label_method)
        row_wf_align.addWidget(self._make_help_button("RPM Label Method — Explanation", self._help_html("wf_rpm_label"), "RPM label method"))
        row_wf_align.addStretch(1)
        lay.addLayout(row_wf_align)

        row_wf_3 = QHBoxLayout()
        row_wf_3.addWidget(QLabel("Buffer Size (NFFT):"))
        row_wf_3.addWidget(self.wf_nfft)
        row_wf_3.addWidget(self._make_help_button("Waterfall NFFT — Explanation", self._help_html("nfft"), "Waterfall NFFT"))
        row_wf_3.addWidget(QLabel("Overlap (%):"))
        row_wf_3.addWidget(self.wf_ovl_spin)
        row_wf_3.addWidget(self._make_help_button("Waterfall Overlap — Explanation", self._help_html("overlap"), "Waterfall Overlap"))
        row_wf_3.addWidget(QLabel("Output:"))
        row_wf_3.addWidget(self.wf_out)
        row_wf_3.addWidget(self._make_help_button("Waterfall Output — Explanation", self._help_html("output"), "Waterfall Output"))
        row_wf_3.addWidget(QLabel("Scale:"))
        row_wf_3.addWidget(self.wf_scale)
        row_wf_3.addWidget(self._make_help_button("Waterfall Scale — Explanation", self._help_html("scale"), "Waterfall Scale"))
        row_wf_3.addWidget(QLabel("Amplitude convention:"))
        row_wf_3.addWidget(self.wf_amp_conv)
        row_wf_3.addWidget(self._make_help_button("Waterfall Amplitude Conversion — Explanation", self._help_html("amp"), "Waterfall Amplitude conversion"))
        row_wf_3.addWidget(QLabel("Num Spectra:"))
        row_wf_3.addWidget(self.wf_num_spectra_spin)
        row_wf_3.addWidget(self.btn_wf_num_spectra_help)
        row_wf_3.addWidget(QLabel("Num Buffers:"))
        row_wf_3.addWidget(self.wf_num_buffers_spin)
        row_wf_3.addWidget(self.btn_wf_num_buffers_help)
        row_wf_3.addStretch(1)
        lay.addLayout(row_wf_3)

        row_wf_4 = QHBoxLayout()
        row_wf_4.addWidget(self.chk_wf_filter)
        row_wf_4.addWidget(self._make_help_button("Waterfall Butterworth Filter — Explanation", self._help_html("filter_enable"), "Waterfall Butterworth filter"))
        row_wf_4.addSpacing(12)
        row_wf_4.addWidget(QLabel("Type:"))
        row_wf_4.addWidget(self.cmb_wf_filter_type)
        row_wf_4.addWidget(self._make_help_button("Waterfall Filter Type — Explanation", self._help_html("filter_type"), "Waterfall Filter Type"))
        row_wf_4.addSpacing(8)
        row_wf_4.addWidget(QLabel("Order:"))
        row_wf_4.addWidget(self.cmb_wf_filter_order)
        row_wf_4.addWidget(self._make_help_button("Waterfall Filter Order — Explanation", self._help_html("filter_order"), "Waterfall Filter Order"))
        row_wf_4.addSpacing(8)
        row_wf_4.addWidget(QLabel("Cutoff 1 (Hz):"))
        row_wf_4.addWidget(self.wf_cut1_spin)
        row_wf_4.addWidget(self.lbl_wf_cut2)
        row_wf_4.addWidget(self.wf_cut2_spin)
        row_wf_4.addWidget(self._make_help_button("Waterfall Filter Cutoffs — Explanation", self._help_html("cutoffs"), "Waterfall cutoffs"))
        row_wf_4.addStretch(1)
        lay.addLayout(row_wf_4)

        row_wf_5 = QHBoxLayout()
        row_wf_5.addWidget(QLabel("Apply filter to:"))
        row_wf_5.addWidget(self.cmb_wf_filter_target)
        row_wf_5.addSpacing(12)
        row_wf_5.addWidget(self.chk_wf_open_3d)
        row_wf_5.addWidget(self._make_help_button("Open 3D Waterfall — Explanation", self._help_html("wf_open_3d"), "Open 3D waterfall"))
        row_wf_5.addStretch(1)
        row_wf_5.addWidget(self.btn_wf_plot)
        lay.addLayout(row_wf_5)

        # ── Order Waterfall / COT / order-extraction rows are kept as
        #    hidden advanced widgets.  The primary user-facing order feature
        #    is the Order Tracking Filter below.  The cmb_wf_domain combo is
        #    pinned to "Frequency Waterfall" so the order-domain code path
        #    in PlotWorker is never taken from the UI.
        self.cmb_wf_domain.setCurrentText("Frequency Waterfall")
        for _w in (
            self.cmb_wf_domain, self.wf_order_max_spin, self.wf_order_bins_spin,
            self.wf_order_bw_spin, self.cmb_wf_order_bw_unit,
            self.cmb_wf_order_extract_mode,
            self.chk_wf_cot, self.wf_cot_spr_spin,
        ):
            _w.setVisible(False)

        # Order Tracking Filter row — the only order-related UI now visible.
        row_wf_otf = QHBoxLayout()
        row_wf_otf.addWidget(self.chk_wf_otf)
        row_wf_otf.addWidget(self._make_help_button("Order Tracking Filter — Explanation", self._help_html("wf_otf_enable"), "Order Tracking Filter"))
        row_wf_otf.addSpacing(10)
        row_wf_otf.addWidget(QLabel("Target order (k):"))
        row_wf_otf.addWidget(self.wf_otf_order_spin)
        row_wf_otf.addWidget(self._make_help_button("Target Order — Explanation", self._help_html("wf_target_order"), "Target order"))
        row_wf_otf.addWidget(QLabel("Half-BW:"))
        row_wf_otf.addWidget(self.wf_otf_bw_spin)
        row_wf_otf.addWidget(self._make_help_button("Half-Bandwidth — Explanation", self._help_html("wf_half_bw"), "Half-bandwidth"))
        row_wf_otf.addWidget(self.cmb_wf_otf_bw_unit)
        row_wf_otf.addWidget(self._make_help_button("Bandwidth Unit — Explanation", self._help_html("wf_bw_unit"), "Bandwidth unit"))
        row_wf_otf.addSpacing(8)
        row_wf_otf.addWidget(QLabel("Type:"))
        row_wf_otf.addWidget(self.cmb_wf_otf_type)
        row_wf_otf.addWidget(self._make_help_button("Order Filter Type — Explanation", self._help_html("wf_otf_type"), "Band-pass vs Band-stop"))
        row_wf_otf.addWidget(QLabel("Filter order:"))
        row_wf_otf.addWidget(self.wf_otf_filter_order_spin)
        row_wf_otf.addWidget(self._make_help_button("Order Filter Order — Explanation", self._help_html("filter_order"), "Order Tracking Filter order"))
        row_wf_otf.addSpacing(8)
        row_wf_otf.addWidget(self.chk_wf_otf_trace)
        row_wf_otf.addWidget(self._make_help_button("Extracted Trace — Explanation", self._help_html("wf_trace"), "Show extracted trace vs RPM"))
        row_wf_otf.addStretch(1)
        lay.addLayout(row_wf_otf)

        lay.addWidget(self.lbl_wf_status)

        self.wf_result_group = QGroupBox("Waterfall View")
        self.wf_result_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #999;"
            " border-radius: 4px; margin-top: 8px; padding: 6px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )
        wf_status_lay = QHBoxLayout(self.wf_result_group)
        self.lbl_wf_result_empty = QLabel(
            "Waterfall result opens in the Waterfall View tab. Configure acceleration/RPM sources above, then click Plot Waterfall."
        )
        self.lbl_wf_result_empty.setWordWrap(True)
        self.lbl_wf_result_empty.setStyleSheet("color: #666;")
        self.btn_go_waterfall_view = QPushButton("Open Waterfall View")
        self.btn_go_waterfall_view.clicked.connect(self._go_to_waterfall_view)
        wf_status_lay.addWidget(self.lbl_wf_result_empty, 1)
        wf_status_lay.addWidget(self.btn_go_waterfall_view)
        lay.addWidget(self.wf_result_group, 0)

        self.btn_wf_acc_open.clicked.connect(lambda: self._open_waterfall_source("acc"))
        self.btn_wf_rpm_open.clicked.connect(lambda: self._open_waterfall_source("rpm"))
        self.cmb_wf_acc_group.currentIndexChanged.connect(lambda *_: self._reload_waterfall_channels("acc"))
        self.cmb_wf_rpm_group.currentIndexChanged.connect(lambda *_: self._reload_waterfall_channels("rpm"))
        self.chk_wf_crop.stateChanged.connect(self._on_wf_crop_toggle)
        self.chk_wf_filter.stateChanged.connect(self._sync_wf_filter_enabled_state)
        self.cmb_wf_filter_type.currentIndexChanged.connect(self._on_wf_filter_type_change)
        self.btn_wf_plot.clicked.connect(self.plot_waterfall)

    def _on_wf_domain_changed(self, _idx):
        is_order = self.cmb_wf_domain.currentText() == "Order Waterfall"
        self.wf_order_max_spin.setEnabled(is_order)
        self.wf_order_bins_spin.setEnabled(is_order)

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
            "waterfall_domain": self.cmb_wf_domain.currentText(),
            "waterfall_order_max": float(self.wf_order_max_spin.value()),
            "waterfall_order_bins": int(self.wf_order_bins_spin.value()),
            "waterfall_order_bw": float(self.wf_order_bw_spin.value()),
            "waterfall_order_bw_unit": self.cmb_wf_order_bw_unit.currentText(),
            "waterfall_order_extract_mode": self.cmb_wf_order_extract_mode.currentText(),
            "waterfall_cot_enabled": self.chk_wf_cot.isChecked(),
            "waterfall_cot_spr": int(self.wf_cot_spr_spin.value()),
            # Order filter / mask — isolates a single order in
            # the displayed waterfall.
            "waterfall_otf_enabled": self.chk_wf_otf.isChecked(),
            "waterfall_otf_mode": self.cmb_wf_otf_mode.currentText(),
            "waterfall_otf_order": float(self.wf_otf_order_spin.value()),
            "waterfall_otf_bw": float(self.wf_otf_bw_spin.value()),
            "waterfall_otf_bw_unit": self.cmb_wf_otf_bw_unit.currentText(),
            "waterfall_otf_type": self.cmb_wf_otf_type.currentText(),
            "waterfall_otf_filter_order": int(self.wf_otf_filter_order_spin.value()),
            "waterfall_otf_block_seconds": 0.5,
            "waterfall_otf_show_trace": self.chk_wf_otf_trace.isChecked(),
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
            "method": "Both (Averaged + STFT)",
            "amp_conv": self.cmb_amp_conv.currentText(),
            "window_kind": self.cmb_window_kind.currentText().lower(),
            "velocity_integration_method": "Time-domain trapezoidal",
            "velocity_highpass_hz": 10.0,
            "velocity_detrend": "Remove mean",

            # Independent AVG / STFT settings (fall back to globals).
            "avg_override": self.chk_avg_override.isChecked(),
            "avg_out_type": self.cmb_avg_out.currentText(),
            "avg_scale": self.cmb_avg_scale.currentText(),
            "avg_amp_conv": self.cmb_avg_amp_conv.currentText(),

            "stft_override": self.chk_stft_override.isChecked(),
            "stft_out_type": self.cmb_stft_out.currentText(),
            "stft_scale": self.cmb_stft_scale.currentText(),
            "stft_amp_conv": self.cmb_stft_amp_conv.currentText(),
            "stft_cmap": self.cmb_spec_cmap.currentText(),
            "stft_dyn_range_db": int(self.spec_levels_spin.value()),
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


# ─────────────────────────────────────────────────────────────────────────────
# TRDSP Signal-Processing Self-Tests
# Call run_signal_processing_self_tests() from a Python console or add
# a button / menu action to trigger them during development.
# ─────────────────────────────────────────────────────────────────────────────

def run_signal_processing_self_tests(verbose: bool = True) -> bool:
    """Synthetic validation tests for core signal-processing functions.

    Tests
    -----
    1. Constant-RPM sine: 1X = RPM/60, known amplitude.
    2. Run-up chirp (1X): frequency scales with RPM.
    3. Two-component signal (1X + 2X): amplitudes recovered independently.
    4. RPM NaN/spike: invalid RPM must NOT become 0 RPM.
    5. Averaged spectrum sanity: known sine amplitude within 5 % tolerance.
    6. Order map: 1X peak at order = 1.0.
    7. compute_order_tracking_spectrum: returns order axis and amplitude > 0.

    Returns True if all tests pass.
    """
    import traceback
    PASS = "[PASS]"
    FAIL = "[FAIL]"
    results = []

    def log(name: str, ok: bool, detail: str = ""):
        tag = PASS if ok else FAIL
        msg = f"  [{tag}] {name}"
        if detail:
            msg += f"  ({detail})"
        if verbose:
            print(msg)
        results.append(ok)

    # Use fs=4096 so that many integer frequencies fall on exact FFT bins.
    # With NFFT=N=4096 and fs=4096: delta_f=1 Hz, so any integer Hz freq is exact.
    fs = 4096.0
    N = 4096
    t = np.arange(N) / fs

    # ── Test 1: Constant RPM, 1X sine ────────────────────────────────
    # Choose f1x=100 Hz (exact bin: bin index 100). RPM=6000.
    try:
        f1x = 100.0
        A_known = 2.0
        sig = A_known * np.sin(2.0 * np.pi * f1x * t)
        freqs, spec = compute_avg_spectrum_from_signal(
            sig, fs, nfft=N, overlap_pct=0.0,
            output_type="Amplitude", scale="Linear",
            amp_conv="Peak", signal_kind="acc",
        )
        peak_idx = int(np.argmax(spec))
        peak_freq = float(freqs[peak_idx])
        peak_amp = float(spec[peak_idx])
        freq_ok = abs(peak_freq - f1x) < 1.5
        amp_ok = abs(peak_amp - A_known) / A_known < 0.05
        log("Test 1: Constant-RPM 1X peak frequency",
            freq_ok, f"found {peak_freq:.2f} Hz, expected {f1x:.2f} Hz")
        log("Test 1: Constant-RPM 1X peak amplitude",
            amp_ok, f"found {peak_amp:.4f}, expected {A_known:.4f}")
    except Exception as e:
        log("Test 1", False, str(e))

    # ── Test 2: Averaged spectrum RMS convention ──────────────────────
    try:
        A_rms_known = 1.0
        A_peak_for_rms = A_rms_known * np.sqrt(2.0)
        sig2 = A_peak_for_rms * np.sin(2.0 * np.pi * 200.0 * t)
        _, spec2 = compute_avg_spectrum_from_signal(
            sig2, fs, nfft=N, overlap_pct=0.0,
            output_type="Amplitude", scale="Linear",
            amp_conv="RMS", signal_kind="acc",
        )
        rms_peak = float(np.max(spec2))
        ok = abs(rms_peak - A_rms_known) / A_rms_known < 0.05
        log("Test 2: RMS amplitude convention", ok,
            f"found {rms_peak:.4f}, expected {A_rms_known:.4f}")
    except Exception as e:
        log("Test 2", False, str(e))

    # ── Test 3: 1X + 2X two-component signal ─────────────────────────
    # Both f1=80 Hz and f2=160 Hz are exact bins (integers with fs=4096, N=4096).
    try:
        A1 = 1.5
        A2 = 0.7
        f1 = 80.0
        f2 = 160.0
        sig3 = A1 * np.sin(2.0 * np.pi * f1 * t) + A2 * np.sin(2.0 * np.pi * f2 * t)
        freqs3, spec3 = compute_avg_spectrum_from_signal(
            sig3, fs, nfft=N, overlap_pct=0.0,
            output_type="Amplitude", scale="Linear",
            amp_conv="Peak", signal_kind="acc",
        )
        p1 = float(spec3[int(np.argmin(np.abs(freqs3 - f1)))])
        p2 = float(spec3[int(np.argmin(np.abs(freqs3 - f2)))])
        ok1 = abs(p1 - A1) / A1 < 0.05
        ok2 = abs(p2 - A2) / A2 < 0.05
        log("Test 3: Two-component 1X amplitude", ok1, f"found {p1:.4f}, expected {A1:.4f}")
        log("Test 3: Two-component 2X amplitude", ok2, f"found {p2:.4f}, expected {A2:.4f}")
    except Exception as e:
        log("Test 3", False, str(e))

    # ── Test 4: RPM NaN/spike handling ───────────────────────────────
    try:
        rpm_test = np.full(100, 3000.0)
        rpm_test[10:15] = np.nan
        rpm_test[50] = 999999.0          # spike
        rpm_test[60] = -100.0            # negative RPM
        rpm_clean = clean_rpm_signal(rpm_test, mode="mask", spike_mad_thresh=5.0)
        # After masking, spikes and invalids must be NaN, not 0
        spike_is_nan = not np.isfinite(rpm_clean[50])
        neg_is_nan = not np.isfinite(rpm_clean[60])
        nan_range_nan = not np.any(np.isfinite(rpm_clean[11:14]))
        ok = spike_is_nan and neg_is_nan and nan_range_nan
        # Make sure valid samples are NOT 0
        valid_nonzero = np.all(rpm_clean[np.isfinite(rpm_clean)] > 0.0)
        log("Test 4: RPM spike becomes NaN (not 0)", spike_is_nan,
            f"rpm_clean[50] = {rpm_clean[50]}")
        log("Test 4: Negative RPM becomes NaN", neg_is_nan,
            f"rpm_clean[60] = {rpm_clean[60]}")
        log("Test 4: NaN gap preserved in mask mode", nan_range_nan)
        log("Test 4: Valid RPM samples remain > 0", valid_nonzero)
    except Exception as e:
        log("Test 4", False, str(e))

    # ── Test 5: Power output sanity ───────────────────────────────────
    # f=300 Hz is an exact bin with fs=4096, N=4096 (since 300 is integer).
    try:
        A5 = 3.0
        sig5 = A5 * np.sin(2.0 * np.pi * 300.0 * t)
        _, spec5 = compute_avg_spectrum_from_signal(
            sig5, fs, nfft=N, overlap_pct=0.0,
            output_type="Power", scale="Linear",
            amp_conv="Peak", signal_kind="acc",
        )
        expected_power = A5 ** 2 / 2.0
        p_found = float(np.max(spec5))
        ok = abs(p_found - expected_power) / expected_power < 0.05
        log("Test 5: Power spectrum amplitude",
            ok, f"found {p_found:.4f}, expected {expected_power:.4f}")
    except Exception as e:
        log("Test 5", False, str(e))

    # ── Test 6: Order map 1X peak at order = 1.0 ─────────────────────
    # rpm6=1800 => f_1x=30 Hz, which is an exact bin with fs=4096, N=4096.
    try:
        rpm6 = 1800.0
        f_1x = rpm6 / 60.0   # 30 Hz
        A6 = 1.0
        sig6 = A6 * np.sin(2.0 * np.pi * f_1x * t)
        freqs6 = np.fft.rfftfreq(N, 1.0 / fs)
        w6 = _hann_window(N, "periodic")
        w_sum6 = float(np.sum(w6))
        X6 = np.fft.rfft(sig6 * w6, n=N)
        A_peak6 = (2.0 / w_sum6) * np.abs(X6)
        A_peak6[0] *= 0.5
        spec6_linear = A_peak6.reshape(1, -1)
        order_ax, _, S_order6 = compute_order_map(
            freqs6, np.array([rpm6]), spec6_linear,
            order_max=5.0, n_orders=512,
            output_type="Amplitude", scale="Linear",
        )
        peak_order_idx = int(np.argmax(S_order6[0]))
        peak_order = float(order_ax[peak_order_idx])
        ok = abs(peak_order - 1.0) < 0.1
        log("Test 6: Order map 1X peak at order=1.0",
            ok, f"peak at order={peak_order:.3f}")
    except Exception as e:
        log("Test 6", False, str(e))

    # ── Test 7: COT function runs without crash ───────────────────────
    # Use fs=4096, rpm=1800 (30 Hz shaft), signal=1X at 30 Hz.
    try:
        rpm7 = np.full(N, 1800.0)
        f_1x7 = 1800.0 / 60.0   # 30 Hz
        sig7 = 1.5 * np.sin(2.0 * np.pi * f_1x7 * t)
        orders7, amp7, amp_vs_rpm7 = compute_order_tracking_spectrum(
            sig7, rpm7, fs, samples_per_rev=512, order_max=5.0, amp_conv="Peak"
        )
        has_output = orders7.size > 0 and amp7.size > 0
        peak_1x_idx = int(np.argmin(np.abs(orders7 - 1.0)))
        peak_1x_amp = float(amp7[peak_1x_idx])
        amp_ok = abs(peak_1x_amp - 1.5) / 1.5 < 0.10
        log("Test 7: COT returns output", has_output)
        log("Test 7: COT 1X amplitude within 10 pct",
            amp_ok, f"found {peak_1x_amp:.4f}, expected 1.5")
    except Exception as e:
        log("Test 7", False, str(e))

    n_pass = sum(results)
    n_total = len(results)
    if verbose:
        print(f"\n  Summary: {n_pass}/{n_total} tests passed.")
        if n_pass == n_total:
            print("  All tests PASSED.")
        else:
            print("  Some tests FAILED — review the output above.")
    return n_pass == n_total


def run_regression_tests(verbose: bool = True):
    """Run headless DSP regression tests without launching the GUI."""
    try:
        from trdsp_regression_tests import run_regression_tests as _run
        return _run(verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"Regression test runner failed to start: {e}")
        return {"passed": 0, "failed": 1, "skipped": 0, "details": [("FAIL", "runner", str(e))]}


def _close_pyinstaller_splash():
    """Close PyInstaller splash screen after the main Qt window is visible."""
    try:
        import pyi_splash
        if pyi_splash.is_alive():
            pyi_splash.close()
    except Exception:
        pass


def main():
    if "--self-test" in sys.argv:
        summary = run_regression_tests(verbose=True)
        sys.exit(0 if int(summary.get("failed", 1)) == 0 else 1)

    app = QApplication(sys.argv)
    try:
        app.setWindowIcon(QIcon(resource_path("logo.png")))
    except Exception:
        pass

    if not _HAVE_PG:
        QMessageBox.critical(None, "Missing dependency", "pyqtgraph is required. Install with: pip install pyqtgraph")
        sys.exit(1)

    w = MainWindow()
    w.resize(1600, 900)
    w.show()
    QTimer.singleShot(300, _close_pyinstaller_splash)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
