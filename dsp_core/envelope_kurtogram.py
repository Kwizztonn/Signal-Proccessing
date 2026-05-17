"""
Envelope analysis, spectral kurtosis, and fast-kurtogram utilities for TRDSP.

The routines in this module are backend-only: no Qt, no matplotlib, and no
application state.  They are designed to be called from PlotWorker/QThread code
and from validation scripts.

Algorithmic references
----------------------
- J. Antoni, "Fast computation of the kurtogram for the detection of transient
  faults", Mechanical Systems and Signal Processing, 2007.
- T. A. Harris and M. N. Kotzalas, Rolling Bearing Analysis, bearing fault
  characteristic-frequency formulae.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import signal


_EPS = np.finfo(float).eps


@dataclass(frozen=True)
class BearingFaultFrequencies:
    """Characteristic bearing fault frequencies in Hz."""

    shaft_hz: float
    bpfo: float
    bpfi: float
    bsf: float
    ftf: float

    def as_dict(self) -> dict[str, float]:
        return {
            "shaft_hz": self.shaft_hz,
            "BPFO": self.bpfo,
            "BPFI": self.bpfi,
            "BSF": self.bsf,
            "FTF": self.ftf,
        }


@dataclass(frozen=True)
class KurtogramBand:
    """Description of one kurtogram analysis band."""

    level: int
    band_index: int
    center_freq: float
    bandwidth: float
    f_low: float
    f_high: float
    kurtosis: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "level": self.level,
            "band_index": self.band_index,
            "center_freq": self.center_freq,
            "bandwidth": self.bandwidth,
            "f_low": self.f_low,
            "f_high": self.f_high,
            "kurtosis": self.kurtosis,
        }


def _as_clean_1d(x: Iterable[float] | np.ndarray, *, name: str = "x") -> np.ndarray:
    arr = np.asarray(x, dtype=float).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def _validate_fs(fs: float) -> float:
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError("fs must be a positive finite sampling frequency")
    return fs


def _band_sos(fs: float, band: tuple[float, float] | list[float], order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    if len(band) != 2:
        raise ValueError("band must be a (low_hz, high_hz) pair")
    low, high = float(band[0]), float(band[1])
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("band limits must be finite")
    if low < 0.0 or high <= 0.0 or low >= high:
        raise ValueError("band must satisfy 0 <= low < high")
    if high >= nyq:
        high = np.nextafter(nyq, 0.0)
    if low <= 0.0:
        return signal.butter(order, high, btype="lowpass", fs=fs, output="sos")
    return signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")


def _safe_zero_phase_filter(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Use zero-phase SOS filtering, falling back for very short records."""
    if x.size < 32:
        return signal.sosfilt(sos, x)
    try:
        return signal.sosfiltfilt(sos, x)
    except ValueError:
        return signal.sosfilt(sos, x)


def _excess_kurtosis(x: np.ndarray) -> float:
    y = np.asarray(x, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 4:
        return 0.0
    y = y - np.mean(y)
    var = float(np.mean(y * y))
    if var <= _EPS:
        return 0.0
    return float(np.mean(y ** 4) / (var * var) - 3.0)


def compute_envelope_spectrum(
    x: Iterable[float] | np.ndarray,
    fs: float,
    band: tuple[float, float] | list[float] | None = None,
    demod_method: str = "hilbert",
) -> dict[str, np.ndarray | float | tuple[float, float] | str | None]:
    """
    Compute a Hilbert-envelope spectrum, optionally after band-pass filtering.

    Parameters
    ----------
    x:
        Input vibration signal.
    fs:
        Sampling frequency in Hz.
    band:
        Optional ``(low_hz, high_hz)`` pre-demodulation filter band.
    demod_method:
        Currently only ``"hilbert"`` is supported.

    Returns
    -------
    dict
        Contains ``freqs``, ``spectrum``, ``envelope``, ``filtered_signal``,
        ``band``, ``fs``, and ``demod_method``.  ``spectrum`` is single-sided
        peak amplitude of the AC envelope, so a unit sinusoidal modulation has
        an approximately unit line amplitude.
    """
    fs = _validate_fs(fs)
    x_arr = _as_clean_1d(x)
    method = str(demod_method).strip().lower()
    if method != "hilbert":
        raise ValueError("Only demod_method='hilbert' is currently supported")

    x_proc = x_arr - np.mean(x_arr)
    if band is not None:
        sos = _band_sos(fs, band)
        x_proc = _safe_zero_phase_filter(sos, x_proc)
        band_out: tuple[float, float] | None = (float(band[0]), float(band[1]))
    else:
        band_out = None

    analytic = signal.hilbert(x_proc)
    envelope = np.abs(analytic)
    envelope_ac = envelope - np.mean(envelope)

    n = envelope_ac.size
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(envelope_ac)) / max(n, 1)
    if n > 1:
        spectrum[1:-1 if n % 2 == 0 else None] *= 2.0

    return {
        "freqs": freqs,
        "spectrum": spectrum,
        "envelope": envelope,
        "filtered_signal": x_proc,
        "band": band_out,
        "fs": fs,
        "demod_method": method,
    }


def compute_fast_kurtogram(
    x: Iterable[float] | np.ndarray,
    fs: float,
    n_levels: int = 8,
) -> dict[str, np.ndarray | dict[str, float | int] | list[dict[str, float | int]] | float]:
    """
    Compute a kurtogram-style dyadic/1-3 band grid and return the best band.

    This is a practical Antoni-style fast-kurtogram backend: candidate bands are
    scanned on a dyadic grid, with additional one-third and two-third splits per
    level to improve localization of impulsive resonance bands.  Each candidate
    is filtered, Hilbert-demodulated, and scored by excess kurtosis of the AC
    envelope.
    """
    fs = _validate_fs(fs)
    x_arr = _as_clean_1d(x)
    n_levels = int(n_levels)
    if n_levels < 1:
        raise ValueError("n_levels must be >= 1")
    if x_arr.size < 16:
        raise ValueError("x must contain at least 16 samples for kurtogram analysis")

    nyq = 0.5 * fs
    max_bins = 2 ** n_levels
    kurtosis_grid = np.full((n_levels + 1, max_bins), np.nan, dtype=float)
    center_grid = np.full_like(kurtosis_grid, np.nan)
    bandwidth_grid = np.full_like(kurtosis_grid, np.nan)
    bands: list[KurtogramBand] = []

    for level in range(n_levels + 1):
        n_bands = 2 ** level
        bw = nyq / n_bands
        starts = [i * bw for i in range(n_bands)]
        candidates: list[tuple[int, float, float]] = [(i, low, low + bw) for i, low in enumerate(starts)]

        if level > 0:
            # Antoni's fast kurtogram uses a mixed binary/ternary partition.
            # Add third-band candidates without changing grid dimensions.
            third_bw = bw / 3.0
            for i, low in enumerate(starts):
                candidates.append((i, low, low + third_bw))
                candidates.append((i, low + third_bw, low + 2.0 * third_bw))
                candidates.append((i, low + 2.0 * third_bw, low + bw))

        best_for_bin: dict[int, KurtogramBand] = {}
        for band_index, f_low, f_high in candidates:
            f_low = max(0.0, float(f_low))
            f_high = min(float(f_high), np.nextafter(nyq, 0.0))
            if f_high - f_low <= max(fs / x_arr.size, 1e-9):
                continue
            try:
                sos = _band_sos(fs, (f_low, f_high), order=3)
                y = _safe_zero_phase_filter(sos, x_arr - np.mean(x_arr))
                env = np.abs(signal.hilbert(y))
                score = _excess_kurtosis(env)
            except (ValueError, FloatingPointError):
                continue

            band_obj = KurtogramBand(
                level=level,
                band_index=band_index,
                center_freq=0.5 * (f_low + f_high),
                bandwidth=f_high - f_low,
                f_low=f_low,
                f_high=f_high,
                kurtosis=score,
            )
            bands.append(band_obj)
            current = best_for_bin.get(band_index)
            if current is None or score > current.kurtosis:
                best_for_bin[band_index] = band_obj

        for band_index, band_obj in best_for_bin.items():
            kurtosis_grid[level, band_index] = band_obj.kurtosis
            center_grid[level, band_index] = band_obj.center_freq
            bandwidth_grid[level, band_index] = band_obj.bandwidth

    if not bands:
        raise ValueError("No valid kurtogram bands could be computed")
    best = max(bands, key=lambda item: item.kurtosis)
    return {
        "kurtosis_grid": kurtosis_grid,
        "center_freq_grid": center_grid,
        "bandwidth_grid": bandwidth_grid,
        "best_band": best.as_dict(),
        "bands": [band.as_dict() for band in bands],
        "fs": fs,
        "n_levels": n_levels,
    }


def compute_spectral_kurtosis(
    x: Iterable[float] | np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int | None = None,
) -> dict[str, np.ndarray | float | int]:
    """Compute Welch/STFT spectral kurtosis per frequency bin."""
    fs = _validate_fs(fs)
    x_arr = _as_clean_1d(x)
    nperseg = int(nperseg)
    if nperseg < 8:
        raise ValueError("nperseg must be >= 8")
    if nperseg > x_arr.size:
        raise ValueError("nperseg cannot exceed signal length")
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(noverlap)
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg")

    freqs, times, zxx = signal.stft(
        x_arr - np.mean(x_arr),
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        return_onesided=True,
        boundary=None,
        padded=False,
    )
    power = np.abs(zxx) ** 2
    m2 = np.mean(power, axis=1)
    m4 = np.mean(power ** 2, axis=1)
    sk = np.zeros_like(m2)
    valid = m2 > _EPS
    # Complex STFT Gaussian baseline is 2, so subtract 2 for excess SK.
    sk[valid] = m4[valid] / (m2[valid] ** 2) - 2.0
    return {
        "freqs": freqs,
        "times": times,
        "spectral_kurtosis": sk,
        "stft_power": power,
        "fs": fs,
        "nperseg": nperseg,
        "noverlap": noverlap,
    }


def extract_bearing_fault_frequencies(
    rpm: float,
    n_balls: int,
    d_ball: float,
    d_pitch: float,
    contact_angle: float,
) -> dict[str, float]:
    """
    Return BPFO/BPFI/BSF/FTF for a rolling-element bearing.

    ``contact_angle`` is in degrees.  ``d_ball`` and ``d_pitch`` may be in any
    common length unit, provided both use the same unit.
    """
    rpm = float(rpm)
    n_balls = int(n_balls)
    d_ball = float(d_ball)
    d_pitch = float(d_pitch)
    contact_angle = float(contact_angle)
    if not np.isfinite(rpm) or rpm < 0.0:
        raise ValueError("rpm must be a non-negative finite value")
    if n_balls <= 0:
        raise ValueError("n_balls must be positive")
    if d_ball <= 0.0 or d_pitch <= 0.0:
        raise ValueError("d_ball and d_pitch must be positive")
    if d_ball >= d_pitch:
        raise ValueError("d_ball must be smaller than d_pitch")
    if not np.isfinite(contact_angle):
        raise ValueError("contact_angle must be finite")

    shaft_hz = rpm / 60.0
    ratio = d_ball / d_pitch
    cos_a = np.cos(np.deg2rad(contact_angle))
    bpfo = 0.5 * n_balls * shaft_hz * (1.0 - ratio * cos_a)
    bpfi = 0.5 * n_balls * shaft_hz * (1.0 + ratio * cos_a)
    bsf = 0.5 * shaft_hz * (d_pitch / d_ball) * (1.0 - (ratio * cos_a) ** 2)
    ftf = 0.5 * shaft_hz * (1.0 - ratio * cos_a)
    return BearingFaultFrequencies(shaft_hz, bpfo, bpfi, bsf, ftf).as_dict()


__all__ = [
    "BearingFaultFrequencies",
    "KurtogramBand",
    "compute_envelope_spectrum",
    "compute_fast_kurtogram",
    "compute_spectral_kurtosis",
    "extract_bearing_fault_frequencies",
]
