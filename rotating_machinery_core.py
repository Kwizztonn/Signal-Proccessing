"""
rotating_machinery_core.py
─────────────────────────────────────────────────────────────────────────────
Rotating-machinery analysis foundation: RPM preprocessing, amplitude
normalization, frequency-vs-RPM mapping, angle-domain order tracking,
and order trace extraction.

This is a MATLAB-like / nCode-style implementation. It is designed for
scientific validation and educational transparency. Exact numerical
equivalence with MATLAB rpmfreqmap / rpmordermap / orderspectrum or
nCode GlyphWorks would require validated reference exports — when no
such reference is provided, this module uses physically-derived
conventions documented inline.

SCOPE OF THIS PATCH (Stages 1-4 + partial 6 + foundations for 10)
-----------------------------------------------------------------
    • RPM cleanup, alignment, smoothing, shaft-phase integration
    • Amplitude normalization with 9 explicit modes (peak/rms/power/
      psd/asd + dB variants), each derived from one unified pipeline
    • rpmfreqmap-style frequency-vs-RPM map
    • Angle-domain resampling and rpmordermap-style order map
    • Order trace extraction (0.5X / 1X / 2X / arbitrary harmonics)

NOT YET IMPLEMENTED (deferred to follow-up patches)
---------------------------------------------------
    • Stage 5  : compute_order_spectrum global wrapper
    • Stage 7  : waterfall builders + plotting helpers
    • Stage 8  : bearing fault frequencies (BPFO / BPFI / BSF / FTF)
    • Stage 9  : envelope analysis (Hilbert, bandpass, envelope spectrum)
    • Stage 11 : MATLAB / nCode CSV reference comparison hooks

AMPLITUDE CONVENTION TABLE (the most common source of misinterpretation)
------------------------------------------------------------------------
For a bin-centred sine x(t) = A * cos(2π f₀ t) at one of the rfft bins,
windowed by a unit-energy-corrected Hann (sum(w) is taken into account):

    mode        unit           returned value at f₀
    -------     -------------  ----------------------
    "peak"      g              A
    "rms"       g RMS          A / √2
    "power"     g²             A² / 2
    "psd"       g²/Hz          (A²/2) / ENBW
    "asd"       g/√Hz          √PSD
    "db_peak"   dB             20·log₁₀(A     / ref)
    "db_rms"    dB             20·log₁₀(A/√2 / ref)
    "db_power"  dB             10·log₁₀(A²/2 / ref²)
    "db_psd"    dB(/Hz)        10·log₁₀(PSD  / ref²)

DC (k=0) and Nyquist (k=N/2 when nfft is even) are NOT doubled because
they have no negative-frequency twin. They are also deterministic for
real signals, so peak == rms there (handled internally).

DEPENDENCIES
------------
    numpy, scipy.signal, scipy.interpolate, scipy.integrate
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Union

import numpy as np
from scipy import signal as _sp_signal
from scipy.interpolate import interp1d as _interp1d

try:
    from scipy.integrate import cumulative_trapezoid as _cumtrapz
except ImportError:  # SciPy < 1.6
    from scipy.integrate import cumtrapz as _cumtrapz  # type: ignore


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TIME / RPM PREPROCESSING                                (Stage 1)
# ═════════════════════════════════════════════════════════════════════════════

def validate_timebase(t, x=None, name: str = "signal") -> dict:
    """Validate a time vector and (optionally) a paired data array.

    Sorts the time vector if needed (carrying ``x`` along), removes exact
    duplicates, detects non-monotonicity, counts NaNs, and reports whether
    the timebase is uniform.

    Parameters
    ----------
    t : array-like
        Time vector in seconds. Must be 1-D and finite.
    x : array-like or None
        Optional data array sharing the same length as ``t``.
    name : str
        Name used in warnings and error messages.

    Returns
    -------
    dict with keys:
        t          : (M,) cleaned time vector, strictly monotonic
        x          : (M,) cleaned data (only present if x was given)
        fs         : float or None — uniform sample rate, else None
        uniform    : bool          — True if dt is uniform within 1e-6 rel tol
        n_input    : int           — original sample count
        n_output   : int           — sample count after cleanup
        warnings   : list[str]
    """
    t_arr = np.asarray(t, dtype=np.float64).ravel()
    if t_arr.ndim != 1 or t_arr.size < 2:
        raise ValueError(
            f"validate_timebase: {name!r} time vector must be 1-D with >= 2 samples "
            f"(got shape {np.asarray(t).shape})."
        )

    if x is not None:
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        if x_arr.size != t_arr.size:
            raise ValueError(
                f"validate_timebase: {name!r} length mismatch — "
                f"t has {t_arr.size}, x has {x_arr.size}."
            )
    else:
        x_arr = None

    msgs: list[str] = []

    # Drop NaN/Inf timestamps
    finite = np.isfinite(t_arr)
    if not finite.all():
        msgs.append(f"Dropped {int((~finite).sum())} non-finite timestamps from {name}.")
        t_arr = t_arr[finite]
        if x_arr is not None:
            x_arr = x_arr[finite]
        if t_arr.size < 2:
            raise ValueError(f"validate_timebase: {name!r} has < 2 finite timestamps.")

    # Sort if needed
    if not np.all(np.diff(t_arr) >= 0):
        msgs.append(f"Time vector for {name} was not monotonic — sorting ascending.")
        order = np.argsort(t_arr, kind="stable")
        t_arr = t_arr[order]
        if x_arr is not None:
            x_arr = x_arr[order]

    # Remove exact duplicates
    keep = np.concatenate(([True], np.diff(t_arr) > 0))
    n_dups = int((~keep).sum())
    if n_dups > 0:
        msgs.append(f"Removed {n_dups} duplicate timestamps from {name}.")
        t_arr = t_arr[keep]
        if x_arr is not None:
            x_arr = x_arr[keep]

    if t_arr.size < 2:
        raise ValueError(
            f"validate_timebase: {name!r} has < 2 unique timestamps after cleanup."
        )

    # Uniform sample-rate detection
    dt = np.diff(t_arr)
    dt_med = float(np.median(dt))
    if dt_med <= 0.0:
        raise ValueError(f"validate_timebase: {name!r} median dt is non-positive.")
    rel_jitter = float(np.max(np.abs(dt - dt_med)) / dt_med)
    uniform = rel_jitter < 1e-6
    fs = (1.0 / dt_med) if uniform else None
    if not uniform and rel_jitter < 1e-3:
        msgs.append(
            f"{name} timebase has small jitter (rel {rel_jitter:.1e}); treating as non-uniform."
        )

    out = {
        "t": t_arr,
        "fs": fs,
        "uniform": uniform,
        "n_input": int(np.asarray(t).size),
        "n_output": int(t_arr.size),
        "warnings": msgs,
    }
    if x_arr is not None:
        out["x"] = x_arr
    return out


def clean_rpm_signal(t_rpm, rpm, *,
                     min_rpm: float = 1.0,
                     max_rpm: Optional[float] = None,
                     spike_mad_thresh: float = 5.0) -> dict:
    """Clean an RPM channel: drop NaN/<min/>max, reject spikes, interpolate gaps.

    Parameters
    ----------
    t_rpm : array-like
        Time vector for the RPM samples (seconds).
    rpm : array-like
        Raw RPM samples (same length as ``t_rpm``).
    min_rpm : float
        Samples with rpm < min_rpm are treated as invalid (default 1.0).
        Use 0.0 to allow zero RPM.
    max_rpm : float or None
        Samples with rpm > max_rpm are treated as invalid. ``None`` disables.
    spike_mad_thresh : float
        Spike rejection threshold in MAD-units on the RPM derivative.
        ``0`` disables spike rejection. Default 5.0 — same convention as the
        legacy ``Last.py:163`` implementation.

    Returns
    -------
    dict with keys:
        t_rpm      : (M,) cleaned time vector
        rpm        : (M,) cleaned RPM, NaN where unrecoverable
        valid_mask : (M,) bool — True where original (or interpolated) was valid
        n_invalid  : int — number of originally-invalid samples
        gaps       : list of (t_start, t_end, n_samples) tuples for interpolated gaps
        warnings   : list[str]
    """
    base = validate_timebase(t_rpm, rpm, name="rpm")
    t = base["t"]
    r = base["x"].copy()
    msgs = list(base["warnings"])

    # Mark invalid
    invalid = ~np.isfinite(r) | (r < float(min_rpm))
    if max_rpm is not None:
        invalid |= r > float(max_rpm)
    n_invalid_raw = int(invalid.sum())

    # MAD-based spike rejection on derivative (mark both surrounding samples)
    if spike_mad_thresh > 0 and r.size > 2:
        with np.errstate(invalid="ignore"):
            dr = np.diff(r)
        med_d = float(np.nanmedian(dr))
        mad_d = float(np.nanmedian(np.abs(dr - med_d))) or 1e-12
        spikes = np.abs(dr - med_d) > spike_mad_thresh * mad_d * 1.4826
        spike_idx = np.where(spikes)[0]
        invalid[spike_idx] = True
        invalid[np.clip(spike_idx + 1, 0, r.size - 1)] = True

    n_invalid = int(invalid.sum())
    r[invalid] = np.nan

    # Detect contiguous invalid spans before interpolation
    gaps: list[tuple[float, float, int]] = []
    in_gap = False
    g_start = 0
    for i, bad in enumerate(invalid):
        if bad and not in_gap:
            in_gap = True
            g_start = i
        elif not bad and in_gap:
            gaps.append((float(t[g_start]), float(t[i - 1]), i - g_start))
            in_gap = False
    if in_gap:
        gaps.append((float(t[g_start]), float(t[-1]), invalid.size - g_start))

    # Linear interpolation across gaps using time-aware interp
    valid = ~invalid
    if valid.sum() >= 2:
        r = np.interp(t, t[valid], r[valid], left=np.nan, right=np.nan)
        # Re-mark out-of-range interpolated values
        if max_rpm is not None:
            r[r > max_rpm] = np.nan
        r[r < min_rpm] = np.nan
    else:
        msgs.append("clean_rpm_signal: < 2 valid RPM samples — cannot interpolate.")

    if n_invalid > 0:
        msgs.append(
            f"clean_rpm_signal: {n_invalid}/{invalid.size} samples were invalid "
            f"(raw {n_invalid_raw}, +{n_invalid - n_invalid_raw} from spike rejection); "
            f"interpolated across {len(gaps)} gap(s)."
        )

    return {
        "t_rpm": t,
        "rpm": r,
        "valid_mask": ~invalid,
        "n_invalid": n_invalid,
        "gaps": gaps,
        "warnings": msgs,
    }


def align_rpm_to_signal(t_signal, t_rpm, rpm, *,
                         method: str = "linear",
                         fill: str = "edge") -> dict:
    """Interpolate an RPM channel onto a target signal time grid.

    Parameters
    ----------
    t_signal : array-like
        Target time vector (e.g. acceleration timebase). Must be monotonic.
    t_rpm, rpm : array-like
        RPM time and value vectors (typically pre-cleaned by ``clean_rpm_signal``).
    method : {"linear", "cubic", "nearest"}
        Interpolation kind for ``scipy.interpolate.interp1d``.
    fill : {"edge", "nan", "zero"}
        Strategy outside the RPM support:
            "edge"  : hold first/last valid value
            "nan"   : fill with NaN (caller must handle)
            "zero"  : fill with 0.0

    Returns
    -------
    dict with keys:
        t_signal : (N,) target time vector
        rpm      : (N,) interpolated RPM aligned to t_signal
        in_support : (N,) bool — True where t_signal is within t_rpm range
        warnings : list[str]
    """
    ts = validate_timebase(t_signal, name="t_signal")["t"]
    tr_v = validate_timebase(t_rpm, rpm, name="rpm")
    tr = tr_v["t"]
    rv = tr_v["x"]
    msgs = list(tr_v["warnings"])

    finite = np.isfinite(rv)
    if finite.sum() < 2:
        raise ValueError("align_rpm_to_signal: need at least 2 finite RPM samples.")
    tr_f = tr[finite]
    rv_f = rv[finite]

    in_support = (ts >= tr_f[0]) & (ts <= tr_f[-1])
    n_out = int((~in_support).sum())
    if n_out > 0:
        frac = n_out / ts.size
        if frac > 0.01:
            msgs.append(
                f"align_rpm_to_signal: {n_out} target samples ({frac:.1%}) lie outside "
                f"the RPM support [{tr_f[0]:.3f}, {tr_f[-1]:.3f}] s — using fill={fill!r}."
            )

    fill_value: Union[str, tuple, float]
    bounds_error = False
    if fill == "edge":
        fill_value = (float(rv_f[0]), float(rv_f[-1]))
    elif fill == "nan":
        fill_value = np.nan
    elif fill == "zero":
        fill_value = 0.0
    else:
        raise ValueError(f"align_rpm_to_signal: unknown fill={fill!r}")

    f = _interp1d(tr_f, rv_f, kind=method, bounds_error=bounds_error,
                  fill_value=fill_value, assume_sorted=True)
    rpm_aligned = f(ts).astype(np.float64)

    return {
        "t_signal": ts,
        "rpm": rpm_aligned,
        "in_support": in_support,
        "warnings": msgs,
    }


def smooth_rpm(rpm, fs_or_time, *,
               method: str = "savgol",
               window_sec: float = 0.1,
               polyorder: int = 2) -> np.ndarray:
    """Smooth an RPM signal that is already aligned to a uniform time grid.

    Parameters
    ----------
    rpm : array-like
        RPM samples on a uniform grid.
    fs_or_time : float or array-like
        Either the sample rate (Hz) or the corresponding time vector. If a
        time vector is given the median dt is used.
    method : {"savgol", "median"}
        Smoothing kernel. "savgol" is a Savitzky-Golay filter (preserves
        peaks); "median" is a centered running median (robust to spikes).
    window_sec : float
        Smoothing window length in seconds.
    polyorder : int
        Polynomial order for Savitzky-Golay (must be < window length).

    Returns
    -------
    np.ndarray (same shape as input)
    """
    r = np.asarray(rpm, dtype=np.float64).ravel()
    if np.isscalar(fs_or_time):
        fs = float(fs_or_time)
    else:
        t_arr = np.asarray(fs_or_time, dtype=np.float64).ravel()
        fs = 1.0 / float(np.median(np.diff(t_arr)))

    win = max(3, int(round(window_sec * fs)))
    if win % 2 == 0:
        win += 1
    if win >= r.size:
        win = max(3, r.size - (r.size + 1) % 2)

    method = method.lower()
    if method == "savgol":
        po = max(0, min(polyorder, win - 1))
        return _sp_signal.savgol_filter(r, window_length=win, polyorder=po, mode="interp")
    if method == "median":
        return _sp_signal.medfilt(r, kernel_size=win)
    raise ValueError(f"smooth_rpm: unknown method={method!r}")


def compute_shaft_phase(t, rpm) -> dict:
    """Integrate RPM(t) to obtain shaft phase θ(t).

    θ(t) = 2π · ∫₀ᵗ RPM(τ)/60 dτ          (radians, unwrapped, monotonic)

    Trapezoidal integration is used (second-order accuracy), in contrast
    with the forward-Euler ``np.cumsum`` accumulator used by the legacy
    GUI implementation in ``Last.py``.

    Parameters
    ----------
    t : array-like
        Time vector in seconds. Must be strictly monotonic increasing.
    rpm : array-like
        RPM samples aligned to ``t``. Must be finite and ≥ 0.

    Returns
    -------
    dict with keys:
        theta      : (N,) shaft phase (radians), starts at 0
        omega      : (N,) angular velocity (rad/s)
        monotonic  : bool — True if θ is strictly increasing
        total_revs : float — number of full revolutions = θ[-1] / (2π)
        warnings   : list[str]
    """
    t_arr = np.asarray(t, dtype=np.float64).ravel()
    r_arr = np.asarray(rpm, dtype=np.float64).ravel()
    if t_arr.size != r_arr.size:
        raise ValueError(
            f"compute_shaft_phase: length mismatch (t={t_arr.size}, rpm={r_arr.size})."
        )
    if t_arr.size < 2:
        raise ValueError("compute_shaft_phase: need at least 2 samples.")
    if not np.all(np.diff(t_arr) > 0):
        raise ValueError("compute_shaft_phase: time vector must be strictly increasing.")

    if not np.all(np.isfinite(r_arr)):
        raise ValueError("compute_shaft_phase: RPM contains NaN/Inf — clean before calling.")
    if np.any(r_arr < 0):
        raise ValueError("compute_shaft_phase: negative RPM not supported (would produce non-monotonic phase).")

    omega = 2.0 * np.pi * r_arr / 60.0          # rad/s
    theta = _cumtrapz(omega, t_arr, initial=0.0)

    msgs: list[str] = []
    diffs = np.diff(theta)
    monotonic = bool(np.all(diffs >= 0.0))
    if not monotonic:
        msgs.append("Computed phase is not strictly monotonic — check for zero/negative RPM segments.")
    total_revs = float(theta[-1]) / (2.0 * np.pi)

    return {
        "theta": theta,
        "omega": omega,
        "monotonic": monotonic,
        "total_revs": total_revs,
        "warnings": msgs,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — AMPLITUDE / SPECTRAL NORMALIZATION                      (Stage 2)
# ═════════════════════════════════════════════════════════════════════════════

_AMPLITUDE_MODES = (
    "peak", "rms", "power", "psd", "asd",
    "db_peak", "db_rms", "db_power", "db_psd",
)
_LINEAR_MODES = ("peak", "rms", "power", "psd", "asd")


def get_window_and_corrections(window_name: str, nperseg: int, fs: float) -> dict:
    """Build a window and pre-compute spectral correction factors.

    Returns
    -------
    dict with keys:
        w             : (nperseg,) window samples
        name          : window descriptor
        sum_w         : Σ w
        sum_w2        : Σ w²
        coherent_gain : Σ w / N      (amplitude scaling for tones)
        win_power     : Σ w² / N     (power scaling for noise)
        nenbw         : N · Σ w² / (Σ w)²  (Normalized ENBW, in bins)
        enbw_hz       : nenbw · fs/N
        nperseg, fs   : echoed inputs
    """
    nperseg = int(nperseg)
    fs = float(fs)
    if nperseg < 2 or fs <= 0:
        raise ValueError("get_window_and_corrections: nperseg ≥ 2 and fs > 0 required.")

    # scipy.signal.get_window default sym=True. For STFT/Welch the periodic
    # form is preferred — but we follow scipy's convention so users can pass
    # the same window names they would to scipy.signal.welch.
    w = _sp_signal.get_window(window_name, nperseg, fftbins=True).astype(np.float64)

    sum_w = float(np.sum(w))
    sum_w2 = float(np.sum(w * w))
    nenbw = nperseg * sum_w2 / (sum_w * sum_w)
    return {
        "w": w,
        "name": str(window_name),
        "sum_w": sum_w,
        "sum_w2": sum_w2,
        "coherent_gain": sum_w / nperseg,
        "win_power": sum_w2 / nperseg,
        "nenbw": nenbw,
        "enbw_hz": nenbw * fs / nperseg,
        "nperseg": nperseg,
        "fs": fs,
    }


def _spectrum_kernel(x_frame: np.ndarray, fs: float, w: np.ndarray, nfft: int):
    """Return rfft frequencies and the four base spectral quantities.

    All amplitude modes are derived from these so they remain mutually
    consistent. Returns (freqs, A_peak, A_rms, power_lin, psd).
    """
    n = w.size
    if x_frame.size != n:
        raise ValueError(f"_spectrum_kernel: frame length {x_frame.size} != window length {n}.")
    if nfft < n:
        raise ValueError(f"_spectrum_kernel: nfft ({nfft}) must be ≥ nperseg ({n}).")

    sum_w = float(np.sum(w))
    sum_w2 = float(np.sum(w * w))

    X = np.fft.rfft(x_frame * w, n=nfft)
    mag = np.abs(X)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    n_bins = freqs.size

    A_peak = (2.0 * mag) / max(sum_w, 1e-30)
    A_peak[0] *= 0.5
    if (nfft % 2 == 0) and n_bins > 1:
        A_peak[-1] *= 0.5

    # RMS: sinusoid has rms = peak/√2, but DC and Nyquist are deterministic
    A_rms = A_peak / np.sqrt(2.0)
    A_rms[0] = A_peak[0]
    if (nfft % 2 == 0) and n_bins > 1:
        A_rms[-1] = A_peak[-1]

    power_lin = A_rms * A_rms

    # One-sided PSD: 2|X|² / (fs · Σw²) ; DC and Nyquist not doubled
    psd = (2.0 * mag * mag) / max(fs * sum_w2, 1e-30)
    psd[0] *= 0.5
    if (nfft % 2 == 0) and n_bins > 1:
        psd[-1] *= 0.5

    return freqs, A_peak, A_rms, power_lin, psd


def single_sided_spectrum(x_frame, fs: float, window, nfft: int, *,
                          amplitude_mode: str = "rms",
                          ref: float = 1.0):
    """Compute the one-sided spectrum of a single windowed frame.

    Parameters
    ----------
    x_frame : array-like
        Time-domain frame. Length ``nperseg``.
    fs : float
        Sampling rate (Hz).
    window : array-like
        Window samples of length ``nperseg``.
    nfft : int
        FFT length (≥ nperseg). Zero-padded by rfft if larger.
    amplitude_mode : str
        One of "peak", "rms", "power", "psd", "asd",
        "db_peak", "db_rms", "db_power", "db_psd".
    ref : float
        Reference amplitude for dB modes (default 1.0).

    Returns
    -------
    freqs : (nfft//2 + 1,) array
    S     : (nfft//2 + 1,) array
        Spectrum in the requested amplitude mode.
    meta  : dict
    """
    mode = str(amplitude_mode).lower()
    if mode not in _AMPLITUDE_MODES:
        raise ValueError(
            f"single_sided_spectrum: amplitude_mode={amplitude_mode!r} unknown. "
            f"Choose from {_AMPLITUDE_MODES}."
        )

    x = np.asarray(x_frame, dtype=np.float64)
    w = np.asarray(window, dtype=np.float64)
    fs = float(fs)
    nfft = int(nfft)

    freqs, A_peak, A_rms, power_lin, psd = _spectrum_kernel(x, fs, w, nfft)

    if mode == "peak":
        S = A_peak
    elif mode == "rms":
        S = A_rms
    elif mode == "power":
        S = power_lin
    elif mode == "psd":
        S = psd
    elif mode == "asd":
        S = np.sqrt(np.maximum(psd, 0.0))
    elif mode == "db_peak":
        S = 20.0 * np.log10(np.maximum(A_peak, 1e-30) / max(ref, 1e-30))
    elif mode == "db_rms":
        S = 20.0 * np.log10(np.maximum(A_rms, 1e-30) / max(ref, 1e-30))
    elif mode == "db_power":
        S = 10.0 * np.log10(np.maximum(power_lin, 1e-30) / max(ref * ref, 1e-30))
    elif mode == "db_psd":
        S = 10.0 * np.log10(np.maximum(psd, 1e-30) / max(ref * ref, 1e-30))

    meta = {
        "amplitude_mode": mode,
        "fs": fs,
        "nperseg": int(w.size),
        "nfft": nfft,
        "ref": float(ref),
    }
    return freqs, S, meta


def convert_amplitude(values, from_mode: str, to_mode: str, *, ref: float = 1.0):
    """Convert spectral values between linear amplitude modes (and dB wrappers).

    Supported conversions for tonal content:
        peak ↔ rms ↔ power ↔ db_peak ↔ db_rms ↔ db_power
    PSD/ASD conversions are bandwidth-dependent and only handled when both
    sides agree (psd ↔ asd ↔ db_psd).

    Mixing tonal modes with PSD modes is not supported here because the
    correct conversion requires the analysis ENBW.
    """
    fm = str(from_mode).lower()
    tm = str(to_mode).lower()
    if fm == tm:
        return np.asarray(values, dtype=np.float64).copy()

    v = np.asarray(values, dtype=np.float64)
    tonal = {"peak", "rms", "power", "db_peak", "db_rms", "db_power"}
    density = {"psd", "asd", "db_psd"}

    if (fm in tonal) != (tm in tonal):
        raise ValueError(
            f"convert_amplitude: cannot convert between tonal ({fm}) and density ({tm}) "
            f"modes without explicit ENBW — recompute the spectrum in the target mode."
        )

    # Step 1: bring v to a canonical linear quantity
    #         tonal canonical = peak (linear amplitude)
    #         density canonical = psd
    if fm in tonal:
        if fm == "peak":
            lin = v
        elif fm == "rms":
            lin = v * np.sqrt(2.0)
        elif fm == "power":
            lin = np.sqrt(np.maximum(v, 0.0)) * np.sqrt(2.0)
        elif fm == "db_peak":
            lin = ref * 10.0 ** (v / 20.0)
        elif fm == "db_rms":
            lin = ref * 10.0 ** (v / 20.0) * np.sqrt(2.0)
        elif fm == "db_power":
            lin = ref * 10.0 ** (v / 20.0) * np.sqrt(2.0)  # 20·log10(rms/ref)
        # convert to target
        if tm == "peak":
            return lin
        if tm == "rms":
            return lin / np.sqrt(2.0)
        if tm == "power":
            return (lin / np.sqrt(2.0)) ** 2
        if tm == "db_peak":
            return 20.0 * np.log10(np.maximum(lin, 1e-30) / max(ref, 1e-30))
        if tm == "db_rms":
            return 20.0 * np.log10(np.maximum(lin / np.sqrt(2.0), 1e-30) / max(ref, 1e-30))
        if tm == "db_power":
            return 20.0 * np.log10(np.maximum(lin / np.sqrt(2.0), 1e-30) / max(ref, 1e-30))

    # Density branch
    if fm == "psd":
        psd = v
    elif fm == "asd":
        psd = v * v
    else:  # db_psd
        psd = (max(ref, 1e-30) ** 2) * 10.0 ** (v / 10.0)

    if tm == "psd":
        return psd
    if tm == "asd":
        return np.sqrt(np.maximum(psd, 0.0))
    return 10.0 * np.log10(np.maximum(psd, 1e-30) / max(ref * ref, 1e-30))


def compute_stft_map(x, fs: float, *,
                     nperseg: int = 2048,
                     noverlap: Optional[int] = None,
                     nfft: Optional[int] = None,
                     window: str = "hann",
                     amplitude_mode: str = "rms",
                     ref: float = 1.0) -> dict:
    """Welch/STFT-style framing → per-frame one-sided spectrum.

    Returns
    -------
    dict with keys:
        freq           : (nfft//2 + 1,) frequency axis (Hz)
        time           : (M,) frame-centre time vector (s)
        S              : (M, nfft//2 + 1) spectra (rows = frames)
        amplitude_mode : str
        window_info    : dict (from get_window_and_corrections)
        warnings       : list[str]
    """
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    fs = float(fs)
    nperseg = int(nperseg)
    if nfft is None:
        nfft = nperseg
    if noverlap is None:
        noverlap = nperseg // 2
    nfft = int(nfft); noverlap = int(noverlap)
    if not (0 <= noverlap < nperseg):
        raise ValueError(f"compute_stft_map: require 0 ≤ noverlap < nperseg.")
    if nperseg > x_arr.size:
        raise ValueError(
            f"compute_stft_map: nperseg ({nperseg}) > signal length ({x_arr.size})."
        )

    win = get_window_and_corrections(window, nperseg, fs)
    w = win["w"]
    hop = nperseg - noverlap
    n_frames = 1 + (x_arr.size - nperseg) // hop

    starts = np.arange(n_frames) * hop
    centres = (starts + nperseg / 2.0) / fs

    msgs: list[str] = []
    n_freq = nfft // 2 + 1
    S = np.empty((n_frames, n_freq), dtype=np.float64)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    for i, s in enumerate(starts):
        frame = x_arr[s:s + nperseg]
        _, S[i, :], _ = single_sided_spectrum(
            frame, fs, w, nfft,
            amplitude_mode=amplitude_mode, ref=ref,
        )

    return {
        "freq": freqs,
        "time": centres,
        "S": S,
        "amplitude_mode": amplitude_mode,
        "window_info": win,
        "warnings": msgs,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — rpmfreqmap-STYLE FREQUENCY-vs-RPM MAP                   (Stage 3)
# ═════════════════════════════════════════════════════════════════════════════

def compute_rpm_frequency_map(x, fs: float, t_signal, rpm, *,
                              nperseg: int = 2048,
                              noverlap: Optional[int] = None,
                              nfft: Optional[int] = None,
                              window: str = "hann",
                              amplitude_mode: str = "rms",
                              rpm_grid: Optional[Sequence[float]] = None,
                              rpm_bin_width: Optional[float] = None,
                              average_mode: str = "linear",
                              sort_by_rpm: bool = True,
                              ref: float = 1.0) -> dict:
    """MATLAB-like rpmfreqmap: STFT spectra labelled by frame-centre RPM.

    Each frame's spectrum is associated with the **centre-sample RPM** of
    that frame — physically meaningful for Hann-windowed data because the
    Hann window has its energy peak at the frame centre.

    Parameters
    ----------
    x, fs : array-like, float
        Acceleration (or other) signal and its sample rate.
    t_signal : array-like
        Time vector matching ``x``.
    rpm : array-like
        RPM aligned to ``t_signal`` (call ``align_rpm_to_signal`` first if not).
    nperseg, noverlap, nfft, window, amplitude_mode, ref
        Forwarded to ``compute_stft_map``.
    rpm_grid : array-like or None
        If given, interpolate per-frequency-row spectra onto this RPM grid
        (after sorting frames by RPM). Mutually exclusive with rpm_bin_width.
    rpm_bin_width : float or None
        If given, average frames whose RPM falls into uniform bins of this
        width (RPM units). Averaging is performed in the **power** domain
        to avoid Jensen-inequality bias, then converted back to the
        requested amplitude mode.
    average_mode : {"linear", "power"}
        Averaging convention when binning. "power" forces power-domain
        averaging even for amplitude modes (the safer choice).
    sort_by_rpm : bool
        Sort the output map rows by ascending RPM (default). The raw
        time-ordered grid is always preserved in the returned dict.

    Returns
    -------
    dict with keys:
        freq           : (Nf,) frequency axis (Hz)
        rpm            : (M,) RPM axis for the map rows
        map            : (M, Nf) the spectrum-vs-RPM map
        time           : (Mraw,) frame-centre times of the raw STFT
        frame_rpm      : (Mraw,) frame-centre RPM of the raw STFT
        raw_S          : (Mraw, Nf) time-ordered raw STFT (always preserved)
        amplitude_mode : str
        window_info    : dict
        warnings       : list[str]
    """
    if rpm_grid is not None and rpm_bin_width is not None:
        raise ValueError("compute_rpm_frequency_map: pass either rpm_grid or rpm_bin_width, not both.")

    base = validate_timebase(t_signal, x, name="signal")
    ts = base["t"]
    xs = base["x"]
    fs = float(fs) if base["fs"] is None else float(base["fs"])
    rpm_arr = np.asarray(rpm, dtype=np.float64).ravel()
    if rpm_arr.size != xs.size:
        raise ValueError(
            f"compute_rpm_frequency_map: rpm length ({rpm_arr.size}) != signal length ({xs.size})."
        )

    msgs: list[str] = list(base["warnings"])

    stft = compute_stft_map(
        xs, fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
        window=window, amplitude_mode=amplitude_mode, ref=ref,
    )
    raw_S = stft["S"]
    times = stft["time"]
    n_frames = times.size
    hop = nperseg - (noverlap if noverlap is not None else nperseg // 2)
    centre_idx = np.arange(n_frames) * hop + nperseg // 2
    centre_idx = np.clip(centre_idx, 0, rpm_arr.size - 1)
    frame_rpm = rpm_arr[centre_idx]

    finite = np.isfinite(frame_rpm)
    if not finite.all():
        msgs.append(f"compute_rpm_frequency_map: dropped {int((~finite).sum())} frames with non-finite RPM.")
    raw_S_full = raw_S
    times_full = times
    frame_rpm_full = frame_rpm
    raw_S = raw_S[finite]
    times = times[finite]
    frame_rpm = frame_rpm[finite]
    if frame_rpm.size < 2:
        raise RuntimeError("compute_rpm_frequency_map: < 2 frames with valid RPM — cannot build map.")

    # Detect direction changes
    drpm = np.diff(frame_rpm)
    if np.any(drpm > 0) and np.any(drpm < 0):
        msgs.append("RPM is non-monotonic across frames — run-up/coast-down direction changes inside the record.")

    # Build the map
    if rpm_grid is not None:
        rpm_axis = np.asarray(rpm_grid, dtype=np.float64).ravel()
        order = np.argsort(frame_rpm)
        rpm_sorted = frame_rpm[order]
        S_sorted = raw_S[order]
        # Power-domain interpolation to avoid bias when amplitudes are skewed
        S_power = _to_power_for_average(S_sorted, amplitude_mode, ref)
        out_pow = np.empty((rpm_axis.size, S_power.shape[1]), dtype=np.float64)
        # Deduplicate identical rpm values
        uniq, idx = np.unique(rpm_sorted, return_index=True)
        if uniq.size < 2:
            msgs.append("compute_rpm_frequency_map: only one unique frame RPM — output map is constant in RPM.")
            for k in range(S_power.shape[1]):
                out_pow[:, k] = S_power[idx[0], k]
        else:
            S_power_uniq = S_power[idx]
            for k in range(S_power.shape[1]):
                out_pow[:, k] = np.interp(rpm_axis, uniq, S_power_uniq[:, k],
                                           left=S_power_uniq[0, k], right=S_power_uniq[-1, k])
        S_map = _from_power_after_average(out_pow, amplitude_mode, ref)
    elif rpm_bin_width is not None:
        rpm_min, rpm_max = float(np.min(frame_rpm)), float(np.max(frame_rpm))
        if rpm_max <= rpm_min:
            rpm_axis = np.array([rpm_min])
            S_map = raw_S.mean(axis=0, keepdims=True)
        else:
            edges = np.arange(rpm_min, rpm_max + rpm_bin_width, rpm_bin_width)
            if edges.size < 2:
                edges = np.array([rpm_min, rpm_max])
            centres = 0.5 * (edges[:-1] + edges[1:])
            S_power = _to_power_for_average(raw_S, amplitude_mode, ref)
            out_pow = np.zeros((centres.size, S_power.shape[1]), dtype=np.float64)
            counts = np.zeros(centres.size, dtype=np.int64)
            bin_idx = np.clip(np.digitize(frame_rpm, edges) - 1, 0, centres.size - 1)
            for i, b in enumerate(bin_idx):
                out_pow[b] += S_power[i]
                counts[b] += 1
            valid = counts > 0
            out_pow[valid] /= counts[valid, None]
            rpm_axis = centres[valid]
            S_map = _from_power_after_average(out_pow[valid], amplitude_mode, ref)
            if average_mode == "linear" and amplitude_mode in ("peak", "rms", "asd"):
                msgs.append("compute_rpm_frequency_map: amplitude-mode bin averaging is performed in the power domain to avoid bias.")
    else:
        if sort_by_rpm:
            order = np.argsort(frame_rpm)
            rpm_axis = frame_rpm[order]
            S_map = raw_S[order]
        else:
            rpm_axis = frame_rpm
            S_map = raw_S

    return {
        "freq": stft["freq"],
        "rpm": rpm_axis,
        "map": S_map,
        "time": times_full,
        "frame_rpm": frame_rpm_full,
        "raw_S": raw_S_full,
        "amplitude_mode": amplitude_mode,
        "window_info": stft["window_info"],
        "warnings": msgs + stft["warnings"],
    }


def _to_power_for_average(S, amplitude_mode: str, ref: float) -> np.ndarray:
    """Map any amplitude mode into a linear power-like quantity for averaging."""
    mode = amplitude_mode.lower()
    if mode == "power" or mode == "psd":
        return np.asarray(S, dtype=np.float64)
    if mode == "rms":
        return np.asarray(S, dtype=np.float64) ** 2
    if mode == "peak":
        v = np.asarray(S, dtype=np.float64)
        return (v / np.sqrt(2.0)) ** 2
    if mode == "asd":
        return np.asarray(S, dtype=np.float64) ** 2
    if mode == "db_psd":
        return (max(ref, 1e-30) ** 2) * 10.0 ** (np.asarray(S, dtype=np.float64) / 10.0)
    if mode == "db_power":
        return (max(ref, 1e-30) ** 2) * 10.0 ** (np.asarray(S, dtype=np.float64) / 10.0)
    if mode == "db_rms":
        rms = max(ref, 1e-30) * 10.0 ** (np.asarray(S, dtype=np.float64) / 20.0)
        return rms * rms
    if mode == "db_peak":
        peak = max(ref, 1e-30) * 10.0 ** (np.asarray(S, dtype=np.float64) / 20.0)
        return (peak / np.sqrt(2.0)) ** 2
    raise ValueError(f"_to_power_for_average: unknown mode {mode!r}")


def _from_power_after_average(P, amplitude_mode: str, ref: float) -> np.ndarray:
    """Inverse of _to_power_for_average."""
    mode = amplitude_mode.lower()
    P = np.maximum(np.asarray(P, dtype=np.float64), 0.0)
    if mode == "power" or mode == "psd":
        return P
    if mode == "rms":
        return np.sqrt(P)
    if mode == "peak":
        return np.sqrt(P) * np.sqrt(2.0)
    if mode == "asd":
        return np.sqrt(P)
    if mode == "db_psd":
        return 10.0 * np.log10(np.maximum(P, 1e-30) / max(ref * ref, 1e-30))
    if mode == "db_power":
        return 10.0 * np.log10(np.maximum(P, 1e-30) / max(ref * ref, 1e-30))
    if mode == "db_rms":
        return 20.0 * np.log10(np.maximum(np.sqrt(P), 1e-30) / max(ref, 1e-30))
    if mode == "db_peak":
        return 20.0 * np.log10(np.maximum(np.sqrt(P) * np.sqrt(2.0), 1e-30) / max(ref, 1e-30))
    raise ValueError(f"_from_power_after_average: unknown mode {mode!r}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ANGLE-DOMAIN ORDER TRACKING                             (Stage 4)
# ═════════════════════════════════════════════════════════════════════════════

def resample_to_angle_domain(t, x, rpm, *, samples_per_rev: int = 1024,
                              interp_kind: str = "cubic") -> dict:
    """Resample a time-domain signal onto a uniform shaft-angle grid.

    Computes θ(t) via ``compute_shaft_phase``, builds θ_uniform with
    ``samples_per_rev`` points per revolution, and resamples ``x`` onto
    that grid using cubic interpolation (better amplitude preservation
    than linear at this stage).

    Parameters
    ----------
    t, x, rpm : array-like
        Same-length time vector, signal, and RPM (already aligned).
    samples_per_rev : int
        Resampling density in samples per shaft revolution. 512 / 1024 typical.
    interp_kind : {"cubic", "linear"}
        ``scipy.interpolate.interp1d`` kind. "cubic" preserves amplitude
        better; "linear" is faster.

    Returns
    -------
    dict with keys:
        x_angle         : (K,) signal sampled uniformly in angle
        theta           : (N,) original-domain shaft phase (rad)
        theta_uniform   : (K,) uniform angle grid (rad)
        time_at_angle   : (K,) time of each angle sample (interpolated)
        samples_per_rev : int
        n_revs          : float
        warnings        : list[str]
    """
    t_arr = np.asarray(t, dtype=np.float64).ravel()
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    rpm_arr = np.asarray(rpm, dtype=np.float64).ravel()
    if not (t_arr.size == x_arr.size == rpm_arr.size):
        raise ValueError(
            f"resample_to_angle_domain: length mismatch "
            f"(t={t_arr.size}, x={x_arr.size}, rpm={rpm_arr.size})."
        )

    phase = compute_shaft_phase(t_arr, rpm_arr)
    theta = phase["theta"]
    if not phase["monotonic"]:
        raise ValueError("resample_to_angle_domain: shaft phase is not monotonic — cannot resample to angle.")

    n_revs = float(phase["total_revs"])
    if n_revs < 0.5:
        raise ValueError(
            f"resample_to_angle_domain: only {n_revs:.3f} shaft revolutions in record — "
            "too short for meaningful order analysis."
        )

    # Strict monotonicity for interp1d
    diffs = np.diff(theta)
    if np.any(diffs <= 0):
        raise ValueError("resample_to_angle_domain: phase has non-positive increments after integration.")

    K = int(round(n_revs * samples_per_rev))
    K = max(K, 4)
    theta_uniform = np.linspace(theta[0], theta[-1], K)

    # x(theta) and t(theta) interpolation
    fx = _interp1d(theta, x_arr, kind=interp_kind, bounds_error=False,
                   fill_value=0.0, assume_sorted=True)
    ft = _interp1d(theta, t_arr, kind="linear", bounds_error=False,
                   fill_value=(t_arr[0], t_arr[-1]), assume_sorted=True)

    x_angle = fx(theta_uniform).astype(np.float64)
    time_at_angle = ft(theta_uniform).astype(np.float64)

    return {
        "x_angle": x_angle,
        "theta": theta,
        "theta_uniform": theta_uniform,
        "time_at_angle": time_at_angle,
        "samples_per_rev": int(samples_per_rev),
        "n_revs": n_revs,
        "warnings": list(phase["warnings"]),
    }


def compute_order_spectrum_angle_domain(x_angle, samples_per_rev: int, *,
                                         n_revs_per_segment: Optional[int] = None,
                                         overlap_revs: Optional[int] = None,
                                         window: str = "hann",
                                         amplitude_mode: str = "rms",
                                         max_order: float = 20.0,
                                         ref: float = 1.0) -> dict:
    """One-sided order spectrum from an angle-domain signal.

    The "frequency" axis of an rfft on an angle-domain signal is in units
    of cycles per revolution = orders. Bin spacing = 1 / n_revs_per_segment.

    Parameters
    ----------
    x_angle : array-like
        Signal sampled uniformly in shaft angle (output of ``resample_to_angle_domain``).
    samples_per_rev : int
        Angle-domain sample rate (samples per revolution).
    n_revs_per_segment : int or None
        Segment length in shaft revolutions. None → use entire record (single FFT).
    overlap_revs : int or None
        Inter-segment overlap in revolutions. Default = half of segment.
    window, amplitude_mode, ref
        See ``single_sided_spectrum``. Note: "psd" mode here returns
        units of (signal²) per (orders), with per-order normalization
        analogous to per-Hz PSD in the time domain.
    max_order : float
        Trim the output to orders ≤ max_order.

    Returns
    -------
    dict with keys:
        order             : (Norder,) order axis
        amp               : averaged spectrum (single-FFT or Welch-averaged in power domain)
        n_segments        : int
        amplitude_mode    : str
        window_info       : dict
        order_resolution  : float — Δorder
    """
    xa = np.asarray(x_angle, dtype=np.float64).ravel()
    spr = int(samples_per_rev)
    if spr < 4:
        raise ValueError("compute_order_spectrum_angle_domain: samples_per_rev must be ≥ 4.")

    total_revs = xa.size / spr
    if n_revs_per_segment is None or n_revs_per_segment >= total_revs:
        n_revs_per_segment = int(total_revs)
    n_revs_per_segment = max(1, int(n_revs_per_segment))
    nperseg = n_revs_per_segment * spr
    nperseg = min(nperseg, xa.size)
    if nperseg < 8:
        raise ValueError("compute_order_spectrum_angle_domain: segment too short (< 8 angle samples).")

    if overlap_revs is None:
        overlap_revs = n_revs_per_segment // 2
    noverlap = max(0, min(int(overlap_revs) * spr, nperseg - 1))
    hop = nperseg - noverlap

    win = get_window_and_corrections(window, nperseg, fs=float(spr))
    w = win["w"]

    n_segments = 1 + (xa.size - nperseg) // hop if xa.size >= nperseg else 1
    nfft = nperseg
    n_bins = nfft // 2 + 1

    # Average in the linear-power domain and convert back at the end.
    P_accum = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_segments):
        s = i * hop
        frame = xa[s:s + nperseg]
        if frame.size < nperseg:
            tmp = np.zeros(nperseg, dtype=np.float64)
            tmp[:frame.size] = frame
            frame = tmp
        _, P_i, _ = single_sided_spectrum(frame, fs=float(spr), window=w, nfft=nfft,
                                           amplitude_mode="power", ref=1.0)
        P_accum += P_i
    P_avg = P_accum / max(n_segments, 1)

    # Recover requested amplitude mode from power
    A_rms = np.sqrt(np.maximum(P_avg, 0.0))
    A_peak = A_rms * np.sqrt(2.0)
    A_peak[0] = A_rms[0]
    if (nfft % 2 == 0) and n_bins > 1:
        A_peak[-1] = A_rms[-1]

    mode = str(amplitude_mode).lower()
    if mode == "rms":
        S = A_rms
    elif mode == "peak":
        S = A_peak
    elif mode == "power":
        S = P_avg
    elif mode == "db_rms":
        S = 20.0 * np.log10(np.maximum(A_rms, 1e-30) / max(ref, 1e-30))
    elif mode == "db_peak":
        S = 20.0 * np.log10(np.maximum(A_peak, 1e-30) / max(ref, 1e-30))
    elif mode == "db_power":
        S = 10.0 * np.log10(np.maximum(P_avg, 1e-30) / max(ref * ref, 1e-30))
    elif mode in ("psd", "asd", "db_psd"):
        # Per-order density (analogous to per-Hz PSD)
        sum_w2 = float(np.sum(w * w))
        # Recover |X|² from P_avg = (A_rms)² and the kernel formula
        # P_avg = (1/2)·A_peak² ; A_peak = 2|X|/Σw → |X|² = (Σw)²·A_peak²/4
        # So PSD = 2|X|²/(spr·Σw²) = (Σw)²·A_peak²/(2·spr·Σw²)
        sum_w = win["sum_w"]
        psd = (sum_w * sum_w) * (A_peak * A_peak) / (2.0 * float(spr) * sum_w2)
        psd[0] *= 0.5
        if (nfft % 2 == 0) and n_bins > 1:
            psd[-1] *= 0.5
        if mode == "psd":
            S = psd
        elif mode == "asd":
            S = np.sqrt(np.maximum(psd, 0.0))
        else:
            S = 10.0 * np.log10(np.maximum(psd, 1e-30) / max(ref * ref, 1e-30))
    else:
        raise ValueError(f"compute_order_spectrum_angle_domain: unknown amplitude_mode={amplitude_mode!r}")

    d_order = float(spr) / float(nfft)
    orders = np.arange(n_bins, dtype=np.float64) * d_order
    keep = orders <= float(max_order)

    return {
        "order": orders[keep],
        "amp": S[keep],
        "n_segments": int(n_segments),
        "amplitude_mode": mode,
        "window_info": win,
        "order_resolution": d_order,
    }


def compute_rpm_order_map(x, t_signal, rpm, *,
                          samples_per_rev: int = 1024,
                          revs_per_segment: int = 16,
                          overlap_revs: int = 8,
                          max_order: float = 20.0,
                          order_resolution: Optional[float] = None,
                          window: str = "hann",
                          amplitude_mode: str = "rms",
                          ref: float = 1.0) -> dict:
    """MATLAB-like rpmordermap: order spectra vs RPM via angle-domain STFT.

    Each segment in the angle domain corresponds to a contiguous time
    window. The mean RPM over that time window is the row label.

    Parameters
    ----------
    x, t_signal, rpm
        Time-domain signal and matching time / RPM vectors.
    samples_per_rev : int
        Angle-domain resampling density.
    revs_per_segment : int
        STFT segment length in revolutions.
    overlap_revs : int
        Inter-segment overlap in revolutions.
    max_order : float
        Trim output to orders ≤ max_order.
    order_resolution : float or None
        If given, force the per-segment FFT length to achieve this
        Δorder = samples_per_rev / nfft. Must be ≥ 1/revs_per_segment.
    window, amplitude_mode, ref
        Forwarded to ``single_sided_spectrum``.

    Returns
    -------
    dict with keys:
        order           : (Norder,) order axis
        rpm             : (M,) RPM axis (one per segment, sorted ascending)
        map             : (M, Norder) map
        angle           : (K,) full angle-domain signal angle vector
        time_at_angle   : (K,) time vector for the angle-domain signal
        samples_per_rev : int
        amplitude_mode  : str
        window_info     : dict
        order_resolution: float
        n_segments      : int
        warnings        : list[str]
    """
    # Angle-domain resample (uses the validated, monotonic phase)
    ang = resample_to_angle_domain(t_signal, x, rpm, samples_per_rev=samples_per_rev)
    xa = ang["x_angle"]
    t_at_a = ang["time_at_angle"]
    spr = ang["samples_per_rev"]
    msgs: list[str] = list(ang["warnings"])

    nperseg = int(revs_per_segment) * spr
    if nperseg > xa.size:
        nperseg = xa.size - (xa.size % spr) or xa.size
        msgs.append(f"compute_rpm_order_map: clamped nperseg to {nperseg} (record only {xa.size/spr:.2f} revs).")
    if nperseg < 2 * spr:
        raise ValueError(
            f"compute_rpm_order_map: need ≥ 2 revolutions per segment, got "
            f"{nperseg/spr:.2f} (record length: {xa.size/spr:.2f} revs)."
        )

    # Force a finer Δorder = spr / nfft via zero-padding
    if order_resolution is not None:
        nfft = int(np.ceil(spr / float(order_resolution)))
        nfft = max(nfft, nperseg)
    else:
        nfft = nperseg

    noverlap = max(0, min(int(overlap_revs) * spr, nperseg - 1))
    hop = nperseg - noverlap
    n_segments = 1 + (xa.size - nperseg) // hop if xa.size >= nperseg else 1

    win = get_window_and_corrections(window, nperseg, fs=float(spr))
    w = win["w"]
    n_bins = nfft // 2 + 1

    # We compute power-domain spectra per segment; row labels are mean
    # RPM over the time interval that the segment spans.
    rpm_arr = np.asarray(rpm, dtype=np.float64).ravel()
    t_arr = np.asarray(t_signal, dtype=np.float64).ravel()

    # Pre-build a fast time → RPM lookup
    rpm_at_time = _interp1d(t_arr, rpm_arr, kind="linear", bounds_error=False,
                              fill_value=(float(rpm_arr[0]), float(rpm_arr[-1])),
                              assume_sorted=False)

    P_rows = np.zeros((n_segments, n_bins), dtype=np.float64)
    seg_rpm = np.zeros(n_segments, dtype=np.float64)
    for i in range(n_segments):
        s = i * hop
        frame = xa[s:s + nperseg]
        if frame.size < nperseg:
            tmp = np.zeros(nperseg, dtype=np.float64)
            tmp[:frame.size] = frame
            frame = tmp
        _, P_i, _ = single_sided_spectrum(frame, fs=float(spr), window=w, nfft=nfft,
                                           amplitude_mode="power", ref=1.0)
        P_rows[i] = P_i

        t_lo = float(t_at_a[s])
        t_hi = float(t_at_a[min(s + nperseg - 1, xa.size - 1)])
        # Use a few samples of RPM in this interval
        n_probe = max(8, int(round(nperseg / spr * 4)))
        t_probe = np.linspace(t_lo, t_hi, n_probe)
        seg_rpm[i] = float(np.mean(rpm_at_time(t_probe)))

    # Sort rows by RPM
    order = np.argsort(seg_rpm)
    seg_rpm = seg_rpm[order]
    P_rows = P_rows[order]

    # Convert to requested amplitude mode (per-row)
    S_rows = np.empty_like(P_rows)
    for i in range(P_rows.shape[0]):
        S_rows[i] = _power_row_to_mode(P_rows[i], amplitude_mode, win, spr, nfft, ref)

    d_order = float(spr) / float(nfft)
    orders = np.arange(n_bins, dtype=np.float64) * d_order
    keep = orders <= float(max_order)

    return {
        "order": orders[keep],
        "rpm": seg_rpm,
        "map": S_rows[:, keep],
        "angle": ang["theta_uniform"],
        "time_at_angle": t_at_a,
        "samples_per_rev": spr,
        "amplitude_mode": amplitude_mode,
        "window_info": win,
        "order_resolution": d_order,
        "n_segments": int(n_segments),
        "warnings": msgs,
    }


def _power_row_to_mode(P, amplitude_mode: str, win: dict, spr: int, nfft: int, ref: float):
    """Helper: convert a single power-spectrum row to the requested amplitude mode."""
    A_rms = np.sqrt(np.maximum(P, 0.0))
    A_peak = A_rms * np.sqrt(2.0)
    A_peak[0] = A_rms[0]
    if (nfft % 2 == 0) and A_peak.size > 1:
        A_peak[-1] = A_rms[-1]

    mode = amplitude_mode.lower()
    if mode == "rms":
        return A_rms
    if mode == "peak":
        return A_peak
    if mode == "power":
        return P
    if mode == "db_rms":
        return 20.0 * np.log10(np.maximum(A_rms, 1e-30) / max(ref, 1e-30))
    if mode == "db_peak":
        return 20.0 * np.log10(np.maximum(A_peak, 1e-30) / max(ref, 1e-30))
    if mode == "db_power":
        return 10.0 * np.log10(np.maximum(P, 1e-30) / max(ref * ref, 1e-30))

    if mode in ("psd", "asd", "db_psd"):
        sum_w2 = float(np.sum(win["w"] ** 2))
        sum_w = win["sum_w"]
        psd = (sum_w * sum_w) * (A_peak * A_peak) / (2.0 * float(spr) * sum_w2)
        psd[0] *= 0.5
        if (nfft % 2 == 0) and psd.size > 1:
            psd[-1] *= 0.5
        if mode == "psd":
            return psd
        if mode == "asd":
            return np.sqrt(np.maximum(psd, 0.0))
        return 10.0 * np.log10(np.maximum(psd, 1e-30) / max(ref * ref, 1e-30))

    raise ValueError(f"_power_row_to_mode: unknown amplitude_mode={amplitude_mode!r}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ORDER TRACE EXTRACTION                            (Stage 6 part 1)
# ═════════════════════════════════════════════════════════════════════════════

def extract_order_traces(order_axis, rpm_axis, order_map,
                          target_orders: Sequence[float], *,
                          method: str = "interpolated",
                          bandwidth: float = 0.05,
                          amplitude_mode: str = "rms",
                          ref: float = 1.0) -> dict:
    """Extract amplitude-vs-RPM traces at requested orders from an order map.

    Parameters
    ----------
    order_axis : (Norder,) array
        Order axis from ``compute_rpm_order_map``.
    rpm_axis : (M,) array
        RPM axis from ``compute_rpm_order_map``.
    order_map : (M, Norder) array
        Map values in the units indicated by ``amplitude_mode``.
    target_orders : sequence of float
        Orders to extract — supports 0.5, 1, 2, fractional, harmonics, etc.
    method : {"nearest_bin", "interpolated", "peak_in_band", "rms_band_power"}
        Extraction strategy:
            - nearest_bin    : nearest order bin per RPM row
            - interpolated   : linear interpolation along the order axis
            - peak_in_band   : max(|map|) within ± bandwidth orders
            - rms_band_power : total band power → RMS within ± bandwidth orders.
              For density modes (psd/asd/db_psd) the integral is
              Σ psd · Δorder, which physically is the band power.
              For tonal modes (peak/rms/power) the integral is Σ power
              over the included bins. WARNING: a windowed line spectrum
              spreads tone energy across the window's main lobe (≈ ENBW
              bins), so summing tonal-mode bin powers across more than one
              bin OVER-counts by roughly the window's NENBW (1.5 for Hann,
              1.0 for rectangular, etc.). Prefer ``peak_in_band`` for tonal
              extraction; reserve ``rms_band_power`` for PSD-like inputs.
    bandwidth : float
        Half-width in orders for "peak_in_band" / "rms_band_power".
    amplitude_mode : str
        Mode of the input map (used by "rms_band_power"). Output traces
        are returned in the same mode.
    ref : float
        Reference for dB modes when converting via ``_to_power_for_average``.

    Returns
    -------
    dict mapping target_order -> {"rpm", "amplitude", "method", "bandwidth"}.
    """
    o_axis = np.asarray(order_axis, dtype=np.float64).ravel()
    r_axis = np.asarray(rpm_axis, dtype=np.float64).ravel()
    M = np.asarray(order_map, dtype=np.float64)
    if M.shape != (r_axis.size, o_axis.size):
        raise ValueError(
            f"extract_order_traces: order_map shape {M.shape} != "
            f"({r_axis.size}, {o_axis.size})."
        )
    if not np.all(np.diff(o_axis) > 0):
        raise ValueError("extract_order_traces: order_axis must be strictly increasing.")

    method = method.lower()
    if method not in ("nearest_bin", "interpolated", "peak_in_band", "rms_band_power"):
        raise ValueError(f"extract_order_traces: unknown method={method!r}")

    out: dict = {}
    d_order = float(o_axis[1] - o_axis[0])

    for tgt in target_orders:
        tgt_f = float(tgt)
        if method == "nearest_bin":
            j = int(np.argmin(np.abs(o_axis - tgt_f)))
            amp = M[:, j].copy()
        elif method == "interpolated":
            amp = np.empty(r_axis.size, dtype=np.float64)
            for i in range(r_axis.size):
                amp[i] = float(np.interp(tgt_f, o_axis, M[i], left=M[i, 0], right=M[i, -1]))
        elif method == "peak_in_band":
            lo, hi = tgt_f - bandwidth, tgt_f + bandwidth
            mask = (o_axis >= lo) & (o_axis <= hi)
            if not mask.any():
                # Fall back to nearest single bin
                j = int(np.argmin(np.abs(o_axis - tgt_f)))
                mask = np.zeros_like(o_axis, dtype=bool)
                mask[j] = True
            amp = np.max(M[:, mask], axis=1)
        else:  # rms_band_power
            lo, hi = tgt_f - bandwidth, tgt_f + bandwidth
            mask = (o_axis >= lo) & (o_axis <= hi)
            if not mask.any():
                j = int(np.argmin(np.abs(o_axis - tgt_f)))
                mask = np.zeros_like(o_axis, dtype=bool)
                mask[j] = True
            mode = amplitude_mode.lower()
            P = _to_power_for_average(M[:, mask], amplitude_mode, ref)
            # Tonal modes (peak/rms/power and their dB variants) carry power per
            # bin already (Parseval). Density modes (psd/asd/db_psd) carry power
            # per unit order, so they must be integrated by Δorder to recover
            # total band power.
            density_mode = mode in ("psd", "asd", "db_psd")
            integ = np.sum(P, axis=1) * (d_order if density_mode else 1.0)
            integ = np.maximum(integ, 0.0)
            # Convert integrated power back to the requested amplitude mode
            if mode in ("rms", "asd"):
                amp = np.sqrt(integ)
            elif mode == "peak":
                amp = np.sqrt(integ) * np.sqrt(2.0)
            elif mode in ("power", "psd"):
                amp = integ
            elif mode == "db_rms":
                amp = 20.0 * np.log10(np.maximum(np.sqrt(integ), 1e-30) / max(ref, 1e-30))
            elif mode == "db_peak":
                amp = 20.0 * np.log10(np.maximum(np.sqrt(integ) * np.sqrt(2.0), 1e-30) / max(ref, 1e-30))
            elif mode in ("db_power", "db_psd"):
                amp = 10.0 * np.log10(np.maximum(integ, 1e-30) / max(ref * ref, 1e-30))
            else:
                amp = integ

        out[float(tgt_f)] = {
            "rpm": r_axis,
            "amplitude": amp,
            "method": method,
            "bandwidth": float(bandwidth) if method in ("peak_in_band", "rms_band_power") else 0.0,
        }
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Module exports
# ═════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Stage 1
    "validate_timebase", "clean_rpm_signal", "align_rpm_to_signal",
    "smooth_rpm", "compute_shaft_phase",
    # Stage 2
    "get_window_and_corrections", "single_sided_spectrum",
    "compute_stft_map", "convert_amplitude",
    # Stage 3
    "compute_rpm_frequency_map",
    # Stage 4
    "resample_to_angle_domain", "compute_order_spectrum_angle_domain",
    "compute_rpm_order_map",
    # Stage 6 (partial)
    "extract_order_traces",
]
