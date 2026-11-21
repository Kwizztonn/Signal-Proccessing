"""
rotating_machinery_validation.py
─────────────────────────────────────────────────────────────────────────────
Synthetic-signal regression tests for ``rotating_machinery_core``.

Mirrors the style of ``wavelet_validation_v2.py``: a small TestRunner,
PASS/FAIL with numerical detail, optional PNG output for visual ridge
inspection, and a CLI.

USAGE
-----
    python rotating_machinery_validation.py                # all tests + figures
    python rotating_machinery_validation.py --no-show      # CI / headless

Tests (12 total, mapped to user spec stages 1-4 + 6 + 10):
    1.  validate_timebase / clean_rpm_signal edge cases
    2.  compute_shaft_phase: constant RPM → linear phase
    3.  compute_shaft_phase: linear-rising RPM → quadratic phase
    4.  Amplitude normalization on bin-centred 10 g sine
    5.  Off-bin sine — total power preserved
    6.  White-noise PSD flatness
    7.  Constant-RPM 1X via compute_rpm_frequency_map
    8.  Run-up 1X via compute_rpm_frequency_map (diagonal ridge)
    9.  Run-up 1X via compute_rpm_order_map (horizontal ridge at order 1)
    10. Run-up 2X via compute_rpm_order_map (ridge at order 2)
    11. Mixed 0.5X + 1X + 2X via extract_order_traces
    12. Frequency-map vs order-map ridge comparison (slope ≠ 0 vs ≈ 0)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib

import rotating_machinery_core as rmc
from rotating_machinery_core import (
    align_rpm_to_signal, clean_rpm_signal, compute_shaft_phase,
    compute_rpm_frequency_map, compute_rpm_order_map,
    convert_amplitude, extract_order_traces,
    get_window_and_corrections, single_sided_spectrum,
    validate_timebase,
)


# ═════════════════════════════════════════════════════════════════════════════
# TestRunner — mirrors wavelet_validation_v2.py:593-620
# ═════════════════════════════════════════════════════════════════════════════

class TestRunner:
    def __init__(self):
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, condition: bool, detail: str = ""):
        tag = "PASS" if bool(condition) else "FAIL"
        msg = f"  [{tag}] {name}"
        if detail:
            msg += f"  ({detail})"
        print(msg)
        self.results.append((name, bool(condition), detail))

    def summary(self) -> bool:
        n_pass = sum(1 for _, ok, _ in self.results if ok)
        n_total = len(self.results)
        print(f"\n{'=' * 70}")
        print(f"  RESULT: {n_pass}/{n_total} tests passed")
        if n_pass == n_total:
            print("  ALL TESTS PASSED.")
        else:
            print("  SOME TESTS FAILED:")
            for name, ok, detail in self.results:
                if not ok:
                    print(f"    - {name}  ({detail})")
        print('=' * 70)
        return n_pass == n_total


# ═════════════════════════════════════════════════════════════════════════════
# Synthetic signal generators (use the module's own shaft-phase routine
# so the generators are self-consistent with the analyser).
# ═════════════════════════════════════════════════════════════════════════════

def make_constant_rpm_signal(fs: float, T: float, rpm: float, *,
                              harmonics: dict[float, float] | None = None) -> tuple:
    """Constant-RPM time signal with a sum of order components.

    harmonics : {order: amplitude_peak}, default {1.0: 1.0}.
    """
    if harmonics is None:
        harmonics = {1.0: 1.0}
    N = int(round(fs * T))
    t = np.arange(N) / fs
    rpm_vec = np.full(N, float(rpm))
    phase = compute_shaft_phase(t, rpm_vec)
    theta = phase["theta"]
    x = np.zeros(N, dtype=np.float64)
    for order, amp in harmonics.items():
        x += float(amp) * np.cos(float(order) * theta)
    return t, x, rpm_vec


def make_runup_signal(fs: float, T: float, rpm0: float, rpm1: float, *,
                       harmonics: dict[float, float] | None = None) -> tuple:
    """Linear-RPM run-up (or coast-down if rpm1 < rpm0)."""
    if harmonics is None:
        harmonics = {1.0: 1.0}
    N = int(round(fs * T))
    t = np.arange(N) / fs
    rpm_vec = np.linspace(float(rpm0), float(rpm1), N)
    phase = compute_shaft_phase(t, rpm_vec)
    theta = phase["theta"]
    x = np.zeros(N, dtype=np.float64)
    for order, amp in harmonics.items():
        x += float(amp) * np.cos(float(order) * theta)
    return t, x, rpm_vec


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _ridge_freq(freqs, mag) -> np.ndarray:
    """Per-row dominant frequency (or order)."""
    return freqs[np.argmax(mag, axis=1)]


def _save_or_close(fig, path: str, save: bool):
    if save:
        fig.savefig(path, dpi=120)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1 — validate_timebase / clean_rpm_signal edge cases       (Stage 1)
# ═════════════════════════════════════════════════════════════════════════════

def test_timebase_and_rpm_cleanup(runner: TestRunner, save: bool, outdir: str):
    print("\n[TEST 1] validate_timebase + clean_rpm_signal edge cases")

    # 1a. Sorting + duplicates
    t = np.array([0.0, 0.1, 0.05, 0.05, 0.2, 0.15], dtype=np.float64)
    x = np.arange(t.size, dtype=np.float64)
    info = validate_timebase(t, x, name="unit")
    runner.check("validate_timebase: sorts & dedups",
                 info["t"].size == 5 and np.all(np.diff(info["t"]) > 0),
                 f"out len={info['t'].size}, monotone={np.all(np.diff(info['t']) > 0)}")

    # 1b. ValueError on broken time vector
    try:
        validate_timebase(np.array([0.0]), name="too_short")
    except ValueError:
        ok = True
    else:
        ok = False
    runner.check("validate_timebase: raises on < 2 samples", ok)

    # 1c. RPM cleanup with NaN, negative, and a spike
    t_rpm = np.linspace(0.0, 1.0, 1001)
    rpm_clean_in = 3000.0 + 50.0 * np.sin(2 * np.pi * 2.0 * t_rpm)  # mild ripple
    rpm_in = rpm_clean_in.copy()
    rpm_in[100:110] = np.nan
    rpm_in[500] = -50.0
    rpm_in[700] = 9999.0
    res = clean_rpm_signal(t_rpm, rpm_in, min_rpm=10.0, max_rpm=8000.0)
    cleaned = res["rpm"]
    finite = np.isfinite(cleaned)
    runner.check("clean_rpm_signal: NaNs/negatives/over-max removed",
                 finite.sum() >= cleaned.size - 5 and not (cleaned[finite] <= 10.0).any(),
                 f"valid={finite.sum()}/{cleaned.size}, min={np.nanmin(cleaned):.1f}")

    # 1d. clean_rpm_signal interpolates the NaN gap correctly
    err = np.nanmax(np.abs(cleaned[100:110] - rpm_clean_in[100:110]))
    runner.check("clean_rpm_signal: interpolates short NaN gap",
                 err < 5.0,
                 f"max abs interp error in NaN gap = {err:.2f} RPM")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2 — compute_shaft_phase: constant RPM → linear phase       (Stage 1)
# ═════════════════════════════════════════════════════════════════════════════

def test_shaft_phase_constant(runner: TestRunner, save: bool, outdir: str):
    print("\n[TEST 2] compute_shaft_phase: constant RPM → linear phase")

    fs = 5000.0
    T = 4.0
    rpm = 3000.0
    N = int(fs * T)
    t = np.arange(N) / fs
    res = compute_shaft_phase(t, np.full(N, rpm))
    theta = res["theta"]

    # Expected slope = 2π · 3000 / 60 = 100π rad/s
    slope = (theta[-1] - theta[0]) / (t[-1] - t[0])
    expected = 2.0 * np.pi * rpm / 60.0
    runner.check("phase slope matches 2π·RPM/60",
                 abs(slope - expected) / expected < 1e-9,
                 f"slope={slope:.6f}, expected={expected:.6f}")

    # Linearity residual: theta - slope·t should be ~0
    resid = theta - slope * t
    rel = float(np.max(np.abs(resid - resid[0]))) / max(theta[-1], 1e-30)
    runner.check("phase is linear (residual ~ 0)",
                 rel < 1e-9,
                 f"max relative residual = {rel:.2e}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3 — compute_shaft_phase: linear-rising RPM → quadratic phase
# ═════════════════════════════════════════════════════════════════════════════

def test_shaft_phase_runup(runner: TestRunner, save: bool, outdir: str):
    print("\n[TEST 3] compute_shaft_phase: linear-rising RPM → quadratic phase")

    fs = 4000.0
    T = 4.0
    rpm0, rpm1 = 1500.0, 4500.0
    N = int(fs * T)
    t = np.arange(N) / fs
    rpm = np.linspace(rpm0, rpm1, N)
    res = compute_shaft_phase(t, rpm)
    theta = res["theta"]

    # Analytic: rpm(t) = a + b·t,  ω(t) = 2π·(a+b·t)/60
    # θ(t) = 2π·(a·t + b·t²/2)/60
    # NB: np.linspace(rpm0, rpm1, N) with t = arange(N)/fs gives t[-1] = (N-1)/fs ≠ T,
    # so the *actual* RPM slope is (rpm[-1]-rpm[0])/(t[-1]-t[0]), not (rpm1-rpm0)/T.
    a = float(rpm[0])
    b = (float(rpm[-1]) - float(rpm[0])) / (float(t[-1]) - float(t[0]))
    theta_true = 2.0 * np.pi * (a * t + 0.5 * b * t * t) / 60.0

    rel = float(np.max(np.abs(theta - theta_true))) / max(np.max(theta_true), 1e-30)
    runner.check("phase quadratic fit residual",
                 rel < 1e-6,
                 f"max relative error vs analytic = {rel:.2e}")

    # Negative RPM should raise
    try:
        compute_shaft_phase(t, -np.abs(rpm))
    except ValueError:
        raised = True
    else:
        raised = False
    runner.check("compute_shaft_phase: rejects negative RPM", raised)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4 — Amplitude normalization on bin-centred 10 g sine        (Stage 2)
# ═════════════════════════════════════════════════════════════════════════════

def test_amplitude_bin_centred(runner: TestRunner, save: bool, outdir: str):
    print("\n[TEST 4] Amplitude normalization on bin-centred 10 g sine")

    fs = 2000.0
    nperseg = 2000
    A = 10.0
    # Place sine exactly on bin k=200 → f0 = 200 Hz (since fs/nperseg = 1 Hz)
    k = 200
    f0 = k * fs / nperseg
    n = np.arange(nperseg)
    x = A * np.cos(2 * np.pi * f0 * n / fs)

    win = get_window_and_corrections("hann", nperseg, fs)

    # Peak mode
    f, S_peak, _ = single_sided_spectrum(x, fs, win["w"], nperseg, amplitude_mode="peak")
    bin_peak = float(S_peak[k])
    runner.check("peak mode: 10 g sine recovered",
                 abs(bin_peak - A) / A < 0.01,
                 f"peak bin = {bin_peak:.4f} g, expected {A:.4f} g (err {abs(bin_peak-A)/A:.2%})")

    # RMS mode
    _, S_rms, _ = single_sided_spectrum(x, fs, win["w"], nperseg, amplitude_mode="rms")
    bin_rms = float(S_rms[k])
    expected_rms = A / np.sqrt(2.0)
    runner.check("rms mode: 10 g sine → 7.071 g RMS",
                 abs(bin_rms - expected_rms) / expected_rms < 0.01,
                 f"rms bin = {bin_rms:.4f} g, expected {expected_rms:.4f} g")

    # PSD integration consistency: ∫ PSD df ≈ rms² = A²/2
    _, S_psd, _ = single_sided_spectrum(x, fs, win["w"], nperseg, amplitude_mode="psd")
    df = f[1] - f[0]
    integrated = float(np.sum(S_psd) * df)
    expected_ms = (A * A) / 2.0
    runner.check("PSD integration ≈ mean square",
                 abs(integrated - expected_ms) / expected_ms < 0.05,
                 f"∫PSD df = {integrated:.4f}, expected {expected_ms:.4f} g²")

    # convert_amplitude consistency
    rms_from_peak = float(convert_amplitude(np.array([bin_peak]), "peak", "rms")[0])
    runner.check("convert_amplitude: peak → rms identity",
                 abs(rms_from_peak - bin_rms) / bin_rms < 1e-9,
                 f"converted rms = {rms_from_peak:.6f}, direct rms = {bin_rms:.6f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5 — Off-bin sine: total power preserved                     (Stage 2)
# ═════════════════════════════════════════════════════════════════════════════

def test_amplitude_off_bin(runner: TestRunner, save: bool, outdir: str):
    print("\n[TEST 5] Off-bin sine — total integrated power within 5%")

    fs = 2000.0
    nperseg = 2000
    A = 10.0
    f0 = 200.5  # exactly between bins 200 and 201
    n = np.arange(nperseg)
    x = A * np.cos(2 * np.pi * f0 * n / fs)

    win = get_window_and_corrections("hann", nperseg, fs)
    f, S_psd, _ = single_sided_spectrum(x, fs, win["w"], nperseg, amplitude_mode="psd")
    df = f[1] - f[0]
    integrated = float(np.sum(S_psd) * df)
    expected_ms = (A * A) / 2.0
    runner.check("off-bin: PSD integral within 5% of A²/2",
                 abs(integrated - expected_ms) / expected_ms < 0.05,
                 f"∫PSD df = {integrated:.4f}, expected {expected_ms:.4f} g²")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6 — White-noise PSD flatness                                (Stage 2)
# ═════════════════════════════════════════════════════════════════════════════

def test_white_noise_psd_flat(runner: TestRunner, save: bool, outdir: str):
    print("\n[TEST 6] White-noise PSD flatness across spectrum")

    fs = 4000.0
    T = 8.0
    N = int(fs * T)
    sigma = 1.0
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, sigma, size=N)

    nperseg = 4096
    stft = rmc.compute_stft_map(x, fs, nperseg=nperseg, noverlap=nperseg // 2,
                                 window="hann", amplitude_mode="psd")
    # Welch-style mean PSD (averaging in linear PSD already approximately
    # equivalent to Welch since each frame is independent for white noise)
    psd_mean = stft["S"].mean(axis=0)
    f = stft["freq"]

    # Discard DC and a small low-frequency band
    band = (f > 50.0) & (f < 0.45 * fs)
    psd_band = psd_mean[band]
    # Theoretical white-noise one-sided PSD = 2 σ² / fs
    psd_th = 2.0 * sigma * sigma / fs
    rel_err = abs(np.median(psd_band) - psd_th) / psd_th
    flat_dispersion = float(np.std(np.log10(np.maximum(psd_band, 1e-30))))

    runner.check("white-noise PSD level within 10% of theory",
                 rel_err < 0.10,
                 f"median PSD = {np.median(psd_band):.4e}, theory = {psd_th:.4e} (err {rel_err:.2%})")
    runner.check("white-noise PSD log-dispersion bounded",
                 flat_dispersion < 0.20,
                 f"std(log10(PSD)) = {flat_dispersion:.3f} (threshold 0.20)")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 7 — Constant-RPM 1X via compute_rpm_frequency_map           (Stage 3)
# ═════════════════════════════════════════════════════════════════════════════

def test_constant_rpm_freq_map(runner: TestRunner, save: bool, outdir: str):
    print("\n[TEST 7] Constant-RPM 1X via compute_rpm_frequency_map")

    fs = 4000.0
    T = 4.0
    rpm = 3000.0
    A_peak = 5.0
    t, x, rpm_vec = make_constant_rpm_signal(fs, T, rpm, harmonics={1.0: A_peak})

    res = compute_rpm_frequency_map(x, fs, t, rpm_vec,
                                     nperseg=4000, noverlap=2000,
                                     amplitude_mode="rms")
    f = res["freq"]
    # Use the time-ordered raw STFT so RPM column is well-defined
    raw = res["raw_S"]
    mean_S = raw.mean(axis=0)
    j = int(np.argmax(mean_S))
    f_peak = float(f[j])
    f_expected = rpm / 60.0  # = 50 Hz

    runner.check("constant RPM: peak frequency ≈ rpm/60",
                 abs(f_peak - f_expected) / f_expected < 0.02,
                 f"f_peak = {f_peak:.3f} Hz, expected {f_expected:.3f} Hz")

    # Amplitude at the peak (RMS) should be A/√2 within ~5%
    expected_rms = A_peak / np.sqrt(2.0)
    runner.check("constant RPM: amplitude within 5%",
                 abs(mean_S[j] - expected_rms) / expected_rms < 0.05,
                 f"|S| = {mean_S[j]:.4f} g RMS, expected {expected_rms:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 8 — Run-up 1X via compute_rpm_frequency_map (diagonal ridge) (Stage 3)
# ═════════════════════════════════════════════════════════════════════════════

def test_runup_freq_map(runner: TestRunner, save: bool, outdir: str):
    print("\n[TEST 8] Run-up 1X frequency map → diagonal ridge")

    fs = 4000.0
    T = 6.0
    rpm0, rpm1 = 1200.0, 4800.0
    t, x, rpm_vec = make_runup_signal(fs, T, rpm0, rpm1, harmonics={1.0: 1.0})

    res = compute_rpm_frequency_map(x, fs, t, rpm_vec,
                                     nperseg=2048, noverlap=1536,
                                     amplitude_mode="rms")
    f = res["freq"]
    M = res["map"]
    rpm_axis = res["rpm"]

    # Per-row dominant frequency
    ridge_f = _ridge_freq(f, M)
    expected_f = rpm_axis / 60.0
    valid = (expected_f > f[1])
    rel = np.abs(ridge_f[valid] - expected_f[valid]) / expected_f[valid]
    mean_rel = float(np.mean(rel))
    runner.check("run-up freq-map ridge follows rpm/60 (mean rel < 8%)",
                 mean_rel < 0.08,
                 f"mean relative ridge error = {mean_rel:.2%}")

    # Slope of f(rpm) should be ~ 1/60
    if rpm_axis.size >= 2:
        slope = float(np.polyfit(rpm_axis, ridge_f, 1)[0])
    else:
        slope = float("nan")
    runner.check("run-up freq-map ridge slope ≈ 1/60",
                 abs(slope - 1.0 / 60.0) / (1.0 / 60.0) < 0.10,
                 f"slope = {slope:.6f}, expected {1.0/60.0:.6f}")

    if save:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
        im = ax.pcolormesh(f, rpm_axis, M, shading="auto", cmap="viridis")
        ax.plot(expected_f, rpm_axis, "r--", linewidth=1, label="true rpm/60")
        ax.set_xlim(0, 1.2 * float(rpm1) / 60.0)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("RPM")
        ax.set_title("Test 8: Run-up frequency map (1X) — DIAGONAL ridge expected")
        ax.legend()
        fig.colorbar(im, ax=ax, label="g RMS")
        _save_or_close(fig, os.path.join(outdir, "rm_test08_runup_freq_map.png"), save)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 9 — Run-up 1X via compute_rpm_order_map (horizontal ridge)   (Stage 4)
# ═════════════════════════════════════════════════════════════════════════════

def test_runup_order_map_1x(runner: TestRunner, save: bool, outdir: str):
    print("\n[TEST 9] Run-up 1X order map → horizontal ridge at order 1")

    fs = 4000.0
    T = 6.0
    rpm0, rpm1 = 1200.0, 4800.0
    t, x, rpm_vec = make_runup_signal(fs, T, rpm0, rpm1, harmonics={1.0: 1.0})

    res = compute_rpm_order_map(x, t, rpm_vec,
                                 samples_per_rev=512,
                                 revs_per_segment=16,
                                 overlap_revs=12,
                                 max_order=8.0,
                                 amplitude_mode="rms")
    o = res["order"]
    M = res["map"]
    rpm_axis = res["rpm"]

    ridge_o = _ridge_freq(o, M)
    err = float(np.mean(np.abs(ridge_o - 1.0)))
    runner.check("run-up order-map ridge ≈ order 1 (mean dev < 0.05)",
                 err < 0.05,
                 f"mean |order - 1| = {err:.4f}")

    # Slope of order vs rpm should be ~ 0
    if rpm_axis.size >= 2:
        slope = float(np.polyfit(rpm_axis, ridge_o, 1)[0])
    else:
        slope = float("nan")
    runner.check("run-up order-map ridge slope ≈ 0",
                 abs(slope) < 5e-5,
                 f"slope = {slope:.2e} order/RPM")

    if save:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
        im = ax.pcolormesh(o, rpm_axis, M, shading="auto", cmap="viridis")
        ax.axvline(1.0, color="red", linestyle="--", linewidth=1, label="true order = 1")
        ax.set_xlabel("Order")
        ax.set_ylabel("RPM")
        ax.set_title("Test 9: Run-up order map (1X) — HORIZONTAL ridge expected")
        ax.legend()
        fig.colorbar(im, ax=ax, label="g RMS")
        _save_or_close(fig, os.path.join(outdir, "rm_test09_runup_order_map.png"), save)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 10 — Run-up 2X via compute_rpm_order_map                    (Stage 4)
# ═════════════════════════════════════════════════════════════════════════════

def test_runup_order_map_2x(runner: TestRunner, save: bool, outdir: str):
    print("\n[TEST 10] Run-up 2X order map → ridge at order 2")

    fs = 4000.0
    T = 6.0
    rpm0, rpm1 = 1200.0, 4800.0
    t, x, rpm_vec = make_runup_signal(fs, T, rpm0, rpm1, harmonics={2.0: 1.0})

    res = compute_rpm_order_map(x, t, rpm_vec,
                                 samples_per_rev=512,
                                 revs_per_segment=16,
                                 overlap_revs=12,
                                 max_order=8.0,
                                 amplitude_mode="rms")
    o = res["order"]
    M = res["map"]
    rpm_axis = res["rpm"]

    ridge_o = _ridge_freq(o, M)
    err = float(np.mean(np.abs(ridge_o - 2.0)))
    runner.check("run-up order-map ridge ≈ order 2 (mean dev < 0.05)",
                 err < 0.05,
                 f"mean |order - 2| = {err:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 11 — Mixed 0.5X + 1X + 2X via extract_order_traces        (Stage 6 part)
# ═════════════════════════════════════════════════════════════════════════════

def test_mixed_order_traces(runner: TestRunner, save: bool, outdir: str):
    print("\n[TEST 11] Mixed 0.5X + 1X + 2X — order trace extraction")

    fs = 6000.0
    T = 6.0
    rpm0, rpm1 = 1500.0, 4500.0
    amps = {0.5: 0.4, 1.0: 1.0, 2.0: 0.3}
    t, x, rpm_vec = make_runup_signal(fs, T, rpm0, rpm1, harmonics=amps)

    res = compute_rpm_order_map(x, t, rpm_vec,
                                 samples_per_rev=1024,
                                 revs_per_segment=24,
                                 overlap_revs=18,
                                 max_order=4.0,
                                 amplitude_mode="rms")

    # Use "peak_in_band" for tonal extraction — for bin-centred tones with a
    # windowed STFT, the central-bin RMS reads back the true RMS exactly.
    # ("rms_band_power" sums rms² across the main lobe and over-counts by the
    # window's ENBW; that mode is meant for density-mode inputs like PSD.)
    traces = extract_order_traces(res["order"], res["rpm"], res["map"],
                                   target_orders=(0.5, 1.0, 2.0),
                                   method="peak_in_band",
                                   bandwidth=0.10,
                                   amplitude_mode="rms")

    expected_rms = {k: v / np.sqrt(2.0) for k, v in amps.items()}
    medians = {k: float(np.median(traces[k]["amplitude"])) for k in (0.5, 1.0, 2.0)}

    for order_k, exp in expected_rms.items():
        got = medians[order_k]
        rel = abs(got - exp) / exp
        runner.check(f"trace at order {order_k}: amplitude within 15%",
                     rel < 0.15,
                     f"got {got:.4f} g RMS, expected {exp:.4f} (err {rel:.2%})")

    if save:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        for k, color in [(0.5, "tab:purple"), (1.0, "tab:blue"), (2.0, "tab:orange")]:
            ax.plot(traces[k]["rpm"], traces[k]["amplitude"], "-", color=color,
                    label=f"order {k} (expected {expected_rms[k]:.3f} g RMS)")
            ax.axhline(expected_rms[k], color=color, linestyle=":", linewidth=0.8)
        ax.set_xlabel("RPM")
        ax.set_ylabel("Amplitude (g RMS)")
        ax.set_title("Test 11: order traces 0.5X / 1X / 2X")
        ax.legend()
        ax.grid(True)
        _save_or_close(fig, os.path.join(outdir, "rm_test11_order_traces.png"), save)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 12 — Frequency-map vs order-map ridge comparison              (Stage 4)
# ═════════════════════════════════════════════════════════════════════════════

def test_freq_vs_order_ridge(runner: TestRunner, save: bool, outdir: str):
    print("\n[TEST 12] Frequency map ridge moves with RPM ; order map ridge stays put")

    fs = 4000.0
    T = 6.0
    rpm0, rpm1 = 1200.0, 4800.0
    t, x, rpm_vec = make_runup_signal(fs, T, rpm0, rpm1, harmonics={1.0: 1.0})

    fmap = compute_rpm_frequency_map(x, fs, t, rpm_vec,
                                      nperseg=2048, noverlap=1536,
                                      amplitude_mode="rms")
    omap = compute_rpm_order_map(x, t, rpm_vec,
                                  samples_per_rev=512,
                                  revs_per_segment=16,
                                  overlap_revs=12,
                                  max_order=4.0,
                                  amplitude_mode="rms")

    f_ridge = _ridge_freq(fmap["freq"], fmap["map"])
    o_ridge = _ridge_freq(omap["order"], omap["map"])
    slope_f = float(np.polyfit(fmap["rpm"], f_ridge, 1)[0])
    slope_o = float(np.polyfit(omap["rpm"], o_ridge, 1)[0])

    runner.check("freq-map ridge has non-zero slope",
                 abs(slope_f - 1.0 / 60.0) / (1.0 / 60.0) < 0.10,
                 f"slope = {slope_f:.6f} Hz/RPM (≈ 1/60 = {1.0/60.0:.6f})")
    runner.check("order-map ridge has near-zero slope",
                 abs(slope_o) < 5e-5,
                 f"slope = {slope_o:.2e} order/RPM")

    if save:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        im0 = axes[0].pcolormesh(fmap["freq"], fmap["rpm"], fmap["map"],
                                  shading="auto", cmap="viridis")
        axes[0].set_xlim(0, 1.2 * float(rpm1) / 60.0)
        axes[0].set_xlabel("Frequency (Hz)")
        axes[0].set_ylabel("RPM")
        axes[0].set_title("Frequency map — diagonal ridge (1X = rpm/60)")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].pcolormesh(omap["order"], omap["rpm"], omap["map"],
                                  shading="auto", cmap="viridis")
        axes[1].axvline(1.0, color="red", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Order")
        axes[1].set_ylabel("RPM")
        axes[1].set_title("Order map — horizontal ridge at order 1")
        fig.colorbar(im1, ax=axes[1])
        _save_or_close(fig, os.path.join(outdir, "rm_test12_freq_vs_order.png"), save)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rotating-machinery validation suite (patch 1)."
    )
    parser.add_argument("--no-show", action="store_true",
                        help="Save figures but do not display them.")
    parser.add_argument("--outdir", type=str, default=".",
                        help="Directory for PNG outputs.")
    args = parser.parse_args()

    if args.no_show:
        matplotlib.use("Agg")
    save = True
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 70)
    print("  Rotating-Machinery Validation Suite — patch 1")
    print("  (foundation: RPM, phase, amplitude, freq map, order map, traces)")
    print("=" * 70)

    runner = TestRunner()

    test_timebase_and_rpm_cleanup(runner, save, args.outdir)
    test_shaft_phase_constant(runner, save, args.outdir)
    test_shaft_phase_runup(runner, save, args.outdir)
    test_amplitude_bin_centred(runner, save, args.outdir)
    test_amplitude_off_bin(runner, save, args.outdir)
    test_white_noise_psd_flat(runner, save, args.outdir)
    test_constant_rpm_freq_map(runner, save, args.outdir)
    test_runup_freq_map(runner, save, args.outdir)
    test_runup_order_map_1x(runner, save, args.outdir)
    test_runup_order_map_2x(runner, save, args.outdir)
    test_mixed_order_traces(runner, save, args.outdir)
    test_freq_vs_order_ridge(runner, save, args.outdir)

    all_passed = runner.summary()

    print("\nGenerated figures in", os.path.abspath(args.outdir) + ":")
    for name in ("rm_test08_runup_freq_map", "rm_test09_runup_order_map",
                 "rm_test11_order_traces", "rm_test12_freq_vs_order"):
        path = os.path.join(args.outdir, f"{name}.png")
        if os.path.exists(path):
            print(f"  {path}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
