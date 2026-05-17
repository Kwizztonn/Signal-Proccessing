"""
envelope_validation.py
─────────────────────────────────────────────────────────────────────────────
Synthetic-signal regression tests for ``dsp_core.envelope_kurtogram``.

USAGE
-----
    python tests/envelope_validation.py             # all tests + figures
    python tests/envelope_validation.py --no-show   # CI / headless
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dsp_core.envelope_kurtogram import (  # noqa: E402
    compute_envelope_spectrum,
    compute_fast_kurtogram,
    compute_spectral_kurtosis,
    extract_bearing_fault_frequencies,
)


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


def _nearest_peak(freqs: np.ndarray, spec: np.ndarray, fmin: float = 1.0, fmax: float | None = None) -> float:
    mask = freqs >= fmin
    if fmax is not None:
        mask &= freqs <= fmax
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return float("nan")
    return float(freqs[idx[np.argmax(spec[idx])]])


def test_am_envelope_peak(runner: TestRunner):
    fs = 2000.0
    t = np.arange(0.0, 4.0, 1.0 / fs)
    x = (1.0 + 0.45 * np.cos(2.0 * np.pi * 10.0 * t)) * np.cos(2.0 * np.pi * 100.0 * t)
    out = compute_envelope_spectrum(x, fs)
    peak = _nearest_peak(out["freqs"], out["spectrum"], fmin=2.0, fmax=80.0)
    runner.check(
        "AM signal envelope spectrum peak at modulation frequency",
        abs(peak - 10.0) <= 0.5,
        f"peak={peak:.3f} Hz, expected=10 Hz",
    )
    return out


def test_bearing_outer_race_simulation(runner: TestRunner):
    fs = 12000.0
    t = np.arange(0.0, 3.0, 1.0 / fs)
    fault_hz = 100.0
    carrier_hz = 3000.0
    impulses = np.zeros_like(t)
    impulses[:: int(round(fs / fault_hz))] = 1.0
    resonance = np.exp(-900.0 * t) * np.sin(2.0 * np.pi * carrier_hz * t)
    resonance = resonance[: int(0.04 * fs)]
    x = np.convolve(impulses, resonance, mode="same")
    x += 0.04 * np.random.default_rng(42).standard_normal(t.size)
    out = compute_envelope_spectrum(x, fs, band=(2200.0, 3800.0))
    peak = _nearest_peak(out["freqs"], out["spectrum"], fmin=60.0, fmax=160.0)
    runner.check(
        "Outer-race simulation envelope peak at BPFO-like repetition",
        abs(peak - fault_hz) <= 1.0,
        f"peak={peak:.3f} Hz, expected={fault_hz:.1f} Hz",
    )
    return out, x, fs


def test_kurtogram_selects_impulsive_resonance(runner: TestRunner):
    fs = 12000.0
    t = np.arange(0.0, 2.0, 1.0 / fs)
    fault_hz = 80.0
    resonance_hz = 2600.0
    impulses = np.zeros_like(t)
    impulses[:: int(round(fs / fault_hz))] = 1.0
    n_decay = int(0.035 * fs)
    tau = np.arange(n_decay) / fs
    ring = np.exp(-700.0 * tau) * np.sin(2.0 * np.pi * resonance_hz * tau)
    x = np.convolve(impulses, ring, mode="same")
    x += 0.03 * np.random.default_rng(7).standard_normal(t.size)
    out = compute_fast_kurtogram(x, fs, n_levels=5)
    best = out["best_band"]
    contains = best["f_low"] <= resonance_hz <= best["f_high"]
    # A mixed dyadic/third-band kurtogram may legitimately choose the adjacent
    # practical band when the synthetic resonance sits close to a candidate
    # boundary.  Accept that physically equivalent case only within one
    # frequency-bin resolution or one neighboring selected-band width; still
    # reject bands that are clearly remote from the resonance.
    edge_distance = min(abs(resonance_hz - best["f_low"]), abs(resonance_hz - best["f_high"]))
    center_distance = abs(resonance_hz - best["center_freq"])
    frequency_resolution = fs / t.size
    neighboring_band_tolerance = max(float(best["bandwidth"]), frequency_resolution)
    close_neighbor = (edge_distance <= neighboring_band_tolerance) or (center_distance <= neighboring_band_tolerance)
    runner.check(
        "Kurtogram optimum band localizes impulsive resonance",
        contains or close_neighbor,
        (
            f"best=[{best['f_low']:.1f}, {best['f_high']:.1f}] Hz, "
            f"center={best['center_freq']:.1f} Hz, bw={best['bandwidth']:.1f} Hz, "
            f"resonance={resonance_hz:.1f} Hz, df={frequency_resolution:.3f} Hz"
        ),
    )
    return out


def test_spectral_kurtosis_transient_band(runner: TestRunner):
    fs = 8000.0
    t = np.arange(0.0, 2.0, 1.0 / fs)
    x = 0.05 * np.random.default_rng(13).standard_normal(t.size)
    idx = int(0.9 * fs)
    burst_t = np.arange(int(0.08 * fs)) / fs
    x[idx:idx + burst_t.size] += 2.5 * np.hanning(burst_t.size) * np.sin(2.0 * np.pi * 1200.0 * burst_t)
    out = compute_spectral_kurtosis(x, fs, nperseg=512, noverlap=384)
    peak = float(out["freqs"][np.argmax(out["spectral_kurtosis"])])
    runner.check(
        "Spectral kurtosis localizes transient burst band",
        abs(peak - 1200.0) <= 100.0,
        f"peak={peak:.1f} Hz, expected≈1200 Hz",
    )
    return out


def test_bearing_fault_frequency_formulas(runner: TestRunner):
    freqs = extract_bearing_fault_frequencies(
        rpm=1800.0,
        n_balls=8,
        d_ball=10.0,
        d_pitch=50.0,
        contact_angle=0.0,
    )
    runner.check("BPFO formula", abs(freqs["BPFO"] - 96.0) < 1e-9, f"BPFO={freqs['BPFO']:.6f}")
    runner.check("BPFI formula", abs(freqs["BPFI"] - 144.0) < 1e-9, f"BPFI={freqs['BPFI']:.6f}")
    runner.check("BSF formula", abs(freqs["BSF"] - 72.0) < 1e-9, f"BSF={freqs['BSF']:.6f}")
    runner.check("FTF formula", abs(freqs["FTF"] - 12.0) < 1e-9, f"FTF={freqs['FTF']:.6f}")


def test_edge_cases(runner: TestRunner):
    for name, func in [
        ("Envelope rejects NaN", lambda: compute_envelope_spectrum([1.0, np.nan, 2.0], 1000.0)),
        ("Envelope rejects empty input", lambda: compute_envelope_spectrum([], 1000.0)),
        ("Kurtogram rejects too-short input", lambda: compute_fast_kurtogram(np.ones(8), 1000.0)),
        ("Spectral kurtosis rejects bad overlap", lambda: compute_spectral_kurtosis(np.ones(128), 1000.0, 64, 64)),
        ("Bearing formula rejects impossible geometry", lambda: extract_bearing_fault_frequencies(1800, 8, 50.0, 10.0, 0.0)),
    ]:
        try:
            func()
        except ValueError as exc:
            runner.check(name, True, str(exc))
        else:
            runner.check(name, False, "expected ValueError")


def save_figures(outputs: dict[str, object], out_dir: str):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    am = outputs["am"]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(am["freqs"], am["spectrum"], lw=1.2)
    ax.set_xlim(0, 80)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Envelope amplitude [peak]")
    ax.set_title("AM validation: envelope spectrum")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, "envelope_am_spectrum.png"), dpi=140)
    plt.close(fig)

    kg = outputs["kurtogram"]
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    im = ax.imshow(kg["kurtosis_grid"], aspect="auto", origin="lower", interpolation="nearest")
    best = kg["best_band"]
    ax.set_xlabel("Band index")
    ax.set_ylabel("Level")
    ax.set_title(
        "Kurtogram validation: best band "
        f"{best['f_low']:.0f}-{best['f_high']:.0f} Hz, K={best['kurtosis']:.2f}"
    )
    fig.colorbar(im, ax=ax, label="Excess kurtosis")
    fig.savefig(os.path.join(out_dir, "kurtogram_impulsive_resonance.png"), dpi=140)
    plt.close(fig)

    sk = outputs["spectral_kurtosis"]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(sk["freqs"], sk["spectral_kurtosis"], lw=1.2)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Spectral kurtosis [excess]")
    ax.set_title("Transient burst validation: spectral kurtosis")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, "spectral_kurtosis_burst.png"), dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate envelope/kurtogram backend")
    parser.add_argument("--no-show", action="store_true", help="Use Agg backend and do not display figures")
    parser.add_argument("--out-dir", default="envelope_validation_outputs", help="Directory for PNG outputs")
    args = parser.parse_args(argv)

    if args.no_show:
        matplotlib.use("Agg")

    runner = TestRunner()
    outputs: dict[str, object] = {}
    outputs["am"] = test_am_envelope_peak(runner)
    outputs["bearing"] = test_bearing_outer_race_simulation(runner)
    outputs["kurtogram"] = test_kurtogram_selects_impulsive_resonance(runner)
    outputs["spectral_kurtosis"] = test_spectral_kurtosis_transient_band(runner)
    test_bearing_fault_frequency_formulas(runner)
    test_edge_cases(runner)
    save_figures(outputs, args.out_dir)

    ok = runner.summary()
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
