"""Headless DSP regression tests for Son.py.

Run with:
    python3 trdsp_regression_tests.py
or:
    python3 Son.py --self-test
"""
import math
import os
import numpy as np

import Son


def _assert_close(name, value, expected, rel=0.05, abs_tol=1e-12):
    if not (abs(value - expected) <= max(abs_tol, rel * abs(expected))):
        raise AssertionError(f"{name}: got {value}, expected {expected}")


def run_regression_tests(verbose=True):
    passed = 0; failed = 0; skipped = 0
    details = []

    def run(name, fn):
        nonlocal passed, failed, skipped
        try:
            r = fn()
            if r == "skip":
                skipped += 1; details.append(("SKIP", name, "optional reference file not present"))
            else:
                passed += 1; details.append(("PASS", name, ""))
        except Exception as e:
            failed += 1; details.append(("FAIL", name, str(e)))

    def t_on_bin_sine():
        fs = 100.0; dur = 10.0; n = int(fs * dur); tt = np.arange(n) / fs
        x = 10.0 * np.sin(2*np.pi*1.0*tt)
        f, a = Son.compute_one_sided_amp_spectrum_from_segment(x, fs, n, "Peak", "acc", "Amplitude")
        _, ar = Son.compute_one_sided_amp_spectrum_from_segment(x, fs, n, "RMS", "acc", "Amplitude")
        _, p = Son.compute_one_sided_amp_spectrum_from_segment(x, fs, n, "Peak", "acc", "Power")
        k = int(np.argmin(np.abs(f - 1.0)))
        _assert_close("peak amplitude", float(a[k]), 10.0, rel=0.02)
        _assert_close("rms amplitude", float(ar[k]), 10.0/math.sqrt(2), rel=0.02)
        _assert_close("line power", float(p[k]), 50.0, rel=0.02)

    def t_off_bin_leakage():
        fs = 100.0; n = 1000; tt = np.arange(n)/fs
        x = 10*np.sin(2*np.pi*1.35*tt)
        _, a = Son.compute_one_sided_amp_spectrum_from_segment(x, fs, n, "Peak", "acc", "Amplitude")
        if float(np.max(a)) >= 9.99:
            raise AssertionError("off-bin Hann spectrum should show leakage/scalloping, not perfect amplitude")

    def t_two_tone():
        fs = 200.0; n = 2000; tt = np.arange(n)/fs
        x = 3*np.sin(2*np.pi*5*tt) + 1.5*np.sin(2*np.pi*17*tt)
        f, a = Son.compute_one_sided_amp_spectrum_from_segment(x, fs, n, "Peak", "acc", "Amplitude")
        _assert_close("5 Hz tone", float(a[np.argmin(abs(f-5))]), 3.0, rel=0.03)
        _assert_close("17 Hz tone", float(a[np.argmin(abs(f-17))]), 1.5, rel=0.03)

    def _runup(order=2.0, amp=10.0):
        fs = 1000.0; dur = 8.0; tt = np.arange(int(fs*dur))/fs
        rpm = 600.0 + 2400.0 * tt / tt[-1]
        theta = np.zeros_like(tt)
        omega = 2*np.pi*rpm/60.0
        theta[1:] = np.cumsum(0.5*(omega[:-1]+omega[1:])/fs)
        x = amp*np.sin(order*theta)
        return fs, tt, rpm, theta, x

    def t_cot_runup():
        fs, _, rpm, _, x = _runup(order=2.0, amp=10.0)
        orders, amp, _, meta = Son.compute_order_tracking_spectrum(x, rpm, fs, 256, 8, return_metadata=True)
        pk = float(orders[int(np.argmax(amp))])
        if abs(pk - 2.0) > 0.08:
            raise AssertionError(f"COT peak at {pk}, expected 2.0")
        if not (meta["rpm_min"] < meta["rpm_max"] and meta["total_revs"] > 0):
            raise AssertionError("COT metadata missing RPM/revolution fields")

    def t_noisy_rpm():
        fs, _, rpm, _, x = _runup()
        rpm = rpm.copy(); rpm[100:140] = np.nan; rpm[500] = 1e7; rpm[900:930] = 0
        orders, amp, _, meta = Son.compute_order_tracking_spectrum(x, rpm, fs, 256, 8, return_metadata=True)
        if orders.size == 0 or amp.size == 0 or not meta["rpm_interpolated"]:
            raise AssertionError("noisy RPM did not produce COT output/interpolation metadata")

    def t_missing_rpm_spans():
        fs, _, rpm, _, x = _runup()
        rpm[:50] = np.nan; rpm[-50:] = 0
        _, _, _, meta = Son.compute_order_tracking_spectrum(x, rpm, fs, 256, 8, return_metadata=True)
        if not meta["rpm_interpolated"]:
            raise AssertionError("missing RPM spans not reported")

    def t_psd_consistency():
        rng = np.random.default_rng(123)
        fs = 512.0; n = 4096; x = rng.normal(0, 2.0, n)
        f, psd = Son.compute_avg_spectrum_from_signal(x, fs, 1024, 50, "PSD", "Linear", "Peak", "acc")
        rms2_psd = float(np.trapz(psd, f))
        rms2_time = float(np.mean(x*x))
        _assert_close("integrated PSD", rms2_psd, rms2_time, rel=0.20)

    def t_velocity_integration_formula():
        fs = 1000.0; n = 5000; tt = np.arange(n)/fs; f0 = 25.0; ag = 1.0
        x = ag*np.sin(2*np.pi*f0*tt)
        cfg = {"fs": fs, "velocity_integration_method": "Frequency-domain integration", "velocity_highpass_hz": 1.0, "velocity_detrend": "Remove mean"}
        worker = Son.PlotWorker(cfg)
        v = worker._integrate_acc_to_velocity(x, fs)
        ff, aa = Son.compute_one_sided_amp_spectrum_from_segment(v, fs, n, "Peak", "vel", "Amplitude")
        expected = ag * Son.G0 / (2*np.pi*f0) * Son.VEL_SCALE
        _assert_close("velocity peak", float(aa[np.argmin(abs(ff-f0))]), expected, rel=0.03)

    def t_otf_modes():
        fs, _, rpm, theta, x1 = _runup(order=1.0, amp=2.0)
        x = x1 + 2.0*np.sin(2.4*theta)
        yf, meta = Son.apply_time_domain_order_tracking_filter(x, rpm, fs, 1.0, 0.2, "Order")
        if yf.size != x.size or meta.get("mode") != "Time-domain adaptive order filter":
            raise AssertionError("time-domain OTF metadata/shape invalid")
        freqs = np.linspace(0, fs/2, 513); frames = np.abs(np.random.default_rng(1).normal(size=(10, 513)))
        rpms = np.linspace(600, 2400, 10)
        _, trace = Son.apply_order_tracking_filter(freqs, rpms, frames, 1.0, 0.2, "Order")
        if trace.size != 10:
            raise AssertionError("frequency-domain order mask did not run")

    def t_external_refs():
        if not (os.path.exists("matlab_reference.csv") or os.path.exists("ncode_export.csv")):
            return "skip"

    tests = [
        ("on-bin sine amplitude/power", t_on_bin_sine),
        ("off-bin leakage", t_off_bin_leakage),
        ("two-tone peaks", t_two_tone),
        ("run-up COT", t_cot_runup),
        ("noisy RPM COT", t_noisy_rpm),
        ("missing RPM spans", t_missing_rpm_spans),
        ("PSD consistency", t_psd_consistency),
        ("velocity integration", t_velocity_integration_formula),
        ("frequency mask vs time OTF", t_otf_modes),
        ("optional external reference hooks", t_external_refs),
    ]
    for name, fn in tests:
        run(name, fn)
    if verbose:
        for status, name, msg in details:
            print(f"[{status}] {name}" + (f" - {msg}" if msg else ""))
        print(f"Summary: {passed} passed, {failed} failed, {skipped} skipped")
    return {"passed": passed, "failed": failed, "skipped": skipped, "details": details}


if __name__ == "__main__":
    summary = run_regression_tests(True)
    raise SystemExit(0 if summary["failed"] == 0 else 1)
