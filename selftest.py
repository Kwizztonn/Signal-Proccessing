"""
Self-test: writes small synthetic UNV files (ASCII 58 and binary 58b, real /
complex / uneven, single / double, LF / CRLF), converts them to TDMS and checks
that every sample and the main header fields come back unchanged.

Run:  python selftest.py      (exit code 0 = success)
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
from nptdms import TdmsFile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from converter import ConvertOptions, convert_file  # noqa: E402
from unv_reader import UnvFile  # noqa: E402


def _hdr151():
    return ["    -1", "   151", "Test model", "Synthetic data for UNV2TDMS selftest",
            "Simcenter Testlab", "16-Nov-26   10:28:00", "", "", "         0",
            "16-Nov-26   10:28:00", "Simcenter Testlab", "16-Nov-26   10:28:00", "    -1"]


def _hdr164():
    return ["    -1", "   164", "         1SI - mks (Newton)  2",
            "  1.00000000000000000E+00  1.00000000000000000E+00  1.00000000000000000E+00",
            "  2.73150000000000000E+02", "    -1"]


def _rec_lines(name, node, direc, otype, n, even, x0, dx, yunit, ylab="Acceleration",
               xunit="s", xlab="Time", xspec=17, yspec=12, ftype=1):
    return [
        f"{name:<80}"[:80].rstrip(),
        "NONE",
        "16-Nov-26 10:28:00",
        "Run 1",
        "NONE",
        f"{ftype:5d}{0:10d}{1:5d}{0:10d} {'Point':<10s}{node:10d}{direc:4d} {'NONE':<10s}{0:10d}{0:4d}",
        f"{otype:10d}{n:10d}{1 if even else 0:10d}{x0:13.5E}{dx:13.5E}{0.0:13.5E}",
        f"{xspec:10d}{0:5d}{0:5d}{0:5d} {xlab:<20s} {xunit:<20s}",
        f"{yspec:10d}{1:5d}{0:5d}{0:5d} {ylab:<20s} {yunit:<20s}",
        f"{0:10d}{0:5d}{0:5d}{0:5d} {'NONE':<20s} {'NONE':<20s}",
        f"{0:10d}{0:5d}{0:5d}{0:5d} {'NONE':<20s} {'NONE':<20s}",
    ]


def write_58_ascii(f, eol, name, node, direc, y, x=None, double=False, fs=52000.0):
    cplx = np.iscomplexobj(y)
    otype = (6 if double else 5) if cplx else (4 if double else 2)
    even = x is None
    lines = ["    -1", "    58"] + _rec_lines(name, node, direc, otype, len(y), even, 0.0, 1 / fs, "g")
    vals = []
    if even:
        flat = np.column_stack([y.real, y.imag]).ravel() if cplx else y
        fmt, per = ("{:20.12E}", 4) if double else ("{:13.5E}", 6)
        for i in range(0, len(flat), per):
            lines.append("".join(fmt.format(v) for v in flat[i:i + per]))
    else:
        for xi, yi in zip(x, y):
            if cplx:
                vals.append(f"{xi:13.5E}{yi.real:20.12E}{yi.imag:20.12E}" if double
                            else f"{xi:13.5E}{yi.real:13.5E}{yi.imag:13.5E}")
            else:
                vals.append(f"{xi:13.5E}{yi:20.12E}" if double else f"{xi:13.5E}{yi:13.5E}")
        lines += vals
    lines.append("    -1")
    f.write((eol.join(lines) + eol).encode("latin-1"))


def write_58_binary(f, eol, name, node, direc, y, x=None, double=False, fs=52000.0,
                    big_endian=False):
    cplx = np.iscomplexobj(y)
    otype = (6 if double else 5) if cplx else (4 if double else 2)
    even = x is None
    dt = np.dtype(np.float64 if double else np.float32).newbyteorder(">" if big_endian else "<")
    if even:
        flat = np.column_stack([y.real, y.imag]).ravel() if cplx else y
    else:
        cols = [x, y.real, y.imag] if cplx else [x, y]
        flat = np.column_stack(cols).ravel()
    data = flat.astype(dt).tobytes()
    head = (f"    58b{2 if big_endian else 1:6d}{2:6d}{11:12d}{len(data):12d}"
            f"{0:6d}{0:6d}{0:12d}{0:12d}")
    lines = ["    -1", head] + _rec_lines(name, node, direc, otype, len(y), even, 0.0, 1 / fs, "g")
    f.write((eol.join(lines) + eol).encode("latin-1"))
    f.write(data)
    f.write((eol + "    -1" + eol).encode("latin-1"))


def make_test_file(path, binary, eol="\n", n=30011):
    rng = np.random.default_rng(1 if binary else 2)
    t = np.arange(n) / 52000.0
    chans = {
        "Point1:+X": (rng.standard_normal(n) * 3.7 + np.sin(2 * np.pi * 120 * t)).astype(np.float32),
        "Point1:+Y": (rng.standard_normal(n) * 1e-3).astype(np.float32),
        "Mic 1 / ön": (rng.standard_normal(n) * 0.25),                  # double
        "FRF 1": (rng.standard_normal(500) + 1j * rng.standard_normal(500)).astype(np.complex64),
    }
    xs_uneven = np.cumsum(rng.uniform(0.5, 1.5, 300)).astype(np.float64)
    y_uneven = rng.standard_normal(300)
    # ASCII single precision only keeps 6 significant digits -> expected values are the
    # rounded text values
    expected = {}
    with open(path, "wb") as f:
        f.write(("\n".join(_hdr151()) + "\n").encode("latin-1").replace(b"\n", eol.encode()))
        f.write(("\n".join(_hdr164()) + "\n").encode("latin-1").replace(b"\n", eol.encode()))
        w = write_58_binary if binary else write_58_ascii
        for i, (nm, y) in enumerate(chans.items()):
            double = y.dtype in (np.float64, np.complex128)
            w(f, eol, nm, 1 + i, 1 + i % 3, y, double=double)
            if binary:
                expected[nm] = y
            else:
                if np.iscomplexobj(y):
                    fmt = "{:20.12E}" if double else "{:13.5E}"
                    re_ = np.array([float(fmt.format(v)) for v in y.real])
                    im_ = np.array([float(fmt.format(v)) for v in y.imag])
                    expected[nm] = re_ + 1j * im_
                else:
                    fmt = "{:20.12E}" if double else "{:13.5E}"
                    expected[nm] = np.array([float(fmt.format(v)) for v in y])
        w(f, eol, "Uneven", 9, 3, y_uneven, x=xs_uneven, double=True)
        if binary:
            expected["Uneven"] = y_uneven
            expected["Uneven_X"] = xs_uneven
        else:
            expected["Uneven"] = np.array([float(f"{v:20.12E}") for v in y_uneven])
            expected["Uneven_X"] = np.array([float(f"{v:13.5E}") for v in xs_uneven])
    return expected


def check(path, expected, out_dir):
    res = convert_file(path, ConvertOptions(out_dir=out_dir, overwrite=True, verify=True),
                       log=lambda s: None)
    tf = TdmsFile.read(res["output"])
    g = tf.groups()[0]
    names = {c.name for c in g.channels()}
    for nm, exp in expected.items():
        safe = nm.replace("/", "_")
        if np.iscomplexobj(exp):
            re_ = g[safe + "_Re"][:].astype(np.float64)
            im_ = g[safe + "_Im"][:].astype(np.float64)
            got = re_ + 1j * im_
        else:
            assert safe in names, f"{safe} kanalı yok ({names})"
            got = g[safe][:].astype(np.float64)
        exp64 = exp.astype(np.complex128 if np.iscomplexobj(exp) else np.float64)
        assert got.shape == exp64.shape, (nm, got.shape, exp64.shape)
        if np.iscomplexobj(exp) and exp.dtype == np.complex64 or exp.dtype == np.float32:
            # binary single precision must be bit exact
            assert np.array_equal(got, exp64), f"{nm}: veri farklı"
        else:
            # ASCII: TDMS value must reproduce the text value at the written precision
            rel = np.max(np.abs(got - exp64) / np.maximum(np.abs(exp64), 1e-30))
            assert rel < 1e-6, f"{nm}: göreli hata {rel}"
            if exp64.dtype == np.float64 and g[safe if safe in names else safe + "_Re"].dtype == np.float64:
                assert np.array_equal(got, exp64), f"{nm}: float64 birebir değil"
    ch = g["Point1:+X"]
    p = ch.properties
    assert abs(1 / p["wf_increment"] - 52000.0) < 1e-6, p["wf_increment"]
    assert p["unit_string"] == "g"
    assert p["UNV_id1"] == "Point1:+X"
    assert "UNV_header_raw" in p
    assert "UNV151_model_name" in tf.properties
    return res


def main():
    with tempfile.TemporaryDirectory() as d:
        for binary in (False, True):
            for eol in ("\n", "\r\n"):
                p = os.path.join(d, f"test_{'bin' if binary else 'asc'}_{'crlf' if len(eol) == 2 else 'lf'}.unv")
                exp = make_test_file(p, binary, eol)
                with UnvFile(p) as uf:
                    assert len(uf.channels) == 5, len(uf.channels)
                check(p, exp, d)
                print(f"  OK  {'58b binary' if binary else '58 ASCII  '}  "
                      f"{'CRLF' if len(eol) == 2 else 'LF  '}  – 5 kanal, veri ve başlıklar doğrulandı")
    print("    Kendi kendini test BAŞARILI.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
