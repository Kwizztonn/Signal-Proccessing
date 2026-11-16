"""
UNV (dataset 58/58b) -> TDMS conversion with lossless verification.

* Data are copied in their native precision (float32 stays float32,
  float64 stays float64) – no rescaling, no resampling.
* Every header field of every dataset is stored as a TDMS property
  (prefix "UNV_"), plus the raw 11 header lines, so nothing is lost.
* NI waveform properties (wf_increment, wf_start_offset, wf_samples,
  unit_string, ...) are written so that nCode / DIAdem / LabVIEW
  recognise the time axis and sample rate.
* Optional verification re-reads the written TDMS file and compares a
  BLAKE2 hash of every channel with the hash of the source data, plus the
  sample counts.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from nptdms import ChannelObject, GroupObject, RootObject, TdmsFile, TdmsWriter

from unv_reader import DIRECTIONS, FUNCTION_TYPES, SPEC_DATA_TYPES, UnvChannel, UnvError, UnvFile

APP_NAME = "UNV2TDMS"
APP_VERSION = "1.0.0"

NAMING_MODES = {
    "id1": "ID1 (Testlab kanal adı)",
    "id1_node": "ID1 + [Nokta:Yön]",
    "node_dir": "Nokta:Yön (ör. Point1:12:+X)",
    "index_id1": "Sıra no + ID1 (Ch001_...)",
}


@dataclass
class ConvertOptions:
    out_dir: Optional[str] = None       # None -> next to the source file
    group_name: str = ""                # "" -> source file name
    naming: str = "id1"
    verify: bool = True
    snap_fs: bool = True
    force_float64: bool = False
    overwrite: bool = False
    chunk_samples: int = 4_000_000


# --------------------------------------------------------------------------
def _sanitize(name: str) -> str:
    name = "".join(c if c.isprintable() else "_" for c in name).strip()
    name = name.replace("/", "_").replace("\\", "_")
    return name or "Kanal"


def channel_name(ch: UnvChannel, mode: str) -> str:
    if mode == "node_dir":
        return _sanitize(ch.node_dir_name)
    if mode == "id1_node":
        return _sanitize(f"{ch.default_name} [{ch.node_dir_name}]")
    if mode == "index_id1":
        return _sanitize(f"Ch{ch.index + 1:03d}_{ch.default_name}")
    return _sanitize(ch.default_name)


def _is_time_axis(ch: UnvChannel) -> bool:
    if ch.rec.get("abscissa_spec_data_type") == 17 or ch.func_type == 1:
        return True
    return (ch.x_unit or "").strip().lower() in ("s", "sec", "second", "seconds", "sn")


def effective_dx(ch: UnvChannel, snap: bool):
    """Return (dx, note). UNV writes dx with only ~6 digits (E13.5), e.g.
    1/52000 -> 1.92308E-05 -> 51999.92 Hz. If snapping is enabled and the
    axis is time, round the sample rate to the integer it obviously is."""
    dx = ch.dx
    if not (snap and ch.even and dx > 0 and _is_time_axis(ch)):
        return dx, ""
    fs = 1.0 / dx
    fs_r = round(fs)
    if fs_r >= 1:
        rel = abs(fs - fs_r) / fs
        if 1e-12 < rel < 5e-6:
            return 1.0 / fs_r, f"Δt düzeltildi: fs {fs:.4f} Hz → {fs_r} Hz (UNV yazım hassasiyeti)"
    return dx, ""


def _prop(v):
    """Convert a value to something npTDMS can store as a property."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        v = int(v)
        return v if -2**31 <= v < 2**31 else np.int64(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    if isinstance(v, _dt.datetime):
        return np.datetime64(v.astimezone(_dt.timezone.utc).replace(tzinfo=None), "us")
    return str(v)


def build_channel_properties(ch: UnvChannel, name: str, dx: float, dx_note: str,
                             out_dtype: np.dtype, part: str = "") -> dict:
    r = ch.rec
    p = {
        "NI_ChannelName": name,
        "description": r.get("id1", "").strip(),
        "unit_string": ch.y_unit,
        "NI_UnitDescription": ch.y_unit,
        "wf_xname": ch.x_label or ("Time" if _is_time_axis(ch) else "X"),
        "wf_xunit_string": ch.x_unit or ("s" if _is_time_axis(ch) else ""),
        "wf_start_offset": float(ch.x0),
        "wf_increment": float(dx),
        "wf_samples": _prop(ch.n),
    }
    if ch.start_time is not None:
        p["wf_start_time"] = _prop(ch.start_time)
    if not ch.even:
        # unevenly spaced: there is a separate X channel; increments are meaningless
        for k in ("wf_increment", "wf_start_offset"):
            p.pop(k, None)
    if part:
        p["UNV_complex_part"] = part
    p["UNV_format"] = ch.fmt
    p["UNV_dataset_index"] = ch.dataset_index + 1
    p["UNV_channel_index"] = ch.index + 1
    p["UNV_function_type_name"] = FUNCTION_TYPES.get(ch.func_type, "")
    p["UNV_ordinate_type_name"] = {2: "real single", 4: "real double",
                                   5: "complex single", 6: "complex double"}[ch.ord_type]
    p["UNV_rsp_dir_name"] = DIRECTIONS.get(r.get("rsp_dir", 0), "")
    p["UNV_ref_dir_name"] = DIRECTIONS.get(r.get("ref_dir", 0), "")
    for k in ("abscissa", "ordinate", "ordinate_den", "z_axis"):
        p[f"UNV_{k}_spec_data_type_name"] = SPEC_DATA_TYPES.get(r.get(f"{k}_spec_data_type"), "")
    for k, v in r.items():
        p[f"UNV_{k}"] = _prop(v)
    if dx != ch.dx:
        p["UNV_abscissa_inc_original"] = float(ch.dx)
        p["UNV_note"] = dx_note
    if ch.start_time is not None:
        p["UNV_start_time_text"] = ch.start_time.isoformat(sep=" ")
    p["UNV_source_precision"] = ch.src_dtype.name
    p["UNV_stored_precision"] = out_dtype.name
    p["UNV_header_raw"] = "\n".join(ch.header_lines)
    return p


def _unique(name: str, used: set) -> str:
    base, i = name, 2
    while name.lower() in used:
        name = f"{base}_{i}"
        i += 1
    used.add(name.lower())
    return name


def output_path(src: str, opts: ConvertOptions) -> str:
    out_dir = opts.out_dir or os.path.dirname(os.path.abspath(src))
    stem = os.path.splitext(os.path.basename(src))[0]
    out = os.path.join(out_dir, stem + ".tdms")
    if not opts.overwrite:
        i = 1
        while os.path.exists(out):
            out = os.path.join(out_dir, f"{stem}_{i}.tdms")
            i += 1
    return out


class Cancelled(Exception):
    pass


# --------------------------------------------------------------------------
def convert_file(src: str, opts: ConvertOptions,
                 log: Callable[[str], None] = print,
                 progress: Optional[Callable[[float, str], None]] = None,
                 cancel: Optional[threading.Event] = None,
                 channel_filter: Optional[set] = None) -> dict:
    """Convert one UNV file. Returns a result dict."""
    t0 = time.time()
    progress = progress or (lambda f, s: None)
    log(f"► {os.path.basename(src)}  ({os.path.getsize(src) / 1e6:,.1f} MB)")
    progress(0.0, "Dosya taranıyor…")
    with UnvFile(src, progress_cb=lambda f: progress(0.05 * f, "Dosya taranıyor…")) as uf:
        for w in uf.warnings:
            log(f"  UYARI: {w}")
        chans = [c for c in uf.channels if channel_filter is None or c.index in channel_filter]
        if not chans:
            raise UnvError("Dosyada dönüştürülecek dataset 58/58b kanalı bulunamadı.")
        log(f"  {len(uf.channels)} kanal bulundu ({len(chans)} dönüştürülecek), "
            f"{len(uf.other_datasets)} diğer dataset.")

        out = output_path(src, opts)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".part"
        group = _sanitize(opts.group_name.strip() or os.path.splitext(os.path.basename(src))[0])

        root_props = {
            "name": os.path.splitext(os.path.basename(src))[0],
            "description": f"{os.path.basename(src)} dosyasından dönüştürüldü",
            "UNV_source_file": os.path.abspath(src),
            "UNV_source_size_bytes": _prop(os.path.getsize(src)),
            "UNV_source_modified": _dt.datetime.fromtimestamp(
                os.path.getmtime(src)).isoformat(sep=" ", timespec="seconds"),
            "UNV_channel_count": len(uf.channels),
            "converter": f"{APP_NAME} {APP_VERSION}",
            "conversion_time": _dt.datetime.now().isoformat(sep=" ", timespec="seconds"),
        }
        root_props.update({k: _prop(v) for k, v in uf.file_properties().items()})

        total = sum(c.n for c in chans) or 1
        weight_write = 0.6 if opts.verify else 0.95
        done = 0
        used: set = set()
        expected = {}   # tdms channel name -> (count, digest, dtype)
        stats_log = []
        try:
            with TdmsWriter(tmp) as w:
                w.write_segment([RootObject(properties=root_props),
                                 GroupObject(group, properties={"UNV_source_file": os.path.basename(src)})])
                for ch in chans:
                    base = _unique(channel_name(ch, opts.naming), used)
                    dx, note = effective_dx(ch, opts.snap_fs)
                    if note:
                        log(f"  {base}: {note}")
                    for wmsg in ch.warnings:
                        log(f"  UYARI {base}: {wmsg}")
                    out_dtype = np.dtype(np.float64) if opts.force_float64 else ch.dtype
                    parts = [("", base)] if not ch.is_complex else \
                        [("real", _unique(base + "_Re", used)), ("imag", _unique(base + "_Im", used))]
                    props = {nm: build_channel_properties(ch, nm, dx, note, out_dtype, part)
                             for part, nm in parts}
                    xname = None
                    if not ch.even:
                        xname = _unique(base + "_X", used)
                        props[xname] = {"NI_ChannelName": xname, "unit_string": ch.x_unit,
                                        "description": f"X (abscissa) of {base}",
                                        "UNV_abscissa_of": base, "wf_samples": _prop(ch.n)}
                    hashers = {nm: hashlib.blake2b(digest_size=32) for nm in props}
                    counts = dict.fromkeys(props, 0)
                    st = {"min": np.inf, "max": -np.inf, "s": 0.0, "ss": 0.0, "nan": 0}
                    first = True
                    for x, y in uf.iter_values(ch, opts.chunk_samples):
                        if cancel is not None and cancel.is_set():
                            raise Cancelled()
                        arrays = {}
                        if ch.is_complex:
                            arrays[parts[0][1]] = np.ascontiguousarray(y.real)
                            arrays[parts[1][1]] = np.ascontiguousarray(y.imag)
                        else:
                            arrays[base] = np.ascontiguousarray(y)
                        if xname:
                            arrays[xname] = np.ascontiguousarray(x)
                        objs = []
                        for nm, a in arrays.items():
                            if opts.force_float64 and a.dtype != np.float64:
                                a = a.astype(np.float64)
                            hashers[nm].update(a.tobytes())
                            counts[nm] += a.size
                            objs.append(ChannelObject(group, nm, a,
                                                      properties=props[nm] if first else {}))
                        first = False
                        w.write_segment(objs)
                        yy = np.abs(y) if ch.is_complex else y
                        fin = np.isfinite(yy)
                        nbad = yy.size - int(fin.sum())
                        if nbad:
                            st["nan"] += nbad
                            yy = yy[fin]
                        if yy.size:
                            st["min"] = min(st["min"], float(yy.min()))
                            st["max"] = max(st["max"], float(yy.max()))
                            yd = yy.astype(np.float64)
                            st["s"] += float(yd.sum())
                            st["ss"] += float(np.dot(yd, yd))
                        done += len(y)
                        progress(0.05 + weight_write * done / total, f"Yazılıyor: {base}")
                    if counts[parts[0][1]] != ch.n:
                        raise UnvError(f"{base}: {counts[parts[0][1]]} nokta yazıldı, {ch.n} bekleniyordu.")
                    for nm in props:
                        expected[nm] = (counts[nm], hashers[nm].hexdigest(), out_dtype)
                    nn = max(ch.n - st["nan"], 1)
                    mean = st["s"] / nn
                    rms = (st["ss"] / nn) ** 0.5
                    fs_txt = f"{1 / dx:,.3f} Hz" if ch.even and dx else "eşit aralıksız"
                    stats_log.append(f"  ✓ {base:<30} n={ch.n:>12,}  {fs_txt:>16}  [{ch.y_unit}]  "
                                     f"min={st['min']:.6g} max={st['max']:.6g} "
                                     f"ort={mean:.6g} rms={rms:.6g}"
                                     + (f"  NaN/Inf={st['nan']}" if st["nan"] else ""))
            for line in stats_log:
                log(line)

            # ---- verification ----
            if opts.verify:
                progress(0.05 + weight_write, "Doğrulanıyor…")
                log("  Doğrulama: TDMS geri okunup kaynak ile karşılaştırılıyor…")
                vdone = 0
                vtotal = sum(c for c, _, _ in expected.values()) or 1
                with TdmsFile.open(tmp) as tf:
                    g = tf[group]
                    names = {c.name for c in g.channels()}
                    for nm, (cnt, dig, dt) in expected.items():
                        if nm not in names:
                            raise UnvError(f"Doğrulama: '{nm}' kanalı TDMS içinde yok!")
                        tch = g[nm]
                        if len(tch) != cnt:
                            raise UnvError(f"Doğrulama: '{nm}' nokta sayısı farklı "
                                           f"({len(tch)} ≠ {cnt}).")
                        h = hashlib.blake2b(digest_size=32)
                        for chunk in tch.data_chunks():
                            if cancel is not None and cancel.is_set():
                                raise Cancelled()
                            a = np.ascontiguousarray(chunk[:])
                            if a.dtype != dt:
                                raise UnvError(f"Doğrulama: '{nm}' veri tipi farklı ({a.dtype} ≠ {dt}).")
                            h.update(a.tobytes())
                            vdone += a.size
                            progress(0.05 + weight_write + (0.95 - weight_write) * vdone / vtotal,
                                     f"Doğrulanıyor: {nm}")
                        if h.hexdigest() != dig:
                            raise UnvError(f"Doğrulama: '{nm}' verisi kaynakla birebir aynı değil!")
                log(f"  ✔ Doğrulama BAŞARILI: {len(expected)} kanal, tüm örnekler bit-bit aynı "
                    f"(BLAKE2 özet + nokta sayısı + veri tipi).")
            if os.path.exists(out):
                os.remove(out)
            os.replace(tmp, out)
        except BaseException:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            raise
    dt_s = time.time() - t0
    size = os.path.getsize(out)
    log(f"  → {out}  ({size / 1e6:,.1f} MB, {dt_s:.1f} s)")
    progress(1.0, "Bitti")
    return {"output": out, "channels": len(expected), "seconds": dt_s, "verified": opts.verify}


# --------------------------------------------------------------------------
# TDMS reader with the same interface as UnvFile (for previewing results)
# --------------------------------------------------------------------------
class TdmsChannelInfo:
    def __init__(self, idx, group, ch):
        self.index = idx
        self.group = group
        self.name = f"{group}/{ch.name}"
        self.tdms_name = ch.name
        p = dict(ch.properties)
        self.props = p
        self.n = len(ch)
        inc = p.get("wf_increment")
        self.even = inc is not None
        self.dx = float(inc) if inc is not None else 1.0
        self.x0 = float(p.get("wf_start_offset", 0.0))
        self.y_unit = str(p.get("unit_string", ""))
        self.x_unit = str(p.get("wf_xunit_string", "s" if self.even else ""))
        self.x_label = str(p.get("wf_xname", "Time"))
        self.is_complex = False
        self.dtype = np.dtype(ch.dtype) if ch.dtype is not None else np.dtype(np.float64)
        self.type_name = "TDMS"
        self.precision_note = self.dtype.name
        if not self.even:
            self.even = True  # plot against sample index
            self.dx = 1.0
            self.x_unit = "örnek"

    @property
    def fs(self):
        return 1.0 / self.dx if self.dx else float("nan")

    @property
    def duration(self):
        return self.n * self.dx

    def header_text(self):
        out = [f"TDMS kanal: {self.name}", f"Nokta sayısı: {self.n:,}",
               f"Veri tipi: {self.dtype}", "", "---- Özellikler ----"]
        for k, v in self.props.items():
            if k == "UNV_header_raw":
                continue
            out.append(f"{k:<34}= {v!r}")
        if "UNV_header_raw" in self.props:
            out += ["", "---- UNV ham başlık ----", str(self.props["UNV_header_raw"])]
        return "\n".join(out)


class TdmsSource:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self._f = TdmsFile.open(self.path)
        self.channels = []
        self.warnings = []
        self.other_datasets = []
        for g in self._f.groups():
            for c in g.channels():
                self.channels.append(TdmsChannelInfo(len(self.channels), g.name, c))

    def _ch(self, ci):
        return self._f[ci.group][ci.tdms_name]

    def iter_values(self, ci, chunk=4_000_000):
        for c in self._ch(ci).data_chunks():
            yield None, c[:]

    def read_range(self, ci, start, count):
        start = max(0, int(start))
        count = max(0, min(int(count), ci.n - start))
        y = self._ch(ci).read_data(start, count)
        return ci.x0 + ci.dx * np.arange(start, start + count), y

    def file_properties(self):
        return dict(self._f.properties)

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


def open_source(path, progress_cb=None):
    if path.lower().endswith(".tdms"):
        return TdmsSource(path)
    return UnvFile(path, progress_cb=progress_cb)


def envelope(src, ch, nbins=4000, cancel=None, progress=None):
    """Min/max envelope + stats over the full channel, read in chunks."""
    n = ch.n
    if n <= 0:
        return np.empty(0), np.empty(0), np.empty(0), {}
    b = max(1, -(-n // nbins))
    nb = -(-n // b)
    mins = np.full(nb, np.inf)
    maxs = np.full(nb, -np.inf)
    s = ss = 0.0
    cnt = 0
    gmin, gmax = np.inf, -np.inf
    pos = 0
    for _, y in src.iter_values(ch, 4_000_000):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        y = np.abs(y) if np.iscomplexobj(y) else y
        y = y.astype(np.float64, copy=False)
        m = len(y)
        if m == 0:
            continue
        off = (-pos) % b
        starts = np.arange(off, m, b)
        if off != 0:
            starts = np.concatenate(([0], starts))
        ids = (pos + starts) // b
        fin = np.where(np.isfinite(y), y, np.nan)
        with np.errstate(invalid="ignore"):
            mn = np.fmin.reduceat(fin, starts)
            mx = np.fmax.reduceat(fin, starts)
        mn = np.where(np.isnan(mn), np.inf, mn)
        mx = np.where(np.isnan(mx), -np.inf, mx)
        np.minimum.at(mins, ids, mn)
        np.maximum.at(maxs, ids, mx)
        good = y[np.isfinite(y)]
        if good.size:
            s += float(good.sum())
            ss += float(np.dot(good, good))
            cnt += good.size
            gmin = min(gmin, float(good.min()))
            gmax = max(gmax, float(good.max()))
        pos += m
        if progress:
            progress(pos / n)
    centers = ch.x0 + ch.dx * (np.arange(nb) * b + (b - 1) / 2.0)
    mins[np.isinf(mins)] = np.nan
    maxs[np.isinf(maxs)] = np.nan
    stats = {"n": n, "min": gmin, "max": gmax,
             "mean": s / cnt if cnt else float("nan"),
             "rms": (ss / cnt) ** 0.5 if cnt else float("nan"),
             "nonfinite": n - cnt, "bin": b}
    return centers, mins, maxs, stats


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="UNV (58/58b) → TDMS dönüştürücü (komut satırı)")
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--out-dir")
    ap.add_argument("-g", "--group", default="")
    ap.add_argument("--naming", choices=list(NAMING_MODES), default="id1")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--no-snap", action="store_true")
    ap.add_argument("--float64", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    opts = ConvertOptions(out_dir=a.out_dir, group_name=a.group, naming=a.naming,
                          verify=not a.no_verify, snap_fs=not a.no_snap,
                          force_float64=a.float64, overwrite=a.overwrite)
    import glob
    files = []
    for pat in a.files:  # Windows komut satırı joker karakterleri genişletmez
        files += sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
    rc = 0
    for f in files:
        try:
            convert_file(f, opts)
        except Exception as e:  # noqa: BLE001
            print(f"HATA: {f}: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
