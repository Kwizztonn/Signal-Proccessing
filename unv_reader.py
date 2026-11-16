"""
Streaming reader for Universal File Format (UFF / .unv) files.

Supports function datasets 58 (ASCII) and 58b (binary) as written by
Simcenter Testlab, LMS, ME'scope, etc. Large files (several GB) are handled
through a memory map: the file is indexed once, and channel data are read in
chunks, so RAM usage stays small regardless of file size.

Other datasets (151 header, 164 units, ...) are indexed and their raw text is
kept so that it can be stored as metadata in the output.
"""
from __future__ import annotations

import datetime as _dt
import mmap
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np


class UnvError(Exception):
    pass


# --------------------------------------------------------------------------
# Lookup tables (UFF dataset 58 specification)
# --------------------------------------------------------------------------
FUNCTION_TYPES = {
    0: "General or Unknown", 1: "Time Response", 2: "Auto Spectrum",
    3: "Cross Spectrum", 4: "Frequency Response Function", 5: "Transmissibility",
    6: "Coherence", 7: "Auto Correlation", 8: "Cross Correlation",
    9: "Power Spectral Density (PSD)", 10: "Energy Spectral Density (ESD)",
    11: "Probability Density Function", 12: "Spectrum",
    13: "Cumulative Frequency Distribution", 14: "Peaks Valley",
    15: "Stress/Cycles", 16: "Strain/Cycles", 17: "Orbit",
    18: "Mode Indicator Function", 19: "Force Pattern", 20: "Partial Power",
    21: "Partial Coherence", 22: "Eigenvalue", 23: "Eigenvector",
    24: "Shock Response Spectrum", 25: "Finite Impulse Response Filter",
    26: "Multiple Coherence", 27: "Order Function", 28: "Phase Compensation",
}

ORDINATE_TYPES = {
    # code: (is_complex, numpy float dtype, description)
    2: (False, np.float32, "real, single precision"),
    4: (False, np.float64, "real, double precision"),
    5: (True, np.float32, "complex, single precision"),
    6: (True, np.float64, "complex, double precision"),
}

SPEC_DATA_TYPES = {
    0: "unknown", 1: "general", 2: "stress", 3: "strain", 5: "temperature",
    6: "heat flux", 8: "displacement", 9: "reaction force", 11: "velocity",
    12: "acceleration", 13: "excitation force", 15: "pressure", 16: "mass",
    17: "time", 18: "frequency", 19: "rpm", 20: "order", 21: "sound pressure",
    22: "sound intensity", 23: "sound power",
}

DIRECTIONS = {0: "S", 1: "+X", -1: "-X", 2: "+Y", -2: "-Y", 3: "+Z", -3: "-Z",
              4: "+RX", -4: "-RX", 5: "+RY", -5: "-RY", 6: "+RZ", -6: "-RZ"}

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MONTHS.update({"oca": 1, "şub": 2, "sub": 2, "nis": 4, "haz": 6, "tem": 7,
                "ağu": 8, "agu": 8, "eyl": 9, "eki": 10, "kas": 11, "ara": 12})

_DELIM_RE = re.compile(rb"^[ \t]*-1[ \t]*\r?$", re.M)
_NUM_RE = re.compile(
    rb"[-+]?(?:(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?|nan|NaN|NAN|inf|Inf|INF)")

ASCII_CHUNK_BYTES = 64 * 1024 * 1024


def _decode(b: bytes) -> str:
    """Header text: UTF-8 if valid, otherwise Windows Turkish/Western code page."""
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("cp1254", "replace")


def _to_float(s: str) -> float:
    return float(s.strip().replace("D", "E").replace("d", "e"))


def _to_int(s: str) -> int:
    return int(s.strip())


def parse_unv_datetime(text: str) -> Optional[_dt.datetime]:
    """Parse typical UNV date strings (e.g. '16-Nov-26 10:28:00')."""
    if not text:
        return None
    t = text.strip()
    m = re.search(r"(\d{1,2})[-./ ]([A-Za-zçğıöşüÇĞİÖŞÜ]{3})[A-Za-z]*[-./ ](\d{2,4})"
                  r"[ T]+(\d{1,2}):(\d{2})(?::(\d{2})(?:[.,](\d+))?)?", t)
    if m:
        d, mon, y, hh, mm_, ss, frac = m.groups()
        mon_i = _MONTHS.get(mon.lower())
        if mon_i:
            y = int(y)
            y = y + (2000 if y < 70 else 1900) if y < 100 else y
            us = int((frac or "0")[:6].ljust(6, "0"))
            try:
                return _dt.datetime(y, mon_i, int(d), int(hh), int(mm_), int(ss or 0), us)
            except ValueError:
                pass
    patterns = [
        (r"(\d{4})-(\d{1,2})-(\d{1,2})[ T]+(\d{1,2}):(\d{2})(?::(\d{2}))?", "ymd"),
        (r"(\d{1,2})\.(\d{1,2})\.(\d{4})[ T]+(\d{1,2}):(\d{2})(?::(\d{2}))?", "dmy"),
        (r"(\d{1,2})/(\d{1,2})/(\d{4})[ T]+(\d{1,2}):(\d{2})(?::(\d{2}))?", "mdy"),
    ]
    for pat, order in patterns:
        m = re.search(pat, t)
        if not m:
            continue
        a, b, c, hh, mm_, ss = m.groups()
        try:
            if order == "ymd":
                return _dt.datetime(int(a), int(b), int(c), int(hh), int(mm_), int(ss or 0))
            if order == "dmy":
                return _dt.datetime(int(c), int(b), int(a), int(hh), int(mm_), int(ss or 0))
            return _dt.datetime(int(c), int(a), int(b), int(hh), int(mm_), int(ss or 0))
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------
@dataclass
class Dataset:
    number: int
    tag: str              # e.g. "58", "58b", "151"
    binary: bool
    body_start: int       # first byte after the dataset-id line
    body_end: int         # first byte of the closing "-1" line
    data_start: int = 0   # for 58/58b: first byte of the data block
    data_end: int = 0
    byte_order: int = 0   # 58b: 1 = little endian, 2 = big endian
    float_format: int = 0 # 58b: 2 = IEEE 754
    header_line: str = ""


@dataclass
class UnvChannel:
    index: int                      # 0-based index among the 58 datasets
    dataset_index: int              # index among all datasets in the file
    fmt: str                        # "58" or "58b"
    header_lines: list              # the 11 ASCII header records
    rec: dict                       # all parsed header fields
    n: int                          # number of data points
    even: bool
    x0: float
    dx: float
    is_complex: bool
    src_dtype: np.dtype             # dtype as stored in the file
    dtype: np.dtype                 # dtype used for output (lossless)
    ord_type: int
    x_label: str
    x_unit: str
    y_label: str
    y_unit: str
    func_type: int
    name: str = ""
    start_time: Optional[_dt.datetime] = None
    byte_order: str = "<"
    data_start: int = 0
    data_end: int = 0
    binary: bool = False
    precision_note: str = ""
    warnings: list = field(default_factory=list)

    # ---- convenience ------------------------------------------------------
    @property
    def fs(self) -> float:
        return 1.0 / self.dx if (self.even and self.dx) else float("nan")

    @property
    def duration(self) -> float:
        return self.n * self.dx if self.even else float("nan")

    @property
    def type_name(self) -> str:
        return FUNCTION_TYPES.get(self.func_type, f"type {self.func_type}")

    @property
    def stride(self) -> int:
        return (2 if self.is_complex else 1) + (0 if self.even else 1)

    @property
    def default_name(self) -> str:
        id1 = self.rec.get("id1", "").strip()
        if id1 and id1.upper() != "NONE":
            return id1
        return self.node_dir_name

    @property
    def node_dir_name(self) -> str:
        r = self.rec
        d = DIRECTIONS.get(r.get("rsp_dir", 0), str(r.get("rsp_dir", "")))
        ent = (r.get("rsp_ent_name") or "").strip()
        node = r.get("rsp_node", "")
        base = f"{ent}:{node}" if ent and ent.upper() != "NONE" else f"{node}"
        return f"{base}:{d}"

    def header_text(self) -> str:
        out = [f"Dataset {self.fmt}  (UNV içindeki #{self.dataset_index + 1}, kanal #{self.index + 1})",
               f"Fonksiyon tipi : {self.func_type} - {self.type_name}",
               f"Veri tipi      : {self.ord_type} - {ORDINATE_TYPES[self.ord_type][2]}"
               f"{'  (' + self.precision_note + ')' if self.precision_note else ''}",
               f"Nokta sayısı   : {self.n:,}",
               f"Aralık         : {'eşit' if self.even else 'eşit değil (X kanalı yazılır)'}",
               f"x0 / dx        : {self.x0!r} / {self.dx!r}",
               ]
        if self.even and self.dx:
            out.append(f"fs / süre      : {self.fs:.6f} Hz / {self.duration:.6f} {self.x_unit or 's'}")
        out.append(f"X ekseni       : {self.x_label} [{self.x_unit}]")
        out.append(f"Y ekseni       : {self.y_label} [{self.y_unit}]")
        if self.start_time:
            out.append(f"Başlangıç      : {self.start_time.isoformat(sep=' ')}")
        for w in self.warnings:
            out.append(f"UYARI          : {w}")
        out.append("")
        out.append("---- Tüm başlık alanları ----")
        for k, v in self.rec.items():
            out.append(f"{k:<24}= {v!r}")
        out.append("")
        out.append("---- Ham başlık satırları (kayıt 1-11) ----")
        for i, line in enumerate(self.header_lines, 1):
            out.append(f"{i:>2}| {line}")
        return "\n".join(out)


# --------------------------------------------------------------------------
# Header parsing
# --------------------------------------------------------------------------
def _parse_rec6(line: str) -> dict:
    keys = ["func_type", "func_id", "ver_num", "load_case_id", "rsp_ent_name",
            "rsp_node", "rsp_dir", "ref_ent_name", "ref_node", "ref_dir"]
    # Fixed width: 2(I5,I10),2(1X,10A1,I10,I4)
    try:
        if len(line.rstrip()) >= 55:
            vals = [_to_int(line[0:5]), _to_int(line[5:15]), _to_int(line[15:20]),
                    _to_int(line[20:30]), line[31:41].strip(), _to_int(line[41:51]),
                    _to_int(line[51:55]), line[56:66].strip(),
                    _to_int(line[66:76] or "0"), _to_int(line[76:80] or "0")]
            return dict(zip(keys, vals))
    except ValueError:
        pass
    tok = line.split()
    if len(tok) >= 10:
        vals = [int(tok[0]), int(tok[1]), int(tok[2]), int(tok[3]), tok[4],
                int(tok[5]), int(tok[6]), tok[7], int(tok[8]), int(tok[9])]
        return dict(zip(keys, vals))
    if len(tok) >= 1:  # very short / non-standard record
        d = dict.fromkeys(keys, 0)
        d.update({"rsp_ent_name": "", "ref_ent_name": ""})
        for k, t in zip(keys, tok):
            try:
                d[k] = int(t)
            except ValueError:
                d[k] = t
        return d
    raise UnvError(f"Kayıt 6 okunamadı: {line!r}")


def _parse_rec7(line: str) -> dict:
    keys = ["ord_data_type", "num_pts", "abscissa_spacing", "abscissa_min",
            "abscissa_inc", "z_axis_value"]
    tok = line.replace("D", "E").split()
    if len(tok) >= 6:
        try:
            return dict(zip(keys, [int(tok[0]), int(tok[1]), int(tok[2]),
                                   float(tok[3]), float(tok[4]), float(tok[5])]))
        except ValueError:
            pass
    try:  # Fixed width 3I10,3E13.5
        return dict(zip(keys, [_to_int(line[0:10]), _to_int(line[10:20]), _to_int(line[20:30]),
                               _to_float(line[30:43]), _to_float(line[43:56]),
                               _to_float(line[56:69] or "0")]))
    except ValueError:
        pass
    # last resort: numbers glued together
    nums = _NUM_RE.findall(line.encode("latin-1", "replace"))
    if len(nums) >= 5:
        f = [float(x.replace(b"D", b"E")) for x in nums]
        return dict(zip(keys, [int(f[0]), int(f[1]), int(f[2]), f[3], f[4],
                               f[5] if len(f) > 5 else 0.0]))
    raise UnvError(f"Kayıt 7 okunamadı: {line!r}")


def _parse_axis(line: str, prefix: str) -> dict:
    keys = ["spec_data_type", "len_unit_exp", "force_unit_exp", "temp_unit_exp",
            "axis_lab", "axis_units_lab"]
    keys = [f"{prefix}_{k}" for k in keys]
    try:  # I10,3I5,2(1X,20A1)
        vals = [_to_int(line[0:10]), _to_int(line[10:15]), _to_int(line[15:20]),
                _to_int(line[20:25]), line[26:46].strip(), line[47:67].strip()]
        return dict(zip(keys, vals))
    except (ValueError, IndexError):
        pass
    tok = line.split()
    vals = [0, 0, 0, 0, "", ""]
    try:
        for i in range(min(4, len(tok))):
            vals[i] = int(tok[i])
        if len(tok) > 4:
            vals[4] = tok[4]
        if len(tok) > 5:
            vals[5] = " ".join(tok[5:])
    except ValueError:
        vals[4] = line.strip()
    return dict(zip(keys, vals))


# --------------------------------------------------------------------------
# Main reader
# --------------------------------------------------------------------------
class UnvFile:
    """Index a UNV file and give chunked access to dataset-58 channels."""

    def __init__(self, path: str, progress_cb=None):
        self.path = os.path.abspath(path)
        self.size = os.path.getsize(self.path)
        if self.size == 0:
            raise UnvError("Dosya boş.")
        self._fh = open(self.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self.datasets: list[Dataset] = []
        self.channels: list[UnvChannel] = []
        self.other_datasets: list[dict] = []   # 151, 164, ... (raw text)
        self.warnings: list[str] = []
        self._delim = b"    -1"
        self._progress_cb = progress_cb
        try:
            self._scan()
        except Exception:
            self.close()
            raise

    # ---- context management ---------------------------------------------
    def close(self):
        try:
            if getattr(self, "_mm", None) is not None:
                self._mm.close()
        except BufferError:
            pass
        self._mm = None
        if getattr(self, "_fh", None) is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- delimiter search -------------------------------------------------
    def _delim_ok(self, i: int) -> Optional[int]:
        """If a '-1' delimiter line starts at i, return the index of its EOL."""
        mm, n = self._mm, len(self._mm)
        j = i + len(self._delim)
        k = mm.find(b"\n", j, min(n, j + 128))
        if k < 0:
            if n - j <= 128:
                k = n
            else:
                return None
        if mm[j:k].strip() == b"":
            return k
        return None

    def _find_delim(self, start: int):
        """Return (line_start, eol) of the next '-1' line at/after start."""
        mm = self._mm
        d = self._delim
        if mm[start:start + len(d)] == d and (start == 0 or mm[start - 1:start] in (b"\n", b"\r")):
            k = self._delim_ok(start)
            if k is not None:
                return start, k
        key = b"\n" + d
        pos = start
        while True:
            i = mm.find(key, pos)
            if i < 0:
                break
            k = self._delim_ok(i + 1)
            if k is not None:
                return i + 1, k
            pos = i + 1
        m = _DELIM_RE.search(mm, start)
        if m:
            e = mm.find(b"\n", m.start())
            return m.start(), (e if e >= 0 else len(mm))
        return None

    def _next_line(self, pos: int):
        """Return (line_text_without_eol, next_pos)."""
        mm = self._mm
        e = mm.find(b"\n", pos)
        if e < 0:
            e = len(mm)
        return _decode(mm[pos:e].rstrip(b"\r")), e + 1

    # ---- scanning -----------------------------------------------------------
    def _scan(self):
        mm = self._mm
        n = len(mm)
        # Detect delimiter style from the first delimiter line
        m = _DELIM_RE.search(mm, 0, min(n, 1 << 20))
        if not m:
            raise UnvError("Geçerli bir UNV/UFF dosyası değil ('-1' ayıracı bulunamadı).")
        self._delim = mm[m.start():m.end()].rstrip(b"\r").rstrip()
        pos = m.start()
        ds_index = 0
        while pos < n:
            d = self._find_delim(pos)
            if d is None:
                break
            _, eol = d
            hstart = eol + 1
            if hstart >= n:
                break
            hline, body_start = self._next_line(hstart)
            tok = hline.split()
            if not tok:
                pos = body_start
                continue
            tag = tok[0]
            binary = tag.lower().endswith("b")
            try:
                number = int(tag[:-1] if binary else tag)
            except ValueError:
                # not a dataset header -> resync on next delimiter
                pos = hstart
                continue
            ds = Dataset(number=number, tag=tag, binary=binary, body_start=body_start,
                         body_end=body_start, header_line=hline)
            if binary:
                try:
                    ds.byte_order = int(tok[1])
                    ds.float_format = int(tok[2])
                    nlines = int(tok[3])
                    nbytes = int(tok[4])
                except (IndexError, ValueError):
                    raise UnvError(f"Binary dataset başlığı okunamadı: {hline!r}")
                p = body_start
                for _ in range(nlines):
                    q = mm.find(b"\n", p)
                    if q < 0:
                        raise UnvError("Dosya beklenmedik şekilde bitti (binary başlık).")
                    p = q + 1
                ds.data_start = p
                ds.data_end = p + nbytes
                if ds.data_end > n:
                    raise UnvError(f"Dataset {tag} #{ds_index + 1}: dosya kesik "
                                   f"({nbytes} bayt bekleniyordu, {n - p} bayt var).")
                close = self._find_delim(ds.data_end)
                if close is None:
                    ds.body_end = n
                    self.warnings.append(f"Dataset #{ds_index + 1}: kapanış '-1' bulunamadı.")
                    pos = n
                else:
                    ds.body_end = close[0]
                    pos = close[1] + 1
            else:
                close = self._find_delim(body_start)
                if close is None:
                    ds.body_end = n
                    self.warnings.append(f"Dataset #{ds_index + 1}: kapanış '-1' bulunamadı.")
                    pos = n
                else:
                    ds.body_end = close[0]
                    pos = close[1] + 1
            self.datasets.append(ds)
            try:
                self._register(ds, ds_index)
            except UnvError as e:
                self.warnings.append(f"Dataset #{ds_index + 1} ({tag}) atlandı: {e}")
            ds_index += 1
            if self._progress_cb:
                self._progress_cb(pos / n)

    def _register(self, ds: Dataset, ds_index: int):
        mm = self._mm
        if ds.number != 58:
            if ds.binary:
                raw = _decode(mm[ds.body_start:ds.data_start])
            else:
                size = ds.body_end - ds.body_start
                raw = _decode(mm[ds.body_start:ds.body_start + min(size, 64 * 1024)])
            self.other_datasets.append({"index": ds_index, "tag": ds.tag, "number": ds.number,
                                        "bytes": ds.body_end - ds.body_start,
                                        "raw": raw.replace("\r", "").rstrip("\n")})
            return

        # ---- dataset 58: 11 ASCII header records ----
        lines = []
        p = ds.body_start
        for _ in range(11):
            line, p = self._next_line(p)
            lines.append(line)
        if not ds.binary:
            ds.data_start = p
            ds.data_end = ds.body_end
        rec = {f"id{i + 1}": lines[i].rstrip() for i in range(5)}
        rec.update(_parse_rec6(lines[5]))
        rec.update(_parse_rec7(lines[6]))
        rec.update(_parse_axis(lines[7], "abscissa"))
        rec.update(_parse_axis(lines[8], "ordinate"))
        rec.update(_parse_axis(lines[9], "ordinate_den"))
        rec.update(_parse_axis(lines[10], "z_axis"))

        ot = rec["ord_data_type"]
        if ot not in ORDINATE_TYPES:
            raise UnvError(f"Desteklenmeyen veri tipi (kayıt 7): {ot}")
        is_complex, fdt, _ = ORDINATE_TYPES[ot]
        n = int(rec["num_pts"])
        even = int(rec["abscissa_spacing"]) == 1
        ch = UnvChannel(
            index=len(self.channels), dataset_index=ds_index, fmt=ds.tag, header_lines=lines,
            rec=rec, n=n, even=even, x0=float(rec["abscissa_min"]), dx=float(rec["abscissa_inc"]),
            is_complex=is_complex, src_dtype=np.dtype(fdt), dtype=np.dtype(fdt), ord_type=ot,
            x_label=rec["abscissa_axis_lab"], x_unit=rec["abscissa_axis_units_lab"],
            y_label=rec["ordinate_axis_lab"], y_unit=rec["ordinate_axis_units_lab"],
            func_type=int(rec["func_type"]), binary=ds.binary,
            data_start=ds.data_start, data_end=ds.data_end,
        )
        for k in ("id3", "id4", "id2", "id5"):
            t = parse_unv_datetime(rec.get(k, ""))
            if t:
                ch.start_time = t
                break
        expected_vals = n * ch.stride
        if ds.binary:
            if ds.float_format != 2:
                raise UnvError(f"Desteklenmeyen kayan nokta formatı ({ds.float_format}); "
                               f"yalnızca IEEE 754 (2) desteklenir.")
            ch.byte_order = "<" if ds.byte_order == 1 else ">"
            nbytes = ds.data_end - ds.data_start
            exp_bytes = expected_vals * ch.src_dtype.itemsize
            if nbytes < exp_bytes:
                raise UnvError(f"Binary veri eksik: {nbytes} bayt var, {exp_bytes} bekleniyor.")
            if nbytes > exp_bytes:
                ch.warnings.append(f"Binary blok {nbytes - exp_bytes} bayt fazla; "
                                   f"başlıktaki {n} nokta okunur.")
            ch.precision_note = "binary, birebir kopya"
        else:
            # Check how many significant digits are written; if a "single precision"
            # dataset has more than 7 digits in text, keep it as float64 (no loss).
            first, _ = self._next_line(ds.data_start)
            toks = first.split()
            digits = 0
            for t in toks[:6]:
                mant = re.split(r"[EeDd]", t)[0]
                digits = max(digits, len(re.sub(r"[^0-9]", "", mant).lstrip("0")) or 1)
            if ch.src_dtype == np.float32 and digits > 7:
                ch.dtype = np.dtype(np.float64)
                ch.precision_note = (f"ASCII {digits} anlamlı basamak → float64 olarak saklanır")
            else:
                ch.precision_note = f"ASCII {digits} anlamlı basamak → {ch.dtype.name} kayıpsız"
        ch.name = ch.default_name
        self.channels.append(ch)

    # ---- data access ---------------------------------------------------------
    def _iter_ascii_values(self, ch: UnvChannel) -> Iterator[np.ndarray]:
        """Yield flat float64 arrays of all numbers in the ASCII data block."""
        mm = self._mm
        pos, end = ch.data_start, ch.data_end
        while pos < end:
            stop = min(end, pos + ASCII_CHUNK_BYTES)
            if stop < end:
                nl = mm.rfind(b"\n", pos, stop)
                if nl <= pos:
                    nl = mm.find(b"\n", stop, end)
                    stop = end if nl < 0 else nl + 1
                else:
                    stop = nl + 1
            block = mm[pos:stop]
            if b"D" in block:
                block = block.replace(b"D", b"E")
            vals = None
            if block.strip():
                try:
                    with warnings.catch_warnings():
                        # older numpy only warns (and truncates) on unparsable text
                        warnings.simplefilter("error")
                        vals = np.fromstring(block, dtype=np.float64, sep=" ")
                except (ValueError, DeprecationWarning):
                    vals = None
                if vals is None:  # fixed-width fields glued together, e.g. "1.0E+00-2.0E+00"
                    toks = _NUM_RE.findall(block)
                    vals = np.array(toks, dtype=np.float64)
                yield vals
            pos = stop

    def iter_values(self, ch: UnvChannel, chunk: int = 4_000_000):
        """Yield (x, y) chunks. x is None for evenly spaced data.
        y is real (ch.dtype) or complex (complex64/128)."""
        stride = ch.stride
        if ch.binary:
            src = ch.src_dtype.newbyteorder(ch.byte_order)
            item = src.itemsize
            for k in range(0, ch.n, chunk):
                m = min(chunk, ch.n - k)
                raw = np.frombuffer(self._mm, dtype=src, count=m * stride,
                                    offset=ch.data_start + k * stride * item).astype(ch.dtype)
                yield self._split(raw, ch)
            return

        need = chunk * stride
        buf: list[np.ndarray] = []
        buffered = 0
        total = 0
        expected = ch.n * stride
        for vals in self._iter_ascii_values(ch):
            buf.append(vals)
            buffered += vals.size
            while buffered >= need and total + need <= expected:
                arr = np.concatenate(buf) if len(buf) > 1 else buf[0]
                out, rest = arr[:need], arr[need:]
                buf = [rest] if rest.size else []
                buffered = rest.size
                total += need
                yield self._split(out.astype(ch.dtype), ch)
        arr = np.concatenate(buf) if buf else np.empty(0)
        if total + arr.size < expected:
            raise UnvError(f"Kanal '{ch.name}': ASCII veri eksik "
                           f"({(total + arr.size) // stride} nokta okundu, {ch.n} bekleniyordu).")
        if total + arr.size > expected:
            extra = total + arr.size - expected
            ch.warnings.append(f"ASCII blokta {extra} fazla değer var; yok sayıldı.")
            arr = arr[:expected - total]
        if arr.size:
            yield self._split(arr.astype(ch.dtype), ch)

    @staticmethod
    def _split(raw: np.ndarray, ch: UnvChannel):
        if ch.even:
            if not ch.is_complex:
                return None, raw
            return None, raw.view(np.complex64 if raw.dtype == np.float32 else np.complex128)
        if not ch.is_complex:
            return raw[0::2].copy(), raw[1::2].copy()
        ctype = np.complex64 if raw.dtype == np.float32 else np.complex128
        y = np.empty(raw.size // 3, dtype=ctype)
        y.real = raw[1::3]
        y.imag = raw[2::3]
        return raw[0::3].copy(), y

    def read_range(self, ch: UnvChannel, start: int, count: int):
        """Return (x, y) arrays for samples [start, start+count)."""
        start = max(0, int(start))
        count = max(0, min(int(count), ch.n - start))
        if count == 0:
            return np.empty(0), np.empty(0, dtype=ch.dtype)
        if ch.binary:
            src = ch.src_dtype.newbyteorder(ch.byte_order)
            raw = np.frombuffer(self._mm, dtype=src, count=count * ch.stride,
                                offset=ch.data_start + start * ch.stride * src.itemsize
                                ).astype(ch.dtype)
            x, y = self._split(raw, ch)
        else:
            xs, ys, pos = [], [], 0
            for x, y in self.iter_values(ch):
                m = len(y)
                a, b = max(start - pos, 0), min(start + count - pos, m)
                if b > a:
                    ys.append(y[a:b])
                    if x is not None:
                        xs.append(x[a:b])
                pos += m
                if pos >= start + count:
                    break
            y = np.concatenate(ys)
            x = np.concatenate(xs) if xs else None
        if x is None:
            x = ch.x0 + ch.dx * np.arange(start, start + count, dtype=np.float64)
        return x, y

    # ---- file-level metadata ---------------------------------------------
    def file_properties(self) -> dict:
        props = {}
        for od in self.other_datasets:
            if od["number"] == 151:
                lines = od["raw"].split("\n")
                keys = ["model_name", "description", "db_app", "date_db_created",
                        "time_db_created", "version_db1", "version_db2", "file_type",
                        "date_db_saved", "time_db_saved", "program", "date_file_written",
                        "time_file_written"]
                for k, v in zip(keys, lines):
                    if v.strip():
                        props[f"UNV151_{k}"] = v.strip()
                props["UNV151_raw"] = od["raw"]
            elif od["number"] == 164:
                lines = od["raw"].split("\n")
                if lines:
                    tok = lines[0].split()
                    if tok:
                        props["UNV164_units_code"] = tok[0]
                        props["UNV164_units_description"] = " ".join(tok[1:])
                props["UNV164_raw"] = od["raw"]
        others = sorted({od["tag"] for od in self.other_datasets if od["number"] not in (151, 164)})
        if others:
            props["UNV_other_datasets"] = ", ".join(others)
        return props
