"""
UNV2TDMS – Simcenter Testlab UNV (dataset 58 / 58b) → NI TDMS dönüştürücü
Arayüz: tkinter + matplotlib (veri önizleme)

Komut satırı kullanımı:  UNV2TDMS.exe --cli dosya1.unv dosya2.unv -o cikis_klasoru
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import traceback

# Komut satırı modu: arayüz kütüphanelerini yüklemeden çalış
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--cli":
    from converter import main as _cli_main
    sys.exit(_cli_main(sys.argv[2:]))

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from converter import (APP_NAME, APP_VERSION, NAMING_MODES, Cancelled, ConvertOptions,  # noqa: E402
                       channel_name, convert_file, effective_dx, envelope, open_source)

MAX_PLOT_CHANNELS = 6
RAW_PLOT_LIMIT = 3_000_000      # en fazla bu kadar ham nokta çizilir
FULL_RAW_IF_BELOW = 200_000     # kanal bundan kısaysa doğrudan ham çizilir
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def _fmt_size(b):
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024 or unit == "GB":
            return f"{b:,.1f} {unit}" if unit != "B" else f"{b} B"
        b /= 1024


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(f"{APP_NAME} {APP_VERSION} – Testlab UNV → TDMS (nCode) Dönüştürücü")
        root.geometry("1500x940")
        root.minsize(1100, 700)

        self.q: queue.Queue = queue.Queue()
        self.files: list[str] = []
        self.status_of: dict[str, str] = {}
        self.sources: dict = {}            # path -> açık kaynak (önizleme için)
        self.loading: set = set()
        self.cur_path = None
        self.cancel_ev = threading.Event()
        self.preview_cancel = threading.Event()
        self.preview_gen = 0
        self.converting = False
        self._sel_after = None
        self.plot_channels = []

        self._build_ui()
        self.root.after(50, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.log(f"{APP_NAME} {APP_VERSION} hazır. UNV dosyalarını ekleyin, bir dosya seçip "
                 f"kanalları önizleyin, sonra DÖNÜŞTÜR'e basın.")

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        style = ttk.Style()
        try:
            if sys.platform == "win32":
                style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure("Big.TButton", font=("Segoe UI", 11, "bold"), padding=6)

        # ---- toolbar
        tb = ttk.Frame(self.root, padding=(6, 6, 6, 0))
        tb.pack(fill="x")
        ttk.Button(tb, text="➕ UNV Dosyası Ekle…", command=self.add_files).pack(side="left")
        ttk.Button(tb, text="📁 Klasörden Ekle…", command=self.add_folder).pack(side="left", padx=4)
        ttk.Button(tb, text="🔍 TDMS Aç (kontrol için)…", command=self.add_tdms).pack(side="left")
        ttk.Button(tb, text="Seçileni Kaldır", command=self.remove_selected).pack(side="left", padx=(12, 4))
        ttk.Button(tb, text="Listeyi Temizle", command=self.clear_files).pack(side="left")
        ttk.Button(tb, text="Logu Kaydet…", command=self.save_log).pack(side="right")

        # ---- settings
        opt = ttk.LabelFrame(self.root, text="Ayarlar", padding=6)
        opt.pack(fill="x", padx=6, pady=6)
        self.v_out = tk.StringVar()
        self.v_group = tk.StringVar()
        self.v_naming = tk.StringVar(value=NAMING_MODES["id1"])
        self.v_verify = tk.BooleanVar(value=True)
        self.v_snap = tk.BooleanVar(value=True)
        self.v_f64 = tk.BooleanVar(value=False)
        self.v_over = tk.BooleanVar(value=False)

        ttk.Label(opt, text="Çıkış klasörü:").grid(row=0, column=0, sticky="w")
        ttk.Entry(opt, textvariable=self.v_out, width=60).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(opt, text="Gözat…", command=self.pick_out).grid(row=0, column=2)
        ttk.Label(opt, text="(boş = UNV dosyasının yanına)", foreground="gray").grid(row=0, column=3, sticky="w", padx=4)
        ttk.Label(opt, text="TDMS grup adı:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(opt, textvariable=self.v_group, width=30).grid(row=1, column=1, sticky="w", padx=4, pady=(4, 0))
        ttk.Label(opt, text="(boş = dosya adı)", foreground="gray").grid(row=1, column=1, sticky="e", pady=(4, 0))
        ttk.Label(opt, text="Kanal adı:").grid(row=1, column=2, sticky="e", pady=(4, 0))
        cb = ttk.Combobox(opt, textvariable=self.v_naming, values=list(NAMING_MODES.values()),
                          state="readonly", width=30)
        cb.grid(row=1, column=3, sticky="w", padx=4, pady=(4, 0))
        cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_channel_names())

        chk = ttk.Frame(opt)
        chk.grid(row=2, column=0, columnspan=6, sticky="w", pady=(6, 0))
        ttk.Checkbutton(chk, text="Yazdıktan sonra doğrula (TDMS geri okunur, bit-bit karşılaştırılır)",
                        variable=self.v_verify).pack(side="left")
        ttk.Checkbutton(chk, text="Δt yuvarlama düzeltmesi (ör. 51999.92 → 52000 Hz)",
                        variable=self.v_snap, command=self._refresh_channel_names).pack(side="left", padx=12)
        ttk.Checkbutton(chk, text="Tüm veriyi float64 yaz", variable=self.v_f64).pack(side="left")
        ttk.Checkbutton(chk, text="Var olan TDMS'in üzerine yaz", variable=self.v_over).pack(side="left", padx=12)
        opt.columnconfigure(1, weight=1)

        # ---- main area
        main = ttk.PanedWindow(self.root, orient="horizontal")
        main.pack(fill="both", expand=True, padx=6)

        left = ttk.PanedWindow(main, orient="vertical")
        main.add(left, weight=2)

        # files
        ff = ttk.LabelFrame(left, text="Dosyalar", padding=4)
        self.tv_files = ttk.Treeview(ff, columns=("size", "status"), show="tree headings",
                                     height=5, selectmode="extended")
        self.tv_files.heading("#0", text="Dosya")
        self.tv_files.heading("size", text="Boyut")
        self.tv_files.heading("status", text="Durum")
        self.tv_files.column("#0", width=330)
        self.tv_files.column("size", width=90, anchor="e")
        self.tv_files.column("status", width=200)
        sb = ttk.Scrollbar(ff, orient="vertical", command=self.tv_files.yview)
        self.tv_files.configure(yscrollcommand=sb.set)
        self.tv_files.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tv_files.bind("<<TreeviewSelect>>", self.on_file_select)
        left.add(ff, weight=1)

        # channels
        cf = ttk.LabelFrame(left, text="Kanallar (önizleme için seçin, Ctrl/Shift ile çoklu seçim)", padding=4)
        cols = ("idx", "name", "type", "n", "fs", "dur", "unit", "xunit", "prec")
        heads = ("#", "Kanal adı (TDMS)", "Tip", "Nokta", "fs [Hz]", "Süre", "Birim", "X birim", "Hassasiyet")
        widths = (40, 220, 110, 95, 90, 80, 60, 55, 150)
        self.tv_ch = ttk.Treeview(cf, columns=cols, show="headings", selectmode="extended")
        for c, h, w in zip(cols, heads, widths):
            self.tv_ch.heading(c, text=h)
            self.tv_ch.column(c, width=w, anchor="e" if c in ("idx", "n", "fs", "dur") else "w",
                              stretch=(c == "name"))
        sb2 = ttk.Scrollbar(cf, orient="vertical", command=self.tv_ch.yview)
        sbx = ttk.Scrollbar(cf, orient="horizontal", command=self.tv_ch.xview)
        self.tv_ch.configure(yscrollcommand=sb2.set, xscrollcommand=sbx.set)
        self.tv_ch.grid(row=0, column=0, sticky="nsew")
        sb2.grid(row=0, column=1, sticky="ns")
        sbx.grid(row=1, column=0, sticky="ew")
        cf.rowconfigure(0, weight=1)
        cf.columnconfigure(0, weight=1)
        self.tv_ch.bind("<<TreeviewSelect>>", self.on_channel_select)
        left.add(cf, weight=3)

        # header text
        hf = ttk.LabelFrame(left, text="Başlık bilgileri (seçili kanal)", padding=4)
        self.txt_hdr = tk.Text(hf, height=10, wrap="none", font=("Consolas", 9))
        sb3 = ttk.Scrollbar(hf, orient="vertical", command=self.txt_hdr.yview)
        sb4 = ttk.Scrollbar(hf, orient="horizontal", command=self.txt_hdr.xview)
        self.txt_hdr.configure(yscrollcommand=sb3.set, xscrollcommand=sb4.set)
        self.txt_hdr.grid(row=0, column=0, sticky="nsew")
        sb3.grid(row=0, column=1, sticky="ns")
        sb4.grid(row=1, column=0, sticky="ew")
        hf.rowconfigure(0, weight=1)
        hf.columnconfigure(0, weight=1)
        left.add(hf, weight=2)

        # ---- right: plot
        right = ttk.LabelFrame(main, text="Veri önizleme", padding=4)
        main.add(right, weight=3)
        ctl = ttk.Frame(right)
        ctl.pack(fill="x")
        ttk.Button(ctl, text="Seçili kanalları çiz", command=self.preview_selected).pack(side="left")
        ttk.Button(ctl, text="Görünen aralığı HAM veriyle çiz", command=self.preview_raw_view).pack(side="left", padx=4)
        ttk.Button(ctl, text="Tümünü göster", command=self.preview_selected).pack(side="left")
        self.v_sep = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctl, text="Ayrı eksenler", variable=self.v_sep,
                        command=self.preview_selected).pack(side="left", padx=10)
        self.v_auto = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctl, text="Otomatik çiz", variable=self.v_auto).pack(side="left")

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        tbf = ttk.Frame(right)
        tbf.pack(fill="x")
        NavigationToolbar2Tk(self.canvas, tbf).update()
        self.lbl_stats = tk.Text(right, height=4, wrap="none", font=("Consolas", 9),
                                 background=self.root.cget("background"), relief="flat")
        self.lbl_stats.pack(fill="x", side="bottom")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._draw_empty("Önizleme için soldan bir dosya ve kanal seçin")

        # ---- bottom: convert + progress + log
        bot = ttk.Frame(self.root, padding=6)
        bot.pack(fill="x", side="bottom", before=main)
        self.btn_conv = ttk.Button(bot, text="▶ TÜMÜNÜ DÖNÜŞTÜR", style="Big.TButton",
                                   command=lambda: self.start_convert(False))
        self.btn_conv.pack(side="left")
        self.btn_conv_sel = ttk.Button(bot, text="Seçili dosyaları dönüştür",
                                       command=lambda: self.start_convert(True))
        self.btn_conv_sel.pack(side="left", padx=6)
        self.btn_cancel = ttk.Button(bot, text="İptal", command=self.cancel, state="disabled")
        self.btn_cancel.pack(side="left")
        self.pb = ttk.Progressbar(bot, mode="determinate", maximum=1000)
        self.pb.pack(side="left", fill="x", expand=True, padx=8)
        self.lbl_status = ttk.Label(bot, text="Hazır", width=45)
        self.lbl_status.pack(side="left")

        lf = ttk.LabelFrame(self.root, text="İşlem kaydı", padding=4)
        lf.pack(fill="x", padx=6, pady=(0, 6), side="bottom", before=bot)
        self.txt_log = tk.Text(lf, height=7, wrap="none", font=("Consolas", 9))
        sb5 = ttk.Scrollbar(lf, orient="vertical", command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=sb5.set)
        self.txt_log.pack(side="left", fill="both", expand=True)
        sb5.pack(side="right", fill="y")
        self.txt_log.tag_configure("ok", foreground="#0a7d16")
        self.txt_log.tag_configure("err", foreground="#c00000")
        self.txt_log.tag_configure("warn", foreground="#b36b00")

    # ------------------------------------------------------- thread helpers
    def ui(self, fn, *a, **kw):
        self.q.put((fn, a, kw))

    def _poll(self):
        try:
            while True:
                fn, a, kw = self.q.get_nowait()
                try:
                    fn(*a, **kw)
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    def log(self, text):
        tag = ()
        if "✔" in text or "BAŞARILI" in text:
            tag = ("ok",)
        elif "HATA" in text or "✖" in text:
            tag = ("err",)
        elif "UYARI" in text or "düzeltildi" in text:
            tag = ("warn",)
        self.txt_log.insert("end", text + "\n", tag)
        self.txt_log.see("end")

    def set_status(self, text, frac=None):
        self.lbl_status.configure(text=text[:60])
        if frac is not None:
            self.pb["value"] = max(0, min(1000, frac * 1000))

    # ------------------------------------------------------------- options
    def options(self) -> ConvertOptions:
        naming = next((k for k, v in NAMING_MODES.items() if v == self.v_naming.get()), "id1")
        return ConvertOptions(out_dir=self.v_out.get().strip() or None,
                              group_name=self.v_group.get().strip(), naming=naming,
                              verify=self.v_verify.get(), snap_fs=self.v_snap.get(),
                              force_float64=self.v_f64.get(), overwrite=self.v_over.get())

    def pick_out(self):
        d = filedialog.askdirectory(title="TDMS çıkış klasörü")
        if d:
            self.v_out.set(d)

    # --------------------------------------------------------------- files
    def _add(self, paths):
        added = 0
        for p in paths:
            p = os.path.abspath(p)
            if p in self.files:
                continue
            self.files.append(p)
            self.status_of[p] = "TDMS (yalnızca önizleme)" if p.lower().endswith(".tdms") else "Bekliyor"
            self.tv_files.insert("", "end", iid=p, text=os.path.basename(p),
                                 values=(_fmt_size(os.path.getsize(p)), self.status_of[p]))
            added += 1
        if added and not self.tv_files.selection():
            first = self.files[-added]
            self.tv_files.selection_set(first)
            self.tv_files.focus(first)

    def add_files(self):
        ps = filedialog.askopenfilenames(title="UNV dosyaları seçin",
                                         filetypes=[("Universal File (UNV/UFF)", "*.unv *.uff *.UNV *.UFF"),
                                                    ("Tüm dosyalar", "*.*")])
        self._add(ps)

    def add_folder(self):
        d = filedialog.askdirectory(title="UNV dosyalarının bulunduğu klasör")
        if not d:
            return
        ps = []
        for r, _, fs in os.walk(d):
            ps += [os.path.join(r, f) for f in sorted(fs) if f.lower().endswith((".unv", ".uff"))]
        if not ps:
            messagebox.showinfo(APP_NAME, "Klasörde .unv / .uff dosyası bulunamadı.")
        self._add(ps)

    def add_tdms(self):
        ps = filedialog.askopenfilenames(title="Kontrol için TDMS açın",
                                         filetypes=[("TDMS", "*.tdms"), ("Tüm dosyalar", "*.*")])
        self._add(ps)
        if ps:
            p = os.path.abspath(ps[-1])
            self.tv_files.selection_set(p)

    def _close_source(self, p):
        src = self.sources.pop(p, None)
        if src is not None:
            try:
                src.close()
            except Exception:  # noqa: BLE001
                pass

    def remove_selected(self):
        if self.converting:
            return
        for p in list(self.tv_files.selection()):
            self.files.remove(p)
            self.tv_files.delete(p)
            self._close_source(p)
            if p == self.cur_path:
                self.cur_path = None
                self._clear_channels()

    def clear_files(self):
        if self.converting:
            return
        for p in list(self.files):
            self._close_source(p)
        self.files.clear()
        self.tv_files.delete(*self.tv_files.get_children())
        self.cur_path = None
        self._clear_channels()

    def _set_file_status(self, p, st):
        self.status_of[p] = st
        if self.tv_files.exists(p):
            self.tv_files.set(p, "status", st)

    # ----------------------------------------------------------- channels
    def _clear_channels(self):
        self.tv_ch.delete(*self.tv_ch.get_children())
        self.txt_hdr.delete("1.0", "end")
        self._draw_empty("Önizleme için soldan bir dosya ve kanal seçin")

    def on_file_select(self, _evt=None):
        sel = self.tv_files.selection()
        if not sel:
            return
        p = sel[0]
        if p == self.cur_path:
            return
        self.cur_path = p
        self._clear_channels()
        if p in self.sources:
            self._show_channels(p)
            return
        if p in self.loading:
            return
        self.loading.add(p)
        self.set_status(f"Okunuyor: {os.path.basename(p)}", 0)
        self.txt_hdr.insert("end", "Dosya indeksleniyor, lütfen bekleyin…")

        def work():
            try:
                src = open_source(p, progress_cb=lambda f: self.ui(self.set_status,
                                                                    f"İndeksleniyor: {os.path.basename(p)}", f))
                self.ui(self._loaded, p, src, None)
            except Exception as e:  # noqa: BLE001
                self.ui(self._loaded, p, None, e)
        threading.Thread(target=work, daemon=True).start()

    def _loaded(self, p, src, err):
        self.loading.discard(p)
        if err is not None:
            self.set_status("Okuma hatası", 0)
            self.log(f"✖ HATA ({os.path.basename(p)}): {err}")
            self._set_file_status(p, "✖ Okunamadı")
            if p == self.cur_path:
                self.txt_hdr.delete("1.0", "end")
                self.txt_hdr.insert("end", f"Okuma hatası:\n{err}")
            return
        if p not in self.files:
            src.close()
            return
        self.sources[p] = src
        nch = len(src.channels)
        self.log(f"{os.path.basename(p)}: {nch} kanal okundu"
                 + (f", diğer datasetler: {', '.join(sorted({d['tag'] for d in src.other_datasets}))}"
                    if getattr(src, "other_datasets", None) else ""))
        for w in getattr(src, "warnings", []):
            self.log(f"  UYARI: {w}")
        self.set_status("Hazır", 0)
        if p == self.cur_path:
            self._show_channels(p)

    def _naming(self):
        return next((k for k, v in NAMING_MODES.items() if v == self.v_naming.get()), "id1")

    def _show_channels(self, p):
        src = self.sources[p]
        self.tv_ch.delete(*self.tv_ch.get_children())
        is_unv = not p.lower().endswith(".tdms")
        used = set()
        for ch in src.channels:
            if is_unv:
                nm = channel_name(ch, self._naming())
                base, i = nm, 2
                while nm.lower() in used:
                    nm = f"{base}_{i}"
                    i += 1
                used.add(nm.lower())
                dx, _ = effective_dx(ch, self.v_snap.get())
                fs = f"{1 / dx:,.3f}" if ch.even and dx else "—"
                dur = f"{ch.n * dx:,.3f} {ch.x_unit}" if ch.even else "—"
                typ = ch.type_name + (" (kompleks)" if ch.is_complex else "")
            else:
                nm = ch.name
                fs = f"{ch.fs:,.3f}" if ch.dx else "—"
                dur = f"{ch.duration:,.3f} {ch.x_unit}"
                typ = "TDMS"
            self.tv_ch.insert("", "end", iid=str(ch.index),
                              values=(ch.index + 1, nm, typ, f"{ch.n:,}", fs, dur, ch.y_unit,
                                      ch.x_unit, ch.precision_note))
        kids = self.tv_ch.get_children()
        if kids:
            self.tv_ch.selection_set(kids[0])
            self.tv_ch.focus(kids[0])

    def _refresh_channel_names(self):
        if self.cur_path in self.sources:
            sel = self.tv_ch.selection()
            self._show_channels(self.cur_path)
            if sel:
                self.tv_ch.selection_set(sel)

    def _selected_channels(self):
        src = self.sources.get(self.cur_path)
        if src is None:
            return None, []
        idx = [int(i) for i in self.tv_ch.selection()]
        return src, [src.channels[i] for i in idx]

    def on_channel_select(self, _evt=None):
        src, chans = self._selected_channels()
        if not chans:
            return
        self.txt_hdr.delete("1.0", "end")
        self.txt_hdr.insert("end", chans[0].header_text())
        if self.v_auto.get():
            if self._sel_after:
                self.root.after_cancel(self._sel_after)
            self._sel_after = self.root.after(350, self.preview_selected)

    # ------------------------------------------------------------- preview
    def _draw_empty(self, msg):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw_idle()
        self.lbl_stats.delete("1.0", "end")

    def preview_selected(self):
        src, chans = self._selected_channels()
        if not chans:
            return
        if len(chans) > MAX_PLOT_CHANNELS:
            self.log(f"UYARI: Önizlemede en fazla {MAX_PLOT_CHANNELS} kanal çizilir.")
            chans = chans[:MAX_PLOT_CHANNELS]
        self.preview_cancel.set()
        self.preview_cancel = threading.Event()
        cancel = self.preview_cancel
        self.preview_gen += 1
        gen = self.preview_gen
        snap = self.v_snap.get()
        is_unv = not self.cur_path.lower().endswith(".tdms")
        self.set_status("Önizleme hazırlanıyor…", 0)

        def work():
            results = []
            try:
                for k, ch in enumerate(chans):
                    dx = effective_dx(ch, snap)[0] if is_unv else ch.dx
                    if ch.n <= FULL_RAW_IF_BELOW:
                        x, y = src.read_range(ch, 0, ch.n)
                        if is_unv and ch.even:
                            x = ch.x0 + dx * np.arange(ch.n)
                        yy = np.abs(y) if np.iscomplexobj(y) else y
                        fin = yy[np.isfinite(yy)].astype(np.float64)
                        st = {"n": ch.n, "min": fin.min() if fin.size else np.nan,
                              "max": fin.max() if fin.size else np.nan,
                              "mean": fin.mean() if fin.size else np.nan,
                              "rms": float(np.sqrt(np.mean(fin ** 2))) if fin.size else np.nan,
                              "nonfinite": yy.size - fin.size, "bin": 1}
                        results.append(("raw", ch, x, y, None, st))
                    else:
                        c, mn, mx, st = envelope(
                            src, ch, 3000, cancel,
                            lambda f, k=k: self.ui(self.set_status, f"Önizleme ({k + 1}/{len(chans)})…",
                                                   (k + f) / len(chans)))
                        if is_unv and ch.even and dx != ch.dx:
                            c = ch.x0 + (c - ch.x0) * (dx / ch.dx)
                        results.append(("env", ch, c, mn, mx, st))
                self.ui(self._plot_results, gen, results, "tam sinyal")
            except Cancelled:
                pass
            except Exception as e:  # noqa: BLE001
                self.ui(self.log, f"✖ HATA önizleme: {e}")
                self.ui(self.set_status, "Önizleme hatası", 0)
        threading.Thread(target=work, daemon=True).start()

    def preview_raw_view(self):
        src, chans = self._selected_channels()
        if not chans or not self.fig.axes:
            return
        chans = chans[:MAX_PLOT_CHANNELS]
        x0, x1 = self.fig.axes[0].get_xlim()
        snap = self.v_snap.get()
        is_unv = not self.cur_path.lower().endswith(".tdms")
        self.preview_gen += 1
        gen = self.preview_gen
        self.set_status("Ham veri okunuyor…", 0)

        def work():
            results = []
            try:
                for ch in chans:
                    dx = effective_dx(ch, snap)[0] if is_unv else ch.dx
                    if ch.even:
                        i0 = int(np.floor((x0 - ch.x0) / dx))
                        i1 = int(np.ceil((x1 - ch.x0) / dx)) + 1
                    else:
                        i0, i1 = 0, ch.n
                    i0 = max(0, i0)
                    i1 = min(ch.n, i1)
                    cnt = i1 - i0
                    if cnt <= 0:
                        continue
                    if cnt > RAW_PLOT_LIMIT:
                        self.ui(messagebox.showinfo, APP_NAME,
                                f"Görünen aralıkta {cnt:,} nokta var. Ham çizim için en fazla "
                                f"{RAW_PLOT_LIMIT:,} nokta – lütfen yakınlaştırın (büyüteç aracı).")
                        self.ui(self.set_status, "Hazır", 0)
                        return
                    x, y = src.read_range(ch, i0, cnt)
                    if is_unv and ch.even:
                        x = ch.x0 + dx * np.arange(i0, i0 + cnt)
                    yy = np.abs(y) if np.iscomplexobj(y) else y
                    fin = yy[np.isfinite(yy)].astype(np.float64)
                    st = {"n": cnt, "min": fin.min() if fin.size else np.nan,
                          "max": fin.max() if fin.size else np.nan,
                          "mean": fin.mean() if fin.size else np.nan,
                          "rms": float(np.sqrt(np.mean(fin ** 2))) if fin.size else np.nan,
                          "nonfinite": yy.size - fin.size, "bin": 1, "range": (i0, i1)}
                    results.append(("raw", ch, x, y, None, st))
                self.ui(self._plot_results, gen, results, "ham veri – görünen aralık", (x0, x1))
            except Exception as e:  # noqa: BLE001
                self.ui(self.log, f"✖ HATA ham önizleme: {e}")
        threading.Thread(target=work, daemon=True).start()

    def _plot_results(self, gen, results, mode, xlim=None):
        if gen != self.preview_gen or not results:
            return
        self.fig.clear()
        sep = self.v_sep.get() and len(results) > 1
        axes = []
        if sep:
            for i in range(len(results)):
                axes.append(self.fig.add_subplot(len(results), 1, i + 1,
                                                 sharex=axes[0] if axes else None))
        else:
            axes = [self.fig.add_subplot(111)] * len(results)
        lines = []
        for i, (kind, ch, x, a, b, st) in enumerate(results):
            ax = axes[i]
            col = COLORS[i % len(COLORS)]
            label = f"{ch.name} [{ch.y_unit}]"
            if kind == "env":
                ax.fill_between(x, a, b, color=col, alpha=0.55, linewidth=0, label=label)
                ax.plot(x, a, color=col, linewidth=0.4)
                ax.plot(x, b, color=col, linewidth=0.4)
            else:
                y = a
                if np.iscomplexobj(y):
                    ax.plot(x, y.real, color=col, linewidth=0.7, label=label + " Re")
                    ax.plot(x, y.imag, color=col, linewidth=0.7, linestyle="--", label=label + " Im")
                else:
                    ax.plot(x, y, color=col, linewidth=0.7, label=label,
                            marker="." if len(y) < 2000 else None, markersize=3)
            ax.grid(True, alpha=0.3)
            if sep or i == 0:
                ax.set_ylabel(ch.y_unit or "")
            if sep or len(results) == 1:
                ax.legend(loc="upper right", fontsize=8)
            lines.append(
                f"{ch.name[:24]:<24} n={st['n']:,}  min={st['min']:.6g}  max={st['max']:.6g}  "
                f"ort={st['mean']:.6g}  rms={st['rms']:.6g}"
                + (f" NaN/Inf={st['nonfinite']}" if st.get("nonfinite") else "")
                + (f"  [örnek {st['range'][0]:,}–{st['range'][1]:,}]" if "range" in st else ""))
        if not sep and len(results) > 1:
            axes[0].legend(loc="upper right", fontsize=8)
        ch0 = results[0][1]
        axes[-1].set_xlabel(f"{ch0.x_label or 'X'} [{ch0.x_unit}]")
        info = "min/max zarf" if any(r[0] == "env" for r in results) else "ham örnekler"
        self.fig.suptitle(f"{os.path.basename(self.cur_path)} – {mode} ({info})", fontsize=9)
        if xlim:
            axes[0].set_xlim(*xlim)
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.lbl_stats.delete("1.0", "end")
        self.lbl_stats.insert("end", "\n".join(lines))
        self.set_status("Önizleme hazır", 1)

    # ------------------------------------------------------------- convert
    def start_convert(self, selected_only):
        if self.converting:
            return
        if selected_only:
            files = [p for p in self.tv_files.selection()]
        else:
            files = list(self.files)
        files = [p for p in files if not p.lower().endswith(".tdms")]
        if not files:
            messagebox.showinfo(APP_NAME, "Dönüştürülecek UNV dosyası yok. Önce dosya ekleyin.")
            return
        opts = self.options()
        if opts.out_dir and not os.path.isdir(opts.out_dir):
            try:
                os.makedirs(opts.out_dir)
            except OSError as e:
                messagebox.showerror(APP_NAME, f"Çıkış klasörü oluşturulamadı:\n{e}")
                return
        self.converting = True
        self.cancel_ev = threading.Event()
        self.btn_conv.configure(state="disabled")
        self.btn_conv_sel.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        cancel = self.cancel_ev

        def work():
            ok, fail, outs = 0, 0, []
            for i, p in enumerate(files):
                if cancel.is_set():
                    break
                self.ui(self._set_file_status, p, "⏳ Dönüştürülüyor…")
                try:
                    res = convert_file(
                        p, opts,
                        log=lambda t: self.ui(self.log, t),
                        progress=lambda f, s, i=i: self.ui(
                            self.set_status, f"[{i + 1}/{len(files)}] {s}", (i + f) / len(files)),
                        cancel=cancel)
                    ok += 1
                    outs.append(res["output"])
                    self.ui(self._set_file_status, p,
                            "✔ Tamam (doğrulandı)" if res["verified"] else "✔ Tamam")
                except Cancelled:
                    self.ui(self.log, "İşlem kullanıcı tarafından iptal edildi.")
                    self.ui(self._set_file_status, p, "İptal edildi")
                    break
                except Exception as e:  # noqa: BLE001
                    fail += 1
                    self.ui(self.log, f"✖ HATA ({os.path.basename(p)}): {e}")
                    self.ui(self.log, traceback.format_exc().strip().splitlines()[-1])
                    self.ui(self._set_file_status, p, "✖ Hata")
            self.ui(self._convert_done, ok, fail, outs, cancel.is_set())
        threading.Thread(target=work, daemon=True).start()

    def _convert_done(self, ok, fail, outs, cancelled):
        self.converting = False
        self.btn_conv.configure(state="normal")
        self.btn_conv_sel.configure(state="normal")
        self.btn_cancel.configure(state="disabled")
        self.set_status("Bitti" if not cancelled else "İptal edildi", 1 if not cancelled else 0)
        msg = f"{ok} dosya dönüştürüldü" + (f", {fail} dosyada hata" if fail else "") + "."
        self.log(("✔ " if not fail else "") + msg)
        if outs:
            msg += "\n\nÇıktı(lar):\n" + "\n".join(outs[:10]) + ("\n…" if len(outs) > 10 else "")
            msg += "\n\nİpucu: TDMS dosyasını 'TDMS Aç' ile listeye ekleyip önizleyebilirsiniz."
        (messagebox.showwarning if fail else messagebox.showinfo)(APP_NAME, msg)

    def cancel(self):
        self.cancel_ev.set()
        self.set_status("İptal ediliyor…")

    # --------------------------------------------------------------- misc
    def save_log(self):
        p = filedialog.asksaveasfilename(title="Logu kaydet", defaultextension=".txt",
                                         filetypes=[("Metin", "*.txt")],
                                         initialfile="UNV2TDMS_log.txt")
        if p:
            with open(p, "w", encoding="utf-8") as f:
                f.write(self.txt_log.get("1.0", "end"))

    def _on_close(self):
        if self.converting and not messagebox.askyesno(
                APP_NAME, "Dönüştürme devam ediyor. Çıkmak istiyor musunuz?"):
            return
        self.cancel_ev.set()
        self.preview_cancel.set()
        for p in list(self.sources):
            self._close_source(p)
        self.root.destroy()


def main():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:  # noqa: BLE001
            pass
    root = tk.Tk()
    app = App(root)
    # Dosyalar exe simgesinin üzerine sürüklenip bırakılırsa listeye ekle
    args = [a for a in sys.argv[1:] if os.path.isfile(a)]
    if args:
        root.after(200, lambda: app._add(args))
    root.mainloop()


if __name__ == "__main__":
    main()
