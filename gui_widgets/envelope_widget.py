"""PyQtGraph display widget for TRDSP envelope/kurtogram states.

This module is intentionally GUI-only.  It renders state dictionaries produced
by the main PlotWorker and does not call the envelope/kurtogram backend.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from PyQt5.QtCore import Qt, QRectF, pyqtSignal
from PyQt5.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

try:
    import pyqtgraph as pg
except Exception as exc:  # pragma: no cover - app already requires pyqtgraph for plotting
    pg = None
    _PG_IMPORT_ERROR = exc
else:
    _PG_IMPORT_ERROR = None


class EnvelopeKurtogramWidget(QWidget):
    """Render precomputed envelope-spectrum and kurtogram state dictionaries."""

    clicked = pyqtSignal()
    color_levels_changed = pyqtSignal(float, float)
    color_levels_change_finished = pyqtSignal(float, float)

    def __init__(self, mini: bool = False, parent=None):
        super().__init__(parent)
        if pg is None:
            raise RuntimeError(f"pyqtgraph is required for EnvelopeKurtogramWidget: {_PG_IMPORT_ERROR}")

        self.mini = bool(mini)
        self.plot_state: dict[str, Any] | None = None
        self.plot_key: str | None = None
        self._rect_zoom_active = False
        self._home_ranges: dict[str, tuple[float, float, float, float]] = {}
        self._color_bar = None
        self._image_item = None

        self.title = QLabel("")
        self.title.setWordWrap(True)
        self.title.setStyleSheet("font-weight: bold;" if not self.mini else "font-weight: bold; font-size: 9pt;")

        self.kurt_plot = pg.PlotWidget(background="w")
        self.kurt_plot.setMenuEnabled(False)
        self.kurt_item = self.kurt_plot.getPlotItem()
        self.kurt_item.showGrid(x=True, y=True, alpha=0.2)
        self.kurt_item.setLabel("bottom", "Band center frequency [Hz]")
        self.kurt_item.setLabel("left", "Kurtogram level")

        self.env_plot = pg.PlotWidget(background="w")
        self.env_plot.setMenuEnabled(False)
        self.env_item = self.env_plot.getPlotItem()
        self.env_item.showGrid(x=True, y=True, alpha=0.25)
        self.env_item.setLabel("bottom", "Frequency [Hz]")
        self.env_item.setLabel("left", "Envelope amplitude [peak]")

        for pw in (self.kurt_plot, self.env_plot):
            pw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            vb = pw.getPlotItem().getViewBox()
            vb.setMouseMode(pg.ViewBox.RectMode if self._rect_zoom_active else pg.ViewBox.PanMode)
            pw.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self.title)
        lay.addWidget(self.kurt_plot, 3)
        lay.addWidget(self.env_plot, 2)

    def _on_mouse_clicked(self, event):
        try:
            if event.button() == Qt.LeftButton:
                self.clicked.emit()
        except Exception:
            self.clicked.emit()

    def clear_plot(self):
        self.plot_state = None
        self.plot_key = None
        self._home_ranges.clear()
        self.kurt_item.clear()
        self.env_item.clear()
        self.title.setText("")
        self._color_bar = None
        self._image_item = None
        self.kurt_plot.setVisible(True)

    def set_rect_zoom_active(self, active: bool):
        self._rect_zoom_active = bool(active)
        for pw in (self.kurt_plot, self.env_plot):
            try:
                pw.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode if active else pg.ViewBox.PanMode)
            except Exception:
                pass

    def zoom_axes(self, factor: float):
        factor = float(factor)
        for pw in (self.kurt_plot, self.env_plot):
            try:
                pw.getPlotItem().getViewBox().scaleBy((factor, factor))
            except Exception:
                pass

    def reset_view(self):
        for name, pw in (("kurt", self.kurt_plot), ("env", self.env_plot)):
            if name not in self._home_ranges:
                continue
            x0, x1, y0, y1 = self._home_ranges[name]
            try:
                vb = pw.getPlotItem().getViewBox()
                vb.setXRange(x0, x1, padding=0.0)
                vb.setYRange(y0, y1, padding=0.0)
            except Exception:
                pass

    def get_view_range(self):
        try:
            xr, yr = self.env_plot.getPlotItem().getViewBox().viewRange()
            return (float(xr[0]), float(xr[1])), (float(yr[0]), float(yr[1]))
        except Exception:
            return None

    def get_export_annotations(self):
        return []

    def _pg_colormap(self, color_cfg: dict | None):
        color_cfg = color_cfg or {}
        name = str(color_cfg.get("cmap", "viridis") or "viridis")
        if name == "jet":
            name = "CET-R4"
        try:
            cm = pg.colormap.get(name)
        except Exception:
            cm = pg.colormap.get("viridis")
        if bool(color_cfg.get("reverse", False)):
            try:
                cm = cm.reverse()
            except Exception:
                pass
        return cm

    @staticmethod
    def _finite_range(values: np.ndarray, default=(0.0, 1.0)):
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return default
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if hi <= lo:
            span = max(abs(lo), 1.0) * 0.05
            lo -= span
            hi += span
        return lo, hi

    def set_state(self, state: dict, color_cfg=None, preserve_view=False, preserve_annotations=False):
        old_ranges = {}
        if preserve_view:
            for name, pw in (("kurt", self.kurt_plot), ("env", self.env_plot)):
                try:
                    xr, yr = pw.getPlotItem().getViewBox().viewRange()
                    old_ranges[name] = (float(xr[0]), float(xr[1]), float(yr[0]), float(yr[1]))
                except Exception:
                    pass

        self.clear_plot()
        self.plot_state = state
        self.plot_key = state.get("key")
        self.title.setText(str(state.get("title") or state.get("short_title") or self.plot_key or ""))

        kind = str(state.get("kind", "")).lower()
        if kind == "kurtogram":
            self.kurt_plot.setVisible(True)
            self._draw_kurtogram(state, color_cfg or {})
            self._draw_envelope(state, envelope_prefix=True)
        else:
            self.kurt_plot.setVisible(False)
            self._draw_envelope(state, envelope_prefix=False)

        if preserve_view and old_ranges:
            self._home_ranges.update(old_ranges)
            for name, pw in (("kurt", self.kurt_plot), ("env", self.env_plot)):
                if name in old_ranges:
                    x0, x1, y0, y1 = old_ranges[name]
                    try:
                        vb = pw.getPlotItem().getViewBox()
                        vb.setXRange(x0, x1, padding=0.0)
                        vb.setYRange(y0, y1, padding=0.0)
                    except Exception:
                        pass
        else:
            self.reset_view()

    def _draw_kurtogram(self, state: dict, color_cfg: dict):
        K = np.asarray(state.get("kurtosis_grid", []), dtype=np.float64)
        centers = np.asarray(state.get("center_freq_grid", []), dtype=np.float64)
        if K.size == 0:
            self._home_ranges["kurt"] = (0.0, 1.0, 0.0, 1.0)
            return

        plot_K = np.nan_to_num(K, nan=np.nanmin(K[np.isfinite(K)]) if np.any(np.isfinite(K)) else 0.0)
        finite_centers = centers[np.isfinite(centers)]
        if finite_centers.size:
            x0, x1 = self._finite_range(finite_centers)
        else:
            x0, x1 = 0.0, float(state.get("fs", 2.0)) * 0.5
        y0, y1 = -0.5, float(K.shape[0]) - 0.5

        image = pg.ImageItem(axisOrder="row-major")
        image.setImage(plot_K, autoLevels=False)
        image.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        finite_K = K[np.isfinite(K)]
        kmin, kmax = self._finite_range(finite_K)
        if color_cfg.get("manual_clim", False):
            kmin = float(color_cfg.get("vmin", kmin))
            kmax = float(color_cfg.get("vmax", kmax))
            if kmax <= kmin:
                kmax = kmin + 1.0
        image.setLevels((kmin, kmax))
        cm = self._pg_colormap(color_cfg)
        image.setColorMap(cm)
        self.kurt_item.addItem(image)
        self._image_item = image

        try:
            cbar = pg.ColorBarItem(values=(kmin, kmax), colorMap=cm, label=state.get("colorbar_label", "Excess kurtosis"), width=12 if self.mini else 18)
            cbar.setImageItem(image, insert_in=self.kurt_item)
            self._color_bar = cbar
        except Exception:
            self._color_bar = None

        best = state.get("best_band") or {}
        try:
            f_low = float(best["f_low"])
            f_high = float(best["f_high"])
            level = float(best.get("level", 0))
            center = 0.5 * (f_low + f_high)
            self.kurt_item.addItem(pg.InfiniteLine(center, angle=90, pen=pg.mkPen("r", width=1.5)))
            self.kurt_item.addItem(pg.InfiniteLine(level, angle=0, pen=pg.mkPen("r", width=1.0, style=Qt.DashLine)))
            band = pg.LinearRegionItem(values=(f_low, f_high), orientation=pg.LinearRegionItem.Vertical, movable=False,
                                       brush=pg.mkBrush(255, 0, 0, 40), pen=pg.mkPen(255, 0, 0, 90))
            self.kurt_item.addItem(band)
            label = pg.TextItem(f"Best: {f_low:.0f}-{f_high:.0f} Hz\nK={float(best.get('kurtosis', 0.0)):.2f}", color="r", anchor=(0, 1))
            label.setPos(center, min(y1, level + 0.4))
            self.kurt_item.addItem(label)
        except Exception:
            pass

        self.kurt_item.setLabel("bottom", "Band center frequency [Hz]")
        self.kurt_item.setLabel("left", "Kurtogram level")
        self._home_ranges["kurt"] = (x0, x1, y0, y1)

    def _draw_envelope(self, state: dict, envelope_prefix: bool):
        if envelope_prefix:
            freqs = np.asarray(state.get("envelope_freqs", []), dtype=np.float64)
            spec = np.asarray(state.get("envelope_spectrum", []), dtype=np.float64)
        else:
            freqs = np.asarray(state.get("freqs", state.get("envelope_freqs", [])), dtype=np.float64)
            spec = np.asarray(state.get("spectrum", state.get("envelope_spectrum", [])), dtype=np.float64)

        self.env_item.setLabel("bottom", state.get("xlabel", "Frequency [Hz]"))
        self.env_item.setLabel("left", state.get("ylabel", "Envelope amplitude [peak]"))

        if freqs.size and spec.size:
            if self.mini and freqs.size > 3500:
                step = max(1, freqs.size // 3500)
                freqs_plot = freqs[::step]
                spec_plot = spec[::step]
            else:
                freqs_plot = freqs
                spec_plot = spec
            self.env_item.plot(freqs_plot, spec_plot, pen=pg.mkPen("#1f77b4", width=1.1), connect="finite")
            x0, x1 = self._finite_range(freqs)
            y0, y1 = self._finite_range(spec, default=(0.0, 1.0))
            y0 = min(0.0, y0)
        else:
            x0, x1, y0, y1 = 0.0, 1.0, 0.0, 1.0

        band = state.get("band")
        if band is None and isinstance(state.get("best_band"), dict):
            best = state["best_band"]
            band = (best.get("f_low"), best.get("f_high"))
        if band is not None:
            try:
                f_low, f_high = float(band[0]), float(band[1])
                self.env_item.setTitle(f"Envelope spectrum | demodulation band {f_low:.3g}-{f_high:.3g} Hz")
            except Exception:
                self.env_item.setTitle("Envelope spectrum")
        else:
            self.env_item.setTitle("Envelope spectrum")

        bearing = state.get("bearing_freqs") or {}
        ymax = y1
        for label, freq in bearing.items():
            if label == "shaft_hz":
                continue
            try:
                f = float(freq)
                if f <= 0.0 or not np.isfinite(f):
                    continue
                line = pg.InfiniteLine(f, angle=90, pen=pg.mkPen("#d71920", width=1.0, style=Qt.DashLine))
                self.env_item.addItem(line)
                txt = pg.TextItem(str(label), color="#d71920", anchor=(0, 1))
                txt.setPos(f, ymax)
                self.env_item.addItem(txt)
            except Exception:
                pass

        self._home_ranges["env"] = (x0, x1, y0, y1)
