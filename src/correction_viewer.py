"""
Shared breach-finder viewer infrastructure
===========================================

Contains:
- Orientation helpers & FreeSurfer LUT loader
- Common constants (colormaps, axis names, dims order)
- ``BreachFinderControls`` — reusable right-panel Qt widget
- ``BreachViewerMixin`` — all breach navigation / overlay logic

Both ``widget.BreachFinderWidget`` (grid mode) and
``widgets.multiple_viewer_widget.BreachFinderMultiViewerWidget``
(multi-viewer mode) inherit from the mixin and only implement the
viewer-specific hooks.
"""

from __future__ import annotations
from utils import load_freesurfer_colormap, orient_for_display, ellipse_bbox_3d, orient_for_save
from data.constants import make_breach_colormap
from numpy.typing import NDArray
from widgets.controls import BreachFinderControls

from pathlib import Path
from typing import Any, TYPE_CHECKING, cast

import numpy as np
import nibabel as nib
from nibabel.dataobj_images import DataobjImage

from breachfinder import detect_breaches, extract_slice, propose_fix

if TYPE_CHECKING:
    from napari.utils.colormaps import DirectLabelColormap

class BreachViewerMixin:
    """
    Mixin providing breach detection / overlay / navigation state.

    The host widget must set these attributes before calling ``_init_breach_state``:
        viewer, seg_path, label_values, axis

    And must expose:
        breach_layer, breach_vol, fix_layer, fix_vol, shapes_layer,
        seg_layer, seg_data, t2_data, _t2_raw, _seg_raw, seg_img, fs_cmap,
        controls  (BreachFinderControls instance)

    The host widget must implement:
        _navigate_to_slice(idx)
        _reorient_volumes()
        _apply_view_for_axis()
    """

    # Declared for type checkers — set by subclasses / _init_breach_state
    seg_path: str | Path
    label_values: tuple[int, int]
    axis: int
    result: dict | None
    current_slice: int | None
    _t2_raw: NDArray
    _seg_raw: NDArray
    seg_img: Any
    fs_cmap: DirectLabelColormap
    breach_cmap: DirectLabelColormap
    t2_data: NDArray
    seg_data: NDArray
    controls: BreachFinderControls
    breach_vol: NDArray
    breach_layer: Any
    fix_vol: NDArray
    fix_layer: Any
    shapes_layer: Any
    seg_layer: Any

    def _init_breach_state(
        self,
        t2_path: str | Path,
        seg_path: str | Path,
        lut_path: str | Path,
        label_values: tuple[int, int],
        axis: int,
    ):
        """Load NIfTI data and orient for display. Call from __init__."""
        self.seg_path = seg_path
        self.label_values = label_values
        self.axis = axis
        self.result = None
        self.current_slice = None

        self._t2_seg = cast(DataobjImage, nib.load(t2_path))
        self._t2_raw = self._t2_seg.get_fdata()
        self.seg_img = cast(DataobjImage, nib.load(seg_path))
        self._seg_raw = self.seg_img.get_fdata()
        self.fs_cmap = load_freesurfer_colormap(lut_path)
        self.breach_cmap = make_breach_colormap()

        self.t2_data = orient_for_display(self._t2_raw, self.axis)
        self.seg_data = orient_for_display(self._seg_raw, self.axis)

    # Abstract hooks — must be implemented by the host widget
    def _navigate_to_slice(self, idx: int) -> None: ...
    def _reorient_volumes(self) -> None: ...
    def _apply_view_for_axis(self) -> None: ...

    def _bind_shortcuts(self, viewer):
        @viewer.bind_key("n")
        def _n(v):
            self._on_next()

        @viewer.bind_key("r")
        def _r(v):
            self._on_recheck()

        @viewer.bind_key("a")
        def _a(v):
            self._on_apply()

    # ── Status / Overlays ─────────────────────────────────────────

    def _set_status(self, text, color="#0f0"):
        self.controls.set_status(text, color)

    def _clear_overlays(self):
        self.breach_vol[:] = 0
        self.breach_layer.data = self.breach_vol.copy()
        self.fix_vol[:] = 0
        self.fix_layer.data = self.fix_vol.copy()
        self.shapes_layer.data = []

    def _write_breach_to_volume(self, result, idx):
        self.breach_vol[:] = 0
        overlay = np.zeros_like(result["breach_mask"], dtype=int)
        overlay[result["cp_mask"] == 1] = 2
        if self.controls.wp_check.isChecked():
            overlay[result["weakpoint_mask"] == 1] = 3
        overlay[result["breach_mask"] == 1] = 1
        if self.axis == 0:
            self.breach_vol[idx, :, :] = overlay
        elif self.axis == 1:
            self.breach_vol[:, idx, :] = overlay
        else:
            self.breach_vol[:, :, idx] = overlay
        self.breach_layer.data = self.breach_vol.copy()

    def _write_fix_to_volume(self, fixed_seg, idx):
        self.fix_vol[:] = 0
        if self.axis == 0:
            self.fix_vol[idx, :, :] = fixed_seg.astype(int)
        elif self.axis == 1:
            self.fix_vol[:, idx, :] = fixed_seg.astype(int)
        else:
            self.fix_vol[:, :, idx] = fixed_seg.astype(int)
        self.fix_layer.data = self.fix_vol.copy()

    def _update_highlights(self, result):
        if not self.controls.hl_check.isChecked() or result is None:
            self.shapes_layer.data = []
            return

        all_ellipses = []
        edge_colors = []

        for c, r in zip(result["holes"]["centroids"], result["holes"]["radii"]):
            all_ellipses.append(ellipse_bbox_3d(c, r, self.current_slice, self.axis))
            edge_colors.append("red")

        if self.controls.wp_check.isChecked():
            for c, r in zip(
                result["weakpoints"]["centroids"], result["weakpoints"]["radii"]
            ):
                all_ellipses.append(
                    ellipse_bbox_3d(c, r, self.current_slice, self.axis)
                )
                edge_colors.append("yellow")

        if all_ellipses:
            self.shapes_layer.data = all_ellipses
            self.shapes_layer.shape_type = ["ellipse"] * len(all_ellipses)
            self.shapes_layer.edge_color = edge_colors
        else:
            self.shapes_layer.data = []

    # ── Navigation ────────────────────────────────────────────────

    def _advance(self, start_from=0):
        n = self.seg_data.shape[self.axis]
        for i in range(start_from, n):
            seg = extract_slice(self.seg_data, i, self.axis)
            res = detect_breaches(
                seg, self.label_values,
                detect_weakpoints=self.controls.wp_check.isChecked(),
            )
            if res is not None:
                self.current_slice = i
                self.result = res
                self._show_breach()
                return
        self.result = None
        self._clear_overlays()
        self._set_status(f"No breaches from slice {start_from} onwards", "#ff0")

    def _show_breach(self):
        idx = self.current_slice
        res = self.result
        assert res is not None and idx is not None

        self._write_breach_to_volume(res, idx)
        self._update_highlights(res)

        seg = extract_slice(self.seg_data, idx, self.axis)
        fixed_seg = propose_fix(seg, res["breach_mask"])
        self._write_fix_to_volume(fixed_seg, idx)

        self._navigate_to_slice(idx)
        n = res["holes"]["count"]
        nw = res["weakpoints"]["count"]
        self._set_status(
            f"Slice {idx} — {n} BREACH{'ES' if n != 1 else ''} and "
            f"{nw} WEAKPOINT{'S' if nw != 1 else ''} detected",
            "#f44",
        )

    # ── Callbacks ─────────────────────────────────────────────────

    def _on_next(self):
        self._advance(0 if self.current_slice is None else self.current_slice + 1)

    def _on_recheck(self):
        if self.current_slice is None:
            return
        self.seg_img = cast(DataobjImage, nib.load(self.seg_path))
        self._seg_raw = self.seg_img.get_fdata()
        self.seg_data = orient_for_display(self._seg_raw, self.axis)
        self.seg_layer.data = self.seg_data.astype(int)

        seg = extract_slice(self.seg_data, self.current_slice, self.axis)
        res = detect_breaches(
            seg, self.label_values,
            detect_weakpoints=self.controls.wp_check.isChecked(),
        )
        if res is None:
            self.result = None
            self._clear_overlays()
            self._set_status(f"✓ Slice {self.current_slice} FIXED!", "#0f0")
        else:
            self.result = res
            self._show_breach()

    def _on_apply(self):
        if self.result is None or "breach_mask" not in self.result:
            self._set_status("Nothing to apply", "#ff0")
            return

        idx = self.current_slice
        seg = extract_slice(self.seg_data, idx, self.axis)
        fixed = propose_fix(seg, self.result["breach_mask"])

        if self.axis == 0:
            self.seg_data[idx, :, :] = fixed
        elif self.axis == 1:
            self.seg_data[:, idx, :] = fixed
        else:
            self.seg_data[:, :, idx] = fixed

        self._seg_raw = orient_for_save(self.seg_data, self.axis)

        try:
            out = nib.Nifti1Image(
                self._seg_raw, self.seg_img.affine, self.seg_img.header
            )
            nib.save(out, self.seg_path)
            self.seg_layer.data = self.seg_data.astype(int)
            self._set_status(f"✓ Fix applied & saved — Slice {idx}", "#0f0")
            self._on_recheck()
        except Exception as e:
            self._set_status(f"✗ Error saving: {e}", "#f44")

    def _on_toggle_hl(self):
        if self.result is not None:
            self._update_highlights(self.result)

    def _on_axis_changed(self, new_axis):
        self.axis = new_axis
        self._clear_overlays()
        self.result = None
        self.current_slice = None
        self._reorient_volumes()
        self._apply_view_for_axis()
        self._advance(0)

    def _connect_controls(self, controls: BreachFinderControls):
        """Wire a BreachFinderControls instance to this viewer's callbacks."""
        self.controls = controls
        controls.next_requested.connect(self._on_next)
        controls.recheck_requested.connect(self._on_recheck)
        controls.apply_requested.connect(self._on_apply)
        controls.axis_changed.connect(self._on_axis_changed)
        controls.highlight_toggled.connect(self._on_toggle_hl)
