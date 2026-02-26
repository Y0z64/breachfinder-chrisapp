#!/usr/bin/env python
"""
Interactive cortical plate breach viewer — napari GUI (3-panel grid mode).

Layout (matching the matplotlib version):
    Panel 1 │ Panel 2              │ Panel 3
    T2 only │ T2 + Seg + Breaches  │ T2 + Proposed Fix

All detection and repair logic lives in ``breachfinder.py``.

Orientation note

The matplotlib viewer displays slices with ``.T`` + ``origin="lower"``, which
is equivalent to a 90° counter-clockwise rotation of the raw array slice.
Napari shows 3-D arrays as-is, so we rotate the volumes on load and undo
the rotation before saving back to NIfTI.
"""

import os
import numpy as np
import nibabel as nib
import napari
from napari.utils.colormaps import DirectLabelColormap
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QCheckBox,
)
from qtpy.QtCore import Qt

from src.breachfinder import (
    detect_breaches,
    propose_fix,
    extract_slice,
    scan_volume,
)

from src.data.constants import FREESURFER_LUT, LEFT_CP, RIGHT_CP

def load_freesurfer_colormap(lut_path):
    color_dict = {0: np.array([0, 0, 0, 0], dtype=np.float32)}
    with open(lut_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    idx = int(parts[0])
                    color_dict[idx] = np.array(
                        [
                            int(parts[2]) / 255,
                            int(parts[3]) / 255,
                            int(parts[4]) / 255,
                            1.0,
                        ],
                        dtype=np.float32,
                    )
                except (ValueError, IndexError):
                    continue
    return DirectLabelColormap(color_dict=color_dict)


#  orientation transforms

# For each scrolling axis, these are the two displayed axes that need rotating.
_ROT_PLANES = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


def _orient_for_display(data, axis):
    """Rotate a 3-D volume 90° CCW in the display plane of *axis*.
    This matches what matplotlib's ``.T`` + ``origin='lower'`` does."""
    return np.rot90(data, k=1, axes=_ROT_PLANES[axis])


def _orient_for_save(data, axis):
    """Undo ``_orient_for_display`` so we can write back to NIfTI."""
    return np.rot90(data, k=-1, axes=_ROT_PLANES[axis])


#  ellipse helper


def _ellipse_bbox_3d(center_rc, radius, slice_idx, axis):
    """4×3 bounding box for a napari 3-D ellipse in *display* volume coords."""
    r, c = center_rc
    if axis == 0:
        vol = [float(slice_idx), r, c]
        rd, cd = 1, 2
    elif axis == 1:
        vol = [r, float(slice_idx), c]
        rd, cd = 0, 2
    else:
        vol = [r, c, float(slice_idx)]
        rd, cd = 0, 1
    corners = []
    for dr, dc in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
        pt = list(vol)
        pt[rd] += dr * radius
        pt[cd] += dc * radius
        corners.append(pt)
    return np.array(corners, dtype=float)


#  dock widget


class BreachFinderWidget(QWidget):
    AXIS_NAMES = {0: "Sagittal", 1: "Coronal", 2: "Axial"}

    # dims.order: the FIRST element is the slider (scrolled) axis,
    # remaining two are the displayed plane.
    DIMS_ORDER = {
        0: (0, 1, 2),
        1: (1, 0, 2),
        2: (2, 0, 1),
    }

    # Grid layout: 3 panels × STRIDE layers each.
    #
    # Layers are added bottom-to-top in the layer list, and grid mode
    # (napari ≥0.6.2) distributes them left-to-right in stride-sized groups.
    #
    #   Panel 1 (T2 only):
    #     0. t2_panel1           (image)
    #     1. _pad1a              (labels – invisible padding)
    #     2. _pad1b              (labels – invisible padding)
    #     3. _pad1c              (labels – invisible padding)
    #
    #   Panel 2 (T2 + Seg + Breaches + Highlights):
    #     4. t2_panel2           (image)
    #     5. seg_layer           (labels – segmentation)
    #     6. breach_layer        (labels – breach + CP)
    #     7. shapes_layer        (shapes – highlight ellipses)
    #
    #   Panel 3 (T2 + Proposed Fix):
    #     8. t2_panel3           (image)
    #     9. fix_layer           (labels – proposed fix)
    #    10. _pad3a              (labels – invisible padding)
    #    11. _pad3b              (labels – invisible padding)
    GRID_STRIDE = 4

    def __init__(
        self,
        viewer,
        t2_path,
        seg_path,
        lut_path,
        label_values=(LEFT_CP, RIGHT_CP),
        axis=0,
    ):
        super().__init__()
        self.viewer = viewer
        self.seg_path = seg_path
        self.label_values = label_values
        self.axis = axis
        self.result = None
        self.current_slice = None

        #  load raw data (disk orientation) 
        self._t2_raw = nib.load(t2_path).get_fdata()
        self.seg_img = nib.load(seg_path)
        self._seg_raw = self.seg_img.get_fdata()
        self.fs_cmap = load_freesurfer_colormap(lut_path)

        #  rotate for display 
        self.t2_data = _orient_for_display(self._t2_raw, self.axis)
        self.seg_data = _orient_for_display(self._seg_raw, self.axis)

        #  build colormaps 
        breach_cmap = DirectLabelColormap(
            color_dict={
                0: np.array([0, 0, 0, 0], dtype=np.float32),
                1: np.array([1, 0, 0, 0.85], dtype=np.float32),  # breach = red
                2: np.array([0, 1, 0, 0.6], dtype=np.float32),  # cp = green
            }
        )
        empty_vol = np.zeros(self.seg_data.shape, dtype=int)

        #  Panel 1: T2 only 
        self.t2_panel1 = viewer.add_image(
            self.t2_data.copy(),
            name="T2 (raw)",
            colormap="gray",
            opacity=1.0,
        )
        self._pad1a = viewer.add_labels(empty_vol.copy(), name="_pad1a", opacity=0.0)
        self._pad1b = viewer.add_labels(empty_vol.copy(), name="_pad1b", opacity=0.0)
        self.seg_layer = viewer.add_labels(
            self.seg_data.astype(int),
            name="Segmentation",
            colormap=self.fs_cmap,
            opacity=1,
        )

        #  Panel 2: T2 + Seg + Breaches + Highlights 
        self.t2_panel2 = viewer.add_image(
            self.t2_data.copy(),
            name="T2 (seg)",
            colormap="gray",
            opacity=1.0,
        )
        self.seg_layer = viewer.add_labels(
            self.seg_data.astype(int),
            name="Segmentation",
            colormap=self.fs_cmap,
            opacity=0,
        )
        self.breach_vol = np.zeros(self.seg_data.shape, dtype=int)
        self.breach_layer = viewer.add_labels(
            self.breach_vol.copy(),
            name="Breaches",
            colormap=breach_cmap,
            opacity=1,
        )
        self.shapes_layer = viewer.add_shapes(
            [],
            ndim=3,
            shape_type="ellipse",
            name="Highlights",
            edge_color="red",
            face_color="transparent",
            edge_width=2,
            opacity=1.0,
        )

        #  Panel 3: T2 + Proposed Fix 
        self._pad3a = viewer.add_labels(empty_vol.copy(), name="_pad3a", opacity=0.0)
        self._pad3b = viewer.add_labels(empty_vol.copy(), name="_pad3b", opacity=0.0)
        self.t2_panel3 = viewer.add_image(
            self.t2_data.copy(),
            name="T2 (fix)",
            colormap="gray",
            opacity=1.0,
        )
        self.fix_vol = np.zeros(self.seg_data.shape, dtype=int)
        self.fix_layer = viewer.add_labels(
            self.fix_vol.copy(),
            name="Proposed Fix",
            colormap=self.fs_cmap,
            opacity=0.5,
        )


        #  enable grid mode 
        viewer.grid.enabled = True
        viewer.grid.stride = self.GRID_STRIDE
        viewer.grid.shape = (1, 3)  # 1 row, 3 columns

        #  Qt layout 
        layout = QVBoxLayout()
        self.setLayout(layout)
        title = QLabel("Breach Finder")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight:bold; font-size:14px;")
        layout.addWidget(title)

        axis_row = QHBoxLayout()
        axis_row.addWidget(QLabel("Axis:"))
        self.axis_combo = QComboBox()
        for k, v in self.AXIS_NAMES.items():
            self.axis_combo.addItem(v, k)
        self.axis_combo.setCurrentIndex(axis)
        self.axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        axis_row.addWidget(self.axis_combo)
        layout.addLayout(axis_row)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "background:#333; color:#0f0; padding:4px; border-radius:4px;"
        )
        layout.addWidget(self.status_label)

        for text, cb in [
            ("▶  Next Breach  (N)", self._on_next),
            ("🔄  Recheck      (R)", self._on_recheck),
            ("✓  Apply Fix     (A)", self._on_apply),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(cb)
            layout.addWidget(btn)

        self.hl_check = QCheckBox("Show highlight circles")
        self.hl_check.setChecked(True)
        self.hl_check.stateChanged.connect(self._on_toggle_hl)
        layout.addWidget(self.hl_check)
        layout.addStretch()

        #  keyboard shortcuts 
        @viewer.bind_key("n")
        def _n(v):
            self._on_next()

        @viewer.bind_key("r")
        def _r(v):
            self._on_recheck()

        @viewer.bind_key("a")
        def _a(v):
            self._on_apply()

        #  go 
        self._apply_view_for_axis()
        self._advance(0)

    #  orientation 

    def _reorient_volumes(self):
        """Re-rotate volumes from raw disk data for the current axis."""
        self.t2_data = _orient_for_display(self._t2_raw, self.axis)
        self.seg_data = _orient_for_display(self._seg_raw, self.axis)

        # Update all T2 panels
        for layer in (self.t2_panel1, self.t2_panel2, self.t2_panel3):
            layer.data = self.t2_data.copy()

        # Update segmentation
        self.seg_layer.data = self.seg_data.astype(int)

        # Reset overlays
        empty = np.zeros(self.seg_data.shape, dtype=int)
        self.breach_vol = empty.copy()
        self.breach_layer.data = self.breach_vol.copy()
        self.fix_vol = empty.copy()
        self.fix_layer.data = self.fix_vol.copy()

        # Reset pads to match new shape
        for pad in [self._pad1a, self._pad1b, self._pad1c, self._pad3a, self._pad3b]:
            pad.data = empty.copy()

    def _apply_view_for_axis(self):
        """Set dims.order so the chosen axis becomes the slider."""
        self.viewer.dims.order = self.DIMS_ORDER[self.axis]

    #  helpers 

    def _set_status(self, text, color="#0f0"):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"background:#333; color:{color}; padding:4px; border-radius:4px;"
        )

    def _clear_overlays(self):
        self.breach_vol[:] = 0
        self.breach_layer.data = self.breach_vol.copy()
        self.fix_vol[:] = 0
        self.fix_layer.data = self.fix_vol.copy()
        self.shapes_layer.data = []

    def _write_breach_to_volume(self, result, idx):
        """Write breach + CP mask into the breach overlay volume."""
        self.breach_vol[:] = 0
        overlay = np.zeros_like(result["breach_mask"], dtype=int)
        overlay[result["cp_mask"] == 1] = 2  # green = CP barrier
        overlay[result["breach_mask"] == 1] = 1  # red = breach
        if self.axis == 0:
            self.breach_vol[idx, :, :] = overlay
        elif self.axis == 1:
            self.breach_vol[:, idx, :] = overlay
        else:
            self.breach_vol[:, :, idx] = overlay
        self.breach_layer.data = self.breach_vol.copy()

    def _write_fix_to_volume(self, fixed_seg, idx):
        """Write the proposed fix into the fix overlay volume."""
        self.fix_vol[:] = 0
        if self.axis == 0:
            self.fix_vol[idx, :, :] = fixed_seg.astype(int)
        elif self.axis == 1:
            self.fix_vol[:, idx, :] = fixed_seg.astype(int)
        else:
            self.fix_vol[:, :, idx] = fixed_seg.astype(int)
        self.fix_layer.data = self.fix_vol.copy()

    def _update_highlights(self, result):
        if not self.hl_check.isChecked() or result is None:
            self.shapes_layer.data = []
            return
        ellipses = [
            _ellipse_bbox_3d(c, r, self.current_slice, self.axis)
            for c, r in zip(result["centroids"], result["radii"])
        ]
        if ellipses:
            self.shapes_layer.data = ellipses
            self.shapes_layer.shape_type = ["ellipse"] * len(ellipses)
        else:
            self.shapes_layer.data = []

    def _navigate_to_slice(self, idx):
        step = list(self.viewer.dims.current_step)
        step[self.axis] = idx
        self.viewer.dims.current_step = tuple(step)

    #  navigation 

    def _advance(self, start_from=0):
        n = self.seg_data.shape[self.axis]
        for i in range(start_from, n):
            seg = extract_slice(self.seg_data, i, self.axis)
            res = detect_breaches(seg, self.label_values)
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

        # Panel 2: breach overlay
        self._write_breach_to_volume(res, idx)
        self._update_highlights(res)

        # Panel 3: proposed fix
        seg = extract_slice(self.seg_data, idx, self.axis)
        fixed_seg = propose_fix(seg, res["breach_mask"])
        self._write_fix_to_volume(fixed_seg, idx)

        self._navigate_to_slice(idx)
        n = res["num_holes"]
        self._set_status(
            f"Slice {idx} — {n} BREACH{'ES' if n != 1 else ''} detected", "#f44"
        )

    #  callbacks 

    def _on_next(self):
        self._advance(0 if self.current_slice is None else self.current_slice + 1)

    def _on_recheck(self):
        if self.current_slice is None:
            return
        # Reload raw data from disk, then re-orient for display
        self.seg_img = nib.load(self.seg_path)
        self._seg_raw = self.seg_img.get_fdata()
        self.seg_data = _orient_for_display(self._seg_raw, self.axis)
        self.seg_layer.data = self.seg_data.astype(int)

        seg = extract_slice(self.seg_data, self.current_slice, self.axis)
        res = detect_breaches(seg, self.label_values)
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

        # Write fix into the display-oriented volume
        if self.axis == 0:
            self.seg_data[idx, :, :] = fixed
        elif self.axis == 1:
            self.seg_data[:, idx, :] = fixed
        else:
            self.seg_data[:, :, idx] = fixed

        # Convert back to disk orientation for saving
        self._seg_raw = _orient_for_save(self.seg_data, self.axis)

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

    def _on_axis_changed(self, index):
        self.axis = self.axis_combo.currentData()
        self._clear_overlays()
        self.result = None
        self.current_slice = None
        self._reorient_volumes()
        self._apply_view_for_axis()
        self._advance(0)
