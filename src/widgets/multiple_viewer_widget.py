"""
Breach Finder Multi-Viewer Widget
==================================

Three-panel viewer for cortical plate breach detection using separate
napari ViewerModel instances for independent zoom/pan per panel.

Layout:
    Panel 1 (left)   │ Panel 2 (center)        │ Panel 3 (right)
    T2 only          │ T2 + Seg + Breaches      │ T2 + Proposed Fix

Control panel (BreachFinderControls) sits to the right of the viewers.

All detection / overlay / navigation logic is inherited from
``breach_viewer_base.BreachViewerMixin``.
"""
from data.constants import DIMS_ORDER

from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import QShortcut, QSplitter

import napari
from napari.components.viewer_model import ViewerModel
from napari.qt import QtViewer

from widgets.controls import BreachFinderControls
from breach_viewer_base import (
    BreachViewerMixin,
    orient_for_display,
)


class QtViewerWrap(QtViewer):
    """QtViewer subclass that redirects drag-and-drop file opens to the main viewer."""

    def __init__(self, main_viewer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.main_viewer = main_viewer

    def _qt_open(
        self,
        filenames: list[str],
        stack: bool | list[list[str]],
        choose_plugin: bool = False,
        plugin: str | None = None,
        layer_type: str | None = None,
        **kwargs,
    ):
        self.main_viewer.window._qt_viewer._qt_open(
            filenames, stack, plugin, layer_type, **kwargs
        )


class BreachFinderCorrectionViewer(QSplitter, BreachViewerMixin):
    """
    Three-panel breach finder using independent ViewerModel instances.

    Panel 1 (left):   T2 raw
    Panel 2 (center): T2 + Segmentation + Breach overlay + Highlights
    Panel 3 (right):  T2 + Proposed Fix overlay
    """

    def __init__(
        self,
        viewer: napari.Viewer,
        t2_path: str | Path,
        seg_path: str | Path,
        lut_path: str | Path,
        controls: BreachFinderControls,
        label_values: tuple[int, int] = (1, 42),
        axis: int = 0,
        show_weakpoints: bool = False,
    ) -> None:
        super().__init__()
        self.viewer = viewer
        self._block = False

        # Shared state initialisation (data loading, orientation)
        self._init_breach_state(t2_path, seg_path, lut_path, label_values, axis)

        # Connect the externally-provided controls panel
        self._connect_controls(controls)

        # --- Three independent viewer models ---
        self.viewer_model1 = ViewerModel(title='T2 Raw')
        self.viewer_model2 = ViewerModel(title='T2 + Breaches')
        self.viewer_model3 = ViewerModel(title='T2 + Fix')

        self.qt_viewer1 = QtViewerWrap(viewer, self.viewer_model1)
        self.qt_viewer2 = QtViewerWrap(viewer, self.viewer_model2)
        self.qt_viewer3 = QtViewerWrap(viewer, self.viewer_model3)

        # --- Layout: self IS the viewer splitter ---
        self.setOrientation(Qt.Orientation.Horizontal)
        self.addWidget(self.qt_viewer1)
        self.addWidget(self.qt_viewer2)
        self.addWidget(self.qt_viewer3)
        self.setContentsMargins(0, 0, 0, 0)

        # --- Populate layers ---
        self._setup_layers()

        # --- Sync dim scrolling across viewers ---
        self.viewer_model1.dims.events.current_step.connect(self._point_update)
        self.viewer_model2.dims.events.current_step.connect(self._point_update)
        self.viewer_model3.dims.events.current_step.connect(self._point_update)

        # Forward status from sub-viewers to main viewer
        self.viewer_model1.events.status.connect(self._status_update)
        self.viewer_model2.events.status.connect(self._status_update)
        self.viewer_model3.events.status.connect(self._status_update)

        # --- Sync zoom across viewers when checkbox is on ---
        self._zoom_block = False
        self.viewer_model1.camera.events.zoom.connect(self._zoom_update)
        self.viewer_model2.camera.events.zoom.connect(self._zoom_update)
        self.viewer_model3.camera.events.zoom.connect(self._zoom_update)

        # --- Keyboard shortcuts (Qt-native, since we replace the central widget) ---
        self._bind_qt_shortcuts(viewer.window._qt_window)
        self._apply_view_for_axis()
        self._advance(0)

    def _bind_qt_shortcuts(self, window):
        """Bind keyboard shortcuts via QShortcut on the main window."""
        QShortcut(QKeySequence("N"), window, self._on_next)
        QShortcut(QKeySequence("R"), window, self._on_recheck)
        QShortcut(QKeySequence("A"), window, self._on_apply)
        QShortcut(QKeySequence("Z"), window, self._toggle_sync_zoom)

    # ── Layer setup ───────────────────────────────────────────────

    def _setup_layers(self):
        empty = np.zeros(self.seg_data.shape, dtype=int)

        # Panel 1: T2 only
        self.viewer_model1.add_image(
            self.t2_data.copy(), name="T2 (raw)", colormap="gray",
        )

        # Panel 2: T2 + Seg + Breaches + Highlights
        self.viewer_model2.add_image(
            self.t2_data.copy(), name="T2 (seg)", colormap="gray",
        )
        self.seg_layer = self.viewer_model2.add_labels(
            self.seg_data.astype(int),
            name="Segmentation",
            colormap=self.fs_cmap,
            opacity=0.5,
        )
        self.breach_vol = empty.copy()
        self.breach_layer = self.viewer_model2.add_labels(
            self.breach_vol.copy(),
            name="Breaches",
            colormap=self.breach_cmap,
            opacity=1,
        )
        self.shapes_layer = self.viewer_model2.add_shapes(
            [],
            ndim=3,
            shape_type="ellipse",
            name="Highlights",
            edge_color="red",
            face_color="transparent",
            edge_width=2,
            opacity=1.0,
        )

        # Panel 3: T2 + Fix
        self.viewer_model3.add_image(
            self.t2_data.copy(), name="T2 (fix)", colormap="gray",
        )
        self.fix_vol = empty.copy()
        self.fix_layer = self.viewer_model3.add_labels(
            self.fix_vol.copy(),
            name="Proposed Fix",
            colormap=self.fs_cmap,
            opacity=0.5,
        )

    # ── Multi-viewer synchronisation ──────────────────────────────

    def _toggle_sync_zoom(self):
        cb = self.controls.zoom_check
        cb.setChecked(not cb.isChecked())
        # When toggled on, immediately sync to current source viewer zoom
        if cb.isChecked():
            zoom = self.viewer_model2.camera.zoom
            self._sync_zoom_to(zoom, source=None)

    def _zoom_update(self, event):
        if self._zoom_block or not self.controls.zoom_check.isChecked():
            return
        self._sync_zoom_to(event.value, source=event.source)

    def _sync_zoom_to(self, zoom, source):
        try:
            self._zoom_block = True
            for model in [self.viewer_model1, self.viewer_model2, self.viewer_model3]:
                if model.camera is source:
                    continue
                model.camera.zoom = zoom
        finally:
            self._zoom_block = False

    def _status_update(self, event):
        self.viewer.status = event.value

    def _point_update(self, event):
        if self._block:
            return
        try:
            self._block = True
            for model in [self.viewer_model1, self.viewer_model2, self.viewer_model3]:
                if model.dims is event.source:
                    continue
                model.dims.current_step = event.value
        finally:
            self._block = False

    # ── Viewer-specific hooks (required by BreachViewerMixin) ─────

    def _navigate_to_slice(self, idx):
        for model in [self.viewer_model1, self.viewer_model2, self.viewer_model3]:
            step = list(model.dims.current_step)
            step[self.axis] = idx
            model.dims.current_step = tuple(step)

    def _apply_view_for_axis(self):
        order = DIMS_ORDER[self.axis]
        for model in [self.viewer_model1, self.viewer_model2, self.viewer_model3]:
            model.dims.order = order

    def _reorient_volumes(self):
        self.t2_data = orient_for_display(self._t2_raw, self.axis)
        self.seg_data = orient_for_display(self._seg_raw, self.axis)

        self.viewer_model1.layers["T2 (raw)"].data = self.t2_data.copy()
        self.viewer_model2.layers["T2 (seg)"].data = self.t2_data.copy()
        self.viewer_model3.layers["T2 (fix)"].data = self.t2_data.copy()

        self.seg_layer.data = self.seg_data.astype(int)

        empty = np.zeros(self.seg_data.shape, dtype=int)
        self.breach_vol = empty.copy()
        self.breach_layer.data = self.breach_vol.copy()
        self.fix_vol = empty.copy()
        self.fix_layer.data = self.fix_vol.copy()


if __name__ == '__main__':
    import sys
    from qtpy.QtWidgets import QApplication
    from data.constants import FREESURFER_LUT

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    if len(sys.argv) < 3:
        print("Usage: python multiple_viewer_widget.py <t2.nii> <seg.nii>")
        sys.exit(1)

    view = napari.Viewer(title="Breach Finder (Multi-Viewer)")
    controls = BreachFinderControls()
    dock_widget = BreachFinderCorrectionViewer(
        view,
        t2_path=sys.argv[1],
        seg_path=sys.argv[2],
        lut_path=FREESURFER_LUT,
        controls=controls,
    )
    view.window._qt_window.setCentralWidget(dock_widget)
    view.window.add_dock_widget(controls, name='Breach Finder', area='right')
    napari.run()
