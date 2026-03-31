"""
Breach Finder Orthogonal Viewer
================================

Traditional three-perspective (Sagittal / Coronal / Axial) medical viewer
with integrated breach detection and navigation.

Layout:
    ┌          ┬          ┐
    │          │ Coronal  │
    │ Sagittal ├          ┤
    │          │  Axial   │
    └          ┴          ┘

Each panel contains:
- T2 image layer
- Segmentation labels layer
- Breach overlay labels (visible in all three perspectives)
- Highlight ellipses (only on the active detection axis)
- Red crosshair showing the intersection of the other two planes

Layers are linked as per: https://napari.org/dev/gallery/linked_layers.html

All detection / overlay / navigation logic is inherited from
``breachviewer_base.BreachViewerMixin``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import QShortcut, QSplitter

import napari
from napari.components.viewer_model import ViewerModel
from napari.layers import Vectors
from napari.qt import QtViewer
from napari.experimental import link_layers

from breachviewer_base import BreachViewerMixin
from data.constants import DIMS_ORDER, _ORIENTATIONS
from utils import orient_for_display, orient_for_save
from widgets.controls import BreachFinderControls

# TODO: Cross has bad performance, thread or subproccess the crosshair updates?  Or switch to Shapes layer with fixed-size lines?
# TODO: Correct brain positioning with 90 degree rotations, udpate saving function too
# TODO: Figure out why config panel wont open in this view


class QtViewerWrap(QtViewer):
    """QtViewer that redirects drag-and-drop opens to the main viewer."""

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


class BreachFinderOrthoViewer(QSplitter, BreachViewerMixin):
    """
    Three-panel orthogonal viewer with breach detection.

    Panels show Sagittal (axis 0), Coronal (axis 1), and Axial (axis 2)
    perspectives simultaneously.  Breach detection runs along the
    user-selected axis.  Breach voxels are visible from all perspectives;
    highlight circles appear only in the active-axis panel.
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
        show_cross: bool = True,
    ) -> None:
        super().__init__()
        self.viewer = viewer
        self._block = False
        self._zoom_block = False

        # Shared state initialisation (raw data — dims.order handles orientation)
        self._init_breach_state(t2_path, seg_path, lut_path, label_values, axis, orient=True)
        self.breach_vol = np.zeros(self.seg_data.shape, dtype=int)

        # Connect the externally-provided controls panel
        self._connect_controls(controls)

        #  Three independent viewer models (keyed by axis)
        self.viewer_models: dict[int, ViewerModel] = {}
        self.qt_viewers: dict[int, QtViewerWrap] = {}
        self._t2_layers: dict[int, Any] = {}
        self._seg_layers: dict[int, Any] = {}
        self._breach_layers: dict[int, Any] = {}
        self._shapes_layers: dict[int, Any] = {}
        self._cross_layers: dict[int, Vectors] = {}

        # Axis 0: Sagittal
        self.viewer_models[0] = viewer
        self.qt_viewers[0] = viewer.window._qt_viewer

        # Axes 1 and 2: Coronal and Axial
        for axis, title in [(1, 'Coronal'), (2, 'Axial')]:
            model = ViewerModel(title=title)
            qt_v = QtViewerWrap(viewer, model)
            self.viewer_models[axis] = model
            self.qt_viewers[axis] = qt_v
        
        # Set correct orientations
        for axis, model in self.viewer_models.items():
            model.camera.orientation2d = _ORIENTATIONS[axis]  # type: ignore
        
        #  Layout 
        self.setOrientation(Qt.Orientation.Horizontal)
        self.addWidget(self.qt_viewers[0])
        self.setContentsMargins(0, 0, 0, 0)
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.qt_viewers[1])  # Coronal
        right.addWidget(self.qt_viewers[2])  # Axial
        right.setContentsMargins(0, 0, 0, 0)
        self._right_splitter = right
        self.addWidget(right)

        #  Populate layers 
        self._setup_layers(show_cross)

        #  Set each panel's perspective 
        for axis, model in self.viewer_models.items():
            model.dims.order = DIMS_ORDER[axis]

        #  Synchronisation events 
        for model in self.viewer_models.values():
            model.dims.events.current_step.connect(self._point_update)
            model.events.status.connect(self._status_update)
            model.camera.events.zoom.connect(self._zoom_update)

        #  Keyboard shortcuts 
        self._bind_qt_shortcuts(viewer.window._qt_window)

        #  Start detection 
        self._advance(0)

    #  Keyboard shortcuts 

    def _bind_qt_shortcuts(self, window):
        QShortcut(QKeySequence("N"), window, self._on_next)
        QShortcut(QKeySequence("R"), window, self._on_recheck)
        QShortcut(QKeySequence("A"), window, self._on_apply)
        QShortcut(QKeySequence("Z"), window, self._toggle_sync_zoom)

    #  Layer setup 

    def _to_axis_display(self, volume: np.ndarray, axis: int) -> np.ndarray:
        """Map active-axis display volume into display coordinates for *axis*."""
        raw = orient_for_save(volume, self.axis)
        return orient_for_display(raw, axis)

    def _setup_layers(self, show_cross: bool):
        for axis, model in self.viewer_models.items():
            t2 = model.add_image(
                self._to_axis_display(self.t2_data, axis).copy(),
                name="T2",
                colormap="gray",
            )
            self._t2_layers[axis] = t2
            link_layers(self._t2_layers.values())

            seg = model.add_labels(
                self._to_axis_display(self.seg_data, axis).astype(int),
                name="Segmentation",
                colormap=self.fs_cmap, opacity=0.5,
            )
            self._seg_layers[axis] = seg
            link_layers(self._seg_layers.values())

            breach = model.add_labels(
                self._to_axis_display(self.breach_vol, axis).copy(),
                name="Breaches",
                colormap=self.breach_cmap, opacity=1,
            )
            self._breach_layers[axis] = breach
            link_layers(self._breach_layers.values())

            shapes = model.add_shapes(
                [], ndim=3, shape_type="ellipse", name="Highlights",
                edge_color="red", face_color="transparent",
                edge_width=2, opacity=1.0,
            )
            self._shapes_layers[axis] = shapes
            link_layers(self._shapes_layers.values())

            cross = Vectors(name='.cross', ndim=3)
            cross.edge_width = 1.5
            cross.edge_color = 'red'
            self._cross_layers[axis] = cross
            link_layers(self._cross_layers.values())
            if show_cross:
                model.layers.append(cross)

    #  Crosshair 

    def _update_crosses(self):
        for axis, model in self.viewer_models.items():
            shape = self._t2_layers[axis].data.shape
            cross = self._cross_layers[axis]
            if cross not in model.layers:
                continue
            step = model.dims.current_step
            vec = []
            for dim in range(3):
                if shape[dim] <= 1:
                    continue
                start = list(step)
                start[dim] = 0
                direction = [0, 0, 0]
                direction[dim] = shape[dim]
                vec.append((start, direction))
            cross.data = vec

    def set_cross_visible(self, visible: bool):
        """Toggle red crosshair in all panels."""
        for axis, model in self.viewer_models.items():
            cross = self._cross_layers[axis]
            if visible and cross not in model.layers:
                model.layers.append(cross)
            elif not visible and cross in model.layers:
                model.layers.remove(cross)
        if visible:
            self._update_crosses()

    #  Multi-viewer synchronisation 

    def _point_update(self, event):
        if self._block:
            return
        try:
            self._block = True
            for model in self.viewer_models.values():
                if model.dims is event.source:
                    continue
                model.dims.current_step = event.value
            self._update_crosses()
        finally:
            self._block = False

    def _status_update(self, event):
        self.viewer.status = event.value

    def _toggle_sync_zoom(self):
        cb = self.controls.zoom_check
        cb.setChecked(not cb.isChecked())
        if cb.isChecked():
            zoom = self.viewer_models[self.axis].camera.zoom
            self._sync_zoom_to(zoom, source=None)

    def _zoom_update(self, event):
        if self._zoom_block or not self.controls.zoom_check.isChecked():
            return
        self._sync_zoom_to(event.value, source=event.source)

    def _sync_zoom_to(self, zoom, source):
        try:
            self._zoom_block = True
            for model in self.viewer_models.values():
                if model.camera is source:
                    continue
                model.camera.zoom = zoom
        finally:
            self._zoom_block = False

    #  Mixin hooks (multi-layer broadcasting) 

    def _broadcast_breach_vol(self):
        for axis, bl in self._breach_layers.items():
            bl.data = self._to_axis_display(self.breach_vol, axis).copy()

    def _broadcast_seg_data(self):
        for axis, sl in self._seg_layers.items():
            sl.data = self._to_axis_display(self.seg_data, axis).astype(int)

    def _apply_highlights(self, all_ellipses, edge_colors):
        # Clear all panels first, then show only on the active-axis panel
        for sl in self._shapes_layers.values():
            sl.data = []
        if all_ellipses:
            active = self._shapes_layers[self.axis]
            active.data = all_ellipses
            active.shape_type = ["ellipse"] * len(all_ellipses)
            active.edge_color = edge_colors

    def _clear_highlights(self):
        for sl in self._shapes_layers.values():
            sl.data = []

    def _navigate_to_slice(self, idx):
        for model in self.viewer_models.values():
            step = list(model.dims.current_step)
            step[self.axis] = idx
            model.dims.current_step = tuple(step)
        self._update_crosses()

    def _reorient_volumes(self):
        self.t2_data = orient_for_display(self._t2_raw, self.axis)
        self.seg_data = orient_for_display(self._seg_raw, self.axis)
        self.breach_vol = np.zeros(self.seg_data.shape, dtype=int)

        for axis, t2_layer in self._t2_layers.items():
            t2_layer.data = self._to_axis_display(self.t2_data, axis).copy()

        self._broadcast_seg_data()
        self._broadcast_breach_vol()
        self._clear_highlights()
        self._update_crosses()

    def _apply_view_for_axis(self):
        pass  # each panel always shows its fixed perspective


if __name__ == '__main__':
    import sys
    from qtpy.QtWidgets import QApplication
    from data.constants import FREESURFER_LUT

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    if len(sys.argv) < 3:
        print("Usage: python ortho_viewer.py <t2.nii> <seg.nii>")
        sys.exit(1)

    view = napari.Viewer(title="Breach Finder (Ortho)")
    controls = BreachFinderControls()
    ortho = BreachFinderOrthoViewer(
        view,
        t2_path=sys.argv[1],
        seg_path=sys.argv[2],
        lut_path=FREESURFER_LUT,
        controls=controls,
    )
    view.window._qt_window.setCentralWidget(ortho)
    view.window.add_dock_widget(controls, name='Breach Finder', area='right')
    napari.run()
