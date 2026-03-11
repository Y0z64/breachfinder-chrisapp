from data.constants import AXIS_NAMES
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QPushButton, QCheckBox


class BreachFinderControls(QWidget):
    """Reusable right-panel control widget for breach navigation.

    Emits Qt signals so it can be created independently and connected
    to any viewer that implements the matching slots.

    Signals:
        next_requested()
        recheck_requested()
        apply_requested()
        axis_changed(int)          — new axis index (combo currentData)
        highlight_toggled()
        weakpoints_toggled()
        sync_zoom_toggled()
    """

    next_requested = Signal()
    recheck_requested = Signal()
    apply_requested = Signal()
    axis_changed = Signal(int)
    highlight_toggled = Signal()
    weakpoints_toggled = Signal()
    sync_zoom_toggled = Signal()

    def __init__(
        self,
        axis: int = 0,
        show_weakpoints: bool = False,
    ) -> None:
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        title = QLabel("Breach Finder")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight:bold; font-size:14px;")
        layout.addWidget(title)

        # Axis selector
        axis_row = QHBoxLayout()
        axis_row.addWidget(QLabel("Axis:"))
        self.axis_combo = QComboBox()
        for k, v in AXIS_NAMES.items():
            self.axis_combo.addItem(v, k)
        self.axis_combo.setCurrentIndex(axis)
        self.axis_combo.currentIndexChanged.connect(
            lambda _: self.axis_changed.emit(self.axis_combo.currentData())
        )
        axis_row.addWidget(self.axis_combo)
        layout.addLayout(axis_row)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            "background:#333; color:#0f0; padding:4px; border-radius:4px;"
        )
        layout.addWidget(self.status_label)

        # Action buttons
        for text, signal in [
            ("▶  Next Breach  (N)", self.next_requested),
            ("🔄  Recheck      (R)", self.recheck_requested),
            ("✓  Apply Fix     (A)", self.apply_requested),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(signal.emit)
            layout.addWidget(btn)

        # Checkboxes
        self.hl_check = QCheckBox("Show highlight circles")
        self.hl_check.setChecked(True)
        self.hl_check.stateChanged.connect(lambda _: self.highlight_toggled.emit())
        layout.addWidget(self.hl_check)

        self.wp_check = QCheckBox("Check for weakpoints (WIP)")
        self.wp_check.setChecked(show_weakpoints)
        self.wp_check.stateChanged.connect(lambda _: self.recheck_requested.emit())
        layout.addWidget(self.wp_check)

        self.zoom_check = QCheckBox("Sync zoom (Z)")
        self.zoom_check.setChecked(False)
        self.zoom_check.stateChanged.connect(lambda _: self.sync_zoom_toggled.emit())
        layout.addWidget(self.zoom_check)

        layout.addStretch()

    def set_status(self, text: str, color: str = "#0f0"):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"background:#333; color:{color}; padding:4px; border-radius:4px;"
        )
