"""
NDViewer: a small, hackable multidimensional image viewer.

Dependencies:
    pip install numpy pyqtgraph PyQt6

Typical use:
    import numpy as np
    from ndviewer import NDViewer

    viewer = NDViewer()
    viewer.add_image(np.random.rand(20, 256, 256), name="Raw")
    viewer.add_image(np.random.rand(20, 256, 256), name="Mask",
                     colormap="plasma", opacity=0.35)

The viewer window is shown on construction and its API is non-blocking.
It relies on the host's Qt event loop; standalone scripts should call
QApplication.exec() after setting up the viewer.

Layers with compatible spatial dimensions can be overlaid. Slice positions
are synchronized by axis name using normalized positions, so arrays with
different lengths remain approximately aligned. Display-axis reverse buttons
flip the rendered image without modifying source arrays.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from itertools import permutations
from typing import Sequence

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets


_APP = None
_OPEN_VIEWERS = {}
_CUSTOM_COLORMAPS = {
    "gray": pg.ColorMap(
        pos=[0.0, 1.0],
        color=[(0, 0, 0), (255, 255, 255)],
    ),
    "hot": pg.ColorMap(
        pos=[0.0, 1 / 3, 2 / 3, 1.0],
        color=[(0, 0, 0), (255, 0, 0), (255, 255, 0), (255, 255, 255)],
    ),
}


def _ensure_qapplication():
    """Create and retain QApplication before constructing any QWidget."""
    global _APP
    try:
        from IPython import get_ipython
    except ImportError:
        get_ipython = None
    if get_ipython is not None:
        shell = get_ipython()
        enable_gui = getattr(shell, "enable_gui", None)
        if (
            callable(enable_gui)
            and getattr(shell, "active_eventloop", None) is None
        ):
            enable_gui("qt")

    app = QtWidgets.QApplication.instance()
    if app is None:
        _APP = QtWidgets.QApplication(sys.argv[:1])
        app = _APP
    else:
        _APP = app
    return app


class SliceSlider(QtWidgets.QSlider):
    """Slider whose wheel navigation advances one slice per wheel notch."""

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self._wheel_delta = 0
        self.setSingleStep(1)

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.angleDelta().x()
        if not delta:
            super().wheelEvent(event)
            return

        total_delta = self._wheel_delta + delta
        steps = int(total_delta / 120)
        self._wheel_delta = total_delta - steps * 120
        if steps:
            self.setValue(self.value() + steps)
        event.accept()


class ContrastRangeSlider(QtWidgets.QWidget):
    """Two-handle slider for selecting image black and white levels."""

    valuesChanged = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._minimum = 0.0
        self._maximum = 1.0
        self._low = 0.0
        self._high = 1.0
        self._active_handle = "low"
        self._dragging = False
        self.setMinimumHeight(26)
        self.setMinimumWidth(120)
        self.setMaximumHeight(30)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Drag either handle to adjust the black and white levels")

    @property
    def bounds(self):
        return self._minimum, self._maximum

    def set_bounds(self, minimum, maximum):
        minimum, maximum = float(minimum), float(maximum)
        if maximum < minimum:
            raise ValueError("Slider maximum must not be less than its minimum.")
        self._minimum, self._maximum = minimum, maximum
        self.update()

    def set_values(self, low, high):
        low = min(max(float(low), self._minimum), self._maximum)
        high = min(max(float(high), self._minimum), self._maximum)
        if low > high:
            low, high = high, low
        self._low, self._high = low, high
        self.update()

    def _value_to_x(self, value):
        left, right = 9.0, max(9.0, self.width() - 9.0)
        span = self._maximum - self._minimum
        if span == 0:
            return (left + right) / 2
        fraction = (value - self._minimum) / span
        return left + min(max(fraction, 0.0), 1.0) * (right - left)

    def _x_to_value(self, x):
        left, right = 9.0, max(9.0, self.width() - 9.0)
        fraction = min(max((x - left) / (right - left), 0.0), 1.0)
        return self._minimum + fraction * (self._maximum - self._minimum)

    def paintEvent(self, event):
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        center_y = self.height() / 2
        left, right = self._value_to_x(self._minimum), self._value_to_x(self._maximum)
        low_x, high_x = self._value_to_x(self._low), self._value_to_x(self._high)

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor("#e1e7ef"))
        painter.drawRoundedRect(QtCore.QRectF(left, center_y - 2, right - left, 4), 2, 2)
        painter.setBrush(QtGui.QColor("#3978c7"))
        painter.drawRoundedRect(
            QtCore.QRectF(low_x, center_y - 2, high_x - low_x, 4), 2, 2
        )

        painter.setPen(QtGui.QPen(QtGui.QColor("#3978c7"), 2))
        painter.setBrush(QtGui.QColor("#ffffff"))
        painter.drawEllipse(QtCore.QPointF(low_x, center_y), 6, 6)
        painter.drawEllipse(QtCore.QPointF(high_x, center_y), 6, 6)

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        x = event.position().x()
        self._active_handle = (
            "low" if abs(x - self._value_to_x(self._low))
            <= abs(x - self._value_to_x(self._high)) else "high"
        )
        self._dragging = True
        self._move_handle(x)
        event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._move_handle(event.position().x())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _move_handle(self, x):
        value = self._x_to_value(x)
        if self._active_handle == "low":
            low, high = min(value, self._high), self._high
        else:
            low, high = self._low, max(value, self._low)
        if (low, high) != (self._low, self._high):
            self._low, self._high = low, high
            self.update()
            self.valuesChanged.emit(low, high)


class ChevronComboBox(QtWidgets.QComboBox):
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor("#637188"), 1.6))
        x, y = self.width() - 13, self.height() / 2
        path = QtGui.QPainterPath()
        path.moveTo(x - 3, y - 1)
        path.lineTo(x, y + 2)
        path.lineTo(x + 3, y - 1)
        painter.drawPath(path)


class ChevronDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor("#637188"), 1.5))
        x = self.width() - 10
        up_y, down_y = self.height() * 0.25, self.height() * 0.75
        up = QtGui.QPainterPath()
        up.moveTo(x - 3, up_y + 1)
        up.lineTo(x, up_y - 2)
        up.lineTo(x + 3, up_y + 1)
        painter.drawPath(up)
        down = QtGui.QPainterPath()
        down.moveTo(x - 3, down_y - 1)
        down.lineTo(x, down_y + 2)
        down.lineTo(x + 3, down_y - 1)
        painter.drawPath(down)


def _finite_data_range(data):
    array = np.asarray(data)
    if not array.size:
        return 0.0, 1.0

    minimum, maximum = np.inf, -np.inf
    for chunk in np.nditer(
        array,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        buffersize=1_000_000,
    ):
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            minimum = min(minimum, float(finite.min()))
            maximum = max(maximum, float(finite.max()))
    if minimum == np.inf:
        return 0.0, 1.0
    return minimum, maximum


@dataclass
class ImageLayer:
    """A single array and its display/navigation state."""

    data: object
    name: str
    axes: tuple[str, ...]
    visible: bool = True
    opacity: float = 1.0
    colormap: str = "gray"
    contrast: tuple[float, float] | None = None
    data_range: tuple[float, float] = field(init=False)
    contrast_bounds: tuple[float, float] = field(init=False)
    indices: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not hasattr(self.data, "ndim") or not hasattr(self.data, "shape"):
            raise TypeError("Layer data must be array-like and expose ndim and shape.")
        if self.data.ndim < 2:
            raise ValueError("Image layers must have at least two dimensions.")
        if len(self.axes) != self.data.ndim:
            raise ValueError("The number of axis names must match data.ndim.")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("Axis names must be unique within a layer.")
        self.data_range = _finite_data_range(self.data)
        self.contrast_bounds = (
            tuple(float(value) for value in self.contrast)
            if self.contrast is not None else self.data_range
        )
        for axis, size in zip(self.axes, self.data.shape):
            self.indices.setdefault(axis, int(size) // 2)
            self.indices[axis] = min(max(0, self.indices[axis]), int(size) - 1)


class LayerCollection:
    """Small ordered layer collection exposed as viewer.layers."""

    def __init__(self, viewer: "NDViewer"):
        self._viewer = viewer
        self._items: list[ImageLayer] = []

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._items[key]
        for layer in self._items:
            if layer.name == key:
                return layer
        raise KeyError(key)

    @property
    def active(self) -> ImageLayer | None:
        if self._viewer._warn_if_closed():
            return None
        if not self._items:
            return None
        return self._items[self._viewer._active_index]

    def set_active(self, layer_or_name):
        if self._viewer._warn_if_closed():
            return
        if isinstance(layer_or_name, str):
            index = next(
                (i for i, layer in enumerate(self._items)
                 if layer.name == layer_or_name),
                None,
            )
            if index is None:
                raise KeyError(layer_or_name)
        elif isinstance(layer_or_name, ImageLayer):
            index = next(
                (i for i, layer in enumerate(self._items)
                 if layer is layer_or_name),
                None,
            )
            if index is None:
                raise ValueError("Image layer is not in this viewer.")
        else:
            index = int(layer_or_name)
        if not 0 <= index < len(self._items):
            raise IndexError(index)
        self._viewer._set_active_index(index)

    def remove(self, layer_or_name):
        if self._viewer._warn_if_closed():
            return None
        if isinstance(layer_or_name, str):
            layer = self[layer_or_name]
        elif isinstance(layer_or_name, int):
            layer = self._items[layer_or_name]
        else:
            layer = layer_or_name
        index = next(
            (i for i, item in enumerate(self._items) if item is layer),
            None,
        )
        if index is None:
            raise ValueError("Image layer is not in this viewer.")
        self._items.pop(index)
        self._viewer._remove_layer_item(layer)
        if not self._items:
            self._viewer._active_index = 0
            self._viewer._refresh_layer_list()
            self._viewer._refresh_controls()
            self._viewer._render()
        else:
            self._viewer._set_active_index(min(index, len(self._items) - 1))
        self._viewer._process_events()
        return layer


class NDViewer(QtWidgets.QMainWindow):
    """Multidimensional image viewer, shown immediately without blocking."""

    def __init__(self, data=None, *, name="Image", axes=None, title="NDViewer"):
        _ensure_qapplication()
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1320, 850)
        self._closed = False
        self.setStyleSheet("""
            QMainWindow {
                background: #f3f5f8;
            }
            QWidget {
                color: #263449;
                font-family: "Segoe UI", "Arial", sans-serif;
                font-size: 13px;
            }
            QWidget#sidePanel {
                background: #ffffff;
                border: 1px solid #e0e6ee;
                border-radius: 10px;
            }
            QLabel#appTitle {
                color: #17263d;
                font-size: 20px;
                font-weight: 700;
            }
            QLabel#sectionHeader {
                color: #52627a;
                font-size: 11px;
                font-weight: 700;
                padding-top: 5px;
            }
            QListWidget, QComboBox, QDoubleSpinBox {
                background: #ffffff;
                border: 1px solid #d7dee8;
                border-radius: 5px;
                padding: 5px 8px;
                selection-background-color: #e7effb;
                selection-color: #193b69;
            }
            QComboBox, QDoubleSpinBox {
                padding-right: 28px;
                min-height: 20px;
            }
            QComboBox::drop-down {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 26px;
                border: none;
                border-left: 1px solid #e4e9f0;
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
                background: #f5f7fa;
            }
            QComboBox::drop-down:hover,
            QDoubleSpinBox::up-button:hover,
            QDoubleSpinBox::down-button:hover {
                background: #eaf0f7;
            }
            QComboBox::down-arrow {
                image: none;
                width: 0;
                height: 0;
                border: none;
            }
            QComboBox QAbstractItemView {
                background: #ffffff;
                border: 1px solid #d7dee8;
                outline: 0;
                selection-background-color: #e7effb;
                selection-color: #193b69;
            }
            QDoubleSpinBox::up-button,
            QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: right;
                width: 20px;
                border: none;
                border-left: 1px solid #e4e9f0;
                background: #f5f7fa;
            }
            QDoubleSpinBox::up-button {
                subcontrol-position: top right;
                border-bottom: 1px solid #e4e9f0;
                border-top-right-radius: 4px;
            }
            QDoubleSpinBox::down-button {
                subcontrol-position: bottom right;
                border-bottom-right-radius: 4px;
            }
            QDoubleSpinBox::up-arrow {
                image: none;
                width: 0;
                height: 0;
                border: none;
            }
            QDoubleSpinBox::down-arrow {
                image: none;
                width: 0;
                height: 0;
                border: none;
            }
            QListWidget::item {
                padding: 5px;
                border-radius: 3px;
            }
            QListWidget::item:selected {
                background-color: #dceaff;
                color: #193b69;
                font-weight: 600;
            }
            QPushButton {
                background: #f5f7fa;
                border: 1px solid #d7dee8;
                border-radius: 5px;
                padding: 6px 10px;
            }
            QPushButton:hover, QToolButton:hover {
                background: #eaf0f7;
            }
            QPushButton:pressed, QToolButton:pressed {
                background: #dce6f2;
            }
            QPushButton#primaryButton {
                background: #2864b4;
                border-color: #2864b4;
                color: #ffffff;
                font-weight: 600;
            }
            QPushButton#primaryButton:hover {
                background: #20569d;
            }
            QToolButton {
                background: #f5f7fa;
                border: 1px solid #d7dee8;
                border-radius: 4px;
                min-width: 24px;
                min-height: 24px;
            }
            QToolButton#clearLayersButton:hover {
                background: #fdecec;
                border-color: #efcaca;
                color: #a53d3d;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #e1e7ef;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 12px;
                margin: -5px 0;
                background: #3978c7;
                border-radius: 6px;
            }
            QLabel#statusPanel {
                background: #f5f7fa;
                border: 1px solid #e4e9f0;
                border-radius: 6px;
                color: #637188;
                padding: 8px;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: transparent;
                border: none;
                width: 8px;
                margin: 3px 1px 3px 1px;
            }
            QScrollBar::handle:vertical {
                background: #c5cfdb;
                border-radius: 4px;
                min-height: 28px;
            }
            QScrollBar::handle:vertical:hover {
                background: #97a6ba;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
                border: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)

        self.layers = LayerCollection(self)
        self._active_index = 0
        self._image_items: dict[int, pg.ImageItem] = {}
        self._controls: dict[str, tuple[QtWidgets.QSlider, QtWidgets.QLabel]] = {}
        self._syncing = False
        self._cursor_scene_pos = None
        self._cursor_indices = None
        self._cursor_value = None

        self._build_ui()
        if data is not None:
            self.add_image(data, name=name, axes=axes)
        _OPEN_VIEWERS[id(self)] = self
        self.show()

    # ---------- Public API ----------

    def add_image(
        self,
        data,
        *,
        name=None,
        axes: Sequence[str] | None = None,
        colormap="gray",
        opacity=1.0,
        visible=True,
        contrast=None,
    ) -> ImageLayer | None:
        """Add an ND array, naming default dimensions Axis 0, Axis 1, etc."""
        if self._warn_if_closed():
            return None
        if axes is None:
            axes = tuple(f"Axis {i}" for i in range(data.ndim))
        axes = tuple(str(a) for a in axes)
        if name is None:
            name = f"Layer {len(self.layers) + 1}"
        if any(layer.name == name for layer in self.layers):
            raise ValueError(f"A layer named {name!r} already exists.")

        layer = ImageLayer(
            data=data,
            name=str(name),
            axes=axes,
            visible=bool(visible),
            opacity=float(np.clip(opacity, 0.0, 1.0)),
            colormap=colormap,
            contrast=contrast,
        )
        if layer.contrast is None:
            layer.contrast = self._estimate_contrast(layer.data)

        self.layers._items.append(layer)
        item = pg.ImageItem(axisOrder="row-major")
        item.setAutoDownsample(True)
        item.setOpts(opacity=layer.opacity)
        item.setZValue(1)
        self.plot.addItem(item)
        self._image_items[id(layer)] = item

        self._active_index = len(self.layers) - 1
        self._refresh_layer_list()
        self._refresh_controls()
        self._render()
        self._process_events()
        return layer

    def add_data(self, data, **kwargs):
        """Alias for add_image()."""
        return self.add_image(data, **kwargs)

    def remove_layer(self, layer_or_name):
        if self._warn_if_closed():
            return None
        return self.layers.remove(layer_or_name)

    def clear_layers(self):
        if self._warn_if_closed():
            return
        for layer in tuple(self.layers):
            self._remove_layer_item(layer)
        self.layers._items.clear()
        self._active_index = 0
        self._refresh_layer_list()
        self._refresh_controls()
        self._render()
        self._process_events()

    def show(self):
        """Show the viewer without starting or blocking on the Qt event loop."""
        if self._warn_if_closed():
            return
        super().show()
        self.raise_()
        self.activateWindow()
        self._process_events()

    @staticmethod
    def _process_events():
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()

    def closeEvent(self, event):
        self._closed = True
        _OPEN_VIEWERS.pop(id(self), None)
        event.accept()

    def _warn_if_closed(self):
        if not self._closed:
            return False
        warnings.warn(
            "This NDViewer has been closed and no longer exists.",
            RuntimeWarning,
            stacklevel=2,
        )
        return True

    # ---------- UI ----------

    def _build_ui(self):
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        outer = QtWidgets.QHBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        side = QtWidgets.QWidget()
        side.setObjectName("sidePanel")
        side.setMinimumWidth(300)
        side.setMaximumWidth(370)
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(14, 14, 14, 14)
        side_layout.setSpacing(9)
        side_scroll = QtWidgets.QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        side_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        side_scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        side_scroll.setFixedWidth(340)
        side_scroll.setWidget(side)
        outer.addWidget(side_scroll)

        title = QtWidgets.QLabel("ND Viewer")
        title.setObjectName("appTitle")
        side_layout.addWidget(title)
        subtitle = QtWidgets.QLabel("Multidimensional image viewer")
        subtitle.setStyleSheet("color: #77849a;")
        side_layout.addWidget(subtitle)

        # Layers pane
        layer_header = QtWidgets.QHBoxLayout()
        layer_header.addWidget(self._section("Layers"))
        layer_header.addStretch(1)
        add_btn = QtWidgets.QToolButton()
        add_btn.setText("+")
        add_btn.setToolTip("Add NumPy array from .npy file")
        add_btn.clicked.connect(self._open_npy)
        remove_btn = QtWidgets.QToolButton()
        remove_btn.setText("−")
        remove_btn.setToolTip("Remove selected layer")
        remove_btn.clicked.connect(self._remove_selected)
        clear_btn = QtWidgets.QToolButton()
        clear_btn.setText("×")
        clear_btn.setToolTip("Remove all layers")
        clear_btn.setObjectName("clearLayersButton")
        clear_btn.clicked.connect(self.clear_layers)
        layer_header.addWidget(add_btn)
        layer_header.addWidget(remove_btn)
        layer_header.addWidget(clear_btn)
        side_layout.addLayout(layer_header)

        self.layer_list = QtWidgets.QListWidget()
        self.layer_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.layer_list.currentRowChanged.connect(self._layer_selected)
        self.layer_list.itemChanged.connect(self._layer_item_changed)
        self.layer_list.setMinimumHeight(115)
        side_layout.addWidget(self.layer_list)

        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.valueChanged.connect(self._opacity_changed)
        side_layout.addWidget(self._labeled("Active layer opacity", self.opacity_slider))

        # Axis selectors
        display_axes_header = QtWidgets.QHBoxLayout()
        display_axes_header.addWidget(self._section("Displayed dimensions"))
        display_axes_header.addStretch(1)
        self.cycle_axes_btn = QtWidgets.QPushButton("Next")
        self.cycle_axes_btn.setObjectName("primaryButton")
        self.cycle_axes_btn.setToolTip(
            "Cycle through each ordered pair of displayed dimensions"
        )
        self.cycle_axes_btn.clicked.connect(self._cycle_display_axes)
        display_axes_header.addWidget(self.cycle_axes_btn)
        side_layout.addLayout(display_axes_header)
        self.x_combo = ChevronComboBox()
        self.y_combo = ChevronComboBox()
        self.x_reverse_btn = QtWidgets.QToolButton()
        self.y_reverse_btn = QtWidgets.QToolButton()
        for button in (self.x_reverse_btn, self.y_reverse_btn):
            button.setText("↔")
            button.setCheckable(True)
            button.setToolTip("Reverse this displayed dimension")
        self.x_combo.currentIndexChanged.connect(self._display_axes_changed)
        self.y_combo.currentIndexChanged.connect(self._display_axes_changed)
        self.x_reverse_btn.toggled.connect(self._render)
        self.y_reverse_btn.toggled.connect(self._render)
        side_layout.addWidget(self._axis_selector_row("Horizontal (X)", self.x_combo,
                                                       self.x_reverse_btn))
        side_layout.addWidget(self._axis_selector_row("Vertical (Y)", self.y_combo,
                                                       self.y_reverse_btn))

        side_layout.addWidget(self._section("Other dimensions"))
        self.axis_widget = QtWidgets.QWidget()
        self.axis_layout = QtWidgets.QVBoxLayout(self.axis_widget)
        self.axis_layout.setContentsMargins(0, 0, 0, 0)
        self.axis_layout.setSpacing(4)
        side_layout.addWidget(self.axis_widget)

        side_layout.addWidget(self._section("Appearance"))
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems([
            "gray", "viridis", "plasma", "inferno", "magma", "cividis",
            "turbo", "hot",
        ])
        self.cmap_combo.currentTextChanged.connect(self._active_appearance_changed)
        side_layout.addWidget(self._labeled("Colormap", self.cmap_combo))

        self.low_spin = self._spin()
        self.high_spin = self._spin()
        self.low_spin.valueChanged.connect(self._active_appearance_changed)
        self.high_spin.valueChanged.connect(self._active_appearance_changed)
        self.contrast_slider = ContrastRangeSlider()
        self.contrast_slider.valuesChanged.connect(self._contrast_slider_changed)
        side_layout.addWidget(QtWidgets.QLabel("Display range"))
        side_layout.addWidget(self.contrast_slider)
        contrast_inputs = QtWidgets.QHBoxLayout()
        contrast_inputs.addWidget(self._labeled("Black level", self.low_spin))
        contrast_inputs.addWidget(self._labeled("White level", self.high_spin))
        side_layout.addLayout(contrast_inputs)

        buttons = QtWidgets.QHBoxLayout()
        auto_btn = QtWidgets.QPushButton("Auto contrast")
        auto_btn.clicked.connect(self._auto_contrast)
        reset_btn = QtWidgets.QPushButton("Reset view")
        reset_btn.clicked.connect(self.plot_auto_range)
        buttons.addWidget(auto_btn)
        buttons.addWidget(reset_btn)
        side_layout.addLayout(buttons)

        self.status = QtWidgets.QLabel("Add an array to get started.")
        self.status.setWordWrap(True)
        self.status.setObjectName("statusPanel")
        side_layout.addWidget(self.status)
        side_layout.addStretch(1)

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("#000000")
        self.plot = self.graphics.addPlot()
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.plot.getViewBox().setBackgroundColor("#000000")
        for axis in ("left", "bottom", "right", "top"):
            self.plot.hideAxis(axis)
        self.plot.hideButtons()
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        outer.addWidget(self.graphics, 1)

    @staticmethod
    def _section(text):
        label = QtWidgets.QLabel(text)
        label.setObjectName("sectionHeader")
        return label

    @staticmethod
    def _axis_selector_row(text, combo, reverse_button):
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)
        layout.addWidget(QtWidgets.QLabel(text))
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(combo, 1)
        row.addWidget(reverse_button)
        layout.addLayout(row)
        return box

    @staticmethod
    def _labeled(text, widget):
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)
        layout.addWidget(QtWidgets.QLabel(text))
        layout.addWidget(widget)
        return box

    @staticmethod
    def _spin():
        spin = ChevronDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(-1e30, 1e30)
        spin.setKeyboardTracking(False)
        return spin

    # ---------- Layer pane ----------

    def _refresh_layer_list(self):
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        for row, layer in enumerate(self.layers):
            item = QtWidgets.QListWidgetItem(layer.name)
            item.setFlags(
                item.flags()
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                | QtCore.Qt.ItemFlag.ItemIsSelectable
                | QtCore.Qt.ItemFlag.ItemIsEnabled
            )
            item.setCheckState(
                QtCore.Qt.CheckState.Checked if layer.visible
                else QtCore.Qt.CheckState.Unchecked
            )
            self.layer_list.addItem(item)
            item.setSelected(row == self._active_index)
        if len(self.layers):
            self.layer_list.setCurrentRow(self._active_index)
        self.layer_list.blockSignals(False)

    def _layer_selected(self, row):
        if 0 <= row < len(self.layers):
            self._active_index = row
            self._refresh_controls()
            self._render()

    def _set_active_index(self, index):
        self._active_index = index
        self._refresh_layer_list()
        self._refresh_controls()
        self._render()
        self._process_events()

    def _layer_item_changed(self, item):
        row = self.layer_list.row(item)
        if 0 <= row < len(self.layers):
            layer = self.layers[row]
            layer.visible = item.checkState() == QtCore.Qt.CheckState.Checked
            self._render()

    def _opacity_changed(self, value):
        layer = self.layers.active
        if layer is None:
            return
        layer.opacity = value / 100.0
        self._image_items[id(layer)].setOpacity(layer.opacity)

    def _remove_selected(self):
        row = self.layer_list.currentRow()
        if 0 <= row < len(self.layers):
            self.layers.remove(row)

    def _remove_layer_item(self, layer):
        item = self._image_items.pop(id(layer), None)
        if item is not None:
            self.plot.removeItem(item)

    def _open_npy(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Add NumPy array", "", "NumPy arrays (*.npy)"
        )
        if path:
            try:
                array = np.load(path, mmap_mode="r")
                self.add_image(array, name=path.rsplit("/", 1)[-1])
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Could not load array", str(exc))

    # ---------- Axis controls and extraction ----------

    def _refresh_controls(self):
        layer = self.layers.active
        self._clear_layout(self.axis_layout)
        self._controls.clear()
        self._syncing = True
        for combo in (self.x_combo, self.y_combo):
            combo.blockSignals(True)
            combo.clear()
        if layer is None:
            self.x_combo.blockSignals(False)
            self.y_combo.blockSignals(False)
            self.cycle_axes_btn.setEnabled(False)
            self._syncing = False
            return

        self.cycle_axes_btn.setEnabled(len(layer.axes) > 1)
        for axis, size in zip(layer.axes, layer.data.shape):
            label = f"{axis} [{size}]"
            self.x_combo.addItem(label, axis)
            self.y_combo.addItem(label, axis)

        # Keep current axis names if possible, otherwise use last two axes.
        old_x = self.x_combo.findData(getattr(self, "_x_axis_name", None))
        old_y = self.y_combo.findData(getattr(self, "_y_axis_name", None))
        self.x_combo.setCurrentIndex(old_x if old_x >= 0 else len(layer.axes) - 1)
        self.y_combo.setCurrentIndex(old_y if old_y >= 0 else len(layer.axes) - 2)
        if self.x_combo.currentData() == self.y_combo.currentData():
            self.y_combo.setCurrentIndex(len(layer.axes) - 2)
        self._x_axis_name = self.x_combo.currentData()
        self._y_axis_name = self.y_combo.currentData()
        self.x_combo.blockSignals(False)
        self.y_combo.blockSignals(False)

        for axis, size in zip(layer.axes, layer.data.shape):
            if axis in (self._x_axis_name, self._y_axis_name):
                continue
            row = QtWidgets.QWidget()
            layout = QtWidgets.QGridLayout(row)
            layout.setContentsMargins(0, 1, 0, 1)
            slider = SliceSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(0, int(size) - 1)
            slider.setValue(layer.indices[axis])
            value_label = QtWidgets.QLabel(str(layer.indices[axis]))
            value_label.setMinimumWidth(34)
            layout.addWidget(QtWidgets.QLabel(f"{axis} [{size}]"), 0, 0)
            layout.addWidget(slider, 0, 1)
            layout.addWidget(value_label, 0, 2)
            slider.valueChanged.connect(
                lambda value, a=axis, label=value_label: self._index_changed(a, value, label)
            )
            self.axis_layout.addWidget(row)
            self._controls[axis] = (slider, value_label)

        self.cmap_combo.setCurrentText(layer.colormap)
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(round(layer.opacity * 100))
        self.opacity_slider.blockSignals(False)
        lo, hi = layer.contrast
        slider_min, slider_max = layer.contrast_bounds
        slider_min, slider_max = min(slider_min, lo), max(slider_max, hi)
        layer.contrast_bounds = (slider_min, slider_max)
        self.contrast_slider.set_bounds(slider_min, slider_max)
        self.contrast_slider.set_values(lo, hi)
        self.low_spin.blockSignals(True)
        self.high_spin.blockSignals(True)
        spin_min, spin_max = min(-1e30, slider_min), max(1e30, slider_max)
        self.low_spin.setRange(spin_min, spin_max)
        self.high_spin.setRange(spin_min, spin_max)
        self.low_spin.setValue(float(lo))
        self.high_spin.setValue(float(hi))
        self.low_spin.blockSignals(False)
        self.high_spin.blockSignals(False)
        self._syncing = False

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _display_axes_changed(self, *_):
        if self._syncing or self.layers.active is None:
            return
        x_axis = self.x_combo.currentData()
        y_axis = self.y_combo.currentData()
        if x_axis == y_axis:
            sender = self.sender()
            other = self.y_combo if sender is self.x_combo else self.x_combo
            other.blockSignals(True)
            other.setCurrentIndex((other.currentIndex() - 1) % other.count())
            other.blockSignals(False)
            x_axis, y_axis = self.x_combo.currentData(), self.y_combo.currentData()
        self._x_axis_name, self._y_axis_name = x_axis, y_axis
        self._refresh_controls()
        self._render()

    def _cycle_display_axes(self):
        layer = self.layers.active
        if layer is None or len(layer.axes) < 2:
            return

        axis_pairs = list(permutations(layer.axes, 2))
        current = (self._x_axis_name, self._y_axis_name)
        next_index = (axis_pairs.index(current) + 1) % len(axis_pairs) \
            if current in axis_pairs else 0
        next_x, next_y = axis_pairs[next_index]

        self.x_combo.blockSignals(True)
        self.y_combo.blockSignals(True)
        self.x_combo.setCurrentIndex(self.x_combo.findData(next_x))
        self.y_combo.setCurrentIndex(self.y_combo.findData(next_y))
        self.x_combo.blockSignals(False)
        self.y_combo.blockSignals(False)
        self._x_axis_name, self._y_axis_name = next_x, next_y
        self._refresh_controls()
        self._render()

    def _index_changed(self, axis, value, label):
        """Synchronize the selected axis across layers by normalized position."""
        active = self.layers.active
        if active is None:
            return
        source_size = int(active.data.shape[active.axes.index(axis)])
        fraction = 0.0 if source_size <= 1 else value / (source_size - 1)

        for layer in self.layers:
            if axis not in layer.axes:
                continue
            target_size = int(layer.data.shape[layer.axes.index(axis)])
            target_index = int(round(fraction * max(0, target_size - 1)))
            layer.indices[axis] = target_index

        label.setText(str(value))
        self._render()

    def _extract_layer(self, layer, x_name, y_name):
        """Extract one Y,X plane. No projections or source-array mutation."""
        if x_name not in layer.axes or y_name not in layer.axes:
            return None
        x_axis = layer.axes.index(x_name)
        y_axis = layer.axes.index(y_name)
        index = []
        remaining = []
        for i, axis in enumerate(layer.axes):
            if i in (x_axis, y_axis):
                index.append(slice(None))
                remaining.append(axis)
            else:
                index.append(layer.indices.get(axis, int(layer.data.shape[i]) // 2))

        image = np.asarray(layer.data[tuple(index)])
        order = (remaining.index(y_name), remaining.index(x_name))
        if order != (0, 1):
            image = np.transpose(image, order)

        # Flip every rendered layer consistently without modifying source arrays.
        if self.x_reverse_btn.isChecked():
            image = image[:, ::-1]
        if self.y_reverse_btn.isChecked():
            image = image[::-1, :]
        return image

    # ---------- Rendering and appearance ----------

    @staticmethod
    def _estimate_contrast(data):
        # Sample rather than materialize a potentially huge memory-mapped array.
        arr = np.asarray(data)
        if arr.size > 2_000_000:
            step = max(1, arr.size // 500_000)
            arr = arr.reshape(-1)[::step]
        finite = arr[np.isfinite(arr)]
        if not finite.size:
            return (0.0, 1.0)
        lo, hi = np.percentile(finite, (1, 99))
        if lo == hi:
            lo, hi = float(np.min(finite)), float(np.max(finite))
        if lo == hi:
            hi = lo + 1.0
        return float(lo), float(hi)

    def _render(self, *_):
        active = self.layers.active
        if active is None:
            self._update_status()
            return
        x_name, y_name = self._x_axis_name, self._y_axis_name
        for layer in self.layers:
            item = self._image_items[id(layer)]
            item.setVisible(layer.visible)
            if not layer.visible:
                continue
            image = self._extract_layer(layer, x_name, y_name)
            if image is None:
                item.setVisible(False)
                continue
            # Layers intended as overlays should share spatial dimensions.
            item.setImage(image, autoLevels=False, levels=layer.contrast)
            cmap = _CUSTOM_COLORMAPS.get(layer.colormap)
            if cmap is None:
                cmap = pg.colormap.get(layer.colormap)
            item.setLookupTable(cmap.getLookupTable(nPts=256, alpha=True))
            item.setOpacity(layer.opacity)

        self._update_cursor_position()
        self._update_status()

    def _mouse_moved(self, scene_pos):
        self._cursor_scene_pos = scene_pos
        self._update_cursor_position()
        self._update_status()

    def _update_cursor_position(self):
        self._cursor_indices = None
        self._cursor_value = None
        layer = self.layers.active
        scene_pos = self._cursor_scene_pos
        if layer is None or scene_pos is None:
            return

        view_box = self.plot.getViewBox()
        if not view_box.sceneBoundingRect().contains(scene_pos):
            return
        point = view_box.mapSceneToView(scene_pos)
        x_name, y_name = self._x_axis_name, self._y_axis_name
        x_axis, y_axis = layer.axes.index(x_name), layer.axes.index(y_name)
        x_index, y_index = int(np.floor(point.x())), int(np.floor(point.y()))
        if self.x_reverse_btn.isChecked():
            x_index = int(layer.data.shape[x_axis]) - 1 - x_index
        if self.y_reverse_btn.isChecked():
            y_index = int(layer.data.shape[y_axis]) - 1 - y_index
        if not (0 <= x_index < layer.data.shape[x_axis]):
            return
        if not (0 <= y_index < layer.data.shape[y_axis]):
            return

        indices = dict(layer.indices)
        indices[x_name] = x_index
        indices[y_name] = y_index
        self._cursor_indices = tuple((axis, indices[axis]) for axis in layer.axes)
        array_index = tuple(indices[axis] for axis in layer.axes)
        self._cursor_value = np.asarray(layer.data[array_index]).item()

    def _update_status(self):
        active = self.layers.active
        if active is None:
            self.status.setText("Add an array to get started.")
            return
        if self._cursor_indices is None:
            voxel_position = "[]"
        else:
            voxel_position = "[" + ", ".join(
                str(index) for _, index in self._cursor_indices
            ) + "]"
        voxel_value = "—" if self._cursor_value is None else str(self._cursor_value)
        self.status.setText(
            f"Source shape: {tuple(active.data.shape)}\n"
            f"{voxel_position}\n"
            f"Value: {voxel_value}"
        )

    def _active_appearance_changed(self, *_):
        if self._syncing:
            return
        layer = self.layers.active
        if layer is None:
            return
        layer.colormap = self.cmap_combo.currentText()
        lo, hi = self.low_spin.value(), self.high_spin.value()
        if hi <= lo:
            return
        layer.contrast = (lo, hi)
        slider_min, slider_max = layer.contrast_bounds
        layer.contrast_bounds = (min(slider_min, lo), max(slider_max, hi))
        self._syncing = True
        self.contrast_slider.set_bounds(*layer.contrast_bounds)
        self.contrast_slider.set_values(lo, hi)
        self._syncing = False
        self._render()

    def _contrast_slider_changed(self, lo, hi):
        if self._syncing:
            return
        self._syncing = True
        self.low_spin.setValue(lo)
        self.high_spin.setValue(hi)
        self._syncing = False
        self._active_appearance_changed()

    def _auto_contrast(self):
        layer = self.layers.active
        if layer is None:
            return
        image = self._extract_layer(layer, self._x_axis_name, self._y_axis_name)
        if image is not None:
            layer.contrast = self._estimate_contrast(image)
            self._refresh_controls()
            self._render()

    def plot_auto_range(self):
        if self._warn_if_closed():
            return
        self.plot.autoRange()
        self.plot.setAspectLocked(True)


if __name__ == "__main__":
    viewer = NDViewer()
    viewer.add_image(np.random.random((12, 4, 256, 256)), name="Random data",
                     axes=("Axis 0", "Axis 1", "Axis 2", "Axis 3"))
    viewer.add_image(np.random.random((256, 256)), name="Overlay",
                     axes=("Axis 2", "Axis 3"), colormap="plasma", opacity=0.3)
    viewer.show()
    QtWidgets.QApplication.instance().exec()
