# ND Viewer

A lightweight Qt viewer for multidimensional NumPy-compatible image data.

(shamelessly vibe-coded because napari keeps causing me problems)

![example](/ressources/example.png)

## Usage

```python
import numpy as np
from nd_viewer import NDViewer

viewer = NDViewer()
viewer.add_image(np.random.random((12, 256, 256)), name="Example")
```

The viewer shows when constructed. In IPython or another Qt-enabled host, it
uses the host event loop. A standalone Python script must run a Qt event loop
to remain interactive.
