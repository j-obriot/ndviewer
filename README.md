# ND Viewer

A lightweight Qt viewer for multidimensional NumPy-compatible image data.

(shamelessly vibe-coded because napari keeps causing me problems)

![example](/ressources/example.png)

## Usage

```python
import numpy as np
from ndviewer import NDViewer

viewer = NDViewer()
example_volume = np.random.random((12, 256, 256))
viewer.add_image(example_volume)
```

The layer defaults to the caller's variable name when it can be inferred;
pass `name=` when you want a specific name.

The viewer shows when constructed. In IPython or another Qt-enabled host, it
uses the host event loop. A standalone Python script must run a Qt event loop
to remain interactive.
