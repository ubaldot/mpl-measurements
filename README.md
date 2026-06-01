# mpl-measurements
Interactive measurement tools for Matplotlib.

Click two points on any plotted line to compute engineering-style measurements
such as Δx, Δy, and signal statistics (min, max, RMS) over a selected
interval.

![mpl-measurements](mpl_measurements.png)

To use it, simply call `InteractiveScope(fig)` where `fig` is your figure.
Example:

```python

import numpy as np
import matplotlib.pyplot as plt

from mpl_measurements import InteractiveScope


x = np.linspace(0, 10, 1000)
fig, axs = plt.subplots(2, 2, sharex=True)
axs = axs.flatten()

for ii, ax in enumerate(axs):
    ax.plot(x, np.sin(x + ii), label=f"sin {ii}")
    ax.plot(x, np.cos(x + ii), label=f"cos {ii}")
    ax.set_title(f"Axes {ii}")
    ax.legend()

InteractiveScope(fig) # This is the line!

# You can also call e.g. InteractiveScope(fig, axes=[axs[0], axs[1]]) if you
# want the measurements tool active only on axes axs[0], axs[1]. See
# docstring for more info.

plt.show()
```
## ⚠️  Notes on layout engines

If a Matplotlib layout engine is used (e.g. `"tight"` or `"constrained"`),
you must reserve space on the right side using `rect`:

```python
    fig.set_layout_engine("tight", rect=[0, 0, 0.75, 1])>[Info]
```

Otherwise, the info panel may overlap with the plot axes.
