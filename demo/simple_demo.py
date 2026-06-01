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

InteractiveScope(fig)

plt.show()
