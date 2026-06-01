"""
InteractiveScope: lightweight interactive measurement tool for Matplotlib figures.

This module provides the `InteractiveScope` class, which adds interactive
capabilities to an existing Matplotlib figure:

Features
--------
- Line selection via mouse click
- Automatic computation of:
    * Δx, Δy
    * Min / Max values within the window
    * RMS value
- Dynamic info panel showing measurements

Usage
-----
Instantiate the scope *after* creating your figure:

    >>> x = np.linspace(0, 10, 1000)
    >>> y = x**2
    >>> fig, ax = plt.subplots(1,1)
    >>> InteractiveScope(fig)

Interaction
-----------
- Click on a line → select it
- Click twice on the plot → place cursors
- Press "r" → reset cursors
- Measurements are displayed in the right info panel

Notes
-----
- Compatible with Matplotlib layout engines if a right-side margin is reserved.
- If a Matplotlib layout engine is used (e.g. "tight", "constrained"),
  the user should reserve space on the right side using `rect`, e.g.:

      fig.set_layout_engine("tight", rect=[0, 0, 0.75, 1])

  Otherwise, the info panel may overlap with the plot axes.

Limitations
-----------
- Assumes line-based plots (Line2D objects)
- Works with mouse-based interaction only
"""

import numpy as np

from typing import TypedDict

# For type-annotations
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.backend_bases import PickEvent, KeyEvent, MouseEvent, Event
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from matplotlib.axes._base import _AxesBase
from numpy.typing import NDArray


def compute_window_stats(
    xdata: NDArray[np.floating],
    ydata: NDArray[np.floating],
    x1: float,
    x2: float,
) -> tuple[np.floating | float, np.floating | float, np.floating | float]:
    xmin, xmax = sorted([x1, x2])
    mask = (xdata >= xmin) & (xdata <= xmax)
    y_win = ydata[mask]

    if len(y_win) == 0:
        return np.nan, np.nan, np.nan

    y_min = np.min(y_win)
    y_max = np.max(y_win)
    rms = np.sqrt(np.mean(y_win**2))

    return y_min, y_max, rms


class AxisState(TypedDict):
    selected_line: Line2D | None
    cursor_lines: list
    cursor_points: list
    positions: list[tuple[float, float]]


class InteractiveScope:
    """
    Add interactive measurement tools to a Matplotlib figure.

    This class enables selecting plotted lines and placing cursors to
    extract measurements such as distances, extrema, and RMS values.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Target figure where interaction is enabled
    axes : Axes or list[Axes], optional
        Axes to enable interaction on (default: all axes in figure)
    box_width_inches : float
        Initial width of the information panel (in inches). Default 1.4
    box_padding_inches : float
        Padding inside the information panel (in inches). Default 0.1

    Notes
    -----
    - The class modifies the figure layout by adding a right-side panel
    - The panel automatically expands to fit its content
    - Interaction is event-driven (pick, click, key press)

    Typical workflow:
        1. Create plot
        2. Instantiate InteractiveScope
        3. Interact with the figure
    """

    def __init__(
        self,
        fig: Figure,
        axes: Axes | list[Axes] | None = None,
        box_width_inches: float = 1.4,
        box_padding_inches: float = 0.1,
    ) -> None:
        self.fig = fig
        self._panel_width = box_width_inches
        self._padding_in = box_padding_inches

        # This is needed otherwise things got garbage collected
        setattr(fig, "_interactive_scope", self)

        # Always list[Axes]
        self.axes: list[Axes]
        if axes is None:
            self.axes = list(fig.axes)
        elif isinstance(axes, Axes):
            self.axes = [axes]
        else:
            self.axes = axes

        if not self.axes:
            raise ValueError("No axes available for InteractiveScope")

        for ax in self.axes:
            for line in ax.get_lines():
                line.set_picker(5)

        # ---- layout adjustment ----
        right_margin = self.get_figure_right_margin()
        width_inches, _ = fig.get_size_inches()

        right_margin_inches = right_margin * width_inches
        box_width = box_width_inches / width_inches

        if right_margin_inches < box_width_inches:
            fig.subplots_adjust(right=1 - box_width)

        left = 1 - box_width
        info_ax = fig.add_axes((left, 0.1, box_width, 0.8))
        info_ax.axis("off")
        info_ax.set_navigate(False)
        info_ax.set_zorder(-1)

        box_padding_fraction = box_padding_inches / box_width_inches
        info_text = info_ax.text(
            box_padding_fraction,
            # 1 - box_padding_fraction,
            1.0,
            "Select a line",
            va="top",
            transform=info_ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat"),
        )
        self.info_text = info_text

        # state per axis
        def new_axis_state() -> AxisState:
            return {
                "selected_line": None,
                "cursor_lines": [],
                "cursor_points": [],
                "positions": [],
            }

        self.state: dict[_AxesBase, AxisState] = {
            ax: new_axis_state() for ax in self.axes
        }

        # connect events (store IDs for future cleanup)
        self.cid_pick = fig.canvas.mpl_connect("pick_event", self.on_pick)
        self.cid_click = fig.canvas.mpl_connect(
            "button_press_event", self.on_click
        )
        self.cid_key = fig.canvas.mpl_connect("key_press_event", self.on_key)

    def ensure_panel_fits_text(self) -> None:
        fig = self.fig
        fig_width_in = fig.get_figwidth()

        renderer = fig.canvas.get_renderer()  # type: ignore
        if renderer is None:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()  # type: ignore

        bbox = self.info_text.get_window_extent(renderer)

        text_width_in = bbox.width / fig.dpi

        padding_in = self._padding_in
        required_width_in = text_width_in + 2 * padding_in

        #  only grow panel width
        EPS = max(0.02, 0.02 * self._panel_width)
        if required_width_in > self._panel_width + EPS:
            panel_fraction = min(required_width_in / fig_width_in, 0.6)

            #  shrink axes
            fig.subplots_adjust(right=1 - panel_fraction)

            #  move + resize panel
            left = 1 - panel_fraction

            if self.info_text.axes is None:
                return

            self.info_text.axes.set_position((left, 0.1, panel_fraction, 0.8))

            fig.canvas.draw_idle()
            self._panel_width = required_width_in

    def get_figure_right_margin(self) -> float:
        # Return as portion of the figure
        fig = self.fig

        renderer = fig.canvas.get_renderer()  # type: ignore
        if renderer is None:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()  # type: ignore

        bboxes: list[Bbox] = []

        for ax in self.axes:
            b = ax.get_tightbbox(renderer)
            if b is not None:
                bboxes.append(b)

        if not bboxes:
            return 1.0  # no axes → full space available

        full_bbox = Bbox.union(bboxes)

        inv = fig.transFigure.inverted()
        full_bbox_fig = inv.transform_bbox(full_bbox)

        return 1 - full_bbox_fig.x1

    # -----------------------------------------------------
    # Line selection
    # -----------------------------------------------------
    def on_pick(self, event: Event) -> None:
        if not isinstance(event, PickEvent):
            return

        toolbar = self.fig.canvas.toolbar
        if toolbar is not None and toolbar.mode:
            return

        ax = event.artist.axes
        if ax not in self.state:
            return

        state = self.state[ax]

        artist = event.artist
        if not isinstance(artist, Line2D):
            return

        state["selected_line"] = artist

        # reset visual emphasis
        for line in ax.get_lines():
            line.set_linewidth(1)

        selected = state["selected_line"]
        if selected is None:
            return

        selected.set_linewidth(3)

        if not isinstance(ax, Axes):
            return
        self.info_text.set_text(
            f"{ax.get_title()}\n"
            f"Selected: {selected.get_label()}\n"
            "Click twice to place cursors"
        )
        self.fig.canvas.draw_idle()

    # -----------------------------------------------------
    # Cursor placement
    # -----------------------------------------------------
    def on_click(self, event: Event) -> None:
        if not isinstance(event, MouseEvent):
            return

        ax = event.inaxes
        if ax not in self.state:
            return

        state = self.state[ax]
        line = state["selected_line"]
        if line is None or event.xdata is None:
            return

        xdata = np.asarray(line.get_xdata())
        ydata = np.asarray(line.get_ydata())

        # snap to nearest sample
        idx = np.argmin(np.abs(xdata - event.xdata))
        x_sel, y_sel = xdata[idx], ydata[idx]

        color = line.get_color()

        vline = ax.axvline(x_sel, color=color, linestyle="--")
        (point,) = ax.plot(x_sel, y_sel, "o", color=color)

        state["cursor_lines"].append(vline)
        state["cursor_points"].append(point)
        state["positions"].append((x_sel, y_sel))

        # keep only last 2 cursors
        if len(state["cursor_lines"]) > 2:
            state["cursor_lines"].pop(0).remove()
            state["cursor_points"].pop(0).remove()
            state["positions"].pop(0)

        self.update_measurements(ax, state)

    # -----------------------------------------------------
    # Measurements + UI update
    # -----------------------------------------------------
    def update_measurements(self, ax: Axes, state: AxisState) -> None:
        line = state["selected_line"]

        if line is None:
            return

        if len(state["positions"]) < 2:
            self.info_text.set_text(
                f"{ax.get_title()}\n{line.get_label()}\nClick second point"
            )

            self.ensure_panel_fits_text()
            return

        (x1, y1), (x2, y2) = state["positions"]

        xdata = np.asarray(line.get_xdata())
        ydata = np.asarray(line.get_ydata())

        y_min, y_max, rms = compute_window_stats(xdata, ydata, x1, x2)

        dx = x2 - x1
        dy = y2 - y1

        self.info_text.set_text(
            f"{ax.get_title()}\n"
            f"{line.get_label()}\n\n"
            f"P1: ({x1:.3f}, {y1:.3f})\n"
            f"P2: ({x2:.3f}, {y2:.3f})\n\n"
            f"Δx = {dx:.3f}\n"
            f"Δy = {dy:.3f}\n\n"
            f"Min = {y_min:.3f}\n"
            f"Max = {y_max:.3f}\n"
            f"RMS = {rms:.3f}"
        )

        self.ensure_panel_fits_text()

    # -----------------------------------------------------
    # Reset
    # -----------------------------------------------------
    def on_key(self, event: Event) -> None:
        if not isinstance(event, KeyEvent):
            return

        if event.key != "r":
            return

        for ax, state in self.state.items():
            for l in state["cursor_lines"]:
                l.remove()
            for p in state["cursor_points"]:
                p.remove()

            state["cursor_lines"].clear()
            state["cursor_points"].clear()
            state["positions"].clear()

        self.info_text.set_text("Reset — select a line")
        self.fig.canvas.draw_idle()
