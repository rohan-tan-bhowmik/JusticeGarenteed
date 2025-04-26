#!/usr/bin/env python3
"""
Interactive ROI picker with zoom support.

▪ Draw a rectangle with the left mouse button
▪ Press **q** (or close the window) to finish
The script then prints the NumPy slice corresponding to the ROI.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image


# ── Globals ────────────────────────────────────────────────────────────────
roi: list[int] = []          # [row_min, row_max, col_min, col_max]


# ── Callbacks ──────────────────────────────────────────────────────────────
def onselect(eclick, erelease):
    """Store ROI and print the slice after the rectangle is drawn."""
    x1, y1 = map(int, (eclick.xdata, eclick.ydata))
    x2, y2 = map(int, (erelease.xdata, erelease.ydata))

    col_min, col_max = sorted([x1, x2])
    row_min, row_max = sorted([y1, y2])

    roi[:] = [row_min, row_max, col_min, col_max]

    print("\nSelected region:")
    print(f"  rows : {row_min}:{row_max}")
    print(f"  cols : {col_min}:{col_max}")
    print(f"  NumPy slice ⇒ img[{row_min}:{row_max}, {col_min}:{col_max}]")


def toggle_quit(event):
    """Close the figure when the user presses 'q' or 'Q'."""
    if event.key and event.key.lower() == "q":
        plt.close(event.canvas.figure)


# ── Main routine ───────────────────────────────────────────────────────────
def main(img_path: str | Path):
    img = np.asarray(Image.open(img_path))

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Scroll to zoom, drag to pan — "
                 "draw rectangle then press 'q' to finish")

    # Build kwargs that exist in this Matplotlib version
    common = dict(useblit=True,
                  button=[1],              # left mouse only
                  minspanx=5, minspany=5,  # ignore tiny drags
                  spancoords="pixels",
                  interactive=True)

        # -------- RectangleSelector (version-agnostic) -----------------
    import inspect
    sel_kwargs = dict(    # common to all versions
        ax=ax,
        onselect=onselect,
        useblit=True,
        button=[1],              # left mouse button
        minspanx=5, minspany=5,  # ignore tiny drags
        spancoords="pixels",
        interactive=True,
    )

    # Style for the rectangle
    rect_style = dict(facecolor="none",
                      edgecolor="red",
                      linestyle="--",
                      linewidth=1)

    sig = inspect.signature(RectangleSelector)
    if "props" in sig.parameters:
        sel_kwargs["props"] = rect_style          # Matplotlib ≥ 3.5
    elif "rectprops" in sig.parameters:
        sel_kwargs["rectprops"] = rect_style      # Matplotlib ≤ 3.4
    # (do NOT add drawtype)

    rect = RectangleSelector(**sel_kwargs)

    fig.canvas.mpl_connect("key_press_event", toggle_quit)
    plt.show()

    if roi:
        r0, r1, c0, c1 = roi
        sub = img[r0:r1, c0:c1]          # NumPy view of the ROI
        # Example: save the ROI if you like
        # Image.fromarray(sub).save("roi.png")
    else:
        print("No region selected.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python zoom_roi.py <image_file>")
        sys.exit(1)

    main(sys.argv[1])
