"""
house_frame_plot.py
Create a wire-frame skeleton for a rectangular Minecraft house
and preview it in 3-D with Matplotlib.

Usage example (run from shell):
    python house_frame_plot.py
"""

from __future__ import annotations
import random
from typing import List, Tuple

# -------------------------------------------------------------------------
# 1.  Decide how many floors and how tall each one is
# -------------------------------------------------------------------------
def generate_floor_heights(
    num_floors: int | None = None,
    min_height: int = 3,
    max_height: int = 5,
    seed: int | None = None,
) -> List[int]:
    """
    Return e.g. [4, 3, 5] meaning 3 floors of height 4, 3 and 5 blocks.
    Omit *num_floors* to let the function choose 1–3 floors at random.
    """
    rng = random.Random(seed)
    if num_floors is None:
        num_floors = rng.randint(1, 3)
    return [rng.randint(min_height, max_height) for _ in range(num_floors)]


# -------------------------------------------------------------------------
# 2.  Compute just the coordinates we’ll need for plotting
# -------------------------------------------------------------------------
def frame_segments(
    area: List[Tuple[float, float]],
    floor_heights: List[int],
    y_base: int = 64,
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    Return a list of line-segment endpoints ((x1, y1, z1), (x2, y2, z2))
    representing pillars and beams for the given *area* polygon.
    """
    # drop duplicate closing point if the user included it
    corners = area[:-1] if area[0] == area[-1] else area
    segments: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []

    # -- vertical pillars ---------------------------------------------------
    y_cursor = y_base
    for h in floor_heights:
        y_top = y_cursor + h
        for x, z in corners:
            segments.append(((x, y_cursor, z), (x, y_top, z)))
        y_cursor = y_top

    # -- horizontal beams on each floor ------------------------------------
    y_cursor = y_base
    for h in floor_heights:
        y_beam = y_cursor + h
        for i in range(len(corners)):
            (x1, z1), (x2, z2) = corners[i], corners[(i + 1) % len(corners)]
            segments.append(((x1, y_beam, z1), (x2, y_beam, z2)))
        y_cursor += h

    return segments


# -------------------------------------------------------------------------
# 3.  Plot the frame (requires Matplotlib with mplot3d)
# -------------------------------------------------------------------------
def plot_frame(
    area: list[tuple[float, float]],
    floor_heights: list[int],
    y_base: int = 64,
) -> None:
    """
    3-D wire-frame preview with Y (height) shown on Matplotlib’s Z-axis.

    X  → east–west in-game  
    Y  → height in-game **→ Z axis in the plot**  
    Z  → north–south in-game **→ Y axis in the plot**
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (side-effect import)

    segs = frame_segments(area, floor_heights, y_base)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("House frame preview (Y↔Z swapped)")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")      # depth
    ax.set_zlabel("Y")      # height

    # draw every pillar/beam with the swapped coordinate order
    for (x1, y1, z1), (x2, y2, z2) in segs:
        ax.plot([x1, x2], [z1, z2], [y1, y2])  #  <-- y & z swapped here

    # rescale so the model isn’t distorted
    xs, zs, ys = zip(*[(p[0], p[2], p[1]) for seg in segs for p in seg])
    ax.auto_scale_xyz(xs, zs, ys)

    plt.show()



# -------------------------------------------------------------------------
# 4.  Example run
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Rectangle from your prompt
    house_area = [[(31, -80), (31, -64), (15, -64), (15, -80), (31, -80)], [(31.0, -80.0), (15.0, -80.0), (15.0, -71.0), (31.0, -71.0), (31.0, -80.0)]]

    # Reproducible random example
    heights = generate_floor_heights(seed=44)
    print("Floor heights picked:", heights)

    plot_frame(house_area, heights)
