"""
minecraft_house_frame.py
Generate a rectangular house “skeleton” for Minecraft, write it as a
.mcfunction full of /setblock commands, and preview it with Matplotlib.

Author: ChatGPT · updated 2025-06-15

Changes (2025-06-15)
--------------------
* Inner outline now uses **stone bricks** for the first floor only and **calcite**
  for every floor above it (if two or more floors exist).
* No other behaviour has changed.
"""

from __future__ import annotations
import random
from pathlib import Path
from typing import List, Tuple
from math import copysign
from HouseGen.detail.windows import *
from HouseGen.roof.gaben import *
from HouseGen.layout.roof_placement_area import roof_areas

# ────────────────────────────────────────────────────────────────
# 1.  Choose how many floors and their heights
# ──────────────────────────────────────────────────────────

def generate_floor_heights(
    num_floors: int | None = None,
    min_height: int = 4,
    max_height: int = 7,
    seed: int | None = None,
) -> List[int]:
    """Return a list like `[5, 4, 6] (3 floors)."""
    rng = random.Random(seed)
    if num_floors is None:
        num_floors = rng.randint(1, 3)
    return [rng.randint(min_height, max_height) for _ in range(num_floors)]



# ─────────────────────────────────────────────────────────────────────────────
def _areas_for_floors(
    areas,                       # Sequence[Sequence[tuple]]  OR  Sequence[tuple]
    n_floors: int
) -> List[List[Tuple[float, float]]]:
    """
    Guarantee a list of *n_floors* polygons:

    * If *areas* is already a list/tuple **of** polygons, keep it (pad by
      repeating the last footprint if it is shorter than *n_floors*).
    * If *areas* is a **single** polygon, copy it *n_floors* times.
    """
    # single polygon? (areas[0] is a pair like (x, z))
    is_single_polygon = areas and isinstance(areas[0], (list, tuple)) \
        and len(areas[0]) == 2 and isinstance(areas[0][0], (int, float))

    if is_single_polygon:
        return [list(areas)] * n_floors

    # already a list of polygons
    out = [list(p) for p in areas]
    if len(out) < n_floors:          # pad if needed
        out.extend([out[-1]] * (n_floors - len(out)))
    return out





# ──────────────────────────────────────────────────────────
# 2.  Frame pillars + beams (oak logs)
# ──────────────────────────────────────────────────────────

def build_frame_commands(
    areas,                          # single poly OR list[poly]
    floor_heights: List[int],
    y_base: int = 64,
    block: str = "minecraft:oak_log",
) -> List[str]:
    """
    Pillars + top-beams, but now each floor may have its *own* footprint.
    """
    footprints = _areas_for_floors(areas, len(floor_heights))
    cmds, y_cursor = [], y_base

    for floor_idx, (h, area) in enumerate(zip(floor_heights, footprints)):
        corners = area[:-1] if area[0] == area[-1] else area

        # pillars for this floor
        for x, z in corners:
            for dy in range(h):
                cmds.append(
                    f"setblock {int(x)} {y_cursor+dy} {int(z)} {block}[axis=y]"
                )

        # top ring of beams
        y_top = y_cursor + h - 1
        for (x1, z1), (x2, z2) in zip(corners, corners[1:] + corners[:1]):
            if x1 == x2:                               # vertical edge (Z runs)
                step = 1 if z2 > z1 else -1
                for z in range(int(z1), int(z2 + step), step):
                    cmds.append(
                        f"setblock {int(x1)} {y_top} {int(z)} {block}[axis=z]"
                    )
            elif z1 == z2:                             # horizontal edge (X runs)
                step = 1 if x2 > x1 else -1
                for x in range(int(x1), int(x2 + step), step):
                    cmds.append(
                        f"setblock {int(x)} {y_top} {int(z1)} {block}[axis=x]"
                    )
        y_cursor += h

    return cmds


# ─────────────────────────────────────────────────────────────────────────────
# 2-bis.  Inner outline that stays INSIDE any axis-aligned (possibly concave)
#         footprint – stone bricks for floor 1, calcite for floors 2+
# ─────────────────────────────────────────────────────────────────────────────

# helper ─────────────────────────────────────────────────────────────────────
def _poly_orientation(pts: List[Tuple[float, float]]) -> int:
    """Shoelace sign: +1 = CCW, -1 = CW (0 not expected)."""
    twice_area = sum(x1 * z2 - x2 * z1
                     for (x1, z1), (x2, z2) in zip(pts, pts[1:] + pts[:1]))
    return 1 if twice_area > 0 else -1


def _offset_polygon_axis_aligned(
    pts: List[Tuple[float, float]],
    d: int,
) -> List[Tuple[int, int]]:
    """
    Return the vertex list of the footprint uniformly *eroded* by ``d`` blocks.
    Works for any axis-aligned, simple (non-self-intersecting) polygon – concave
    shapes totally fine.
    """
    ori = _poly_orientation(pts)               # +1 = CCW → “left is inside”

    def normal(dx: int, dz: int) -> Tuple[int, int]:
        # inward unit normal for axis-aligned edge
        if   dx: return (0,  ori * int(copysign(1, dx)))
        elif dz: return (-ori * int(copysign(1, dz)), 0)
        raise ValueError("non-axis-aligned edge")

    n = len(pts)
    result: list[tuple[int, int]] = []

    for i in range(n):
        x_prev, z_prev = pts[i - 1]
        x_cur,  z_cur  = pts[i]
        x_next, z_next = pts[(i + 1) % n]

        # outgoing & incoming direction vectors
        v_prev = (x_cur - x_prev,  z_cur - z_prev)
        v_next = (x_next - x_cur,  z_next - z_cur)

        n_prev = normal(*v_prev)               # inward normal of previous edge
        n_next = normal(*v_next)               # inward normal of next edge

        # Each offset edge is a line:  x = const   or   z = const
        # Take their intersection ⇒ the corner of the shrunken polygon.
        if v_prev[0] == 0:                     # prev edge vertical ⇒ x = …
            x_off = int(x_cur + n_prev[0] * d)
            z_off = int(z_cur + n_next[1] * d) # next must be horizontal
        else:                                  # prev edge horizontal ⇒ z = …
            z_off = int(z_cur + n_prev[1] * d)
            x_off = int(x_cur + n_next[0] * d) # next must be vertical

        result.append((x_off, z_off))

    return result


# ─────────────────────────────────────────────────────────────────────────────
def build_inner_outline_commands(
    areas,                          # single poly OR list[poly]
    floor_heights: List[int],
    y_base: int = 64,
    offset: int = 1,
    block_first: str = "minecraft:stone_bricks",
    block_upper: str = "minecraft:calcite",
) -> List[str]:
    """
    • Footprint may change per floor.
    • First floor uses *block_first*, higher floors use *block_upper*.
    """
    footprints = _areas_for_floors(areas, len(floor_heights))

    cmds_set: set[str] = set()
    y_cursor = y_base

    for floor_idx, (h, area) in enumerate(zip(floor_heights, footprints)):
        material = block_first if floor_idx == 0 else block_upper
        inner = _offset_polygon_axis_aligned(
            area[:-1] if area[0] == area[-1] else area,
            offset,
        )

        for y in range(y_cursor, y_cursor + h):
            for (x1, z1), (x2, z2) in zip(inner, inner[1:] + inner[:1]):
                if x1 == x2:                         # vertical segment
                    step = 1 if z2 > z1 else -1
                    for z in range(z1, z2 + step, step):
                        cmds_set.add(f"setblock {x1} {y} {z} {material}")
                else:                                # horizontal segment
                    step = 1 if x2 > x1 else -1
                    for x in range(x1, x2 + step, step):
                        cmds_set.add(f"setblock {x} {y} {z1} {material}")

        y_cursor += h

    return sorted(cmds_set)


# ──────────────────────────────────────────────────────────
# 3.  Save commands as a .mcfunction
# ──────────────────────────────────────────────────────────

def save_mcfunction(commands: List[str], filename: str | Path = "house_frame.mcfunction") -> Path:
    path = Path(filename)
    path.write_text("\n".join(commands), encoding="utf-8")
    return path


# ──────────────────────────────────────────────────────────
# 4.  Matplotlib preview (Y → Z)
# ──────────────────────────────────────────────────────────

def frame_segments(
    areas,                       # single poly OR list[poly]
    floor_heights: List[int],
    y_base: int = 64,
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    Return segments [((x1,y1,z1),(x2,y2,z2)), …] for *all* floors.

    • Accepts either one footprint or one per floor (same `_areas_for_floors`
      logic used by the builders).
    """
    footprints = _areas_for_floors(areas, len(floor_heights))
    segs: list[tuple[tuple[float, float, float],
                     tuple[float, float, float]]] = []

    y_cursor = y_base
    for h, area in zip(floor_heights, footprints):
        corners = area[:-1] if area[0] == area[-1] else area
        y_top   = y_cursor + h

        # pillars
        for x, z in corners:
            segs.append(((x, y_cursor, z), (x, y_top, z)))

        # beams at top of this floor
        for (x1, z1), (x2, z2) in zip(corners, corners[1:] + corners[:1]):
            segs.append(((x1, y_top, z1), (x2, y_top, z2)))

        y_cursor = y_top

    return segs


def plot_frame(
    areas,                       # single poly OR list[poly]
    floor_heights: List[int],
    y_base: int = 64,
) -> None:
    """Matplotlib 3-D wireframe preview (Minecraft Y → Matplotlib Z)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    segs = frame_segments(areas, floor_heights, y_base)

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_title("House frame preview (Y→Z)")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")       # depth in-game
    ax.set_zlabel("Y")       # height in-game

    for (x1, y1, z1), (x2, y2, z2) in segs:
        ax.plot([x1, x2], [z1, z2], [y1, y2])   # note swap Z↔Y

    xs, zs, ys = zip(*[(p[0], p[2], p[1]) for s in segs for p in s])
    ax.auto_scale_xyz(xs, zs, ys)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Example usage when you run “python minecraft_house_frame.py”
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ❶ Define your building site (rectangle from your original prompt)
    house_area = [[(31, -80), (31, -64), (15, -64), (15, -80), (31, -80)], 
                  [(15.0, -80.0), (15.0, -64.0), (31.0, -64.0), (31.0, -73.0), (24.0, -73.0), (24.0, -80.0), (15.0, -80.0)],
                    [(15.0, -64.0), (31.0, -64.0), (31.0, -73.0), (15.0, -73.0), (15.0, -64.0)]]

    # ❷ Decide floor heights (reproducible because we set a seed)
    heights = generate_floor_heights(seed=44)
    y0 = -60
    print("Floor heights:", heights)
    heights = [4, 5, 5]

    # ❸ Produce the command list & save it
    frame_cmds = build_frame_commands(house_area, heights, y_base=y0)
    outline_cmds = build_inner_outline_commands(house_area, heights, y_base=y0)

    roof_y = y0 + sum(heights)          # same y you passed to outline
    # last_footprint = house_area[-1]              # topmost storey
    footprint = roof_areas(house_area)
    last_footprint = footprint[-1]
    roof_cmds = build_roof_commands(last_footprint, roof_y)
    gable_cmds = build_gable_fill_commands(last_footprint, roof_y)
    overhang_cmds1 = build_side_eave_overhang_commands(last_footprint, roof_y)
    overhang_cmds2 = build_gable_overhang_commands(last_footprint, roof_y)
    roof_cmds = roof_cmds + gable_cmds + overhang_cmds1 + overhang_cmds2



    window_cmds = build_window_frames(house_area, heights, y0)
    windows = build_window_panes(house_area, heights, y0, skip_floors=[1])
    trim = build_window_trim_frameplane(house_area, heights, y_base=y0)
    detail = build_window_greenery(house_area, heights, y_base=y0, skip_floors=[1])

    cmds = frame_cmds + outline_cmds + roof_cmds + window_cmds + windows + trim + detail # combine everything

    # out_path = save_mcfunction(cmds)
    # print(f"Saved {len(cmds):,} commands to {out_path}")

    # # ❹ Optional preview
    # plot_frame(house_area, heights)