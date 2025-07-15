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
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 2-ter-rev.  Gable roof without intrusive upside-down stairs
# ─────────────────────────────────────────────────────────────────────────────
def build_roof_commands(
    last_area: List[Tuple[float, float]],
    roof_base_y: int,
    block: str = "minecraft:oak_stairs",
    *,
    add_ridge_cap: bool = False,              # <-- default OFF
    cap_block: str = "minecraft:oak_slab",    # thin slab, keeps attic hollow
) -> List[str]:
    """
    Classic gable roof.

    Parameters
    ----------
    last_area
        Footprint of the top storey (any axis-aligned polygon; bounding box is
        used).
    roof_base_y
        Height of the *ceiling* of the highest floor (first row of stairs
        sits right on this Y).
    block
        Stair block id for the slopes.
    add_ridge_cap
        If True, a one-block-thick strip of *cap_block* is placed on the ridge.
        Default False (prevents the upside-down-stair issue).
    cap_block
        Block id for the ridge trim (ignored when *add_ridge_cap* is False).
        Using slabs keeps the interior unobstructed.
    """
    # ── bounding rectangle ───────────────────────────────────────────────
    xs = [int(x) for x, _ in last_area]
    zs = [int(z) for _, z in last_area]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)

    w_x = max_x - min_x + 1
    w_z = max_z - min_z + 1

    ridge_axis  = "x" if w_x >= w_z else "z"
    slope_span  = w_z if ridge_axis == "x" else w_x
    slope_depth = (slope_span + 1) // 2        # ceil(span / 2)

    cmds: list[str] = []

    # ── slope layers ─────────────────────────────────────────────────────
    for i in range(slope_depth):
        y = roof_base_y + i

        if ridge_axis == "x":                  # slopes north/south
            nz, sz = min_z + i, max_z - i
            for x in range(min_x, max_x + 1):
                cmds.append(
                    f"setblock {x} {y} {nz} {block}[facing=south,shape=straight]"
                )
                if nz != sz:
                    cmds.append(
                        f"setblock {x} {y} {sz} {block}[facing=north,shape=straight]"
                    )
        else:                                  # slopes west/east
            wx, ex = min_x + i, max_x - i
            for z in range(min_z, max_z + 1):
                cmds.append(
                    f"setblock {wx} {y} {z} {block}[facing=east,shape=straight]"
                )
                if wx != ex:
                    cmds.append(
                        f"setblock {ex} {y} {z} {block}[facing=west,shape=straight]"
                    )

    # ── optional ridge cap (now slabs, OFF by default) ──────────────────
    if add_ridge_cap:
        cap_y = roof_base_y + slope_depth      # exactly one layer above last stair
        if ridge_axis == "x":
            rz = min_z + slope_depth - 1
            for x in range(min_x, max_x + 1):
                cmds.append(f"setblock {x} {cap_y} {rz} {cap_block}")
        else:
            rx = min_x + slope_depth - 1
            for z in range(min_z, max_z + 1):
                cmds.append(f"setblock {rx} {cap_y} {z} {cap_block}")

    return cmds


# ─────────────────────────────────────────────────────────────────────────────
# 2-quater.  Close the two gable ends with oak-plank triangles
# ─────────────────────────────────────────────────────────────────────────────
def build_gable_fill_commands(
    last_area: List[Tuple[float, float]],
    roof_base_y: int,
    block: str = "minecraft:oak_planks",
) -> List[str]:
    """
    Add one-block-thick triangular walls at both gable ends (west/east or
    north/south, depending on which axis the ridge runs).

    • Sits flush *inside* the outer row of stairs → no overlap.  
    • Keeps the roof cavity hollow.  
    • Works for any rectangular top footprint (convex plans are fine too;
      we only use the bounding box to locate the gable planes).
    """
    # ── bounding rectangle of the top footprint ───────────────────────────
    xs = [int(x) for x, _ in last_area]
    zs = [int(z) for _, z in last_area]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)

    width_x = max_x - min_x + 1
    width_z = max_z - min_z + 1

    # decide which side is the ridge (longer) and which is the slope (shorter)
    if width_x >= width_z:
        ridge_axis   = "x"
        slope_depth  = (width_z + 1) // 2           # ⌈width/2⌉
        gable_planes = [min_x, max_x]               # constant-X walls
    else:
        ridge_axis   = "z"
        slope_depth  = (width_x + 1) // 2
        gable_planes = [min_z, max_z]               # constant-Z walls

    cmds: list[str] = []

    for i in range(slope_depth):
        y = roof_base_y + i                         # layer height

        if ridge_axis == "x":                       # roof slopes N/S → gables W/E
            z_start = min_z + i + 1                 # +1 to skip the stair block
            z_end   = max_z - i - 1                 # -1 idem on the far side
            if z_start > z_end:                     # nothing to fill on this tier
                continue

            for z in range(z_start, z_end + 1):
                for x in gable_planes:              # west *and* east walls
                    cmds.append(f"setblock {x} {y} {z} {block}")

        else:                                       # roof slopes W/E → gables N/S
            x_start = min_x + i + 1
            x_end   = max_x - i - 1
            if x_start > x_end:
                continue

            for x in range(x_start, x_end + 1):
                for z in gable_planes:              # north *and* south walls
                    cmds.append(f"setblock {x} {y} {z} {block}")

    return cmds


# ─────────────────────────────────────────────────────────────────────────────
# 2-sexies-rev.  Single-row cobblestone eave (1 Y lower)
# ─────────────────────────────────────────────────────────────────────────────
def build_side_eave_overhang_commands(
    last_area: List[Tuple[float, float]],
    roof_base_y: int,
    stair_block: str = "minecraft:cobblestone_stairs",
) -> List[str]:
    """
    Put a single row of *stair_block* along the long eaves, **one level below**
    the first oak stair layer.
    """
    # bounding rectangle
    xs = [int(x) for x, _ in last_area]
    zs = [int(z) for _, z in last_area]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)

    w_x = max_x - min_x + 1
    w_z = max_z - min_z + 1

    y_eave = roof_base_y - 1           # ↓ one block lower than oak roof

    cmds: list[str] = []

    if w_x >= w_z:                     # ridge E-W → eaves N & S
        for x in range(min_x, max_x + 1):
            cmds.append(
                f"setblock {x} {y_eave} {min_z-1} {stair_block}[facing=south,shape=straight]"
            )
            cmds.append(
                f"setblock {x} {y_eave} {max_z+1} {stair_block}[facing=north,shape=straight]"
            )
    else:                              # ridge N-S → eaves W & E
        for z in range(min_z, max_z + 1):
            cmds.append(
                f"setblock {min_x-1} {y_eave} {z} {stair_block}[facing=east,shape=straight]"
            )
            cmds.append(
                f"setblock {max_x+1} {y_eave} {z} {stair_block}[facing=west,shape=straight]"
            )

    return cmds


# ─────────────────────────────────────────────────────────────────────────────
# 2-septies (FINAL) — gable overhang that matches roof facing & height
# ─────────────────────────────────────────────────────────────────────────────
def build_gable_overhang_commands(
    last_area: list[tuple[float, float]],
    roof_base_y: int,
    stair_block: str = "minecraft:cobblestone_stairs",
) -> list[str]:
    """
    Extend the gable roof by one block with cobblestone stairs whose *facing*
    matches the underlying oak roof.

    • Works for any rectangular top footprint.
    • Leaves interior blocks untouched.
    """
    # ── bounding rectangle ────────────────────────────────────────────────
    xs = [int(x) for x, _ in last_area]
    zs = [int(z) for _, z in last_area]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)

    w_x = max_x - min_x + 1
    w_z = max_z - min_z + 1
    ridge_axis = "x" if w_x >= w_z else "z"        # long side ⇒ ridge
    slope_span = w_z if ridge_axis == "x" else w_x
    layers     = (slope_span + 1) // 2             # ⌈span/2⌉

    cmds: list[str] = []

    # helper: N ↔ S,  E ↔ W
    def _opposite_face(face: str) -> str:
        return {"north": "south", "south": "north",
                "east":  "west",  "west":  "east"}[face]

    def add_pair(x: int, y: int, z: int, face: str) -> None:
        """
        Upright stair faces *face* (outwards).  
        Upside-down stair faces the **opposite** direction so its smooth side
        shows from outside.
        """
        # upright
        cmds.append(
            f"setblock {x} {y} {z} {stair_block}[facing={face},shape=straight]"
        )
        # upside-down, opposite facing
        cmds.append(
            f"setblock {x} {y-1} {z} "
            f"{stair_block}[facing={_opposite_face(face)},half=top,shape=straight]"
        )
    
    # ── build layer-by-layer ──────────────────────────────────────────────
    for i in range(layers):
        y_layer = roof_base_y + i       # first upright stair sits at roof_base_y

        if ridge_axis == "x":           # gables = west & east walls
            z_n = min_z + i             # north slope edge
            z_s = max_z - i             # south slope edge
            x_w = min_x - 1             # west overhang column
            x_e = max_x + 1             # east overhang column

            # west gable: north-slope stair faces SOUTH, south-slope faces NORTH
            add_pair(x_w, y_layer, z_n, "south")
            if z_s != z_n:
                add_pair(x_w, y_layer, z_s, "north")

            # east gable: same facings
            add_pair(x_e, y_layer, z_n, "south")
            if z_s != z_n:
                add_pair(x_e, y_layer, z_s, "north")

        else:                           # ridge N-S ⇒ gables = north & south
            x_w = min_x + i             # west slope edge
            x_e = max_x - i             # east slope edge
            z_n = min_z - 1             # north overhang row
            z_s = max_z + 1             # south overhang row

            # north gable: west-slope stair faces EAST, east-slope faces WEST
            add_pair(x_w, y_layer, z_n, "east")
            if x_e != x_w:
                add_pair(x_e, y_layer, z_n, "west")

            # south gable: same facings
            add_pair(x_w, y_layer, z_s, "east")
            if x_e != x_w:
                add_pair(x_e, y_layer, z_s, "west")

    return cmds