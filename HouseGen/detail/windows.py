# ──────────────────────────────────────────────────────────
# Centered window frames (double pillar always in the MIDDLE)
# ──────────────────────────────────────────────────────────

def build_window_frames(
    areas,                      # single poly OR list[poly]
    floor_heights,              # e.g. [4,5,5]
    y_base,                     # base Y
    pillar_block: str = "minecraft:oak_log",
    trim_block:   str = "minecraft:spruce_planks",
) -> list[str]:
    """
    Pillars are placed symmetrically from BOTH ends every 4th block.
    • Start & end always pillars.
    • Interior pillars march inward (… 0,4,8 … and … L-1, L-5, …) until they meet:
        - meet on the same index  → single **center** pillar
        - meet on adjacent indices → **center double** pillar
    • Any gap between consecutive pillars that isn't exactly 3 gets spruce
      header/sill rows.
    """

    # local helpers (trimmed version of your utilities)
    def _areas_for_floors_simple(areas, n):
        is_single = areas and isinstance(areas[0], (list, tuple)) \
            and len(areas[0]) == 2 and isinstance(areas[0][0], (int, float))
        if is_single:
            return [list(areas)] * n
        out = [list(p) for p in areas]
        if len(out) < n:
            out.extend([out[-1]] * (n - len(out)))
        return out

    def _len_inclusive(p1, p2) -> int:
        (x1, z1), (x2, z2) = p1, p2
        if x1 != x2 and z1 != z2:
            raise ValueError("Non axis-aligned edge.")
        return abs(int(x2) - int(x1)) + abs(int(z2) - int(z1)) + 1

    def _centered_pillar_indices(L: int) -> list[int]:
        """
        Symmetric placement:
          pillars = {0, L-1} ∪ {4,8,12,…} ∪ {L-1-4, L-1-8, …} until they meet.
        Ensures any leftover forms a center **double** (or single) pillar.
        """
        if L <= 1:
            return [0] if L == 1 else []
        pillars = {0, L - 1}

        left, right = 4, L - 1 - 4
        while left <= right:
            if left == right:
                pillars.add(left)             # single center pillar
                break
            elif left == right + 1:
                pillars.add(right)            # center DOUBLE pillar
                pillars.add(left)
                break
            else:
                pillars.add(left)
                pillars.add(right)
                left += 4
                right -= 4

        return sorted(pillars)

    cmds: list[str] = []
    footprints = _areas_for_floors_simple(areas, len(floor_heights))

    y_cursor = int(y_base)
    for h, area in zip(floor_heights, footprints):
        h = int(h)
        y_top = y_cursor + h - 1
        corners = area[:-1] if area[0] == area[-1] else area

        for (x1f, z1f), (x2f, z2f) in zip(corners, corners[1:] + corners[:1]):
            x1, z1 = int(round(x1f)), int(round(z1f))
            x2, z2 = int(round(x2f)), int(round(z2f))
            if x1 == x2 and z1 == z2:
                continue  # degenerate edge

            along_x = (z1 == z2)
            step = 1 if (x2 > x1 or z2 > z1) else -1
            L = _len_inclusive((x1, z1), (x2, z2))
            pillars = _centered_pillar_indices(L)

            def world_at(i: int) -> tuple[int, int]:
                return ((x1 + step * i, z1) if along_x else (x1, z1 + step * i))

            # pillars (vertical logs)
            for idx in pillars:
                px, pz = world_at(idx)
                for dy in range(h):
                    cmds.append(f"setblock {px} {y_cursor+dy} {pz} {pillar_block}[axis=y]")

            # spruce trims over/under any non-3 gap
            for a, b in zip(pillars, pillars[1:]):
                span = b - a - 1
                if span <= 0:
                    continue
                if span != 3:
                    for off in range(1, span + 1):
                        tx, tz = world_at(a + off)
                        cmds.append(f"setblock {tx} {y_cursor} {tz} {trim_block}")   # sill
                        cmds.append(f"setblock {tx} {y_top} {tz} {trim_block}")      # header

        y_cursor += h

    return cmds

def build_window_panes(
    areas,                 # single poly OR list[poly]
    floor_heights,         # e.g. [4,5,5]
    y_base,                # base Y
    pane_block: str = "minecraft:glass_pane",
    *,
    # skip_floors: list[int] | set[int] | tuple[int, ...] | None = None,  # 1-based
    skip_floors = None
) -> list[str]:
    """Place centered panes in each 3-wide bay, recessed one block inward.
       Skip any floors listed in `skip_floors` (1-based: 1 = ground)."""
    cmds: list[str] = []
    skip = {int(f) for f in (skip_floors or [])}

    # helpers (same as before) ────────────────────────────────────────────
    def _areas_for_floors_simple(areas, n):
        is_single = areas and isinstance(areas[0], (list, tuple)) \
            and len(areas[0]) == 2 and isinstance(areas[0][0], (int, float))
        if is_single:
            return [list(areas)] * n
        out = [list(p) for p in areas]
        if len(out) < n:
            out.extend([out[-1]] * (n - len(out)))
        return out
    def _len_inclusive(p1, p2) -> int:
        (x1, z1), (x2, z2) = p1, p2
        if x1 != x2 and z1 != z2: raise ValueError("Non axis-aligned edge.")
        return abs(int(x2) - int(x1)) + abs(int(z2) - int(z1)) + 1
    def _centered_pillar_indices(L: int) -> list[int]:
        if L <= 1: return [0] if L == 1 else []
        pillars = {0, L-1}; left, right = 4, L-1-4
        while left <= right:
            if left == right: pillars.add(left); break
            if left == right + 1: pillars.add(right); pillars.add(left); break
            pillars.add(left); pillars.add(right); left += 4; right -= 4
        return sorted(pillars)
    def _poly_orientation(pts):
        s = sum(x1*z2 - x2*z1 for (x1,z1),(x2,z2) in zip(pts, pts[1:]+pts[:1]))
        return 1 if s > 0 else -1
    def _inward_normal(dx: int, dz: int, ori: int) -> tuple[int, int]:
        if dx != 0:  return (0, ori * (1 if dx > 0 else -1))
        else:        return (-ori * (1 if dz > 0 else -1), 0)

    footprints = _areas_for_floors_simple(areas, len(floor_heights))

    y_cursor = int(y_base)
    for floor_no, (h, area) in enumerate(zip(floor_heights, footprints), start=1):
        h = int(h)

        # skip floors requested or too short to show panes
        if floor_no in skip or h < 4:
            y_cursor += h
            continue

        # panes per rule; centered within visible (h-1); recessed 1 block
        num_panes = max(1, h - 3)        # 4→1, 5→2, …
        visible_h = h - 1
        start_y = y_cursor + (visible_h - num_panes) // 2
        end_y   = start_y + num_panes - 1

        corners = area[:-1] if area[0] == area[-1] else area
        ori = _poly_orientation([(int(x), int(z)) for x, z in corners])

        for (x1f, z1f), (x2f, z2f) in zip(corners, corners[1:] + corners[:1]):
            x1, z1 = int(round(x1f)), int(round(z1f))
            x2, z2 = int(round(x2f)), int(round(z2f))
            if x1 == x2 and z1 == z2: continue

            along_x = (z1 == z2)
            dx = (1 if x2 > x1 else -1) if along_x else 0
            dz = 0 if along_x else (1 if z2 > z1 else -1)
            nx, nz = _inward_normal(dx, dz, ori)      # recess direction
            step = 1 if (dx > 0 or dz > 0) else -1

            L = _len_inclusive((x1, z1), (x2, z2))
            pillars = _centered_pillar_indices(L)

            def world_at(i: int) -> tuple[int, int]:
                return ((x1 + step * i, z1) if along_x else (x1, z1 + step * i))

            for a, b in zip(pillars, pillars[1:]):
                if b - a - 1 != 3: continue
                mid_i = a + 2
                px, pz = world_at(mid_i)
                rx, rz = px + nx, pz + nz             # recessed pane column
                for y in range(start_y, end_y + 1):
                    cmds.append(f"setblock {rx} {y} {rz} {pane_block}")

        y_cursor += h

    return cmds


from typing import Optional, Iterable

def build_window_trim_frameplane(
    areas,                      # single poly OR list[poly]
    floor_heights,              # e.g. [4,5,5]
    y_base,                     # base Y
    *,
    stair_block: str = "minecraft:spruce_stairs",
    trap_block:  str = "minecraft:spruce_trapdoor",
    skip_floors: Optional[Iterable[int]] = None,   # 1-based, e.g. [1] to skip ground
) -> list[str]:
    """
    Window trim placed on the *frame* plane (no recess):
      • header trapdoor above the top pane,
      • upside-down stairs at both sides, facing AWAY from window center,
      • pillar trapdoors one block lower, OPEN and facing TOWARD the window.
    Pane height rule assumed: panes = max(1, floor_height - 3); trim aligns to top pane.
    """
    cmds: list[str] = []
    skip = {int(f) for f in (skip_floors or [])}

    # -------- helpers (self-contained) -----------------------------------
    def _areas_for_floors_simple(areas, n):
        is_single = areas and isinstance(areas[0], (list, tuple)) \
            and len(areas[0]) == 2 and isinstance(areas[0][0], (int, float))
        if is_single:
            return [list(areas)] * n
        out = [list(p) for p in areas]
        if len(out) < n:
            out.extend([out[-1]] * (n - len(out)))
        return out

    def _len_inclusive(p1, p2) -> int:
        (x1, z1), (x2, z2) = p1, p2
        if x1 != x2 and z1 != z2:
            raise ValueError("Non axis-aligned edge.")
        return abs(int(x2) - int(x1)) + abs(int(z2) - int(z1)) + 1

    def _centered_pillar_indices(L: int) -> list[int]:
        if L <= 1:
            return [0] if L == 1 else []
        pillars = {0, L - 1}
        left, right = 4, L - 1 - 4
        while left <= right:
            if left == right:                 # single center pillar
                pillars.add(left); break
            if left == right + 1:             # center double pillar
                pillars.add(right); pillars.add(left); break
            pillars.add(left); pillars.add(right)
            left += 4; right -= 4
        return sorted(pillars)

    footprints = _areas_for_floors_simple(areas, len(floor_heights))

    # -------- per floor ---------------------------------------------------
    y_cursor = int(y_base)
    for floor_no, (h, area) in enumerate(zip(floor_heights, footprints), start=1):
        h = int(h)
        if floor_no in skip or h < 4:
            y_cursor += h
            continue

        # vertical alignment (same as your panes rule)
        num_panes = max(1, h - 3)         # 4→1, 5→2, …
        visible_h = h - 1
        y_first = y_cursor + (visible_h - num_panes) // 2
        y_last  = y_first + num_panes - 1
        y_hdr   = y_last + 1              # header trapdoor
        y_sideS = y_last + 1              # side stairs level (upside-down)
        y_sideT = y_last                  # pillar trapdoors under the stairs

        corners = area[:-1] if area[0] == area[-1] else area

        for (x1f, z1f), (x2f, z2f) in zip(corners, corners[1:] + corners[:1]):
            x1, z1 = int(round(x1f)), int(round(z1f))
            x2, z2 = int(round(x2f)), int(round(z2f))
            if x1 == x2 and z1 == z2:
                continue

            along_x = (z1 == z2)
            step = 1 if (x2 > x1 or z2 > z1) else -1
            L = _len_inclusive((x1, z1), (x2, z2))
            pillars = _centered_pillar_indices(L)

            def world_at(i: int) -> tuple[int, int]:
                return ((x1 + step * i, z1) if along_x else (x1, z1 + step * i))

            # act only on true 3-wide bays
            for a, b in zip(pillars, pillars[1:]):
                if b - a - 1 != 3:
                    continue

                left_i, mid_i, right_i = a + 1, a + 2, a + 3
                lx, lz = world_at(left_i)
                mx, mz = world_at(mid_i)
                rx, rz = world_at(right_i)

                # 1) header trapdoor (frame plane)
                cmds.append(f"setblock {mx} {y_hdr} {mz} {trap_block}[half=top]")

                # 2) side stairs (frame plane), upside-down, facing AWAY from the window
                if along_x:
                    face_left  = "west" if (lx < mx) else "east"
                    face_right = "east" if (rx > mx) else "west"
                    cmds.append(f"setblock {lx} {y_sideS} {lz} {stair_block}[facing={face_left},half=top,shape=straight]")
                    cmds.append(f"setblock {rx} {y_sideS} {rz} {stair_block}[facing={face_right},half=top,shape=straight]")

                    # 3) pillar trapdoors (frame plane), open TOWARD the window
                    face_to_mid_left  = "east" if (lx < mx) else "west"
                    face_to_mid_right = "west" if (rx > mx) else "east"
                    cmds.append(f"setblock {lx} {y_sideT} {lz} {trap_block}[facing={face_to_mid_left},open=true]")
                    cmds.append(f"setblock {rx} {y_sideT} {rz} {trap_block}[facing={face_to_mid_right},open=true]")

                else:  # along Z
                    face_left  = "north" if (lz < mz) else "south"
                    face_right = "south" if (rz > mz) else "north"
                    cmds.append(f"setblock {lx} {y_sideS} {lz} {stair_block}[facing={face_left},half=top,shape=straight]")
                    cmds.append(f"setblock {rx} {y_sideS} {rz} {stair_block}[facing={face_right},half=top,shape=straight]")

                    face_to_mid_left  = "south" if (lz < mz) else "north"
                    face_to_mid_right = "north" if (rz > mz) else "south"
                    cmds.append(f"setblock {lx} {y_sideT} {lz} {trap_block}[facing={face_to_mid_left},open=true]")
                    cmds.append(f"setblock {rx} {y_sideT} {rz} {trap_block}[facing={face_to_mid_right},open=true]")

        y_cursor += h

    return cmds


import random

from typing import Optional, Iterable
import random

def build_window_greenery(
    areas,                      # single poly OR list[poly]
    floor_heights,              # e.g. [4,5,5]
    y_base,                     # base Y
    *,
    moss_block: str = "minecraft:moss_block",
    plant_block_list: Optional[Iterable[str]] = None,
    plant_keep_prob: float = 0.8,                 # 80% chance to place a plant
    trap_block: str = "minecraft:spruce_trapdoor",
    skip_floors: Optional[Iterable[int]] = None,  # 1-based: floors to skip
    seed: Optional[int] = None,                   # set for deterministic output
) -> list[str]:
    """
    For each 3-wide window bay:
      • place a 3-block row of MOSS on the *frame plane* directly below the lowest pane,
      • 80% chance to place a random plant on top of each moss,
      • place a CLOSED spruce trapdoor one block *outward* from each moss tile,
        oriented to face back toward the wall.
    """
    rng = random.Random(seed)
    cmds: list[str] = []
    skip = {int(f) for f in (skip_floors or [])}

    # ── helpers ───────────────────────────────────────────────────────────
    def _areas_for_floors_simple(areas, n):
        is_single = areas and isinstance(areas[0], (list, tuple)) \
            and len(areas[0]) == 2 and isinstance(areas[0][0], (int, float))
        if is_single:
            return [list(areas)] * n
        out = [list(p) for p in areas]
        if len(out) < n:
            out.extend([out[-1]] * (n - len(out)))
        return out

    def _len_inclusive(p1, p2) -> int:
        (x1, z1), (x2, z2) = p1, p2
        if x1 != x2 and z1 != z2:
            raise ValueError("Non axis-aligned edge.")
        return abs(int(x2) - int(x1)) + abs(int(z2) - int(z1)) + 1

    def _centered_pillar_indices(L: int) -> list[int]:
        if L <= 1:
            return [0] if L == 1 else []
        pillars = {0, L - 1}
        left, right = 4, L - 1 - 4
        while left <= right:
            if left == right: pillars.add(left); break
            if left == right + 1: pillars.add(right); pillars.add(left); break
            pillars.add(left); pillars.add(right); left += 4; right -= 4
        return sorted(pillars)

    def _poly_orientation(pts):
        s = sum(x1*z2 - x2*z1 for (x1, z1), (x2, z2) in zip(pts, pts[1:] + pts[:1]))
        return 1 if s > 0 else -1  # +1 CCW, -1 CW

    def _inward_normal(dx: int, dz: int, ori: int) -> tuple[int, int]:
        # inward unit normal for axis-aligned edge (frame → wall)
        if dx != 0:   # horizontal edge
            return (0, ori * (1 if dx > 0 else -1))
        else:         # vertical edge
            return (-ori * (1 if dz > 0 else -1), 0)

    def _facing_from_vec(vx: int, vz: int) -> str:
        # map a 4-neighbour vector to a facing string
        if   vx ==  1 and vz == 0: return "east"
        elif vx == -1 and vz == 0: return "west"
        elif vx ==  0 and vz == 1: return "south"
        elif vx ==  0 and vz ==-1: return "north"
        else: return "north"  # fallback (shouldn't happen)

    if plant_block_list is None:
        plant_block_list = [
            "minecraft:dandelion", "minecraft:poppy", "minecraft:azure_bluet",
            "minecraft:cornflower", "minecraft:allium", "minecraft:oxeye_daisy",
            "minecraft:fern"
        ]
    plants = list(plant_block_list)

    # ── main ──────────────────────────────────────────────────────────────
    footprints = _areas_for_floors_simple(areas, len(floor_heights))
    y_cursor = int(y_base)

    for floor_no, (h, area) in enumerate(zip(floor_heights, footprints), start=1):
        h = int(h)
        corners = area[:-1] if area[0] == area[-1] else area

        if floor_no in skip or h < 4:
            y_cursor += h
            continue

        # vertical: panes are centered in visible (h-1); moss sits just below the first pane
        num_panes = max(1, h - 3)             # 4→1, 5→2, …
        visible_h = h - 1
        y_first_pane = y_cursor + (visible_h - num_panes) // 2
        y_moss = y_first_pane - 1

        ori = _poly_orientation([(int(x), int(z)) for x, z in corners])

        for (x1f, z1f), (x2f, z2f) in zip(corners, corners[1:] + corners[:1]):
            x1, z1 = int(round(x1f)), int(round(z1f))
            x2, z2 = int(round(x2f)), int(round(z2f))
            if x1 == x2 and z1 == z2:
                continue

            along_x = (z1 == z2)
            dx = (1 if x2 > x1 else -1) if along_x else 0
            dz = 0 if along_x else (1 if z2 > z1 else -1)
            nx, nz = _inward_normal(dx, dz, ori)      # inward (toward wall)
            ox, oz = -nx, -nz                         # outward (exterior)
            face_outward = _facing_from_vec(ox, oz)   # trapdoor faces outward direction?
            # We want the trapdoor plate to face BACK toward the wall:
            face_to_wall = _facing_from_vec(nx, nz)

            step = 1 if (dx > 0 or dz > 0) else -1
            L = _len_inclusive((x1, z1), (x2, z2))
            pillars = _centered_pillar_indices(L)

            def world_at(i: int) -> tuple[int, int]:
                return ((x1 + step * i, z1) if along_x else (x1, z1 + step * i))

            for a, b in zip(pillars, pillars[1:]):
                if b - a - 1 != 3:
                    continue  # only true 3-wide windows

                # three columns in the bay on the FRAME plane (no recess)
                for i in (a + 1, a + 2, a + 3):
                    px, pz = world_at(i)

                    # 1) moss
                    cmds.append(f"setblock {px} {y_moss} {pz} {moss_block}")

                    # 2) plant (80% chance)
                    if rng.random() < plant_keep_prob:
                        plant = rng.choice(plants)
                        cmds.append(f"setblock {px} {y_moss+1} {pz} {plant}")
                    else:
                        cmds.append(f"setblock {px} {y_moss+1} {pz} minecraft:air")

                    # 3) outward CLOSED trapdoor, one block outside the frame
                    tx, tz = px + ox, pz + oz
                    # Closed trapdoor placed at the outward block, facing back toward the wall.
                    cmds.append(f"setblock {tx} {y_moss} {tz} {trap_block}[facing={face_outward}, open=true]")

        y_cursor += h

    return cmds

