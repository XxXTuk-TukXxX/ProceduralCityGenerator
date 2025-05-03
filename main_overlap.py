# ─── Imports ───────────────────────────────────────────────────────────────────
import random
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry.polygon import orient
from shapely.geometry import Polygon
from matplotlib.patches import Polygon as MplPoly

from area        import Area
from territory   import Territory
from houses.side_house   import SideHouse
from houses.center_region import CenterRegion          # ← NEW

# ─── Helper ────────────────────────────────────────────────────────────────────
def random_hex_color() -> str:
    """Return a random hex colour, e.g.  '#3f7ab9'."""
    return f"#{random.randint(0, 0xFFFFFF):06x}"


# ─── main() ────────────────────────────────────────────────────────────────────
def main() -> None:

    # 1️⃣  Build the Voronoi‐like regions you already have
    area       = Area()
    territory  = Territory(area)
    points     = area.generate_poisson_disk_points()
    territory.generate_territory(points)
    regions    = territory.get_regions()

    # 2️⃣  Go through every region and decorate it with houses
    for region in regions:
        fig, ax = plt.subplots(figsize=(10, 10))

        # ——— region polygon (force CCW so “inward” is consistent)
        poly             = orient(region, sign=1.0)            # 1.0 ⇒ CCW
        vertices         = np.asarray(poly.exterior.coords[:-1], dtype=float)
        vertices_closed  = np.vstack([vertices, vertices[0]])

        ax.plot(vertices_closed[:, 0], vertices_closed[:, 1],
                'tab:blue', linewidth=2, label='Area Polygon')

        # ——— generate *side houses* along the boundary
        sh = SideHouse(vertices_closed)
        sh.initialise_vertex_houses(randomize_size=True,
                                    min_size=5, max_size=10,
                                    overlap_threshold=90)

        polygon_vertices = [tuple(v) for v in vertices]         # drop the closing pt
        ordered_houses   = sh.get_ordered_vertex_houses(polygon_vertices)
        all_houses       = sh.get_side_and_vertex_houses(polygon_vertices,
                                                         ordered_houses)

        # ——— mark overlapping houses (they will be skipped here)
        overlap_map      = sh.overlap_dict(all_houses, area_tol=1e-6)
        overlapping_keys = set(overlap_map)

        # ——— draw every *non-overlapping* side house
        edge_colour = random_hex_color()
        # ---------------- draw every remaining (non-overlapping) side-house ------------
        for idx, house in enumerate(all_houses.values()):
            ring = house + [house[0]]
            xs, ys = zip(*ring)
            ax.plot(xs, ys, color=edge_colour, linewidth=2,
                    label='Side House' if idx == 0 else "")


        # # ——— NEW: build the interior outline that dodges the houses
        # cr = CenterRegion(vertices_closed, all_houses, clearance=1.0)
        # for r_idx, ring in enumerate(cr.rings()):
        #     print(ring)
        #     xs, ys = zip(*ring)
        #     ax.plot(xs, ys,
        #             color="lime", lw=2.5,
        #             label="Interior Outline" if r_idx == 0 else "")

    # fig, ax = plt.subplots(figsize=(10, 10))
    # # ——— region polygon (force CCW so “inward” is consistent)
    # poly             = orient(regions[0], sign=1.0)            # 1.0 ⇒ CCW
    # vertices         = np.asarray(poly.exterior.coords[:-1], dtype=float)
    # vertices_closed  = np.vstack([vertices, vertices[0]])

    # ax.plot(vertices_closed[:, 0], vertices_closed[:, 1],
    #         'tab:blue', linewidth=2, label='Area Polygon')

    # # ——— generate *side houses* along the boundary
    # sh = SideHouse(vertices_closed)
    # sh.initialise_vertex_houses(randomize_size=True,
    #                             min_size=5, max_size=10,
    #                             overlap_threshold=90)

    # polygon_vertices = [tuple(v) for v in vertices]         # drop the closing pt
    # ordered_houses   = sh.get_ordered_vertex_houses(polygon_vertices)
    # all_houses       = sh.get_side_and_vertex_houses(polygon_vertices,
    #                                                     ordered_houses)

    # # ——— mark overlapping houses (they will be skipped here)
    # overlap_map      = sh.overlap_dict(all_houses, area_tol=1e-6)
    # overlapping_keys = set(overlap_map)

    # # ——— draw every *non-overlapping* side house
    # edge_colour = random_hex_color()
    # # ---------------- draw every remaining (non-overlapping) side-house ------------
    # for idx, house in enumerate(all_houses.values()):
    #     ring = house + [house[0]]
    #     xs, ys = zip(*ring)
    #     ax.plot(xs, ys, color=edge_colour, linewidth=2,
    #             label='Side House' if idx == 0 else "")


    # # ——— NEW: build the interior outline that dodges the houses
    # cr = CenterRegion(vertices_closed, all_houses, clearance=1.0)
    # for r_idx, ring in enumerate(cr.rings()):
    #     print(ring)
    #     xs, ys = zip(*ring)
    #     ax.plot(xs, ys,
    #             color="lime", lw=2.5,
    #             label="Interior Outline" if r_idx == 0 else "")

        # —— cosmetics for this region
        ax.set_aspect('equal', 'box')
        ax.set_title('City Generator: Area with Houses')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)
        ax.legend(loc='upper right')
        plt.show()               # or comment out to draw all on one figure

# ─── run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()
