# ─── Imports ───────────────────────────────────────────────────────────────────
import random
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry.polygon import orient   # ← NEW
from shapely.geometry import Polygon, LineString
from matplotlib.patches import Polygon as MplPoly
from area import Area
from territory import Territory
from houses.side_house import SideHouse

# ─── main() ────────────────────────────────────────────────────────────────────
def main():
    def random_hex_color():
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))
    
    # --- Generate the Area (Polygon) ------------------------------------------
    area = Area()
    territory = Territory(area)
    points = area.generate_poisson_disk_points()
    territory.generate_territory(points)
    regions = territory.get_regions()
    
    # Force the ring to be counter-clockwise; keeps “inward” pointing inward
    fig, ax = plt.subplots(figsize=(10, 10))
    for region in regions:
        poly            = orient(region, sign=1.0)          # 1.0 ⇒ CCW
        vertices        = np.array(poly.exterior.coords[:-1], dtype=float)
        vertices_closed = np.vstack([vertices, vertices[0]])
        # ───────────────────────────────────────────────────────────────────────────
        
        # (the rest of your pipeline is unchanged)
        plt.plot(vertices_closed[:, 0], vertices_closed[:, 1],
                'b-', linewidth=2, label='Area Polygon')

        side_houses = SideHouse(vertices_closed)
        side_houses.initialise_vertex_houses(
            randomize_size=True, min_size=5, max_size=10, overlap_threshold=90)
            

        polygon_vertices = [tuple(v) for v in vertices]
        ordered_houses = side_houses.get_ordered_vertex_houses(polygon_vertices)

        all_houses = side_houses.get_side_and_vertex_houses(polygon_vertices, ordered_houses)

        overlap_map = side_houses.overlap_dict(all_houses, area_tol=1e-6)
        overlapping_keys = set(overlap_map.keys())

        rand_edge_color = random_hex_color()

        # ───────────── draw every side house ───────────────
        for idx, (center, house) in enumerate(all_houses.items()):
            poly_coords = house + [house[0]]             # close the ring
            xs, ys = zip(*poly_coords)

            if center in overlapping_keys:              
                # filled, solid red
                # ax.add_patch(MplPoly(poly_coords, closed=True,
                #                     facecolor="red", edgecolor="red", alpha=0.6,
                #                     label='Overlapping House' if idx == 0 else ""))
                continue
            else:
                # your previous look: outline only, random colour
                ax.plot(xs, ys, color=rand_edge_color, linewidth=2,
                        label='Additional House' if idx == 0 else "")

    plt.title('City Generator: Area with Houses')
    plt.xlabel('X'); plt.ylabel('Y')
    plt.grid(True); plt.axis('equal'); plt.legend(); plt.show()

if __name__ == '__main__':
    main()