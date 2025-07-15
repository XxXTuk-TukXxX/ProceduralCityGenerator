import math
import random
from typing import List, Tuple
import numpy as np
from shapely.geometry import Point, Polygon, box, MultiPolygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt

Coord = Tuple[float, float]
Ring  = List[Coord]

def generate_random_polygon(seed, radius, num_vertices):
    random.seed(seed)
    angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(num_vertices)])
    polygon_points = []
    for angle in angles:
        r = random.uniform(radius * 0.5, radius)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        polygon_points.append((x, y))
    polygon_points.append(polygon_points[0])
    return polygon_points

class CityPlanner:
    def __init__(self, outline_ring: Ring) -> None:
        ring = [tuple(np.asarray(pt, float)) for pt in outline_ring]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        self.outline_ring: Ring    = ring
        self.outline_poly: Polygon = Polygon(ring).buffer(0)

    def build_grid(self, spacing = 5.0, include_edge: bool = True,) -> List[Tuple[float, float]]:
        h = max(spacing, 5.0)
        minx, miny, maxx, maxy = self.outline_poly.bounds
        xs = np.arange(minx, maxx + h * 0.5, h)
        ys = np.arange(miny, maxy + h * 0.5, h)
        keep = []
        poly  = self.outline_poly
        for x in xs:
            for y in ys:
                pt = Point(x, y)
                if poly.contains(pt) or (include_edge and poly.touches(pt)):
                    keep.append((x, y))
        return keep

    @staticmethod
    def grid_outline_from_points(grid_pts, spacing = 5.0):
        h = max(spacing, 5.0)
        nodes = {(float(x), float(y)) for x, y in grid_pts}
        if not nodes:
            return Polygon()
        tiles = []
        for x, y in nodes:
            if (
                (x + h, y)     in nodes and
                (x,     y + h) in nodes and
                (x + h, y + h) in nodes
            ):
                tiles.append(box(x, y, x + h, y + h))
        if not tiles:
            return Polygon()
        return unary_union(tiles).buffer(0)



# --- Mask-and-Fill Rectangle Packing ---
def mask_and_fill_rectangles(
    outline: Polygon,
    w_range=(5, 16), h_range=(5, 16),
    step=2.0, spacing=5.0, max_houses=None, seed=0
):
    """
    Fast greedy fill of axis-aligned rectangles, largest-first, using masking.
    """
    random.seed(seed)
    available_poly = outline
    houses = []

    # Precompute sorted rectangle sizes (large-to-small, prioritize squares)
    sizes = []
    for w in np.arange(w_range[1], w_range[0]-1e-6, -step):
        for h in np.arange(h_range[1], h_range[0]-1e-6, -step):
            sizes.append((w, h))
    sizes.sort(key=lambda wh: (-min(wh), -max(wh)))  # prioritize squares

    placed = True
    while placed:
        placed = False
        minx, miny, maxx, maxy = available_poly.bounds
        xs = np.arange(minx, maxx, spacing)
        ys = np.arange(miny, maxy, spacing)
        candidates = [(x, y) for x in xs for y in ys if available_poly.contains(Point(x, y))]
        random.shuffle(candidates)

        for x, y in candidates:
            for w, h in sizes:
                rect = box(x, y, x + w, y + h)
                if available_poly.contains(rect):
                    houses.append(rect)
                    available_poly = available_poly.difference(rect)
                    placed = True
                    if max_houses and len(houses) >= max_houses:
                        return houses
                    break  # Don't try smaller rectangles at this position
            if placed:
                break  # Restart search after every placement for efficiency

    return houses
