# pip install shapely
from shapely.geometry import Polygon, GeometryCollection, MultiPolygon
from shapely.ops import unary_union

def _to_polygon(ring):
    """Convert a closed ring [(x,z), ...] to a shapely Polygon."""
    pts = list(ring)
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]  # shapely doesn't need the closing point
    return Polygon(pts)

def _poly_to_exterior_rings(geom, ndigits=6):
    """Return list of exterior rings (closed) from Polygon/MultiPolygon."""
    if geom.is_empty:
        return []
    geoms = []
    if geom.geom_type == "Polygon":
        geoms = [geom]
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        gems = getattr(geom, "geoms", [])
        geoms = [g for g in gems if g.geom_type == "Polygon"]

    out = []
    for g in geoms:
        coords = [(round(x, ndigits), round(y, ndigits)) for x, y in g.exterior.coords]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        out.append(coords)
    return out

def roof_areas(house_area):
    """
    For each polygon, subtract the union of all polygons above it and
    return the list of *visible* (unblocked) areas as closed rings.
    """
    polys = [_to_polygon(r) for r in house_area]
    n = len(polys)
    visible = []

    for i in range(n):
        above = unary_union(polys[i+1:]) if i + 1 < n else GeometryCollection()
        vis = polys[i].difference(above)
        visible.extend(_poly_to_exterior_rings(vis))

    return visible

# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon as MplPolygon

# def _closed(ring):
#     pts = list(ring)
#     if not pts:
#         return pts
#     if pts[0] != pts[-1]:
#         pts.append(pts[0])
#     return pts

# def plot_ring(ring, ax=None, show=True, fill=False, **kwargs):
#     """
#     Plot a single closed ring [(x, y), ...].
#     - fill=False plots an outline; fill=True uses a filled patch.
#     - **kwargs are passed to plot() or Polygon().
#     """
#     pts = _closed(ring)
#     if ax is None:
#         fig, ax = plt.subplots()

#     if fill:
#         ax.add_patch(MplPolygon(pts, closed=True, fill=True, **kwargs))
#     else:
#         xs, ys = zip(*pts)
#         ax.plot(xs, ys, **kwargs)

#     ax.set_aspect('equal', adjustable='box')
#     ax.grid(True)
#     ax.margins(0.05)
#     if show:
#         plt.show()
#     return ax

# def plot_rings(rings, ax=None, show=True, fill=False, **kwargs):
#     """
#     Plot multiple rings. `rings` is an iterable of rings (each is [(x, y), ...]).
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     for ring in rings:
#         plot_ring(ring, ax=ax, show=False, fill=fill, **kwargs)
#     if show:
#         plt.show()
#     return ax

# house_area = [[(31, -80), (31, -64), (15, -64), (15, -80), (31, -80)], [(15.0, -80.0), (15.0, -64.0), (31.0, -64.0), (31.0, -73.0), (24.0, -73.0), (24.0, -80.0), (15.0, -80.0)], [(15.0, -64.0), (31.0, -64.0), (31.0, -73.0), (15.0, -73.0), (15.0, -64.0)]]
# v1 = roof_areas(house_area)
# print(v1[2])