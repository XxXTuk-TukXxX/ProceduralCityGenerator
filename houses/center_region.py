# center_region.py
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations
from typing import Sequence, Dict, List, Tuple
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly


Coord     = Tuple[float, float]
Ring      = List[Coord]
HouseDict = Dict[Coord, Ring]     # {house_center : [vertex,…]}


def _to_ring(seq: Sequence) -> Ring:
    """Convert np.ndarray / list / tuple to a *plain* list[tuple]."""
    return [tuple(np.asarray(pt).tolist()) for pt in seq]


# ──────────────────────────────────────────────────────────────────────
class CenterRegion:
    """
    Build an *interior* outline of a region while treating the given houses
    as solid obstacles.

    Parameters
    ----------
    region_vertices : Sequence[(x, y)]
        The region’s vertices (closed or open ring is fine).
    side_houses : dict | list
        Either the dict returned by your `SideHouse` pipeline **or**
        a simple list of rings – each ring being the house outline.
    clearance : float, default 1.0
        The amount by which to pull every boundary **inward** (region) and
        push every house **outward** before computing the outline.
        Set it to 0 to hug the raw edges.
    """

    # ───────── construction ─────────────────────────────────────────
    def __init__(
        self,
        region_vertices: Sequence[Coord],
        side_houses: HouseDict | List[Ring],
        clearance: float = 1.0,
    ) -> None:

        # 1) normalise input --------------------------------------------------
        region_ring = _to_ring(region_vertices)
        if region_ring[0] != region_ring[-1]:
            region_ring.append(region_ring[0])

        if isinstance(side_houses, dict):
            house_rings = [ _to_ring(r) for r in side_houses.values() ]
        else:
            house_rings = [ _to_ring(r) for r in side_houses ]

        self.region_poly     = Polygon(region_ring)
        self.house_polys     = [Polygon(r) for r in house_rings]
        self.clearance: float = float(clearance)

        self._outline: MultiPolygon | Polygon | None = None

    # ───────── geometry ------------------------------------------------------
    def _build_outline(self) -> None:
        """
        Build and cache the interior outline as a (Multi)Polygon.
        """
        if self._outline is not None:
            return                                                  # already done

        cl = self.clearance
        region_shrunk   = self.region_poly.buffer(-cl)
        houses_inflated = unary_union(self.house_polys).buffer(+cl) \
                          if self.house_polys else None

        free_space = (region_shrunk
                      if houses_inflated is None
                      else region_shrunk.difference(houses_inflated))

        # topo-fix in case buffer() produced artefacts
        self._outline = free_space.buffer(0)

    # ───────── public helpers -----------------------------------------------
    def rings(self) -> List[Ring]:
        """
        Return all exterior rings of the *free* area,
        ready for plotting.  (Holes, if any, are ignored.)
        """
        self._build_outline()

        geom        = self._outline
        extract     = (lambda p: [list(p.exterior.coords)])
        all_rings   = []

        if geom.is_empty:
            return []

        if geom.geom_type == "Polygon":
            all_rings += extract(geom)
        else:                                   # MultiPolygon
            for poly in geom.geoms:
                all_rings += extract(poly)

        return all_rings
