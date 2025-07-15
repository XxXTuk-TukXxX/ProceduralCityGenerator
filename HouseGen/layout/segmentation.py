import matplotlib.pyplot as plt
from typing import List, Tuple, Union
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from unique_picker import UniquePicker

class HouseBuilder:
    def __init__(self) -> None:
        pass

    def square_side_length(self, coords, *, grid_points=True):
        """
        Return the side length of an axis-aligned square footprint.

        Parameters
        ----------
        coords : list[tuple[int, int]]
            The four distinct corner vertices of the square, in any order.
            (If the first vertex is repeated at the end, it is ignored.)
        grid_points : bool, default True
            • True  → length is counted in **grid points**, i.e. the number of
                    integer lattice points on one edge (endpoints included).
            • False → length is returned as Euclidean distance (|Δx| or |Δy|).

        Returns
        -------
        int | float
            Side length of the square.

        Raises
        ------
        ValueError
            If the provided coordinates do not form an axis-aligned square.
        """
        # Remove duplicate closing vertex, if present
        if coords[0] == coords[-1]:
            coords = coords[:-1]

        if len(coords) != 4:
            raise ValueError("Expect four distinct vertices for a square.")

        xs = sorted({x for x, _ in coords})
        ys = sorted({y for _, y in coords})

        if len(xs) != 2 or len(ys) != 2:
            raise ValueError("Vertices must form an axis-aligned square.")

        side_xy = abs(xs[1] - xs[0]), abs(ys[1] - ys[0])
        if side_xy[0] != side_xy[1]:
            raise ValueError("Provided vertices form a rectangle, not a square.")

        side = side_xy[0]             # either Δx or Δy
        return side + 1 if grid_points else side


    # -----------------------------------------------------------
    # 2) *** NEW: return the four quadrants as four small squares ***
    # -----------------------------------------------------------
    def house_quadrants(self, coords):
        """
        Split the square footprint into its four quadrants.

        Returns
        -------
        list[list[tuple[int, int]]]
            A list of four closed polygons (UL, UR, LL, LR), each represented
            by the corner vertices **including the start point again at the end**.

            • If the side length (in grid points) is **odd**, the middle row
            and column are skipped, so each quadrant is an (L//2)×(L//2)
            block of grid cells with a one-grid-point gap between them.
            • If the side length is **even**, the square is split exactly in
            half without any gap.
        """
        # Basic square facts
        L = self.square_side_length(coords, grid_points=True)
        half = L // 2
        is_odd = L % 2 == 1

        # Unique min/max x, y
        if coords[0] == coords[-1]:
            coords = coords[:-1]
        xs = sorted({x for x, _ in coords})
        ys = sorted({y for _, y in coords})
        x_min, x_max = xs
        y_min, y_max = ys      # y_min is “bottom” (smaller), y_max is “top”

        # Where to cut
        left_x_max  = x_min + half - 1
        right_x_min = left_x_max + (2 if is_odd else 1)

        bottom_y_max = y_min + half - 1
        top_y_min    = bottom_y_max + (2 if is_odd else 1)

        # Quadrants (UL, UR, LL, LR)
        quads = [
            # upper-left
            [
                (x_min,     top_y_min),
                (left_x_max, top_y_min),
                (left_x_max, y_max),
                (x_min,     y_max),
                (x_min,     top_y_min)
            ],
            # upper-right
            [
                (right_x_min, top_y_min),
                (x_max,       top_y_min),
                (x_max,       y_max),
                (right_x_min, y_max),
                (right_x_min, top_y_min)
            ],
            # lower-left
            [
                (x_min,     y_min),
                (left_x_max, y_min),
                (left_x_max, bottom_y_max),
                (x_min,     bottom_y_max),
                (x_min,     y_min)
            ],
            # lower-right
            [
                (right_x_min, y_min),
                (x_max,       y_min),
                (x_max,       bottom_y_max),
                (right_x_min, bottom_y_max),
                (right_x_min, y_min)
            ],
        ]
        return quads







    Coord      = Tuple[float, float]
    CoordList  = List[Coord]
    CoordLists = List[CoordList]

    def _merge_touching(self,
                        polygons: List[Polygon],
                        tol: float = 1e-6) -> Polygon | MultiPolygon:
        """
        Glue polygons that touch only at a *corner* together.
        `tol` is the buffer distance; keep it very small compared
        with the size of your coordinate grid.
        """
        # ① grow every polygon by +tol,
        # ② merge overlaps,
        # ③ shrink the result by –tol back to “true” size
        return unary_union([p.buffer(tol, join_style=2) for p in polygons]).buffer(
            -tol, join_style=2
        )


    def subtract_polygons(self,
                        base: CoordList,
                        to_remove: CoordLists,
                        *,               # make the keywords below explicit
                        fuse_corners: bool = True,
                        corner_tol: float = 1e-6,
                        close_gaps = None
                        ) -> Union[CoordList, CoordLists]:
        """
        Subtract **many** polygons from *base*; optionally fuse polygons
        that only touch at a point so the little middle square disappears.

        Parameters
        ----------
        base : closed ring of (x, y) tuples
        to_remove : *list* of closed rings to subtract
        fuse_corners : if True, make corner-touching rectangles behave
                    as one continuous cut-out (default True)
        corner_tol   : buffer distance used for the corner fuse

        Returns
        -------
        A single coordinate list, or a list-of-lists when several islands
        remain – identical interface to the earlier function.
        """
        if not to_remove:
            return base[:]

        base_poly = Polygon(base)

        removal_polys = [Polygon(r) for r in to_remove]
        if fuse_corners:
            removal_mask = self._merge_touching(removal_polys, tol=corner_tol)
        else:
            removal_mask = unary_union(removal_polys)

        if close_gaps is not None and close_gaps > 0:
            r = close_gaps / 2.0
            removal_mask = removal_mask.buffer(r, join_style=2).buffer(-r, join_style=2)

        diff = base_poly.difference(removal_mask)
        if diff.is_empty:
            return []

        def _ring(p: Polygon):
            return [(x, y) for x, y in zip(*p.exterior.coords.xy)]

        if isinstance(diff, Polygon):
            return _ring(diff)

        # MultiPolygon → list[ring]
        return [_ring(p) for p in diff.geoms]



    def plot_house_area(self, list_coords):
        """
        Plots a closed polygon that outlines a house area.

        Parameters
        ----------
        coords : list[tuple[float, float]]
            Sequence of (x, y) points. The first and last points
            should be identical to “close” the shape.
        """
        plt.figure()
        for coords in  list_coords:
            x_vals, y_vals = zip(*coords)
            plt.plot(x_vals, y_vals, marker='o')

        plt.axis('equal')         # keep aspect ratio square
        plt.title("House Area")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.grid(True)
        plt.show()

    def generate_house_vertices(self, house_area, house_height):
        picker = UniquePicker()
        quadrants = self.house_quadrants(house_area)

        arr = []
        remove_quad = []
        for idx, height in enumerate(house_height):
            if idx == 0:
                arr.append(house_area)
            else:
                list_of_remove_quadrant_idx = picker.pick_range()
                print(list_of_remove_quadrant_idx)
                
                for i in list_of_remove_quadrant_idx:
                    remove_quad.append(quadrants[i])

                new_house_area = self.subtract_polygons(house_area, remove_quad, close_gaps=2.0)
                arr.append(new_house_area)
        
        return arr



# Example usage
house_area = [(31, -80), (31, -64), (15, -64), (15, -80), (31, -80)]
house_height = [5, 5, 5]

house_builder = HouseBuilder()
vertices = house_builder.generate_house_vertices(house_area, house_height)

print(vertices)

house_builder.plot_house_area(vertices)

# quadrants = house_quadrants(house_area)

# arr = []
# for idx, height in enumerate(house_height):
#     if idx == 0:
#         arr.append(house_area)
#     else:
#         remove_quad = [quadrants[1]]
#         new_house_area = subtract_polygons(house_area, remove_quad, close_gaps=2.0)
#         arr.append(new_house_area)


# plot_house_area(arr)
