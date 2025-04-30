import random
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
from area import Area

class Territory:
    def __init__(self, area: Area):
        self.area = area
        self.regions = []

    def add_bounding_box_points(self, points, polygon, padding=10):
        """Adds bounding points outside the polygon to ensure Voronoi regions exist for all points."""
        minx, miny, maxx, maxy = polygon.bounds
        extra_points = [
            (minx - padding, miny - padding),
            (minx - padding, maxy + padding),
            (maxx + padding, miny - padding),
            (maxx + padding, maxy + padding)
        ]
        return points + extra_points
    
    def calculate_angle(self, a, b, c):
        """Returns the angle (in degrees) at vertex b formed by points a-b-c."""
        ab = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        
        dot_product = np.dot(ab, bc)
        magnitude_ab = np.linalg.norm(ab)
        magnitude_bc = np.linalg.norm(bc)
        
        if magnitude_ab == 0 or magnitude_bc == 0:
            return 0
        cos_angle = dot_product / (magnitude_ab * magnitude_bc)
        cos_angle = max(-1, min(1, cos_angle))
        angle_radians = np.arccos(cos_angle)
        return np.degrees(angle_radians)
    
    def find_acute_angle_vertices(self, polygons, threshold=89):
        """Finds vertices in polygons that form angles smaller than the threshold."""
        acute_vertices = []
        for poly in polygons:
            coords = list(poly.exterior.coords)
            if coords[0] == coords[-1]:
                coords = coords[:-1]
            n = len(coords)
            for i in range(n):
                a = coords[i - 1]
                b = coords[i]
                c = coords[(i + 1) % n]
                angle = self.calculate_angle(a, b, c)
                if angle < threshold:
                    acute_vertices.append(b)
        return acute_vertices
    
    def get_acute_edges_dict(self, polygons, area, threshold=90):
        """
        For each polygon, finds the acute angle vertices (with angle < threshold) and maps each acute vertex 
        to its adjacent edges (as tuples of endpoints). Edges that lie fully on the territory boundary are omitted.
        """
        acute_edges_dict = {}
        for poly in polygons:
            coords = list(poly.exterior.coords)
            # Remove duplicate last point if it equals the first
            if coords[0] == coords[-1]:
                coords = coords[:-1]
            n = len(coords)
            for i in range(n):
                p1 = coords[i - 1]
                p2 = coords[i]
                p3 = coords[(i + 1) % n]
                angle = self.calculate_angle(p1, p2, p3)
                if angle < threshold:
                    edges = []
                    # Edge from previous vertex to p2
                    if not (self.is_on_boundary(p1, area.polygon) and self.is_on_boundary(p2, area.polygon)):
                        edges.append((p1, p2))
                    # Edge from p2 to next vertex
                    if not (self.is_on_boundary(p2, area.polygon) and self.is_on_boundary(p3, area.polygon)):
                        edges.append((p2, p3))
                    if edges:
                        if p2 in acute_edges_dict:
                            acute_edges_dict[p2].extend(edges)
                        else:
                            acute_edges_dict[p2] = edges
        return acute_edges_dict

    def eliminate_acute_angle_vertex(self, p1, p2, p3, min_angle=90, max_d=100, step=0.5):
        """
        Moves p2 along the bisector direction until the angle at p2 (formed by p1, p2, p3)
        is at least min_angle. Increments by small steps up to a maximum displacement.
        """
        mx, my = (p1[0] + p3[0]) / 2, (p1[1] + p3[1]) / 2
        dx, dy = mx - p2[0], my - p2[1]
        magnitude = math.hypot(dx, dy)
        if magnitude == 0:
            return p2
        u = (dx / magnitude, dy / magnitude)
        
        d = 0
        new_p2 = p2
        while self.calculate_angle(p1, new_p2, p3) < min_angle and d < max_d:
            d += step
            new_p2 = (p2[0] + u[0] * d, p2[1] + u[1] * d)
        return new_p2
    
    def adjust_voronoi_angles_until_no_acute(self, vor, area, min_angle=90, max_iterations=50):
        """
        Iteratively adjusts the Voronoi vertices such that all polygon vertices (except boundary ones)
        have angles of at least min_angle.
        """
        adjusted_vertices = {tuple(v): tuple(v) for v in vor.vertices}
        boundary_vertices = set(area.get_vertices())

        iteration = 0
        while iteration < max_iterations:
            adjustment_made = False
            
            for region_index in vor.point_region[:-4]:
                region = vor.regions[region_index]
                if not region or -1 in region:
                    continue
                poly_region = [adjusted_vertices.get(tuple(vor.vertices[i]), tuple(vor.vertices[i]))
                            for i in region]
                n = len(poly_region)
                for i in range(n):
                    p1 = poly_region[i - 1]
                    p2 = poly_region[i]
                    p3 = poly_region[(i + 1) % n]
                    current_angle = self.calculate_angle(p1, p2, p3)
                    if current_angle < min_angle and p2 not in boundary_vertices:
                        new_p2 = self.eliminate_acute_angle_vertex(p1, p2, p3, min_angle)
                        if math.dist(new_p2, p2) > 0.1:
                            adjusted_vertices[p2] = new_p2
                            adjustment_made = True

            adjusted_polygons = []
            for region_index in vor.point_region[:-4]:
                region = vor.regions[region_index]
                if not region or -1 in region:
                    continue
                poly_region = [adjusted_vertices.get(tuple(vor.vertices[i]), tuple(vor.vertices[i]))
                            for i in region]
                poly = Polygon(poly_region).intersection(area.polygon)
                if isinstance(poly, Polygon):
                    adjusted_polygons.append(poly)

            acute_vertices = self.find_acute_angle_vertices(adjusted_polygons, threshold=min_angle-1)
            if not acute_vertices:
                break
            iteration += 1

        return adjusted_polygons
    
    def is_on_boundary(self, point, polygon, tol=1e-6):
        """
        Returns True if the point lies on the boundary of the polygon within a tolerance.
        """
        return polygon.exterior.distance(Point(point)) < tol
    
    def adjust_acute_edges_on_boundary(self, acute_edges_dict, boundary_polygon, tol=1e-6):
        """
        For every acute vertex (key) that lies on the boundary polygon, adjust it so that the edge
        that has it as an endpoint forms a perpendicular (altitude) to the boundary segment on which
        it lies.
        
        Returns:
        updated_dict: dict
            A new dictionary with the acute points on the boundary adjusted so that the corresponding
            edge(s) are perpendicular to the boundary segment.
        update_map: dict
            A mapping from the original acute vertex to its new (adjusted) coordinate.
        """
        def find_boundary_segment(pt, polygon, tol):
            coords = list(polygon.exterior.coords)
            if coords[0] == coords[-1]:
                coords = coords[:-1]
            min_dist = float('inf')
            best_segment = None
            for i in range(len(coords)):
                seg = (coords[i], coords[(i + 1) % len(coords)])
                dist = LineString(seg).distance(Point(pt))
                if dist < min_dist:
                    min_dist = dist
                    best_segment = seg
            return best_segment

        updated_dict = {}
        update_map = {}
        for acute_point, edges in acute_edges_dict.items():
            if boundary_polygon.exterior.distance(Point(acute_point)) < tol:
                base_segment = find_boundary_segment(acute_point, boundary_polygon, tol)
                if base_segment is None:
                    updated_dict[acute_point] = edges
                    continue

                B1, B2 = base_segment
                d = (B2[0] - B1[0], B2[1] - B1[1])
                d_norm = math.hypot(d[0], d[1])
                if d_norm == 0:
                    updated_dict[acute_point] = edges
                    continue
                d_unit = (d[0] / d_norm, d[1] / d_norm)
                d_perp = (-d_unit[1], d_unit[0])

                new_edges = []
                new_point_candidates = []
                for edge in edges:
                    pt1, pt2 = edge
                    if math.dist(acute_point, pt1) < tol:
                        partner = pt2
                    elif math.dist(acute_point, pt2) < tol:
                        partner = pt1
                    else:
                        continue

                    v = (acute_point[0] - partner[0], acute_point[1] - partner[1])
                    t = v[0]*d_perp[0] + v[1]*d_perp[1]
                    candidate = (partner[0] + t*d_perp[0], partner[1] + t*d_perp[1])
                    new_point_candidates.append(candidate)
                    if math.dist(acute_point, pt1) < tol:
                        new_edge = (candidate, pt2)
                    else:
                        new_edge = (pt1, candidate)
                    new_edges.append(new_edge)

                if new_point_candidates:
                    avg_x = sum(pt[0] for pt in new_point_candidates) / len(new_point_candidates)
                    avg_y = sum(pt[1] for pt in new_point_candidates) / len(new_point_candidates)
                    new_acute_point = (avg_x, avg_y)
                    update_map[acute_point] = new_acute_point
                    adjusted_edges = []
                    for edge in new_edges:
                        pt1, pt2 = edge
                        if math.dist(pt1, new_acute_point) < tol or math.dist(pt1, acute_point) < tol:
                            adjusted_edges.append((new_acute_point, pt2))
                        elif math.dist(pt2, new_acute_point) < tol or math.dist(pt2, acute_point) < tol:
                            adjusted_edges.append((pt1, new_acute_point))
                        else:
                            adjusted_edges.append(edge)
                    updated_dict[new_acute_point] = adjusted_edges
                else:
                    updated_dict[acute_point] = edges
            else:
                updated_dict[acute_point] = edges
        return updated_dict, update_map
    
    def merge_polygons_by_deleted_edges(self, polygons, delete_edges, tol=1e-6, decimals=6):
        """
        Given an initial Voronoi-cell list and a set of edges that must disappear
        (`delete_edges`), merge every pair of polygons that share one of those
        edges and return the new polygon list.

        Parameters
        ----------
        polygons : list[shapely.geometry.Polygon]
        delete_edges : list[tuple[tuple, tuple]]
            Each edge is ((x1, y1), (x2, y2)).  Order is irrelevant.
        tol : float
            Distance tolerance used only when *very* close duplicates appear.
        decimals : int
            How many decimal places to keep when canonicalising coordinates
            (avoids floating-point noise when matching edges).

        Returns
        -------
        list[Polygon]  – the Voronoi cells after every requested merge.
        """
        # ── canonical helpers ────────────────────────────────────────────────────
        def canon_pt(pt):
            return tuple(round(c, decimals) for c in pt)

        def canon_edge(edge):
            a, b = map(canon_pt, edge)
            return tuple(sorted((a, b)))          # order-independent key

        # map: edge_key  →  list of polygon indices that contain it
        edge_to_polys: dict[tuple, list[int]] = {}

        for idx, poly in enumerate(polygons):
            coords = list(poly.exterior.coords)
            if coords[0] == coords[-1]:           # drop duplicate last vertex
                coords = coords[:-1]

            for i in range(len(coords)):
                edge_key = canon_edge((coords[i], coords[(i + 1) % len(coords)]))
                edge_to_polys.setdefault(edge_key, []).append(idx)

        # ── DSU / Union–Find to collect merging components ───────────────────────
        parent = list(range(len(polygons)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        # merge every pair of polygons that share a *deleted* edge
        for edge in delete_edges:
            key = canon_edge(edge)
            polys_here = edge_to_polys.get(key, [])
            if len(polys_here) >= 2:              # edge appears in ≥ 2 cells
                base = polys_here[0]
                for other in polys_here[1:]:
                    union(base, other)

        # ── gather components and build the merged geometries ────────────────────
        groups: dict[int, list[int]] = {}
        for i in range(len(polygons)):
            groups.setdefault(find(i), []).append(i)

        merged_polys = []
        for idxs in groups.values():
            if len(idxs) == 1:
                merged_polys.append(polygons[idxs[0]])
            else:
                merged_polys.append(unary_union([polygons[i] for i in idxs]))

        return merged_polys
    
    def update_polygon_vertices_with_altitude_lines(self, adjusted_polygons, update_map):
        final_polygons = []
        tol = 1e-6
        for poly in adjusted_polygons:
            coords, new_coords = list(poly.exterior.coords), []
            for pt in coords:
                for orig, new in update_map.items():
                    if math.dist(pt, orig) < tol:
                        new_coords.append(new)
                        break
                else:
                    new_coords.append(pt)
            final_polygons.append(
                Polygon(new_coords).intersection(self.area.polygon)
            )
        return final_polygons
    
    def find_acute_double_or_more_edges(self, adjusted_acute_edges_dict):

        def are_tuple_pairs_equivalent(t1, t2, decimals=9):
            if t1 == None or t2 ==None:
                return None
            return sorted(tuple(round(x, decimals) for x in point) for point in t1) == sorted(tuple(round(x, decimals) for x in point) for point in t2)
        
        adjusted_acute_edges_dict_new = adjusted_acute_edges_dict.copy()
        # print(adjusted_acute_edges_dict_new)
        for acute_points in adjusted_acute_edges_dict.keys():
            if self.is_on_boundary(acute_points, self.area.polygon):
                deleted_lines: list[tuple[tuple]] = adjusted_acute_edges_dict_new[acute_points] # Deleted lines
                adjusted_acute_edges_dict_new.pop(acute_points)
                
                for elements in adjusted_acute_edges_dict_new:
                    for index_line, lines in enumerate(adjusted_acute_edges_dict_new[elements]):
                        # print(lines)
                        # print(items[0])
                        if are_tuple_pairs_equivalent(lines, deleted_lines[0]):
                            adjusted_acute_edges_dict_new[elements][index_line] = None

        return adjusted_acute_edges_dict_new
    
    def get_delete_lines(self, adjusted_acute_edges_dict_new, acute_edges_dict):
        canon = lambda e: tuple(sorted(e))        #  ((x1,y1),(x2,y2))  →  canonical order
        delete_edges:list[tuple[tuple], tuple[tuple]] = []

        # ---- canonical NEW edges ----
        new_edges = {
            canon(e)
            for edges in adjusted_acute_edges_dict_new.values()
            for e in edges
            if e is not None
        }

        # ---- walk through OLD edges, but keep a "seen" set so each undirected edge is handled once ----
        seen = set()

        for edges in acute_edges_dict.values():
            for e in edges:
                if e is None:
                    continue                       # skip empty slots

                key = canon(e)                     # undirected form

                if key in seen:                    # already handled (reverse duplicate)
                    continue
                seen.add(key)

                if key not in new_edges:           # only print if truly missing
                    delete_edges.append(e)
        
        return delete_edges
    
    def generate_territory(self, points):
        """
        Computes and plots a Voronoi diagram with all acute angles eliminated inside the given polygon.
        Then, for those acute angles on the boundary, updates the polygon edges so that the acute vertex is
        moved to form an altitude (90°) relative to the boundary edge. The final polygons are plotted.
        """
        points = self.add_bounding_box_points(points, self.area.polygon)
        vor = Voronoi(points)

        adjusted_polygons = self.adjust_voronoi_angles_until_no_acute(vor, self.area)

        # acute_vertices = find_acute_angle_vertices(adjusted_polygons, threshold=90)
        acute_edges_dict = self.get_acute_edges_dict(adjusted_polygons, self.area, threshold=90)
        adjusted_acute_edges_dict, update_map = self.adjust_acute_edges_on_boundary(acute_edges_dict, self.area.polygon)
        
        # Update the Voronoi polygon vertices with the new altitude (adjusted acute) vertices.
        final_polygons = self.update_polygon_vertices_with_altitude_lines(adjusted_polygons, update_map)


        acute_edges_dict = self.get_acute_edges_dict(final_polygons, self.area, threshold=90)
        adjusted_acute_edges_dict, update_map = self.adjust_acute_edges_on_boundary(acute_edges_dict, self.area.polygon)   

        adjusted_acute_edges_dict_new = self.find_acute_double_or_more_edges(adjusted_acute_edges_dict)

        delete_edges = self.get_delete_lines(adjusted_acute_edges_dict_new, acute_edges_dict)

        # print(delete_edges)
        merged_polygons = self.merge_polygons_by_deleted_edges(final_polygons, delete_edges)

        # print(len(merged_polygons))

        acute_edges_dict = self.get_acute_edges_dict(merged_polygons, self.area, threshold=90) 

        delete_random_edges = []
        # Plot the new altitude edges in violet.
        for elm in acute_edges_dict:
            edges_random_delete = acute_edges_dict[elm]
            edges_to_delete = edges_random_delete.pop(random.randrange(len(edges_random_delete)))
            delete_random_edges.append(edges_to_delete)

        merged_polygons = self.merge_polygons_by_deleted_edges(merged_polygons, delete_random_edges)

        self.regions = merged_polygons
        return merged_polygons
    
    def get_regions(self):
        return self.regions
    
    # def generate_territory_without_deletes(self, points):
    #     """
    #     Computes and plots a Voronoi diagram with all acute angles eliminated inside the given polygon.
    #     Then, for those acute angles on the boundary, updates the polygon edges so that the acute vertex is
    #     moved to form an altitude (90°) relative to the boundary edge. The final polygons are plotted.
    #     """
    #     points = self.add_bounding_box_points(points, area.polygon)
    #     vor = Voronoi(points)

    #     adjusted_polygons = self.adjust_voronoi_angles_until_no_acute(vor, area)

    #     # acute_vertices = find_acute_angle_vertices(adjusted_polygons, threshold=90)
    #     acute_edges_dict = self.get_acute_edges_dict(adjusted_polygons, area, threshold=90)
    #     adjusted_acute_edges_dict, update_map = self.adjust_acute_edges_on_boundary(acute_edges_dict, area.polygon)
        
    #     # Update the Voronoi polygon vertices with the new altitude (adjusted acute) vertices.
    #     final_polygons = self.update_polygon_vertices_with_altitude_lines(adjusted_polygons, update_map)

    #     return final_polygons, adjusted_acute_edges_dict


def plot_voronoi_with_adjusted_angles(merged_polygons, acute_points: dict = None):
    fig, ax = plt.subplots(figsize=(6, 6))
    # # Plot the original territory boundary.
    # polygon_vertices = territory.get_vertices() + [territory.get_vertices()[0]]
    # poly_x, poly_y = zip(*polygon_vertices)
    # ax.plot(poly_x, poly_y, 'k-', linewidth=2, label="Boundary Polygon")
    
    # # Plot original Voronoi points (without bounding box extras).
    # x, y = zip(*points[:-4])
    # ax.scatter(x, y, c='red', marker='o', label="Voronoi Points")

    # Plot the final, updated Voronoi polygons.
    for poly in merged_polygons:
        if not poly.is_empty:
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.4, edgecolor="blue", facecolor="cyan")

    if acute_points != None:
        canon = lambda a, b: tuple(sorted((a, b)))  # order-independent key
        unique_edges = {
            canon(p1, p2)
            for lst in acute_points.values()
            for p1, p2 in lst
        }

        # ── plot every segment ──────────────────────────────────────────────────

        for p1, p2 in unique_edges:
            x = (p1[0], p2[0])
            y = (p1[1], p2[1])
            ax.plot(x, y, linewidth=1, color="red")

        key_pts = list(acute_points.keys())
        xk, yk = zip(*key_pts)
        ax.scatter(xk, yk, c='green', marker='o')


    ax.set_xlim(-110, 110)
    ax.set_ylim(-110, 110)
    ax.set_aspect('equal')
    ax.set_title("Voronoi Diagram with Altitude Edges on Boundary")
    ax.grid(True)
    plt.show()


# # Example usage
# if __name__ == "__main__":
#     area = Area()
#     territory = Territory(area)
#     points = area.generate_poisson_disk_points()
#     # final_polygons, acute_edges_dict = territory.generate_territory_without_deletes(points)
#     territory.generate_territory(points)
#     # plot_voronoi_with_adjusted_angles(final_polygons, acute_edges_dict)
#     plot_voronoi_with_adjusted_angles(territory.get_regions())