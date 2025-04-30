import random
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.geometry.polygon import orient
from scipy.spatial import distance
from matplotlib.path import Path
from collections import defaultdict

class SideHouse:
    def __init__(self, vertices):
        """
        Initialize House with the area (polygon) vertices.
        
        Args:
            vertices (np.ndarray or list of tuples): A closed polygon's vertices 
                (with the first vertex repeated at the end).
        """
        self.vertices = np.array(vertices)
        self.x = self.vertices[:, 0]
        self.y = self.vertices[:, 1]
        self.houses = None

    # ───── Basic House Creation Methods ─────

    def create_house_on_edge(self, vertex1, vertex2, house_size=5, orientation='clockwise'):
        """
        Creates a house (square) on the edge defined by two vertices with a given orientation.
        
        Args:
            vertex1 (tuple): The first vertex (anchor point).
            vertex2 (tuple): The second vertex.
            house_size (float): The size of the house (square) to create.
            orientation (str): 'clockwise' or 'counterclockwise'.
        
        Returns:
            list: List of house (square) vertices [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
        """
        x1, y1 = vertex1
        x2, y2 = vertex2
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            raise ValueError("vertex1 and vertex2 cannot be the same point.")
        unit_dx, unit_dy = dx / length, dy / length
        if orientation == 'clockwise':
            perp_dx, perp_dy = unit_dy, -unit_dx
        elif orientation == 'counterclockwise':
            perp_dx, perp_dy = -unit_dy, unit_dx
        else:
            raise ValueError("Orientation must be 'clockwise' or 'counterclockwise'.")
        house_vertices = [
            (x1, y1),
            (x1 + house_size * unit_dx, y1 + house_size * unit_dy),
            (x1 + house_size * (unit_dx + perp_dx), y1 + house_size * (unit_dy + perp_dy)),
            (x1 + house_size * perp_dx, y1 + house_size * perp_dy),
        ]
        return house_vertices

    def create_all_elbow_houses(self, house_size=5, randomize_size=False, min_size=5, max_size=16):
        """
        Creates houses on all edges of the area. For each unique vertex, if
        randomize_size=True a random house size between min_size and max_size is chosen.
        (Both houses sharing a vertex will use the same size.)
        
        Args:
            house_size (float): Default size (used if randomize_size is False).
            randomize_size (bool): If True, choose a random house size for each vertex.
            min_size (int): Minimum house size.
            max_size (int): Maximum house size.
            
        Returns:
            list: A list of tuples (vertex, house) where 'house' is a list of vertices.
        """
        houses = []
        num_edges = len(self.x) - 1  # Assumes polygon is closed.
        vertex_size_map = {}
        for i in range(num_edges):
            vertex1 = (self.x[i], self.y[i])
            if randomize_size:
                if vertex1 not in vertex_size_map:
                    vertex_size_map[vertex1] = np.random.randint(min_size, max_size + 1)
                size1 = vertex_size_map[vertex1]
            else:
                size1 = house_size
            vertex2 = (self.x[i + 1], self.y[i + 1])
            house_ccw = self.create_house_on_edge(vertex1, vertex2, house_size=size1, orientation='counterclockwise')
            houses.append((vertex1, house_ccw))
            prev_index1 = (i - 1) % num_edges
            vertex1_prev = (self.x[prev_index1], self.y[prev_index1])
            if randomize_size:
                if vertex1_prev not in vertex_size_map:
                    vertex_size_map[vertex1_prev] = np.random.randint(min_size, max_size + 1)
                size_prev = vertex_size_map[vertex1_prev]
            else:
                size_prev = house_size
            prev_index2 = (i - 2) % num_edges
            vertex2_prev = (self.x[prev_index2], self.y[prev_index2])
            house_cw = self.create_house_on_edge(vertex1_prev, vertex2_prev, house_size=size_prev, orientation='clockwise')
            houses.append((vertex1_prev, house_cw))
        return houses

    def group_houses_by_vertex(self, houses):
        """
        Groups houses by the area vertex from which they originate.
        
        Args:
            houses (list): List of tuples (vertex, house).
            
        Returns:
            dict: Keys are vertices; values are lists of houses.
        """
        grouped = {}
        for vertex, house in houses:
            if vertex not in grouped:
                grouped[vertex] = []
            grouped[vertex].append(house)
        return grouped

    def print_overlap_percentages(self, grouped_houses):
        """
        For each vertex group, prints pair-wise overlap percentages between houses.
        Overlap percentage = (intersection area / union area) * 100.
        """
        for vertex, houses in grouped_houses.items():
            if len(houses) < 2:
                continue
            for i in range(len(houses)):
                for j in range(i + 1, len(houses)):
                    poly1 = Polygon(houses[i])
                    poly2 = Polygon(houses[j])
                    intersection = poly1.intersection(poly2)
                    union = poly1.union(poly2)
                    overlap_pct = (intersection.area / union.area) * 100 if union.area != 0 else 0
                    print(f"Overlap at vertex {vertex} between house {i+1} & {j+1}: {overlap_pct:.2f}%")

    def remove_highly_overlapping_houses(self, grouped_houses, threshold=90):
        """
        Removes from each vertex group houses that overlap ≥ threshold% with one already kept.
        
        Args:
            grouped_houses (dict): Dictionary of houses keyed by vertex.
            threshold (float): Overlap threshold percentage.
            
        Returns:
            dict: A new dictionary with pruned house groups.
        """
        pruned = {}
        for vertex, houses in grouped_houses.items():
            kept = []
            for house in houses:
                house_poly = Polygon(house)
                add = True
                for kept_house in kept:
                    kept_poly = Polygon(kept_house)
                    overlap_pct = (house_poly.intersection(kept_poly).area /
                                   house_poly.union(kept_poly).area) * 100
                    if overlap_pct >= threshold:
                        add = False
                        break
                if add:
                    kept.append(house)
            pruned[vertex] = kept
        return pruned

    def merge_overlapping_houses(self, grouped_houses, threshold=90):
        """
        For each vertex group, merges houses that overlap (but not by ≥ threshold) 
        and forces the resulting polygon to have six vertices.
        
        Args:
            grouped_houses (dict): Dictionary of houses keyed by vertex.
            threshold (float): Overlap threshold percentage.
            
        Returns:
            dict: Mapping from vertex to a merged six-vertex house (list of 6 vertices).
        """
        merged_houses = {}
        for vertex, houses in grouped_houses.items():
            if len(houses) == 0:
                continue
            if len(houses) == 1:
                merged_poly = Polygon(houses[0])
            else:
                merged_poly = Polygon(houses[0])
                for house in houses[1:]:
                    merged_poly = merged_poly.union(Polygon(house))
            if merged_poly.geom_type == 'MultiPolygon':
                merged_poly = max(merged_poly.geoms, key=lambda poly: poly.area)
            coords = list(merged_poly.exterior.coords)[:-1]
            if len(coords) == 6:
                merged_houses[vertex] = coords
            else:
                tol = 0.001
                simplified = merged_poly.simplify(tol, preserve_topology=True)
                coords = list(simplified.exterior.coords)[:-1]
                while len(coords) != 6 and tol < 1:
                    tol *= 1.5
                    simplified = merged_poly.simplify(tol, preserve_topology=True)
                    coords = list(simplified.exterior.coords)[:-1]
                if len(coords) == 6:
                    merged_houses[vertex] = coords
                else:
                    hull = merged_poly.convex_hull
                    coords = list(hull.exterior.coords)[:-1]
                    while len(coords) < 6:
                        max_len = 0
                        max_idx = 0
                        for i in range(len(coords)):
                            p1 = coords[i]
                            p2 = coords[(i + 1) % len(coords)]
                            d = np.linalg.norm(np.array(p2) - np.array(p1))
                            if d > max_len:
                                max_len = d
                                max_idx = i
                        p1 = coords[max_idx]
                        p2 = coords[(max_idx + 1) % len(coords)]
                        midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                        coords.insert(max_idx + 1, midpoint)
                    merged_houses[vertex] = coords[:6]
        return merged_houses

    def force_target_vertices(self, poly, target_count):
        """
        Force the given polygon to have exactly target_count vertices.
        
        Args:
            poly (Polygon): The polygon to be forced.
            target_count (int): The desired number of vertices.
            
        Returns:
            list: A list of vertices for the forced polygon.
        """
        coords = list(poly.exterior.coords)[:-1]
        if len(coords) == target_count:
            return coords
        tol = 0.001
        simplified = poly.simplify(tol, preserve_topology=True)
        coords = list(simplified.exterior.coords)[:-1]
        while len(coords) != target_count and tol < 10:
            tol *= 1.5
            simplified = poly.simplify(tol, preserve_topology=True)
            coords = list(simplified.exterior.coords)[:-1]
        if len(coords) == target_count:
            return coords
        else:
            hull = poly.convex_hull
            coords = list(hull.exterior.coords)[:-1]
            while len(coords) < target_count:
                max_len = 0
                max_idx = 0
                for i in range(len(coords)):
                    p1 = coords[i]
                    p2 = coords[(i + 1) % len(coords)]
                    d = np.linalg.norm(np.array(p2) - np.array(p1))
                    if d > max_len:
                        max_len = d
                        max_idx = i
                p1 = coords[max_idx]
                p2 = coords[(max_idx + 1) % len(coords)]
                midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                coords.insert(max_idx + 1, midpoint)
            if len(coords) > target_count:
                coords = coords[:target_count]
            return coords



    # Merge any overlapping 6-vertex houses into a single 8-vertex house
        
    def merge_overlapping_six_vertex_houses(
        self,
        six_vertex_houses: list[list[tuple[float, float]]],
        threshold: float = 0,
    ) -> list[list[tuple[float, float]]]:
        """
        Looks for intersecting pairs of six-vertex houses and replaces each
        pair with a single 8-vertex outline produced by `force_target_vertices`.
        """
        merged_houses = six_vertex_houses[:]       # shallow copy
        changed = True

        while changed:                             # keep looping until no
            changed = False                        # more merges occur
            i = 0
            while i < len(merged_houses):
                j = i + 1
                while j < len(merged_houses):
                    poly_i = Polygon(merged_houses[i])
                    poly_j = Polygon(merged_houses[j])

                    # test for “real” overlap
                    if (poly_i.intersects(poly_j) and
                        poly_i.intersection(poly_j).area > threshold):

                        merged_poly = poly_i.union(poly_j)

                        # union can be a MultiPolygon if shapes only touch
                        if merged_poly.geom_type == 'MultiPolygon':
                            merged_poly = max(merged_poly.geoms,
                                              key=lambda p: p.area)

                        merged_poly = merged_poly.buffer(0)  # topological fix
                        new_coords  = self.force_target_vertices(merged_poly, 8)

                        # replace the two originals with the merged outline
                        merged_houses.pop(j)
                        merged_houses.pop(i)
                        merged_houses.append(new_coords)

                        changed = True
                        break          # restart inner loop with fresh list
                    j += 1
                if changed:
                    break              # restart outer loop
                i += 1

        return merged_houses
    


    # ───── New Method for Extracting Edge Points ─────

    def is_point_on_segment(self, px, py, x1, y1, x2, y2, tol=1e-5):
        """
        Checks if a point (px, py) lies on the line segment between (x1, y1) and (x2, y2).
        """
        # Compute cross product to check collinearity
        cross_product = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)
        if abs(cross_product) > tol:
            return False

        # Check if the point lies within the segment bounds
        dot_product = (px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)
        if dot_product < 0:
            return False

        squared_length = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if dot_product > squared_length:
            return False

        return True

    def get_house_points_on_polygon_edges(self, polygon_vertices, houses):
        """
        Finds house points that lie exactly on the edges of the main polygon.

        Parameters:
        polygon_vertices (list of tuples): The vertices of the polygon.
        houses (dict): Dictionary where keys are house centers and values are lists of house vertices.

        Returns:
        set: A set of house points that lie on the polygon's edges.
        """
        polygon_vertices = np.array(polygon_vertices)
        edge_points = set()

        for i in range(len(polygon_vertices)):
            x1, y1 = polygon_vertices[i]
            x2, y2 = polygon_vertices[(i + 1) % len(polygon_vertices)]  # Wrap around to close the polygon

            for house_vertices in houses.values():
                for px, py in house_vertices:
                    if self.is_point_on_segment(px, py, x1, y1, x2, y2):
                        edge_points.add((px, py))

        return edge_points
    
    def is_approx_square(self, vertices, angle_tolerance=10, zero_angle_tolerance=5):
        if len(vertices) != 6:
            return False

        angles = []
        for i in range(len(vertices)):
            a = np.array(vertices[i - 1])
            b = np.array(vertices[i])
            c = np.array(vertices[(i + 1) % len(vertices)])

            ab = b - a
            bc = c - b

            dot_product = np.dot(ab, bc)
            norm_ab = np.linalg.norm(ab)
            norm_bc = np.linalg.norm(bc)

            if norm_ab == 0 or norm_bc == 0:
                return False

            cos_theta = dot_product / (norm_ab * norm_bc)
            theta = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
            angles.append(theta)

        near_90 = sum(90 - angle_tolerance <= angle <= 90 + angle_tolerance for angle in angles)
        near_0 = sum(-zero_angle_tolerance <= angle <= zero_angle_tolerance for angle in angles)

        return near_90 == 4 and near_0 == 2
    
    def find_square_houses(self, houses):
        square_houses = {}
        for center, vertices in houses.items():
            if self.is_approx_square(vertices):
                square_houses[center] = vertices
        return square_houses
    
    def remove_extra_vertices(self, vertices, zero_angle_tolerance=5):
        """
        Removes extra vertices that form near-zero-degree angles in a 6-vertex square.
        """
        if len(vertices) != 6:
            return vertices  # Only process 6-vertex squares

        filtered_vertices = []
        for i in range(len(vertices)):
            a = np.array(vertices[i - 1])
            b = np.array(vertices[i])
            c = np.array(vertices[(i + 1) % len(vertices)])

            ab = b - a
            bc = c - b

            dot_product = np.dot(ab, bc)
            norm_ab = np.linalg.norm(ab)
            norm_bc = np.linalg.norm(bc)

            if norm_ab == 0 or norm_bc == 0:
                continue

            cos_theta = dot_product / (norm_ab * norm_bc)
            theta = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

            if not (-zero_angle_tolerance <= theta <= zero_angle_tolerance):
                filtered_vertices.append(tuple(b))

        return filtered_vertices
    
    def convert_six_vertex_squares_to_normal(self, houses, cleaned_houses):
        """
        Converts 6-vertex square houses into normal 4-vertex square houses.
        """
        all_houses = houses.copy()
        for center, vertices in cleaned_houses.items():
            all_houses[center] = vertices  # Replace 6-vertex squares with cleaned versions
        return all_houses
    
    def project_point_onto_segment(self, px, py, x1, y1, x2, y2):
        """
        Projects a point (px, py) onto the nearest point on the segment (x1, y1) - (x2, y2).
        """
        seg_vec = np.array([x2 - x1, y2 - y1])
        point_vec = np.array([px - x1, py - y1])
        
        seg_len = np.dot(seg_vec, seg_vec)
        if seg_len == 0:
            return (x1, y1)  # The segment is a point
        
        t = np.dot(point_vec, seg_vec) / seg_len
        t = max(0, min(1, t))  # Clamp to segment range
        
        nearest_x = x1 + t * seg_vec[0]
        nearest_y = y1 + t * seg_vec[1]
        
        return (nearest_x, nearest_y)
    
    # New methods

    def is_point_inside_polygon(self, px, py, polygon):
        """
        Checks if a point is inside a given polygon using the matplotlib Path.contains_point method.
        """
        poly_path = Path(polygon)
        return poly_path.contains_point((px, py))
    
    def move_houses_onto_polygon_edges(self, polygon_vertices, houses):
        """
        Moves house points that are outside the polygon onto the nearest polygon edge.
        """
        polygon_vertices_arr = np.array(polygon_vertices)
        adjusted_houses = {}
        
        for center, vertices in houses.items():
            adjusted_vertices = []
            for px, py in vertices:
                if self.is_point_inside_polygon(px, py, polygon_vertices_arr):
                    adjusted_vertices.append((px, py))  # Keep inside points unchanged
                else:
                    min_dist = float('inf')
                    closest_point = (px, py)
                    
                    for i in range(len(polygon_vertices_arr)):
                        x1, y1 = polygon_vertices_arr[i]
                        x2, y2 = polygon_vertices_arr[(i + 1) % len(polygon_vertices_arr)]
                        projected_point = self.project_point_onto_segment(px, py, x1, y1, x2, y2)
                        d = distance.euclidean((px, py), projected_point)
                        
                        if d < min_dist:
                            min_dist = d
                            closest_point = projected_point
                    
                    adjusted_vertices.append(closest_point)
            
            adjusted_houses[center] = adjusted_vertices
        
        return adjusted_houses
    
    def generate_houses_along_edges(self, edge_points, polygon_vertices, min_house=5, max_house=16):
        """
        For each polygon edge, this function:
        1. Finds all edge_points lying on that edge.
        2. Determines the extreme projection parameters (t0 and t1) along the edge.
        3. Computes the available segment length (t1 - t0) and partitions that length
            into a series of house widths that are each at least `min_house` and at most `max_house`.
            The partitioning is done greedily with a randomized selection.
        4. For each house, a square (with side equal to the house width) is created so that
            its bottom edge lies on the polygon edge and it projects inward (toward the polygon’s interior).

        Returns a dictionary mapping the house center to its vertices.
        """
        additional_houses = {}
        
        # Determine polygon orientation via the shoelace formula.
        # For a counterclockwise polygon, the interior is to the left.
        area = 0
        for i in range(len(polygon_vertices)):
            x1, y1 = polygon_vertices[i]
            x2, y2 = polygon_vertices[(i + 1) % len(polygon_vertices)]
            area += (x1 * y2 - y1 * x2)
        ccw = area > 0

        # Process each edge of the polygon.
        for i in range(len(polygon_vertices)):
            A = np.array(polygon_vertices[i])
            B = np.array(polygon_vertices[(i + 1) % len(polygon_vertices)])
            edge_vec = B - A
            edge_length = np.linalg.norm(edge_vec)
            if edge_length == 0:
                continue
            edge_unit = edge_vec / edge_length

            # Compute the inward normal.
            # For a CCW polygon, the interior is left of the edge, so inward = (-dy, dx).
            if ccw:
                inward = np.array([-edge_unit[1], edge_unit[0]])
            else:
                inward = np.array([edge_unit[1], -edge_unit[0]])

            # Collect all edge_points that lie on this edge.
            points_on_edge = []
            for p in edge_points:
                if self.is_point_on_segment(p[0], p[1], A[0], A[1], B[0], B[1]):
                    # Compute the projection parameter (distance from A along the edge).
                    t = np.dot(np.array(p) - A, edge_unit)
                    points_on_edge.append((t, np.array(p)))
            if len(points_on_edge) < 2:
                continue  # Need at least two points to define a segment

            # Sort points along the edge.
            points_on_edge.sort(key=lambda x: x[0])
            t0 = points_on_edge[0][0]
            t1 = points_on_edge[-1][0]
            segment_length = t1 - t0
            if segment_length < min_house:
                continue  # Not enough room for even one house

            # Partition the available segment length into house widths.
            sizes = []
            R = segment_length
            while R > 1e-6:  # while remaining length is significant
                if R < min_house:
                    # If remainder is too small, add it to the last house.
                    if sizes:
                        sizes[-1] += R
                    else:
                        sizes.append(R)
                    R = 0
                else:
                    if R <= max_house:
                        size = R
                    else:
                        # Randomly select a size between min_house and max_house.
                        size = random.uniform(min_house, max_house)
                        # Ensure that the remaining segment after subtracting size is either 0 or at least min_house.
                        if 0 < (R - size) < min_house:
                            size = R - min_house
                            if size < min_house:
                                size = min_house
                    sizes.append(size)
                    R -= size

            # Generate houses along the edge using the computed sizes.
            cumulative_offset = t0
            for house_width in sizes:
                # Define the four corners of the house square.
                bottom_left = A + edge_unit * cumulative_offset
                bottom_right = bottom_left + edge_unit * house_width
                top_right = bottom_right + inward * house_width
                top_left = bottom_left + inward * house_width

                house_vertices = [
                    tuple(bottom_left),
                    tuple(bottom_right),
                    tuple(top_right),
                    tuple(top_left)
                ]
                center = tuple((np.array(bottom_left) + np.array(bottom_right) +
                                np.array(top_right) + np.array(top_left)) / 4)
                additional_houses[center] = house_vertices

                cumulative_offset += house_width

        return additional_houses
    
    def resolve_overlapping_additional_houses(self, additional_houses, step=0.5, min_size=5.0, max_iterations=100):
        """
        Checks for overlaps among the additional houses and, if found, iteratively
        reduces the size (i.e. the square’s side length) of one of the overlapping houses
        until there is no overlap (or until a minimum size is reached).

        Parameters:
        additional_houses: dict mapping house centers to a list of 4 vertices (tuples)
        step: the amount by which to reduce a house’s size if an overlap is detected.
        min_size: the smallest allowed side length for a house.
        max_iterations: maximum number of iterations to attempt resolving overlaps.

        Returns:
        A new dictionary of additional houses (mapping updated centers to vertices)
        with overlaps resolved.
        """
        # First, convert each house to a structure that holds its base parameters.
        # We assume each house is a square defined by:
        # [bottom_left, bottom_right, top_right, top_left]
        house_list = []
        for center, vertices in additional_houses.items():
            # Extract the four corners as numpy arrays.
            bottom_left = np.array(vertices[0])
            bottom_right = np.array(vertices[1])
            top_right = np.array(vertices[2])
            top_left = np.array(vertices[3])
            # Compute the current side length (assumed equal to the distance along the bottom edge)
            size = np.linalg.norm(bottom_right - bottom_left)
            # Compute the direction along the base (bottom edge)
            edge_vec = bottom_right - bottom_left
            if np.linalg.norm(edge_vec) != 0:
                edge_unit = edge_vec / np.linalg.norm(edge_vec)
            else:
                edge_unit = np.array([1, 0])
            # Compute the inward (vertical) direction (from bottom_left to top_left)
            inward_vec = top_left - bottom_left
            if np.linalg.norm(inward_vec) != 0:
                inward_unit = inward_vec / np.linalg.norm(inward_vec)
            else:
                inward_unit = np.array([0, 1])
            # Create the Shapely polygon
            poly = Polygon(vertices)
            house_list.append({
                'center': center,
                'vertices': vertices,
                'size': size,
                'bottom_left': bottom_left,
                'edge_unit': edge_unit,
                'inward_unit': inward_unit,
                'polygon': poly
            })

        iteration = 0
        overlap_found = True

        while overlap_found and iteration < max_iterations:
            overlap_found = False
            # Check every pair of houses for intersection
            for i in range(len(house_list)):
                for j in range(i + 1, len(house_list)):
                    poly1 = house_list[i]['polygon']
                    poly2 = house_list[j]['polygon']
                    if poly1.intersects(poly2):
                        # Only consider intersections with a meaningful area.
                        inter_area = poly1.intersection(poly2).area
                        if inter_area > 1e-5:
                            overlap_found = True
                            # For this example, adjust the house with the larger size.
                            if house_list[i]['size'] >= house_list[j]['size']:
                                house_to_adjust = house_list[i]
                            else:
                                house_to_adjust = house_list[j]
                            # Calculate the new size, ensuring we do not go below min_size.
                            new_size = max(house_to_adjust['size'] - step, min_size)
                            # If already at min_size, we cannot reduce further.
                            if new_size == house_to_adjust['size']:
                                continue
                            house_to_adjust['size'] = new_size
                            # Recalculate the vertices keeping the bottom edge fixed.
                            bl = house_to_adjust['bottom_left']
                            eu = house_to_adjust['edge_unit']
                            iu = house_to_adjust['inward_unit']
                            new_bottom_left = bl  # remains the same
                            new_bottom_right = bl + eu * new_size
                            new_top_left = bl + iu * new_size
                            new_top_right = new_bottom_right + iu * new_size
                            new_vertices = [
                                tuple(new_bottom_left),
                                tuple(new_bottom_right),
                                tuple(new_top_right),
                                tuple(new_top_left)
                            ]
                            house_to_adjust['vertices'] = new_vertices
                            # Recalculate the center as the average of the vertices.
                            new_center = tuple(np.mean(np.array(new_vertices), axis=0))
                            house_to_adjust['center'] = new_center
                            # Update the Shapely polygon.
                            house_to_adjust['polygon'] = Polygon(new_vertices)
            iteration += 1

        # Convert the list back into a dictionary mapping centers to vertices.
        resolved_houses = {}
        for house in house_list:
            resolved_houses[house['center']] = house['vertices']
        return resolved_houses
    
    def clear_edge_points(self, edge_points, polygon_vertices):
        """
        Removes any edge points that are exactly the same as polygon vertices.

        Args:
            edge_points (set of tuples): Set of edge points (x, y).
            polygon_vertices (list of tuples): List of polygon vertices (x, y).

        Returns:
            set: A new set with the unwanted edge points removed.
        """
        polygon_vertex_set = set(polygon_vertices)  # Convert list to a set for fast lookup
        cleaned_edge_points = edge_points - polygon_vertex_set  # Remove matching points

        return cleaned_edge_points

    # ---------------------------------------------------------------------------------
    # OVERLAPS
    # ---------------------------------------------------------------------------------

    def overlap_pairs(
        self,
        houses: dict | None = None,
        *,
        area_tol: float = 1e-6,
    ) -> list[tuple[tuple, tuple]]:
        """
        Return a list of *pairs* that truly overlap.

        Each item is  ((centerA, verticesA), (centerB, verticesB)).
        """
        if houses is None:
            if getattr(self, "houses", None) is None:
                raise ValueError("No house data – pass a dict or run the pipeline first.")
            houses = self.houses

        items = list(houses.items())
        n      = len(items)
        pairs  = []

        for i in range(n):
            c_i, v_i = items[i]
            poly_i   = Polygon(v_i)
            for j in range(i + 1, n):
                c_j, v_j = items[j]
                poly_j   = Polygon(v_j)

                if poly_i.intersects(poly_j) and \
                   poly_i.intersection(poly_j).area > area_tol:
                    pairs.append(((c_i, v_i), (c_j, v_j)))
        return pairs


    def overlap_dict(
        self,
        houses: dict | None = None,
        *,
        area_tol: float = 1e-6,
        drop_solo: bool = True,
    ) -> dict[tuple, list]:
        """
        Return a dictionary   { house_center : [centers it overlaps] }.

        Parameters
        ----------
        drop_solo : if True (default) omit houses that overlap nobody.
        """
        if houses is None:
            if getattr(self, "houses", None) is None:
                raise ValueError("No house data – pass a dict or run the pipeline first.")
            houses = self.houses

        # initialise adjacency list
        adj = defaultdict(list)
        for (c1, v1), (c2, v2) in self.overlap_pairs(houses, area_tol=area_tol):
            adj[c1].append(c2)
            adj[c2].append(c1)

        if drop_solo:
            return dict(adj)          # only the ones with neighbours

        # add non-overlapping houses with an empty list
        full = {c: [] for c in houses}
        full.update(adj)
        return full

    # ───── Pipeline Methods ─────

    def initialise_vertex_houses(self, randomize_size=True, min_size=5, max_size=16, overlap_threshold=90):
        """
        Runs the complete pipeline to create vertex houses and returns the 
        eight-vertex houses.
        
        Steps:
          1. Create houses along all edges.
          2. Group by vertex.
          3. Remove houses with overlap ≥ overlap_threshold%.
          4. Merge remaining houses into six-vertex houses.
          5. Merge overlapping six-vertex houses into eight-vertex houses.
          
        The final eight-vertex houses are stored in self.eight_vertex_houses.
        
        Returns:
            list: A list of eight-vertex houses (each is a list of vertices).
        """
        houses = self.create_all_elbow_houses(randomize_size=randomize_size, min_size=min_size, max_size=max_size)
        grouped = self.group_houses_by_vertex(houses)
        pruned = self.remove_highly_overlapping_houses(grouped, threshold=overlap_threshold)
        merged_six = self.merge_overlapping_houses(pruned, threshold=overlap_threshold)
        # merged_six_vert = list(merged_six.values())
        eight_vertex_houses = self.merge_overlapping_six_vertex_houses(list(merged_six.values()), threshold=0)
        self.eight_vertex_houses = eight_vertex_houses
        return eight_vertex_houses
    

    def get_vertex_house_points(self):
        """
        Returns the eight-vertex houses (vertex house points) if they have been initialised.
        
        Returns:
            list or None: The eight-vertex houses as a list of vertex lists, or None if not initialised.
        """
        return self.eight_vertex_houses if hasattr(self, 'eight_vertex_houses') else None

    def get_ordered_vertex_houses(self, polygon_vertices):
        """
        Given an ordered list of the area polygon's vertices (not closed),
        returns a dictionary mapping each vertex to the eight-vertex house whose centroid is closest.
        
        Args:
            polygon_vertices (list): List of vertices (tuples) in order.
            
        Returns:
            dict: Mapping from vertex (tuple) to eight-vertex house (list of vertices).
        """
        ordered = {}
        if not hasattr(self, 'eight_vertex_houses'):
            return ordered
        for v in polygon_vertices:
            v_arr = np.array(v)
            best_house = None
            best_dist = np.inf
            for h in self.eight_vertex_houses:
                h_poly = Polygon(h)
                centroid = np.array(h_poly.centroid.coords[0])
                dist = np.linalg.norm(centroid - v_arr)
                if dist < best_dist:
                    best_dist = dist
                    best_house = h
            ordered[v] = best_house
        return ordered


    def get_side_and_vertex_houses(self, polygon_vertices, houses):
        """
        Runs the complete pipeline to return both corrected houses and additional houses as a dictionary.

        Args:
            polygon_vertices (list): List of vertices defining the main polygon.
            houses (dict): Dictionary of initial houses, where keys are centers and values are vertices.

        Returns:
            dict: A dictionary where keys are house centers and values are lists of house vertices.
        """
        # Process the original houses (convert 6-vertex approximations to 4-vertex squares if applicable).
        square_houses = self.find_square_houses(houses)
        cleaned_houses = {center: self.remove_extra_vertices(vertices) for center, vertices in square_houses.items()}
        houses = self.convert_six_vertex_squares_to_normal(houses, cleaned_houses)

        # Adjust house vertices by moving points outside the polygon onto the nearest edge.
        houses = self.move_houses_onto_polygon_edges(polygon_vertices, houses)

        # Identify points from houses that lie exactly on the polygon edges.
        edge_points = self.get_house_points_on_polygon_edges(polygon_vertices, houses)
        edge_points = self.clear_edge_points(edge_points, polygon_vertices)

        # Generate additional houses along each polygon edge.
        additional_houses = self.generate_houses_along_edges(edge_points, polygon_vertices, min_house=5, max_house=16)
        additional_houses = self.resolve_overlapping_additional_houses(additional_houses, step=0.5, min_size=5.0, max_iterations=100)

        # Merge both sets of houses into a single dictionary.
        # additional_houses
        all_houses = {**houses, **additional_houses}

        self.houses = all_houses
        return all_houses

    def get_houses(self):
        return self.houses