import random
import math
from shapely.geometry import Point, Polygon

class Area:
    def __init__(self, num_vertices=5, seed=42, radius=100):
        self.num_vertices = num_vertices
        self.seed = seed
        self.radius = radius
        self.vertices = self.generate_random_polygon()
        self.polygon = Polygon(self.vertices)          # Shapely polygon
        self.points = self.generate_poisson_disk_points()

    def generate_random_polygon(self):
        random.seed(self.seed)
        angles = sorted(random.uniform(0, 2 * math.pi) for _ in range(self.num_vertices))
        return [(self.radius * math.cos(a), self.radius * math.sin(a)) for a in angles]

    def generate_poisson_disk_points(self, min_distance=30):
        points = []
        while len(points) < self.num_vertices * 3:
            candidate = (random.uniform(-self.radius, self.radius),
                         random.uniform(-self.radius, self.radius))
            if self.polygon.contains(Point(candidate)) and \
               all(math.dist(candidate, p) >= min_distance for p in points):
                points.append(candidate)
        return points
    
    def get_vertices(self):
        return self.vertices

    def get_polygon(self):
        return self.polygon

    def get_points(self):
        return self.points