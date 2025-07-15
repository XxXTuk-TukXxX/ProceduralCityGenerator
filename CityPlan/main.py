# Generate a random area
from CityPlan.CityPlanner import generate_random_polygon, CityPlanner
from CityPlan.adjust import extract_vertices
from CityPlan.HouseImplement import greedy_integer_grid_fill, plot_filled_polygon_and_squares, print_house_data


def polygon_to_tuples(polygon):
    """
    Converts a Shapely Polygon object to a list of (x, y) tuples.
    """
    return list(polygon.exterior.coords)

demo_ring = generate_random_polygon(seed=125, radius=100, num_vertices=5)
cp = CityPlanner(demo_ring)

# Build grid (for plotting only, not needed for mask-and-fill)
grid_xy = cp.build_grid(spacing=5)
gx, gy = zip(*grid_xy)

# Grid outline refactor
grid_outline = CityPlanner.grid_outline_from_points(grid_xy, spacing=5)
grid_outline = polygon_to_tuples(grid_outline)
grid_outline = extract_vertices(grid_outline)

print(grid_outline)

houses = greedy_integer_grid_fill(grid_outline, min_size=5, max_size=16)
plot_filled_polygon_and_squares(grid_outline, houses)