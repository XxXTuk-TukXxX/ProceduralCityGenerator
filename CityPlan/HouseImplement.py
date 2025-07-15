import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box, Point
from shapely.strtree import STRtree
import random

def greedy_integer_grid_fill_random_sizes(outline, min_size=5, max_size=16):
    poly = Polygon(outline)
    placed = []
    minx, miny, maxx, maxy = poly.bounds

    # Work with integer bounds
    minx = int(np.ceil(minx))
    miny = int(np.ceil(miny))
    maxx = int(np.floor(maxx))
    maxy = int(np.floor(maxy))

    y = miny
    while y <= maxy - min_size:
        x = minx
        while x <= maxx - min_size:
            # Randomize the order of sizes for more organic placement
            sizes = list(range(min_size, max_size + 1))
            random.shuffle(sizes)
            for size in sizes:
                candidate = box(x, y, x + size, y + size)
                if poly.contains(candidate) and not any(house.intersects(candidate) for house in placed):
                    placed.append(candidate)
                    x += size  # move to next position after the house
                    break
            else:
                x += 1
        y += 1
    return placed

# ----------------------------------------------------------------


def greedy_integer_grid_fill(outline, min_size=5, max_size=16):
    poly = Polygon(outline)
    placed = []
    minx, miny, maxx, maxy = poly.bounds

    # Ensure we work with integer boundaries
    minx = int(np.ceil(minx))
    miny = int(np.ceil(miny))
    maxx = int(np.floor(maxx))
    maxy = int(np.floor(maxy))

    y = miny
    while y <= maxy - min_size:
        x = minx
        while x <= maxx - min_size:
            # Try biggest possible integer house at this integer coordinate
            for size in range(max_size, min_size - 1, -1):
                # Only integer sizes, only integer-aligned placement
                candidate = box(x, y, x + size, y + size)
                # Check polygon containment and overlap
                if poly.contains(candidate) and not any(house.intersects(candidate) for house in placed):
                    placed.append(candidate)
                    x += size  # skip to the next non-overlapping coordinate
                    break
            else:
                x += 1  # can't fit any, try next integer x
        y += 1  # move to next integer y
    return placed


# ----------------------------------------------------------------


def partition_length(length, min_size, max_size):
    """Partition an integer length into random squares [min_size, max_size] that sum exactly."""
    result = []
    n = length
    while n > 0:
        max_part = min(max_size, n)
        if max_part < min_size:
            return None  # not possible
        size = random.randint(min_size, max_part)
        result.append(size)
        n -= size
    return result

def gapless_perfect_square_rows(outline, min_size=5, max_size=16):
    poly = Polygon(outline)
    minx, miny, maxx, maxy = poly.bounds

    minx = int(np.floor(minx))
    miny = int(np.floor(miny))
    maxx = int(np.ceil(maxx))
    maxy = int(np.ceil(maxy))

    houses = []
    y = miny
    while y <= maxy - min_size:
        # Always pick a row_height for this y!
        row_height = random.randint(min_size, max_size)
        # Find all contiguous [x_start, x_end) intervals inside the polygon at this y
        x_intervals = []
        x = minx
        while x <= maxx - min_size:
            pt = Point(x + 0.5, y + 0.5)
            if poly.contains(pt):
                x_start = x
                while x <= maxx - min_size and poly.contains(Point(x + 0.5, y + 0.5)):
                    x += 1
                x_end = x
                if x_end - x_start >= min_size:
                    x_intervals.append((x_start, x_end))
            x += 1
        # For each interval, partition and place houses
        for x0, x1 in x_intervals:
            interval_width = x1 - x0
            partitions = partition_length(interval_width, min_size, max_size)
            if partitions is None:
                continue
            xpos = x0
            for p in partitions:
                candidate = box(xpos, y, xpos + p, y + row_height)
                if poly.contains(candidate):
                    houses.append(candidate)
                xpos += p
        y += row_height
    return houses










def plot_filled_polygon_and_squares(outline, squares):
    plt.figure(figsize=(12, 12))
    poly = Polygon(outline)
    xs, ys = poly.exterior.xy
    plt.plot(xs, ys, color='black', linewidth=2)
    colors = ['orange', 'skyblue', 'lightgreen', 'salmon', 'violet', 'khaki', 'gray']
    for sq in squares:
        xs, ys = sq.exterior.xy
        color = random.choice(colors)
        plt.fill(xs, ys, alpha=0.5, edgecolor='blue', color=color)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def print_house_data(squares):
    # Print all house origins and sizes as integers
    for sq in squares:
        minx, miny, maxx, maxy = map(int, sq.bounds)
        size = maxx - minx  # since it's a square
        print(f'House at ({minx}, {miny}), size {size}')
