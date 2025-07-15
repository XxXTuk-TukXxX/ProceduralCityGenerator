import numpy as np
import matplotlib.pyplot as plt

def extract_vertices(grid):
    """
    Keeps only points where the direction changes (i.e., true vertices).
    The first point is always kept.
    """
    if len(grid) < 3:
        return grid.copy()
    grid_np = np.array(grid)
    vertices = [tuple(grid_np[0])]  # Always keep the first point

    for i in range(1, len(grid)-1):
        prev_vec = grid_np[i] - grid_np[i-1]
        next_vec = grid_np[i+1] - grid_np[i]
        # Normalize for direction (avoid length, just direction)
        prev_dir = np.sign(prev_vec)
        next_dir = np.sign(next_vec)
        if not np.array_equal(prev_dir, next_dir):
            vertices.append(tuple(grid_np[i]))
    vertices.append(tuple(grid_np[-1]))  # Always keep the last point
    return vertices