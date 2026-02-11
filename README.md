# ProceduralCityGenerator

A small procedural “city” prototype that:

1. Generates a random **boundary polygon** (`Area`)
2. Creates interior seed points via **Poisson-disk sampling**
3. Builds **Voronoi-like regions** clipped to the boundary (`Territory`)
4. Adjusts regions to reduce/remove **acute interior angles** and optionally merges regions
5. Places simple **house footprints** along region edges (`houses/side_house.py`)

The output is visualized with Matplotlib (plots of regions + house outlines).

## Project structure

- `main.py` — end-to-end demo: generate territory regions and draw side/vertex houses.
- `main_overlap.py` — alternative demo focusing on overlap handling / experiments.
- `area.py` — random boundary polygon + Poisson-disk point generation.
- `territory.py` — Voronoi generation, angle adjustment, edge deletion/merging, and region output.
- `houses/side_house.py` — constructs square-ish houses along polygon edges and handles overlap.
- `houses/center_region.py` — (optional/experimental) computes an interior “free space” outline by subtracting houses.
- `houses/center_house.py` — (optional/experimental) builds an interior grid and outline from a center region ring.

## Requirements

Python 3.10+ recommended.

Install dependencies:

```bash
python3 -m pip install numpy scipy shapely matplotlib
```

## Run

From this directory:

```bash
python3 main.py
```

Or:

```bash
python3 main_overlap.py
```

## Tuning

Common places to tweak behavior:

- `Area(num_vertices=..., seed=..., radius=...)` in `area.py`
- Poisson-disk spacing via `Area.generate_poisson_disk_points(min_distance=...)`
- House sizing/overlap in `SideHouse.initialise_vertex_houses(...)` (see `main.py`)

