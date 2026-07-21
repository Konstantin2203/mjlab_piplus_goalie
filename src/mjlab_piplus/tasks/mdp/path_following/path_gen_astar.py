"""Obstacle-world path generation with A*.

An alternative to the parametric b-spline generator in ``path_gen.py``. Instead
of drawing a smooth free-space squiggle, each path here is *planned*: a world is
filled with randomly sized cubes and cylinders, a start and a goal pose are
sampled in free space, and A* finds a route through the obstacles (inflated by a
configurable radius, so the reference path keeps clearance). The result looks
like a real navigation path — corridors, corners, detours — rather than an
arbitrary curve.

The obstacles are **virtual**: they are never added to the MuJoCo scene, so the
robot cannot collide with them and no physics cost is paid. They exist to shape
the path, and are stored next to it so the command term can draw them in the
viewer (see ``PathCommand._debug_vis_impl``).

Output format is identical to ``generate_parametric_paths`` so the two are
drop-in alternatives:

- ``paths``: ``(num_paths, max_points, 6)`` — x, y, dx, dy, ddx, ddy, sampled at
  a fixed arc-length spacing and padded with the goal point.
- ``waypoints``: ``(num_paths, num_waypoints, 2)`` — evenly spaced along the
  path; the last one is the goal.

plus one extra tensor:

- ``obstacles``: ``(num_paths, max_obstacles, 7)`` — see ``OBSTACLE_*`` below.

Everything is expressed in a frame whose origin is the start pose, matching the
parametric generator (whose paths also start at the origin).

Pipeline per world: sample obstacles -> rasterize to an occupancy grid ->
Euclidean distance transform (obstacle inflation) -> sample start/goal -> A* on
the 8-connected free grid -> greedy line-of-sight shortcutting -> Chaikin corner
rounding, verified against the exact obstacle geometry -> arc-length resampling.
Straight paths (max lateral deviation from the start->goal line below a
threshold) are rejected once a configurable fraction of the library is already
straight.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

# Obstacle encoding: one row per obstacle, [kind, x, y, yaw, a, b, half_height].
# For a box, (a, b) are the half extents along the obstacle's local x/y and
# ``yaw`` its rotation; for a cylinder, ``a == b`` is the radius and yaw is 0.
# Padding rows (unused slots) are all zeros with kind == OBSTACLE_KIND_NONE.
OBSTACLE_DIM = 7
OBSTACLE_KIND_NONE = -1.0
OBSTACLE_KIND_BOX = 0.0
OBSTACLE_KIND_CYLINDER = 1.0

_SQRT2 = float(np.sqrt(2.0))
# (di, dj, step cost) for an 8-connected grid.
_NEIGHBORS = (
  (-1, 0, 1.0),
  (1, 0, 1.0),
  (0, -1, 1.0),
  (0, 1, 1.0),
  (-1, -1, _SQRT2),
  (-1, 1, _SQRT2),
  (1, -1, _SQRT2),
  (1, 1, _SQRT2),
)


@dataclass
class ObstacleWorldCfg:
  """Sampling and planning parameters for one obstacle world."""

  # World.
  world_size: tuple[float, float] = (5.0, 5.0)
  """Extent of the world in meters. Obstacles and start/goal are sampled here."""
  grid_resolution: float = 0.05
  """Cell size of the planning grid, in meters."""

  # Obstacles.
  density_range: tuple[float, float] = (0.3, 0.50)
  """Obstacles per square meter, sampled uniformly per world."""
  max_obstacles: int = 64
  """Hard cap on obstacles per world; also the width of the obstacle tensor."""
  cylinder_fraction: float = 0.5
  """Probability that an obstacle is a cylinder rather than a box."""
  box_size_range: tuple[float, float] = (0.25, 1.2)
  """Full side length of a box, in meters. x and y are sampled independently."""
  cylinder_radius_range: tuple[float, float] = (0.1, 0.5)
  """Radius of a cylinder, in meters."""
  height: float = 1.0
  """Obstacle height, in meters. Visualization only — nothing plans in 3D."""

  # Planning.
  inflation_radius: float = 0.35
  """Obstacles are inflated by this many meters before planning, so the path
  keeps at least this much clearance from any obstacle surface."""
  smoothing_clearance: float = 0.2
  """Minimum clearance the *smoothed* path must keep, verified against the exact
  obstacle geometry. Corner rounding cuts inside the inflated polyline, so it is
  validated against this looser bound and backed off (fewer smoothing passes)
  until it holds."""
  smoothing_iterations: int = 3
  """Chaikin corner-rounding passes applied to the shortcut polyline."""

  # Start / goal sampling.
  start_goal_distance_range: tuple[float, float] = (3.0, 8.0)
  """Allowed straight-line distance between start and goal, in meters."""

  # Straight-path budget.
  max_straight_fraction: float = 0.2
  """Upper bound on the fraction of paths allowed to be straight (see
  ``straight_deviation``). Paths that come out straight beyond this budget are
  rejected and regenerated."""
  straight_deviation: float = 0.5
  """A path counts as straight when its maximum lateral deviation from the
  start->goal line stays below this many meters."""

  # Retries.
  max_world_attempts: int = 20
  """Obstacle layouts to try per path before giving up."""
  max_query_attempts: int = 20
  """Start/goal pairs to try per obstacle layout."""

  def __post_init__(self):
    if self.smoothing_clearance > self.inflation_radius:
      raise ValueError(
        "smoothing_clearance must be <= inflation_radius "
        f"(got {self.smoothing_clearance} > {self.inflation_radius})."
      )
    if not 0.0 <= self.max_straight_fraction <= 1.0:
      raise ValueError("max_straight_fraction must be in [0, 1].")

  def cache_key(self) -> tuple:
    """Hashable snapshot of every field that affects generation."""
    return tuple(sorted((k, repr(v)) for k, v in vars(self).items()))


@dataclass
class _WorldResult:
  path: np.ndarray  # (max_points, 6)
  waypoints: np.ndarray  # (num_waypoints, 2)
  obstacles: np.ndarray  # (max_obstacles, 7)
  is_straight: bool
  forced: bool  # True if accepted despite violating the straight budget.


##
# Obstacle sampling and rasterization.
##


def _sample_obstacles(cfg: ObstacleWorldCfg, rng: np.random.Generator) -> np.ndarray:
  """Sample a padded ``(max_obstacles, 7)`` obstacle array for one world."""
  world_x, world_y = cfg.world_size
  density = rng.uniform(*cfg.density_range)
  num = int(min(round(density * world_x * world_y), cfg.max_obstacles))

  obstacles = np.zeros((cfg.max_obstacles, OBSTACLE_DIM), dtype=np.float64)
  obstacles[:, 0] = OBSTACLE_KIND_NONE
  if num == 0:
    return obstacles

  is_cylinder = rng.random(num) < cfg.cylinder_fraction
  obstacles[:num, 0] = np.where(is_cylinder, OBSTACLE_KIND_CYLINDER, OBSTACLE_KIND_BOX)
  obstacles[:num, 1] = rng.uniform(0.0, world_x, num)
  obstacles[:num, 2] = rng.uniform(0.0, world_y, num)
  obstacles[:num, 3] = np.where(is_cylinder, 0.0, rng.uniform(-np.pi, np.pi, num))

  radius = rng.uniform(*cfg.cylinder_radius_range, num)
  half_x = 0.5 * rng.uniform(*cfg.box_size_range, num)
  half_y = 0.5 * rng.uniform(*cfg.box_size_range, num)
  obstacles[:num, 4] = np.where(is_cylinder, radius, half_x)
  obstacles[:num, 5] = np.where(is_cylinder, radius, half_y)
  obstacles[:num, 6] = 0.5 * cfg.height
  return obstacles


def _occupancy_grid(obstacles: np.ndarray, cfg: ObstacleWorldCfg) -> np.ndarray:
  """Rasterize obstacles into a ``(nx, ny)`` boolean occupancy grid.

  Cell ``(i, j)`` covers the square centered at ``((i + .5) * res, (j + .5) * res)``.
  Only the cells inside each obstacle's bounding box are tested.
  """
  res = cfg.grid_resolution
  nx = max(int(round(cfg.world_size[0] / res)), 1)
  ny = max(int(round(cfg.world_size[1] / res)), 1)
  occ = np.zeros((nx, ny), dtype=bool)

  for kind, x, y, yaw, a, b, _ in obstacles:
    if kind == OBSTACLE_KIND_NONE:
      continue
    reach = a if kind == OBSTACLE_KIND_CYLINDER else float(np.hypot(a, b))
    i0 = int(np.clip(np.floor((x - reach) / res), 0, nx))
    i1 = int(np.clip(np.ceil((x + reach) / res) + 1, 0, nx))
    j0 = int(np.clip(np.floor((y - reach) / res), 0, ny))
    j1 = int(np.clip(np.ceil((y + reach) / res) + 1, 0, ny))
    if i0 >= i1 or j0 >= j1:
      continue
    cx = (np.arange(i0, i1) + 0.5) * res - x
    cy = (np.arange(j0, j1) + 0.5) * res - y
    dx = cx[:, None]
    dy = cy[None, :]
    if kind == OBSTACLE_KIND_CYLINDER:
      mask = dx**2 + dy**2 <= a**2
    else:
      cos_y, sin_y = np.cos(yaw), np.sin(yaw)
      local_x = cos_y * dx + sin_y * dy
      local_y = -sin_y * dx + cos_y * dy
      mask = (np.abs(local_x) <= a) & (np.abs(local_y) <= b)
    occ[i0:i1, j0:j1] |= mask
  return occ


def _clearance_grid(occ: np.ndarray, resolution: float) -> np.ndarray:
  """Conservative clearance of each cell from the obstacles, in meters.

  The distance transform measures cell center to cell *center*, and a cell is
  only occupied when its center is inside an obstacle, so the true obstacle
  surface can sit up to half a cell diagonal closer than the transform says.
  That much is subtracted here to keep the planning margin honest. It is an
  estimate, not a bound (a thin obstacle can slip between cell centers) — the
  final path clearance is verified against the exact geometry in ``_smooth``.
  """
  edt = distance_transform_edt(~occ, sampling=resolution)
  return np.maximum(edt - 0.5 * _SQRT2 * resolution, 0.0)  # type: ignore[operator]


def _obstacle_distance(xy: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
  """Exact distance from each point to the nearest obstacle surface.

  Negative inside an obstacle. Boxes use the usual rotated-box SDF.
  """
  best = np.full(len(xy), np.inf)
  for kind, x, y, yaw, a, b, _ in obstacles:
    if kind == OBSTACLE_KIND_NONE:
      continue
    dx, dy = xy[:, 0] - x, xy[:, 1] - y
    if kind == OBSTACLE_KIND_CYLINDER:
      dist = np.hypot(dx, dy) - a
    else:
      cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
      local_x = np.abs(cos_yaw * dx + sin_yaw * dy) - a
      local_y = np.abs(-sin_yaw * dx + cos_yaw * dy) - b
      dist = np.hypot(np.maximum(local_x, 0.0), np.maximum(local_y, 0.0)) + np.minimum(
        np.maximum(local_x, local_y), 0.0
      )
    best = np.minimum(best, dist)
  return best


##
# Planning.
##


def _astar(free: np.ndarray, start: tuple[int, int], goal: tuple[int, int]):
  """A* over an 8-connected grid of free cells. Returns cell indices or None."""
  nx, ny = free.shape
  flat_free = free.reshape(-1)
  start_id = start[0] * ny + start[1]
  goal_id = goal[0] * ny + goal[1]
  if not (flat_free[start_id] and flat_free[goal_id]):
    return None

  g = np.full(nx * ny, np.inf, dtype=np.float64)
  came_from = np.full(nx * ny, -1, dtype=np.int64)
  closed = np.zeros(nx * ny, dtype=bool)
  g[start_id] = 0.0
  heap: list[tuple[float, int]] = [(0.0, start_id)]
  gi, gj = goal
  found = False

  while heap:
    _, current = heappop(heap)
    if closed[current]:
      continue
    if current == goal_id:
      found = True
      break
    closed[current] = True
    ci, cj = divmod(current, ny)
    g_cur = g[current]
    for di, dj, step in _NEIGHBORS:
      ni, nj = ci + di, cj + dj
      if not (0 <= ni < nx and 0 <= nj < ny):
        continue
      nid = ni * ny + nj
      if closed[nid] or not flat_free[nid]:
        continue
      # No cutting diagonally between two blocked cells.
      if di and dj and not (flat_free[ci * ny + nj] and flat_free[ni * ny + cj]):
        continue
      tentative = g_cur + step
      if tentative < g[nid]:
        g[nid] = tentative
        came_from[nid] = current
        dx, dy = abs(ni - gi), abs(nj - gj)
        h = (dx + dy) + (_SQRT2 - 2.0) * min(dx, dy)
        heappush(heap, (tentative + h, nid))

  if not found:
    return None

  cells = []
  node = goal_id
  while node != -1:
    cells.append(divmod(node, ny))
    node = came_from[node]
  return np.array(cells[::-1], dtype=np.float64)


def _line_is_free(free: np.ndarray, p: np.ndarray, q: np.ndarray) -> bool:
  """True if every cell along the segment ``p -> q`` (cell coords) is free."""
  delta = q - p
  num = int(np.ceil(np.hypot(delta[0], delta[1]) * 2.0)) + 1
  ts = np.linspace(0.0, 1.0, num)
  pts = p[None, :] + ts[:, None] * delta[None, :]
  ii = np.clip(np.rint(pts[:, 0]).astype(int), 0, free.shape[0] - 1)
  jj = np.clip(np.rint(pts[:, 1]).astype(int), 0, free.shape[1] - 1)
  return bool(free[ii, jj].all())


def _shortcut(cells: np.ndarray, free: np.ndarray) -> np.ndarray:
  """Greedy line-of-sight shortcutting: keep only the corners of the A* path."""
  pruned = [cells[0]]
  i = 0
  last = len(cells) - 1
  while i < last:
    j = last
    while j > i + 1 and not _line_is_free(free, cells[i], cells[j]):
      j -= 1
    pruned.append(cells[j])
    i = j
  return np.array(pruned, dtype=np.float64)


def _chaikin(polyline: np.ndarray, iterations: int) -> np.ndarray:
  """Chaikin corner cutting with the endpoints pinned."""
  for _ in range(iterations):
    if len(polyline) < 3:
      break
    head, tail = polyline[:-1], polyline[1:]
    cut = np.empty((2 * len(head), 2), dtype=np.float64)
    cut[0::2] = 0.75 * head + 0.25 * tail
    cut[1::2] = 0.25 * head + 0.75 * tail
    polyline = np.vstack([polyline[:1], cut, polyline[-1:]])
  return polyline


def _polyline_clearance(
  polyline: np.ndarray, obstacles: np.ndarray, step: float
) -> float:
  """Minimum exact obstacle clearance along a metric polyline, in meters.

  Sampled every ``step`` meters. Distance to an obstacle is 1-Lipschitz along
  the path, so the true minimum is at most ``step / 2`` below what is returned.
  """
  seg = np.hypot(*np.diff(polyline, axis=0).T)
  s_cum = np.concatenate(([0.0], np.cumsum(seg)))
  total = s_cum[-1]
  if total <= 0.0:
    return float(_obstacle_distance(polyline[:1], obstacles)[0])
  keep = np.concatenate(([True], seg > 1e-12))
  targets = np.linspace(0.0, total, int(np.ceil(total / step)) + 1)
  xy = np.stack(
    [
      np.interp(targets, s_cum[keep], polyline[keep, 0]),
      np.interp(targets, s_cum[keep], polyline[keep, 1]),
    ],
    axis=-1,
  )
  return float(_obstacle_distance(xy, obstacles).min())


def _smooth(
  cells: np.ndarray, free: np.ndarray, obstacles: np.ndarray, cfg: ObstacleWorldCfg
) -> np.ndarray | None:
  """Shortcut then round the A* cell path; returns a metric xy polyline.

  Corner cutting pulls the path inside the shortcut polyline, which can eat into
  the inflation margin at tight corners, so each candidate is checked against the
  real obstacle geometry (not the grid, whose discretization error is comparable
  to the margin at stake) and the number of passes reduced until the clearance
  holds. Returns None if even the unrounded shortcut polyline is too tight,
  which means the grid was optimistic about this route.
  """
  polyline = (_shortcut(cells, free) + 0.5) * cfg.grid_resolution
  step = 0.25 * cfg.grid_resolution
  for n in range(cfg.smoothing_iterations, -1, -1):
    candidate = _chaikin(polyline, n)
    if _polyline_clearance(candidate, obstacles, step) >= cfg.smoothing_clearance:
      return candidate
  return None


##
# Path post-processing.
##


def _resample(
  points: np.ndarray, point_spacing: float, max_points: int
) -> np.ndarray | None:
  """Resample a polyline at fixed arc-length spacing into ``(max_points, 6)``.

  Mirrors ``generate_parametric_path``: points are evenly spaced by
  ``point_spacing`` meters, the exact endpoint is always included, and the tail
  is padded with the final (goal) row. Returns None if the path does not fit in
  ``max_points`` samples.
  """
  seg = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
  s_cum = np.concatenate(([0.0], np.cumsum(seg)))
  total_length = s_cum[-1]
  if total_length <= 0.0:
    return None

  target_s = np.arange(0.0, total_length, point_spacing)
  if target_s.size == 0 or target_s[-1] < total_length - 1e-9:
    target_s = np.append(target_s, total_length)
  if target_s.size > max_points:
    return None

  # ``s_cum`` is strictly increasing except for coincident points; drop repeats
  # so np.interp stays well defined.
  keep = np.concatenate(([True], seg > 1e-12))
  s_keep, pts_keep = s_cum[keep], points[keep]
  x = np.interp(target_s, s_keep, pts_keep[:, 0])
  y = np.interp(target_s, s_keep, pts_keep[:, 1])

  if target_s.size >= 2:
    dx = np.gradient(x, target_s)
    dy = np.gradient(y, target_s)
    ddx = np.gradient(dx, target_s)
    ddy = np.gradient(dy, target_s)
  else:
    dx = dy = ddx = ddy = np.zeros_like(x)

  pts = np.stack([x, y, dx, dy, ddx, ddy], axis=-1)
  if pts.shape[0] < max_points:
    pad = np.repeat(pts[-1:], max_points - pts.shape[0], axis=0)
    pts = np.concatenate([pts, pad], axis=0)
  return pts


def _even_waypoints(xy: np.ndarray, num_waypoints: int) -> np.ndarray:
  """``num_waypoints`` points evenly spaced along the path; last one is the goal."""
  seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
  s_cum = np.concatenate(([0.0], np.cumsum(seg)))
  keep = np.concatenate(([True], seg > 1e-12))
  targets = np.linspace(0.0, s_cum[-1], num_waypoints)
  return np.stack(
    [
      np.interp(targets, s_cum[keep], xy[keep, 0]),
      np.interp(targets, s_cum[keep], xy[keep, 1]),
    ],
    axis=-1,
  )


def _max_lateral_deviation(xy: np.ndarray) -> float:
  """Largest perpendicular distance from the straight start->goal line."""
  start, goal = xy[0], xy[-1]
  chord = goal - start
  length = float(np.hypot(chord[0], chord[1]))
  if length < 1e-9:
    return 0.0
  normal = np.array([-chord[1], chord[0]]) / length
  return float(np.max(np.abs((xy - start) @ normal)))


##
# World generation.
##


def _generate_world(
  cfg: ObstacleWorldCfg,
  rng: np.random.Generator,
  point_spacing: float,
  num_waypoints: int,
  max_points: int,
  allow_straight: bool,
) -> _WorldResult:
  """Generate one obstacle world and an A* path through it.

  Retries on: no free space, no start/goal pair at the requested separation, no
  A* solution, a path too long for ``max_points``, or a straight path when the
  straight budget is exhausted. A straight path found while retrying is kept as
  a fallback so generation always returns something.
  """
  res = cfg.grid_resolution
  min_dist, max_dist = cfg.start_goal_distance_range
  fallback: _WorldResult | None = None

  for _ in range(cfg.max_world_attempts):
    obstacles = _sample_obstacles(cfg, rng)
    occ = _occupancy_grid(obstacles, cfg)
    free = _clearance_grid(occ, res) > cfg.inflation_radius
    free_cells = np.argwhere(free)
    if len(free_cells) < 2:
      continue

    for _ in range(cfg.max_query_attempts):
      start_cell = free_cells[rng.integers(len(free_cells))]
      dists = np.hypot(*(free_cells - start_cell).T) * res
      candidates = free_cells[(dists >= min_dist) & (dists <= max_dist)]
      if len(candidates) == 0:
        continue
      goal_cell = candidates[rng.integers(len(candidates))]

      cells = _astar(free, tuple(start_cell), tuple(goal_cell))
      if cells is None:
        continue
      polyline = _smooth(cells, free, obstacles, cfg)
      if polyline is None:
        continue
      # Arc-length resampling keeps every sample on the polyline, so the
      # clearance verified in _smooth carries over to the stored path.
      pts = _resample(polyline, point_spacing, max_points)
      if pts is None:
        continue

      # Shift into the start frame: the path starts at the origin, like the
      # parametric generator's. Obstacles move with it so the two stay aligned.
      # The shift goes to a copy — this attempt may still be rejected below, and
      # the next one reuses ``obstacles`` unshifted.
      offset = pts[0, :2].copy()
      pts[:, :2] -= offset
      shifted = obstacles.copy()
      shifted[shifted[:, 0] != OBSTACLE_KIND_NONE, 1:3] -= offset

      # ``pts`` is padded with the goal; measure straightness on the real path.
      num_real = int(np.argmax(np.all(pts[:, :2] == pts[-1, :2], axis=-1))) + 1
      xy = pts[:num_real, :2]
      is_straight = _max_lateral_deviation(xy) < cfg.straight_deviation
      result = _WorldResult(
        path=pts,
        waypoints=_even_waypoints(xy, num_waypoints),
        obstacles=shifted,
        is_straight=is_straight,
        forced=False,
      )
      if not is_straight or allow_straight:
        return result
      fallback = result

  if fallback is not None:
    fallback.forced = True
    return fallback
  raise RuntimeError(
    "Failed to generate an obstacle world path. The world is likely too "
    "cluttered for the requested start/goal distance: check world_size, "
    "density_range, inflation_radius and start_goal_distance_range."
  )


def _generate_chunk(args) -> list[tuple[int, _WorldResult]]:
  """Worker entry point: generate the worlds for a chunk of path indices."""
  cfg, indices, allow_straight, point_spacing, num_waypoints, max_points, seed = args
  out = []
  for index, allow in zip(indices, allow_straight, strict=True):
    # Seed per index, not per worker, so the library is identical regardless of
    # how the work is split across processes.
    rng = np.random.default_rng([seed, int(index)])
    out.append(
      (
        int(index),
        _generate_world(
          cfg, rng, point_spacing, num_waypoints, max_points, bool(allow)
        ),
      )
    )
  return out


def generate_astar_paths(
  num_paths: int,
  cfg: ObstacleWorldCfg | None = None,
  point_spacing: float = 0.01,
  num_waypoints: int = 4,
  max_points: int = 1500,
  seed: int = 0,
  device=None,
  progress: bool = False,
  num_workers: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Generate ``num_paths`` obstacle worlds with an A* path through each.

  Args:
    num_paths: Number of worlds (and paths) to generate.
    cfg: Obstacle world sampling/planning parameters.
    point_spacing: Arc-length spacing of the stored path samples, in meters.
    num_waypoints: Number of stored waypoints per path (last one is the goal).
    max_points: Fixed length of the stored path; longer paths are rejected.
    seed: Master seed. Generation is deterministic in (seed, path index).
    device: Torch device for the returned tensors.
    progress: Show a tqdm progress bar.
    num_workers: Processes to generate with. None picks a sensible default, 1
      runs serially.

  Returns:
    ``(paths, waypoints, obstacles)`` — see the module docstring for shapes.
  """
  assert device is not None, "Device must be specified for generate_astar_paths"
  cfg = cfg or ObstacleWorldCfg()

  # Which indices are *allowed* to end up straight. This caps the straight
  # fraction; it never forces a path to be straight.
  num_straight = int(round(cfg.max_straight_fraction * num_paths))
  allow_straight = np.zeros(num_paths, dtype=bool)
  allow_straight[np.random.default_rng(seed).permutation(num_paths)[:num_straight]] = (
    True
  )

  if num_workers is None:
    num_workers = max(1, min(os.cpu_count() or 1, 16))
  num_workers = max(1, min(num_workers, num_paths))

  chunks = np.array_split(np.arange(num_paths), num_workers * 8)
  chunks = [c for c in chunks if len(c) > 0]
  jobs = [
    (cfg, c, allow_straight[c], point_spacing, num_waypoints, max_points, seed)
    for c in chunks
  ]

  paths = np.zeros((num_paths, max_points, 6), dtype=np.float32)
  waypoints = np.zeros((num_paths, num_waypoints, 2), dtype=np.float32)
  obstacles = np.zeros((num_paths, cfg.max_obstacles, OBSTACLE_DIM), dtype=np.float32)
  num_forced = 0
  num_straight_actual = 0

  def _collect(results):
    nonlocal num_forced, num_straight_actual
    for index, world in results:
      paths[index] = world.path
      waypoints[index] = world.waypoints
      obstacles[index] = world.obstacles
      num_forced += world.forced
      num_straight_actual += world.is_straight

  if progress:
    from tqdm import tqdm

    bar = tqdm(total=num_paths, desc="Generating A* paths")
  else:
    bar = None

  if num_workers == 1:
    for job in jobs:
      _collect(_generate_chunk(job))
      if bar is not None:
        bar.update(len(job[1]))
  else:
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
      for results in pool.map(_generate_chunk, jobs):
        _collect(results)
        if bar is not None:
          bar.update(len(results))
  if bar is not None:
    bar.close()

  if num_forced:
    print(
      f"Warning: {num_forced}/{num_paths} paths are straight despite the "
      f"max_straight_fraction={cfg.max_straight_fraction} budget "
      "(no curved path found within the retry limits)."
    )
  if progress:
    print(
      f"Generated {num_paths} A* paths "
      f"({num_straight_actual / num_paths:.1%} straight)."
    )

  return (
    torch.from_numpy(paths).to(device),
    torch.from_numpy(waypoints).to(device),
    torch.from_numpy(obstacles).to(device),
  )


# Visualization: `uv run python -m mjlab_piplus.tasks.mdp.path_following.path_gen_astar`
if __name__ == "__main__":
  import matplotlib.pyplot as plt
  from matplotlib.patches import Circle, Rectangle

  cfg = ObstacleWorldCfg()
  num_show = 6
  fig, axes = plt.subplots(2, 3, figsize=(16, 11))
  for k, ax in enumerate(axes.flat):
    rng = np.random.default_rng([0, k])
    world = _generate_world(cfg, rng, 0.01, 4, 1500, allow_straight=False)
    for kind, x, y, yaw, a, b, _ in world.obstacles:
      if kind == OBSTACLE_KIND_NONE:
        continue
      if kind == OBSTACLE_KIND_CYLINDER:
        ax.add_patch(Circle((x, y), a, color="0.4"))
      else:
        rect = Rectangle(
          (x - a, y - b), 2 * a, 2 * b, color="0.4", rotation_point="center"
        )
        rect.set_angle(np.degrees(yaw))
        ax.add_patch(rect)
    ax.plot(world.path[:, 0], world.path[:, 1], "-", color="tab:blue", lw=2)
    ax.plot(world.waypoints[:, 0], world.waypoints[:, 1], "o", color="tab:green")
    ax.plot(*world.path[0, :2], "*", color="tab:orange", markersize=16)
    ax.set_title(f"world {k} ({'straight' if world.is_straight else 'curved'})")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
  fig.suptitle("A* paths through random obstacle worlds (start frame)")
  fig.tight_layout()
  fig.savefig("astar_paths.png", dpi=110)
  print("wrote astar_paths.png")
