import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def generate_parametric_paths(num_paths,
                              max_point_dxy=0.5,
                              num_waypoints=6,
                              curvature_limit=50.0,
                              max_points=1500,
                              point_spacing=0.01,
                              device=None,
                              progress=False
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    assert device is not None, "Device must be specified for generate_parametric_paths"
    paths = torch.zeros((num_paths, max_points, 6), device=device) # x, y, dx, dy, ddx, ddy
    waypoints = torch.zeros((num_paths, num_waypoints, 2), device=device)
    np.random.seed(0)  # for reproducibility
    if progress:
        from tqdm import trange
        path_range = trange(num_paths, desc="Generating paths")
    else:
        path_range = range(num_paths)
    for i in path_range:
        pts, x_points, y_points = generate_parametric_path(
            max_point_dxy, num_waypoints, curvature_limit, point_spacing, max_points)
        paths[i] = torch.tensor(pts, device=device)
        waypoints[i] = torch.tensor(np.stack([x_points, y_points], axis=-1), device=device)
    return paths, waypoints

def generate_parametric_path(max_point_dxy=0.5,
                             num_waypoints=6,
                             curvature_limit=50.0,
                             point_spacing=0.01,
                             max_points=1500,
                             fine_resolution=2000):
    """Generate one random b-spline path sampled at a fixed euclidean spacing.

    Points are evenly spaced by ``point_spacing`` meters of arc length (rather
    than by a uniform spline parameter, which bunches points on tight curves and
    stretches them on straights). The resulting variable-length point list is
    padded with the final (goal) point up to ``max_points`` so every path is a
    fixed ``(max_points, 6)`` array; a path that needs more than ``max_points``
    points at this spacing is rejected and regenerated.
    """
    dxs = np.cumsum(np.random.uniform(-max_point_dxy,
                                      max_point_dxy,
                                      (num_waypoints-1,)))
    x_points = np.concatenate(([0.0], dxs))
    dys = np.cumsum(np.random.uniform(-max_point_dxy,
                                      max_point_dxy,
                                      (num_waypoints-1,)))
    y_points = np.concatenate(([0.0], dys))


    tck, _ = splprep([x_points, y_points], s=0)

    # Evaluate on a fine grid (uniform in the spline parameter) to measure arc
    # length and check curvature.
    u_fine = np.linspace(0, 1, fine_resolution)
    x_fine, y_fine = splev(u_fine, tck)

    # 3. Calculate Derivatives for Curvature
    dx, dy = splev(u_fine, tck, der=1)
    ddx, ddy = splev(u_fine, tck, der=2)

    curvature = np.abs(dx*ddy - dy*ddx) / (dx**2 + dy**2)**1.5

    # Check if the path is "too sharp"
    if np.max(curvature) > curvature_limit:
        # Try again (recursive) or smooth the points
        return generate_parametric_path(max_point_dxy, num_waypoints, curvature_limit,
                                        point_spacing, max_points, fine_resolution)

    # Cumulative arc length along the fine grid.
    seg = np.hypot(np.diff(x_fine), np.diff(y_fine))
    s_cum = np.concatenate(([0.0], np.cumsum(seg)))
    total_length = s_cum[-1]

    # Target arc-length positions, evenly spaced by ``point_spacing``. Always
    # include the exact endpoint so the final stored point is the true goal.
    target_s = np.arange(0.0, total_length, point_spacing)
    if target_s.size == 0 or target_s[-1] < total_length - 1e-9:
        target_s = np.append(target_s, total_length)
    n_points = target_s.size

    if n_points > max_points:
        # Path too long to fit at this spacing; regenerate a shorter one.
        return generate_parametric_path(max_point_dxy, num_waypoints, curvature_limit,
                                        point_spacing, max_points, fine_resolution)

    # Map each target arc length back to a spline parameter u, then evaluate the
    # spline (and its derivatives) there so all stored quantities are consistent
    # with the even euclidean spacing.
    u_target = np.interp(target_s, s_cum, u_fine)
    x_e, y_e = splev(u_target, tck)
    dx_e, dy_e = splev(u_target, tck, der=1)
    ddx_e, ddy_e = splev(u_target, tck, der=2)
    pts = np.stack([x_e, y_e, dx_e, dy_e, ddx_e, ddy_e], axis=-1)  # (n_points, 6)

    # Pad to fixed max_points by repeating the final (goal) point so the tensor
    # shape stays fixed and the last point remains the goal.
    if n_points < max_points:
        pad = np.repeat(pts[-1:], max_points - n_points, axis=0)
        pts = np.concatenate([pts, pad], axis=0)

    return pts, x_points, y_points

# Visualization
if __name__ == "__main__":
    plt.figure(figsize=(8, 8))
    for i in range(3):
        pts, x_points, y_points = generate_parametric_path()
        plt.plot(pts[:, 0], pts[:, 1], ".-", markersize=2, label=f"Path {i+1}")

    plt.title("Parametric Random Paths (Arc-Length-Spaced Samples)")
    plt.axis('equal')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("parametric_paths.png")
