import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def generate_parametric_paths(num_paths,
                              max_point_dxy=0.5,
                              num_waypoints=6,
                              curvature_limit=50.0,
                              resolution=1000,
                              device=None,
                              progress=False
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    assert device is not None, "Device must be specified for generate_parametric_paths"
    paths = torch.zeros((num_paths, resolution, 6), device=device) # x, y, dx, dy, ddx, ddy
    waypoints = torch.zeros((num_paths, num_waypoints, 2), device=device)
    np.random.seed(0)  # for reproducibility
    if progress:
        from tqdm import trange
        path_range = trange(num_paths, desc="Generating paths")
    else:
        path_range = range(num_paths)
    for i in path_range:
        x_fine, y_fine, dx, dy, ddx, ddy, curv, x_points, y_points = generate_parametric_path(max_point_dxy, num_waypoints, curvature_limit, resolution)
        paths[i] = torch.tensor(np.stack([x_fine, y_fine, dx, dy, ddx, ddy], axis=-1), device=device)
        waypoints[i] = torch.tensor(np.stack([x_points, y_points], axis=-1), device=device)
    return paths, waypoints

def generate_parametric_path(max_point_dxy=0.5,
                             num_waypoints=6,
                             curvature_limit=50.0,
                             resolution=1000):
    dxs = np.cumsum(np.random.uniform(-max_point_dxy,
                                      max_point_dxy,
                                      (num_waypoints-1,)))
    x_points = np.concatenate(([0.0], dxs))
    dys = np.cumsum(np.random.uniform(-max_point_dxy,
                                      max_point_dxy,
                                      (num_waypoints-1,)))
    y_points = np.concatenate(([0.0], dys))
    
    
    tck, _ = splprep([x_points, y_points], s=0)
    
    # Evaluate on a fine grid
    u_fine = np.linspace(0, 1, resolution)
    x_fine, y_fine = splev(u_fine, tck)
    
    # 3. Calculate Derivatives for Curvature
    dx, dy = splev(u_fine, tck, der=1)
    ddx, ddy = splev(u_fine, tck, der=2)
    
    curvature = np.abs(dx*ddy - dy*ddx) / (dx**2 + dy**2)**1.5
    
    # Check if the path is "too sharp"
    if np.max(curvature) > curvature_limit:
        # Try again (recursive) or smooth the points
        return generate_parametric_path(max_point_dxy, num_waypoints, curvature_limit, resolution)
        
    return x_fine, y_fine, dx, dy, ddx, ddy, curvature, x_points, y_points

# Visualization
if __name__ == "__main__":
    plt.figure(figsize=(8, 8))
    for i in range(3):
        xf, yf, dx, dy, ddx, ddy, curv, x_points, y_points = generate_parametric_path()
        plt.plot(xf, yf, label=f"Path {i+1} (Max $\\kappa$: {np.max(curv):.3f})")

    plt.title("Parametric Random Paths (Curvature Constrained)")
    plt.axis('equal')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("parametric_paths.png")