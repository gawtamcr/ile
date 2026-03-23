import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

def plot_manifold_level_sets(ax, phi, formula, space='Z', t_eval=0.0, grid_limits=(-6, 6), grid_n=100):
    """
    Evaluates and plots the zero-level sets of the composed logical manifolds.
    space: 'Z' for latent manifolds, 'X' for physical pullback manifolds.
    """
    xs = np.linspace(grid_limits[0], grid_limits[1], grid_n)
    ys = np.linspace(grid_limits[0], grid_limits[1], grid_n)
    Xg, Yg = np.meshgrid(xs, ys)
    
    # Flatten the grid for batched PyTorch evaluation
    grid_pts = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], axis=1), dtype=torch.float32)
    t_tensor = torch.full((grid_pts.shape[0],), t_eval, dtype=torch.float32)

    with torch.no_grad():
        if space == 'Z':
            # Evaluate logic directly on the latent grid
            z_pts = grid_pts
        elif space == 'X':
            # Evaluate logic by pulling the physical grid through the diffeomorphism
            z_pts = phi(grid_pts)
        else:
            raise ValueError("Space must be 'Z' or 'X'")
            
        # Compute the global composed barrier function
        h_vals = formula.compute_h(z_pts, t_tensor)
        
        # Compute individual predicate manifolds if composed
        child_h_vals = []
        if hasattr(formula, 'children'):
            child_h_vals = [child.compute_h(z_pts, t_tensor) for child in formula.children]
        
    H_grid = h_vals.numpy().reshape(grid_n, grid_n)

    # Plot the strict zero-level set boundary (h = 0)
    # Using a slight epsilon in levels to avoid numerical artifacting exactly at 0
    contour = ax.contour(Xg, Yg, H_grid, levels=[0.01], colors='magenta', linewidths=2.0, zorder=2, linestyles='dashed')
    
    # Fill the valid region (h > 0) to highlight the safe manifold
    ax.contourf(Xg, Yg, H_grid, levels=[0.01, np.inf], colors='magenta', alpha=0.1, zorder=1)
    
    # Plot individual predicate manifolds
    for child_h in child_h_vals:
        child_H_grid = child_h.numpy().reshape(grid_n, grid_n)
        ax.contour(Xg, Yg, child_H_grid, levels=[0.01], colors='gray', linewidths=1.0, zorder=1, linestyles='dotted', alpha=0.8)
    
    return contour

def plot_full_analysis(phi, z_traj, x_traj, time_steps, start_state, reach_targets, avoid_targets, formula, avoid_margin=1.0, figsize=(16, 12), grid_n=20, grid_limits=(-6, 6)):
    """
    Orchestrates the complete visual proof of the ILE framework.
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.suptitle("Isomorphic Latent Execution: Geometric Composition and Diffeomorphic Pullback", 
                 fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # 1. Forward map the physical targets to latent coordinates for Z-space tracking
    with torch.no_grad():
        z_stars_reach = {name: phi(torch.tensor([pos], dtype=torch.float32)).squeeze().numpy() for name, pos in reach_targets.items()}
        z_stars_avoid = {name: phi(torch.tensor([pos], dtype=torch.float32)).squeeze().numpy() for name, pos in avoid_targets.items()}
        
    z_start = z_traj[0]

    # 1. Physical Space X Trajectory and Manifold Pullback
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Physical Space $\mathcal{X}$: Diffeomorphic Pullback")
    
    plot_manifold_level_sets(ax1, phi, formula, space='X', t_eval=time_steps[0], grid_limits=grid_limits, grid_n=100)
    sc1 = ax1.scatter(x_traj[:, 0], x_traj[:, 1], c=time_steps, cmap="viridis", s=15, zorder=4)
    ax1.plot(*start_state, "g^", markersize=10, zorder=5, label="Start")
    
    # Plot raw config targets directly in physical space
    for name, pos in reach_targets.items():
        ax1.plot(*pos, "b*", markersize=12, zorder=5, label=f"Reach {name}")
    for name, pos in avoid_targets.items():
        ax1.plot(*pos, "rx", markersize=12, zorder=5, label=f"Avoid {name}")
    
    ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$")
    ax1.legend(loc="upper left"); ax1.grid(True, alpha=0.3)
    fig.colorbar(sc1, ax=ax1, label="Time $t$")

    # 2. Latent Space Z Trajectory and Native Manifold
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Latent Space $\mathcal{Z}$: Native Geometric Logic")
    
    plot_manifold_level_sets(ax2, phi, formula, space='Z', t_eval=time_steps[0], grid_limits=grid_limits, grid_n=100)
    sc2 = ax2.scatter(z_traj[:, 0], z_traj[:, 1], c=time_steps, cmap="viridis", s=15, zorder=4)
    ax2.plot(*z_start, "g^", markersize=10, zorder=5, label="Start")
    
    # Plot forward-mapped targets in latent space
    for name, pos in z_stars_reach.items():
        ax2.plot(*pos, "b*", markersize=12, zorder=5, label=f"Reach {name}")
    for name, pos in z_stars_avoid.items():
        ax2.plot(*pos, "rx", markersize=12, zorder=5, label=f"Avoid {name}")

    ax2.set_xlabel("$z_1$"); ax2.set_ylabel("$z_2$")
    ax2.legend(loc="upper left"); ax2.grid(True, alpha=0.3)
    fig.colorbar(sc2, ax=ax2, label="Time $t$")
    # 3. Diffeomorphism Grid Warp
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("Topological Substrate: $\phi: \mathcal{X} \\to \mathcal{Z}$")
    N = grid_n
    xs = ys = np.linspace(grid_limits[0], grid_limits[1], N)
    Xg, Yg = np.meshgrid(xs, ys)
    grid_pts = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], axis=1), dtype=torch.float32)

    with torch.no_grad():
        z_grid = phi(grid_pts).numpy()
    Zx = z_grid[:, 0].reshape(N, N)
    Zy = z_grid[:, 1].reshape(N, N)

    for i in range(N):
        ax3.plot(Xg[i, :], Yg[i, :], color="steelblue", alpha=0.3, linewidth=0.8)
        ax3.plot(Xg[:, i], Yg[:, i], color="steelblue", alpha=0.3, linewidth=0.8)
        ax3.plot(Zx[i, :], Zy[i, :], color="darkorange", alpha=0.6, linewidth=0.8)
        ax3.plot(Zx[:, i], Zy[:, i], color="darkorange", alpha=0.6, linewidth=0.8)
        
    ax3.set_xlabel("Dimension 1"); ax3.set_ylabel("Dimension 2")
    ax3.grid(True, alpha=0.2)
    custom_lines = [Line2D([0], [0], color="steelblue", lw=2, alpha=0.6), 
                    Line2D([0], [0], color="darkorange", lw=2, alpha=0.8)]
    ax3.legend(custom_lines, ['Physical Grid ($\mathcal{X}$)', 'Warped Grid ($\mathcal{Z}$)'], loc="upper left")

    # 4. Temporal Distance and CBF Compliance
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("Temporal Compliance in Latent Space")
    
    for name, pos in reach_targets.items():
        distances = np.linalg.norm(z_traj - np.array(pos), axis=1)
        ax4.plot(time_steps, distances, linewidth=2, label=f"Distance to {name}")
    for name, pos in avoid_targets.items():
        distances = np.linalg.norm(z_traj - np.array(pos), axis=1)
        ax4.plot(time_steps, distances, linewidth=2, linestyle=':', label=f"Distance to {name}")
        
    ax4.axhline(np.sqrt(avoid_margin), color="red", linestyle="--", alpha=0.7, label="Avoid Barrier $\partial\mathcal{M}$")
    
    ax4.set_xlabel("Time $t$"); ax4.set_ylabel("Euclidean Distance $\|z(t) - z_{target}\|$")
    ax4.legend(loc="upper right"); ax4.grid(True, alpha=0.3)

    plt.savefig("ile_geometric_proof.png", dpi=300, bbox_inches='tight')
    print("Saved publication figure: ile_geometric_proof.png")
    plt.show()