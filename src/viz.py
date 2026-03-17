import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

def plot_full_analysis(phi, z_traj, x_traj, time_steps, start_state, reach_targets, avoid_targets, avoid_margin=1.0, figsize=(16, 10), grid_n=20, grid_limits=(-6, 6)):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.suptitle("STL Planning: Eventually Reach A [0,10] while Always Avoiding B [0,10]", 
                 fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # Precompute target preimages for physical space tracking
    with torch.no_grad():
        x_stars_reach = {name: phi.inverse(torch.tensor([pos], dtype=torch.float32)).squeeze().numpy() for name, pos in reach_targets.items()}
        x_stars_avoid = {name: phi.inverse(torch.tensor([pos], dtype=torch.float32)).squeeze().numpy() for name, pos in avoid_targets.items()}
        
    z_start = z_traj[0]

    # 1. Physical Trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Physical Space X Trajectory")
    sc1 = ax1.scatter(x_traj[:, 0], x_traj[:, 1], c=time_steps, cmap="plasma", s=20, zorder=3)
    ax1.plot(*start_state, "g^", markersize=10, zorder=5, label="Start")
    
    for name, pos in x_stars_reach.items():
        ax1.plot(*pos, "b*", markersize=12, zorder=5, label=f"Reach {name}")
    for name, pos in x_stars_avoid.items():
        ax1.plot(*pos, "rx", markersize=12, zorder=5, label=f"Avoid {name}")
        circle = plt.Circle((pos[0], pos[1]), np.sqrt(avoid_margin), color='r', fill=False, linestyle='--', alpha=0.5)
        ax1.add_patch(circle)
    
    ax1.set_xlabel("x1"); ax1.set_ylabel("x2")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    fig.colorbar(sc1, ax=ax1, label="Time t")

    # 2. Latent Trajectory
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Latent Space Z Trajectory")
    sc2 = ax2.scatter(z_traj[:, 0], z_traj[:, 1], c=time_steps, cmap="plasma", s=20, zorder=3)
    ax2.plot(*z_start, "g^", markersize=10, zorder=5, label="Start")
    
    for name, pos in reach_targets.items():
        ax2.plot(*pos, "b*", markersize=12, zorder=5, label=f"Reach {name}")
    for name, pos in avoid_targets.items():
        ax2.plot(*pos, "rx", markersize=12, zorder=5, label=f"Avoid {name}")
        circle = plt.Circle((pos[0], pos[1]), np.sqrt(avoid_margin), color='r', fill=False, linestyle='--', alpha=0.8)
        ax2.add_patch(circle)

    ax2.set_xlabel("z1"); ax2.set_ylabel("z2")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.colorbar(sc2, ax=ax2, label="Time t")

    # 3. Diffeomorphism Grid Warp
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("Diffeomorphism Warp (X -> Z)")
    N = grid_n
    xs = ys = np.linspace(grid_limits[0], grid_limits[1], N)
    Xg, Yg = np.meshgrid(xs, ys)
    grid_pts = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], axis=1), dtype=torch.float32)

    with torch.no_grad():
        z_grid = phi(grid_pts).numpy()
    Zx = z_grid[:, 0].reshape(N, N)
    Zy = z_grid[:, 1].reshape(N, N)

    for i in range(N):
        ax3.plot(Xg[i, :], Yg[i, :], color="steelblue", alpha=0.2, linewidth=0.8)
        ax3.plot(Xg[:, i], Yg[:, i], color="steelblue", alpha=0.2, linewidth=0.8)
        ax3.plot(Zx[i, :], Zy[i, :], color="darkorange", alpha=0.5, linewidth=0.8)
        ax3.plot(Zx[:, i], Zy[:, i], color="darkorange", alpha=0.5, linewidth=0.8)
        
    ax3.set_xlabel("dim 1"); ax3.set_ylabel("dim 2")
    ax3.grid(True, alpha=0.2)
    custom_lines = [Line2D([0], [0], color="steelblue", lw=2, alpha=0.5), Line2D([0], [0], color="darkorange", lw=2, alpha=0.5)]
    ax3.legend(custom_lines, ['Physical Grid (X)', 'Warped Grid (Z)'])

    # 4. Convergence / Distance over Time
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("Distance to Targets over Time (Latent Space)")
    
    for name, pos in reach_targets.items():
        ax4.plot(time_steps, np.linalg.norm(z_traj - np.array(pos), axis=1), linewidth=2, label=f"||z(t) - z_{name}||")
    for name, pos in avoid_targets.items():
        ax4.plot(time_steps, np.linalg.norm(z_traj - np.array(pos), axis=1), linewidth=2, linestyle=':', label=f"||z(t) - z_{name}||")
        
    ax4.axhline(np.sqrt(avoid_margin), color="red", linestyle="--", alpha=0.5, label="Avoid Margin Limit")
    
    ax4.set_xlabel("Time t"); ax4.set_ylabel("Distance")
    ax4.legend(); ax4.grid(True, alpha=0.3)

    plt.savefig("stl_trajectory_analysis.png", dpi=150)
    print("Saved: stl_trajectory_analysis.png")
    plt.show()