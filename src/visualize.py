import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from main import DiffeomorphicFlow, generate_trajectory

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)

# ── Run the model ─────────────────────────────────────────────────────────────
phi = DiffeomorphicFlow(layers=4)

start_state        = [-4.0,  4.0]
target_latent      = [ 0.0,  0.0]

z_target_t = torch.tensor([target_latent], dtype=torch.float32)
loss_fn    = lambda z: torch.norm(z - z_target_t) ** 2
z_traj, x_traj = generate_trajectory(
    phi, start_state, loss_fn, steps=100, alpha=0.05
)

z_arr = np.array(z_traj)   # (101, 2)
x_arr = np.array(x_traj)   # (101, 2)

# ── Build a dense grid to show the flow field ─────────────────────────────────
grid_range = 6
N = 20
xs = np.linspace(-grid_range, grid_range, N)
ys = np.linspace(-grid_range, grid_range, N)
Xg, Yg = np.meshgrid(xs, ys)
grid_pts = torch.tensor(
    np.stack([Xg.ravel(), Yg.ravel()], axis=1), dtype=torch.float32
)

with torch.no_grad():
    z_grid = phi(grid_pts).numpy()           # forward:  X → Z
    x_back = phi.inverse(                    # inverse:  Z → X  (identity check)
        phi(grid_pts)
    ).numpy()

Zx = z_grid[:, 0].reshape(N, N)
Zy = z_grid[:, 1].reshape(N, N)

# ── Latent potential U(z) = ||z - z*||² over a fine grid ─────────────────────
fine = 200
zs_lin = np.linspace(-6, 6, fine)
Zfx, Zfy = np.meshgrid(zs_lin, zs_lin)
U = Zfx**2 + Zfy**2   # target is origin

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle("ILE Proof-of-Concept  —  Diffeomorphic Flow & Latent Gradient Descent",
             fontsize=14, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# ── Panel 1: Physical space trajectory ───────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("Physical Space  X\nTrajectory via φ⁻¹(z(t))")
ax1.plot(x_arr[:, 0], x_arr[:, 1], "o-", color="steelblue",
         markersize=3, linewidth=1.5, alpha=0.8)
ax1.plot(*x_arr[0],  "g^", markersize=10, zorder=5, label="start")
ax1.plot(*x_arr[-1], "r*", markersize=12, zorder=5, label="end")
ax1.set_xlabel("x₁"); ax1.set_ylabel("x₂")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

# ── Panel 2: Latent space trajectory + potential ──────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title("Latent Space  Z\nGradient descent on U(z) = ‖z−z*‖²")
cf = ax2.contourf(Zfx, Zfy, U, levels=30, cmap="YlOrRd", alpha=0.6)
fig.colorbar(cf, ax=ax2, shrink=0.8, label="U(z)")
ax2.plot(z_arr[:, 0], z_arr[:, 1], "o-", color="navy",
         markersize=3, linewidth=1.5, alpha=0.9)
ax2.plot(*z_arr[0],  "g^", markersize=10, zorder=5, label="start")
ax2.plot(*z_arr[-1], "r*", markersize=12, zorder=5, label="end")
ax2.plot(*target_latent, "rx", markersize=14, markeredgewidth=3, zorder=6, label="z*")
ax2.set_xlabel("z₁"); ax2.set_ylabel("z₂")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.2)

# ── Panel 3: Diffeomorphic grid warp (X → Z) ──────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_title("Diffeomorphism φ: X → Z\nGrid warp (blue=X, orange=Z)")
for i in range(N):
    ax3.plot(Xg[i, :], Yg[i, :], color="steelblue", alpha=0.4, linewidth=0.8)
    ax3.plot(Xg[:, i], Yg[:, i], color="steelblue", alpha=0.4, linewidth=0.8)
    ax3.plot(Zx[i, :], Zy[i, :], color="darkorange", alpha=0.6, linewidth=0.8)
    ax3.plot(Zx[:, i], Zy[:, i], color="darkorange", alpha=0.6, linewidth=0.8)
ax3.set_xlabel("dim 1"); ax3.set_ylabel("dim 2")
ax3.set_xlim(-8, 8); ax3.set_ylim(-8, 8)
ax3.grid(True, alpha=0.2)

# ── Panel 4: Distance to target over time (latent) ───────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_title("Convergence in Latent Space\n‖z(t) − z*‖")
z_star = np.array(target_latent)
dist_z = np.linalg.norm(z_arr - z_star, axis=1)
ax4.plot(dist_z, color="navy", linewidth=2)
ax4.set_xlabel("Step t"); ax4.set_ylabel("‖z(t) − z*‖")
ax4.grid(True, alpha=0.3)

# ── Panel 5: Distance to target over time (physical) ─────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_title("Convergence in Physical Space\n‖x(t) − x*‖")
# x* is the preimage of z* under φ⁻¹
with torch.no_grad():
    x_star = phi.inverse(
        torch.tensor([target_latent], dtype=torch.float32)
    ).squeeze().numpy()
dist_x = np.linalg.norm(x_arr - x_star, axis=1)
ax5.plot(dist_x, color="steelblue", linewidth=2)
ax5.set_xlabel("Step t"); ax5.set_ylabel("‖x(t) − x*‖")
ax5.grid(True, alpha=0.3)
ax5.annotate(f"x* ≈ ({x_star[0]:.2f}, {x_star[1]:.2f})",
             xy=(0.05, 0.85), xycoords="axes fraction", fontsize=8)

# ── Panel 6: Per-coordinate evolution ────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_title("Per-Coordinate Evolution\nin Physical Space")
ax6.plot(x_arr[:, 0], label="x₁(t)", color="steelblue", linewidth=2)
ax6.plot(x_arr[:, 1], label="x₂(t)", color="darkorange", linewidth=2)
ax6.axhline(x_star[0], linestyle="--", color="steelblue", alpha=0.5, label=f"x₁*={x_star[0]:.2f}")
ax6.axhline(x_star[1], linestyle="--", color="darkorange", alpha=0.5, label=f"x₂*={x_star[1]:.2f}")
ax6.set_xlabel("Step t"); ax6.set_ylabel("value")
ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)

plt.savefig("ile_visualization.png", dpi=150, bbox_inches="tight")
print("Saved: ile_visualization.png")
plt.show()
