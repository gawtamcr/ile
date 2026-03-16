import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from main import DiffeomorphicFlow
from temporal_main import generate_temporal_logical_trajectory, smooth_temporal_window

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)

# ── Run the model ─────────────────────────────────────────────────────────────
phi = DiffeomorphicFlow(layers=4)

start_state = [-2.0, 2.0]
target_latent = [2.0, -12.0]

# Using the [0.6, 0.8] window from the example execution
a, b = 0, 0.1
steps = 150

z_traj, x_traj = generate_temporal_logical_trajectory(
    phi, start_state, target_latent, a=a, b=b, steps=steps, alpha_lr=0.05
)

z_arr = np.array(z_traj)
x_arr = np.array(x_traj)
time_steps = np.linspace(0.0, 1.0, steps)

with torch.no_grad():
    x_star = phi.inverse(
        torch.tensor([target_latent], dtype=torch.float32)
    ).squeeze().numpy()
    z_start = phi(
        torch.tensor([start_state], dtype=torch.float32)
    ).squeeze().numpy()

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle(f"Bounded 'Eventually' F_[{a}, {b}]  —  Temporal Latent Evolution",
             fontsize=14, fontweight="bold", y=0.96)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

# ── Panel 1: Physical Space ───────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("Physical Space  X\nTrajectory over time")
sc1 = ax1.scatter(x_arr[:, 0], x_arr[:, 1], c=time_steps, cmap="viridis", s=15, zorder=3)
ax1.plot(*start_state, "g^", markersize=10, zorder=5, label="start (t=0)")
ax1.plot(*x_star, "rx", markersize=12, markeredgewidth=2, zorder=5, label="x* (target)")
ax1.set_xlabel("x₁"); ax1.set_ylabel("x₂")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)
fig.colorbar(sc1, ax=ax1, label="Normalized Time t")

# ── Panel 2: Latent Space ─────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title("Latent Space  Z\nTrajectory over time")
sc2 = ax2.scatter(z_arr[:, 0], z_arr[:, 1], c=time_steps, cmap="viridis", s=15, zorder=3)
ax2.plot(*z_start, "g^", markersize=10, zorder=5, label="start (t=0)")
ax2.plot(*target_latent, "rx", markersize=12, markeredgewidth=2, zorder=5, label="z* (target)")
ax2.set_xlabel("z₁"); ax2.set_ylabel("z₂")
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.3)
fig.colorbar(sc2, ax=ax2, label="Normalized Time t")

# ── Panel 3: Convergence ──────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_title("Distance to Target Over Time\n‖z(t) − z*‖")
dist_z = np.linalg.norm(z_arr - np.array(target_latent), axis=1)
ax3.plot(time_steps, dist_z, color="navy", linewidth=2)
ax3.axvspan(a, b, color="darkorange", alpha=0.2, label=f"Active Window [{a}, {b}]")
ax3.set_xlabel("Normalized Time t"); ax3.set_ylabel("Distance")
ax3.legend()
ax3.grid(True, alpha=0.3)

# ── Panel 4: Temporal Window ──────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_title("Smooth Temporal Window\nw(t)")
window_vals = [smooth_temporal_window(torch.tensor(t), a, b).item() for t in time_steps]
ax4.plot(time_steps, window_vals, color="darkorange", linewidth=2)
ax4.fill_between(time_steps, window_vals, color="darkorange", alpha=0.2)
ax4.axvline(a, color="gray", linestyle="--", alpha=0.7)
ax4.axvline(b, color="gray", linestyle="--", alpha=0.7)
ax4.set_xlabel("Normalized Time t"); ax4.set_ylabel("Weight")
ax4.grid(True, alpha=0.3)

plt.savefig("ile_temporal_viz.png", dpi=150, bbox_inches="tight")
print("Saved: ile_temporal_viz.png")
plt.show()