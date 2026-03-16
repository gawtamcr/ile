"""
Combined ILE Visualization
==========================
Demonstrates all three STL operators implemented across main.py + stl_operators.py:
  - Eventually  (single latent attractor)
  - OR          (nearest of two attractors wins)
  - AND         (must satisfy both attractors -- settles at constrained minimum)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

from main          import DiffeomorphicFlow, generate_logical_trajectory
from stl_operators import generate_compositional_trajectory, smooth_max, smooth_min

# ── Shared setup ──────────────────────────────────────────────────────────────
torch.manual_seed(42)
phi = DiffeomorphicFlow(layers=4)

START    = [-4.0,  4.0]
Z_SINGLE = [ 0.0,  0.0]   # Eventually target
Z_A      = [ 2.0,  2.0]   # Compositional target A
Z_B      = [-2.0, -2.0]   # Compositional target B
STEPS    = 150
ALPHA    = 0.05
BETA     = 10.0

# ── Run all three trajectories ────────────────────────────────────────────────
z_ev,  x_ev  = generate_logical_trajectory(
    phi, START, Z_SINGLE, steps=STEPS, alpha=ALPHA)

z_or,  x_or  = generate_compositional_trajectory(
    phi, START, Z_A, Z_B, operator="OR",  steps=STEPS, alpha=ALPHA, beta=BETA)

z_and, x_and = generate_compositional_trajectory(
    phi, START, Z_A, Z_B, operator="AND", steps=STEPS, alpha=ALPHA, beta=BETA)

z_ev  = np.array(z_ev);  x_ev  = np.array(x_ev)
z_or  = np.array(z_or);  x_or  = np.array(x_or)
z_and = np.array(z_and); x_and = np.array(x_and)

# ── Potential landscapes in Z ─────────────────────────────────────────────────
fine = 300
lim  = 5.5
ls   = np.linspace(-lim, lim, fine)
Zgx, Zgy = np.meshgrid(ls, ls)
Zg = torch.tensor(np.stack([Zgx.ravel(), Zgy.ravel()], 1), dtype=torch.float32)

zA_t = torch.tensor([Z_A],      dtype=torch.float32)
zB_t = torch.tensor([Z_B],      dtype=torch.float32)
zS_t = torch.tensor([Z_SINGLE], dtype=torch.float32)

with torch.no_grad():
    u_single = torch.norm(Zg - zS_t, dim=1) ** 2
    u_A      = torch.norm(Zg - zA_t, dim=1) ** 2
    u_B      = torch.norm(Zg - zB_t, dim=1) ** 2
    U_ev     = u_single.numpy().reshape(fine, fine)
    U_or     = smooth_min(u_A, u_B, BETA).numpy().reshape(fine, fine)
    U_and    = smooth_max(u_A, u_B, BETA).numpy().reshape(fine, fine)

# ── Preimage of targets in X via φ⁻¹ ─────────────────────────────────────────
with torch.no_grad():
    x_star_ev = phi.inverse(torch.tensor([Z_SINGLE], dtype=torch.float32)).squeeze().numpy()
    x_star_A  = phi.inverse(torch.tensor([Z_A],      dtype=torch.float32)).squeeze().numpy()
    x_star_B  = phi.inverse(torch.tensor([Z_B],      dtype=torch.float32)).squeeze().numpy()

# ── Helper: colour trajectory segments by time using a shared colormap ────────
TRAJ_CMAP = "plasma"

def plot_traj(ax, traj, lw=1.5, zorder=4):
    T = len(traj) - 1
    cmap = plt.get_cmap(TRAJ_CMAP)
    for i in range(T):
        ax.plot(traj[i:i+2, 0], traj[i:i+2, 1],
                color=cmap(i / T), linewidth=lw, zorder=zorder)

def mark_start_end(ax, traj):
    ax.plot(*traj[0],  "g^", ms=9,  zorder=6, label="start")
    ax.plot(*traj[-1], "r*", ms=11, zorder=6, label="end")

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor("#0f0f1a")
fig.suptitle(
    "ILE  --  Invertible Latent Evolution\n"
    "Compositional STL Trajectory Planning via Diffeomorphic Flows",
    fontsize=15, fontweight="bold", color="white", y=0.99,
)

gs = gridspec.GridSpec(3, 4, figure=fig,
                       hspace=0.45, wspace=0.35,
                       left=0.06, right=0.95, top=0.93, bottom=0.06)

operator_labels = ["Eventually  phi(x*)=z*", "OR  (nearest target)", "AND  (both targets)"]
bg_cmaps        = ["Blues_r", "Purples_r", "Greens_r"]
latent_Us       = [U_ev,  U_or,  U_and]
z_trajs         = [z_ev,  z_or,  z_and]
x_trajs         = [x_ev,  x_or,  x_and]

# ── Rows 0-1: Latent and Physical panels for each operator ────────────────────
for col, (label, U, zt, xt, bg_cm) in enumerate(
        zip(operator_labels, latent_Us, z_trajs, x_trajs, bg_cmaps)):

    # --- Latent space ---
    ax_z = fig.add_subplot(gs[0, col])
    ax_z.set_facecolor("#0f0f1a")
    ax_z.contourf(Zgx, Zgy, U, levels=40, cmap=bg_cm, alpha=0.85)
    ax_z.contour( Zgx, Zgy, U, levels=12, colors="white", linewidths=0.3, alpha=0.4)
    plot_traj(ax_z, zt)
    mark_start_end(ax_z, zt)
    if col == 0:
        ax_z.plot(*Z_SINGLE, "wx", ms=12, markeredgewidth=2.5, zorder=7, label="z*")
    else:
        ax_z.plot(*Z_A, "w+", ms=12, markeredgewidth=2.5, zorder=7, label="z_A")
        ax_z.plot(*Z_B, "wx", ms=12, markeredgewidth=2.5, zorder=7, label="z_B")
    ax_z.set_title(f"Latent  Z\n{label}", color="white", fontsize=10)
    ax_z.set_xlabel("z1", color="#aaa"); ax_z.set_ylabel("z2", color="#aaa")
    ax_z.tick_params(colors="#888")
    ax_z.set_xlim(-lim, lim); ax_z.set_ylim(-lim, lim)
    for spine in ax_z.spines.values(): spine.set_edgecolor("#444")
    ax_z.legend(fontsize=7, loc="upper right",
                facecolor="#111", labelcolor="white", edgecolor="#444")

    # --- Physical space ---
    ax_x = fig.add_subplot(gs[1, col])
    ax_x.set_facecolor("#0f0f1a")
    plot_traj(ax_x, xt)
    mark_start_end(ax_x, xt)
    if col == 0:
        ax_x.plot(*x_star_ev, "wx", ms=12, markeredgewidth=2.5, zorder=7,
                  label="x*=phi^-1(z*)")
    else:
        ax_x.plot(*x_star_A, "w+", ms=12, markeredgewidth=2.5, zorder=7, label="phi^-1(z_A)")
        ax_x.plot(*x_star_B, "wx", ms=12, markeredgewidth=2.5, zorder=7, label="phi^-1(z_B)")
    ax_x.set_title("Physical  X\nPullback  x(t) = phi^-1(z(t))", color="white", fontsize=10)
    ax_x.set_xlabel("x1", color="#aaa"); ax_x.set_ylabel("x2", color="#aaa")
    ax_x.tick_params(colors="#888")
    ax_x.grid(True, alpha=0.15, color="#555")
    for spine in ax_x.spines.values(): spine.set_edgecolor("#444")
    ax_x.legend(fontsize=7, loc="upper right",
                facecolor="#111", labelcolor="white", edgecolor="#444")

# ── Row 2 left: Convergence comparison ───────────────────────────────────────
ax_conv = fig.add_subplot(gs[2, :2])
ax_conv.set_facecolor("#0f0f1a")

steps_arr = np.arange(STEPS + 1)

d_ev    = np.linalg.norm(z_ev  - np.array(Z_SINGLE), axis=1)
d_or    = np.minimum(
    np.linalg.norm(z_or  - np.array(Z_A), axis=1),
    np.linalg.norm(z_or  - np.array(Z_B), axis=1),
)
d_and_A = np.linalg.norm(z_and - np.array(Z_A), axis=1)
d_and_B = np.linalg.norm(z_and - np.array(Z_B), axis=1)

ax_conv.plot(steps_arr, d_ev,    color="#00bfff", lw=2,   label="Eventually  ||z-z*||")
ax_conv.plot(steps_arr, d_or,    color="#ff69b4", lw=2,   label="OR  min(||z-zA||, ||z-zB||)")
ax_conv.plot(steps_arr, d_and_A, color="#7fff00", lw=1.5, ls="--", label="AND  ||z-zA||")
ax_conv.plot(steps_arr, d_and_B, color="#adff2f", lw=1.5, ls=":",  label="AND  ||z-zB||")

ax_conv.set_title("Convergence in Latent Space  Z", color="white", fontsize=11)
ax_conv.set_xlabel("Gradient step t", color="#aaa")
ax_conv.set_ylabel("Distance to target(s)", color="#aaa")
ax_conv.tick_params(colors="#888")
ax_conv.grid(True, alpha=0.2, color="#555")
for spine in ax_conv.spines.values(): spine.set_edgecolor("#444")
ax_conv.legend(fontsize=8, facecolor="#111", labelcolor="white", edgecolor="#444")

# ── Row 2 right: Operator semantics reference ─────────────────────────────────
ax_diag = fig.add_subplot(gs[2, 2:])
ax_diag.set_facecolor("#0f0f1a")
ax_diag.axis("off")

diagram_text = (
    "ILE Operator Semantics\n"
    "--------------------------------------------------\n\n"
    "  Atomic predicate:   u_i(z) = ||z - z_i||^2\n\n"
    "  Eventually:         U(z) = u_s(z)\n"
    "    -> Gradient pulls z to single basin z*\n\n"
    "  phi |= psi_A  OR  psi_B:\n"
    "    U(z) = smooth_min(u_A, u_B)\n"
    "    -> Converges to whichever target is nearest\n\n"
    "  phi |= psi_A  AND  psi_B:\n"
    "    U(z) = smooth_max(u_A, u_B)\n"
    "    -> Must reduce the larger residual first\n"
    "    -> Settles at minimax point between targets\n\n"
    "  Pullback guarantee:  x(t) = phi^-1(z(t))\n"
    "    phi diffeomorphic => physical trajectory\n"
    "    inherits convergence & smoothness from Z"
)

ax_diag.text(0.03, 0.97, diagram_text,
             transform=ax_diag.transAxes,
             fontsize=8.5, color="#d0d0d0",
             verticalalignment="top",
             fontfamily="monospace",
             bbox=dict(facecolor="#161628", edgecolor="#444", boxstyle="round,pad=0.5"))

# ── Colorbar: trajectory time (matches plasma colormap used in plot_traj) ─────
sm = plt.cm.ScalarMappable(cmap=TRAJ_CMAP, norm=Normalize(0, STEPS))
sm.set_array([])
cbar = fig.colorbar(sm, ax=[fig.axes[i] for i in range(6)],
                    orientation="vertical", shrink=0.6, pad=0.02, aspect=30)
cbar.set_label("Time step  t", color="white", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

plt.savefig("ile_combined.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved: ile_combined.png")
plt.show()
