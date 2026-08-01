"""
Generate a FDFO-style Figure 1 for our CARLA QGF setup using REAL experiment data.

Panel (a): Unguided pi0 denoising — x_t_bc trajectories from traj_capture files,
           projected to 2D via PCA of the 40-D action space.
Panel (b): QGF-guided denoising — x_t_after trajectories (lambda=0.1),
           with real Q-value contours from the pretrained critic.
"""

import os, sys, pickle
import numpy as np
import jax, jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
CAPTURE_BASE = (
    "/home/celinet/carla_exps/OGBench-CARLA/qgf_sweep_20260625_165348"
)
_D = CAPTURE_BASE

# Use w=0.1 captures for the most dramatic guidance effect
_W1_DIR = os.path.join(_D, "sd000_20260625_165521_merger-into-slow-traffic-v2-005")
W1_FILES = sorted([
    os.path.join(_W1_DIR, f) for f in os.listdir(_W1_DIR)
    if f.startswith("traj_capture_w0.1_") and f.endswith(".pkl")
]) if os.path.isdir(_W1_DIR) else []

CRITIC_CKPT = (
    "/scratch/current/celinet/critic_pretrain/"
    "run_20260625_032253/step_0004000.pkl"
)

# --------------------------------------------------------------------------- #
# Colour palette
# --------------------------------------------------------------------------- #
BG    = "#f7f9fc"
WHITE = "#ffffff"
DARK  = "#1a1a2e"
MID   = "#888888"
GREY_BG = "#ebebeb"
BLUE_BG = "#ddeeff"
TEAL  = "#00bfae"
BLUE  = "#2979ff"

# --------------------------------------------------------------------------- #
# Load captures
# --------------------------------------------------------------------------- #
def load_captures(files):
    steps = []
    for fpath in files:
        with open(fpath, "rb") as fh:
            d = pickle.load(fh)
        steps.extend(d["steps"])
    return steps


steps = load_captures(W1_FILES)
print(f"Loaded {len(steps)} env steps from w=0.1 captures")

# Collect all x_t_bc vectors for PCA (shape: n_env_steps * 10, 40)
all_xbc = np.array([
    di["x_t_bc"]
    for s in steps
    for di in s["denoising"]
])
print(f"  all_xbc shape: {all_xbc.shape}")

# Fit PCA — 2 components capturing most denoising variance
pca = PCA(n_components=2, random_state=42)
pca.fit(all_xbc)
print(f"  PCA explained variance ratio: {pca.explained_variance_ratio_}")


def proj(x40):
    """Project (N, 40) or (40,) array to 2D PCA space."""
    x = np.atleast_2d(x40)
    return pca.transform(x)


def proj_traj(traj_list):
    """Project list of (40,) arrays to 2D. Returns (T, 2)."""
    return proj(np.array(traj_list))


# --------------------------------------------------------------------------- #
# Pick interesting env steps: most divergence between BC and guided in 2D space
# --------------------------------------------------------------------------- #
def traj_divergence_2d(env_step_dict):
    """Mean L2 distance between BC and guided trajectory in PCA 2D space."""
    bc  = proj_traj([di["x_t_bc"]    for di in env_step_dict["denoising"]])
    gd  = proj_traj([di["x_t_after"] for di in env_step_dict["denoising"]])
    return float(np.mean(np.linalg.norm(bc - gd, axis=1)))


divs     = np.array([traj_divergence_2d(s) for s in steps])
q_bc_fin = np.array([s["denoising"][-1]["q_bc"] for s in steps])

# Pick 4 steps with largest PCA-space divergence, spread across Q range for variety
sel_idx = np.argsort(divs)[-4:]
print(f"Selected env steps {sel_idx} (by max BC-vs-guided divergence in 2D):")
for i in sel_idx:
    print(f"  step {steps[i]['step']:4d}: divergence={divs[i]:.3f}  "
          f"q_bc_final={q_bc_fin[i]:.1f}")


# --------------------------------------------------------------------------- #
# Load pretrained critic for real Q contours
# --------------------------------------------------------------------------- #
from jax_agents.dsrl import Critic   # noqa: E402

with open(CRITIC_CKPT, "rb") as fh:
    ckpt = pickle.load(fh)
critic_params = ckpt["params"]["modules_critic"]
critic_def    = Critic(hidden_dims=(256, 256), layer_norm=True, ensemble_size=2)

@jax.jit
def q_ensemble_min(obs_enc_batch, actions_phys_batch):
    qs = critic_def.apply(
        {"params": critic_params}, obs_enc_batch, actions_phys_batch
    )
    return jnp.min(qs, axis=0)          # (B,)


XY_SCALE = 7.0                           # model → physical for first 2 action dims


def model40_to_phys40(a40):
    """(N, 40) model actions → physical (× 7 for XY dims)."""
    a = a40.reshape(-1, 10, 4).copy()
    a[:, :, :2] *= XY_SCALE
    return a.reshape(-1, 40)


# Use obs_enc from the highest-divergence step as contour reference — that's
# the most interesting state to show the Q landscape at.
obs_enc_ref = jnp.array(steps[int(np.argmax(divs))]["obs_enc"])  # (1152,)

# Build a 2D grid in PCA space and back-project to 40D for Q evaluation
GRID = 30
pc_min = all_xbc @ pca.components_.T
x_range = (pc_min[:, 0].min() * 1.3, pc_min[:, 0].max() * 1.3)
y_range = (pc_min[:, 1].min() * 1.3, pc_min[:, 1].max() * 1.3)
gx = np.linspace(*x_range, GRID)
gy = np.linspace(*y_range, GRID)
GX, GY = np.meshgrid(gx, gy)
grid2d  = np.stack([GX.ravel(), GY.ravel()], axis=1)              # (GRID^2, 2)
grid40  = pca.inverse_transform(grid2d)                            # (GRID^2, 40)
grid_ph = jnp.array(model40_to_phys40(grid40))                    # (GRID^2, 40)
obs_rep = jnp.broadcast_to(obs_enc_ref[None], (GRID * GRID, 1152))

print("Computing Q contour grid …")
Q_flat  = np.array(q_ensemble_min(obs_rep, grid_ph))              # (GRID^2,)
Q_map   = Q_flat.reshape(GRID, GRID)
Q_map   = gaussian_filter(Q_map, sigma=1.5)
print(f"  Q contour range: {Q_flat.min():.1f} … {Q_flat.max():.1f}")


# --------------------------------------------------------------------------- #
# Helpers: build per-step trajectory arrays in 2D
# --------------------------------------------------------------------------- #
def get_bc_traj_2d(env_step_dict):
    """(10, 2): PCA projection of x_t_bc across denoising steps 0..9."""
    pts = [di["x_t_bc"] for di in env_step_dict["denoising"]]
    return proj_traj(pts)


def get_guided_traj_2d(env_step_dict):
    """(10, 2): PCA projection of x_t_after across denoising steps 0..9."""
    pts = [di["x_t_after"] for di in env_step_dict["denoising"]]
    return proj_traj(pts)


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def draw_traj(ax, traj2d, color, lw=2.0, alpha=0.85, zorder=4):
    """Draw a denoising trajectory as a segmented line + arrowhead."""
    ax.plot(traj2d[:, 0], traj2d[:, 1],
            color=color, lw=lw, alpha=alpha, zorder=zorder,
            solid_capstyle="round")
    for pt in traj2d[1:-1]:
        ax.plot(*pt, "o", color=color, markersize=3.5, alpha=0.6, zorder=zorder+1)
    # Arrow at final step
    if len(traj2d) >= 2:
        ax.annotate("", xy=traj2d[-1], xytext=traj2d[-2],
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=lw, mutation_scale=13),
                    zorder=zorder+2)


def draw_q_contours(ax, alpha=0.45):
    cmap = LinearSegmentedColormap.from_list(
        "q_cmap", [BLUE_BG, "#2979ff"], N=256
    )
    ax.contourf(GX, GY, Q_map, levels=10, cmap=cmap, alpha=alpha, zorder=1)
    ax.contour(GX, GY, Q_map, levels=10, colors=BLUE,
               alpha=0.22, linewidths=0.7, zorder=2)


def panel_setup(ax, title, bg_color):
    ax.set_facecolor(bg_color)
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("PC 1  (action projection)", fontsize=9, color=MID, labelpad=4)
    for sp in ax.spines.values():
        sp.set_linewidth(1.5); sp.set_edgecolor("#cccccc")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=9,
                 color=DARK, fontfamily="DejaVu Sans")


# --------------------------------------------------------------------------- #
# Build figure
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2),
                         gridspec_kw={"wspace": 0.06})
fig.patch.set_facecolor(WHITE)

# Trajectory colours (one per selected env step)
palette_a = ["#333355", "#555577", "#444466", "#222244"]
palette_b = [BLUE, "#1565c0", "#0d47a1", "#1a237e"]

# ======================== Panel (a) — Unguided ========================= #
ax = axes[0]
panel_setup(ax, "(a)  Unguided  $\\hat{\\pi}_{\\theta}(\\cdot | s)$", GREY_BG)
draw_q_contours(ax, alpha=0.20)   # faint, to show space structure

for j, idx in enumerate(sel_idx):
    traj = get_bc_traj_2d(steps[idx])
    draw_traj(ax, traj, palette_a[j], lw=2.2)
    # Mark noise start
    ax.plot(*traj[0], "*", color=MID, markersize=11, zorder=6, alpha=0.8)
    # Mark BC mode
    ax.plot(*traj[-1], "o", color=DARK, markersize=9, zorder=8)

# Label
ax.text(0.97, 0.04,
        "Flow converges to $\\mathbf{BC}$ mode;\nQ-gradient ignored",
        transform=ax.transAxes, fontsize=8.5, color=DARK,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", fc=WHITE, ec="#cccccc", alpha=0.85))
ax.text(0.03, 0.97, "$Q(s,a)$", transform=ax.transAxes,
        fontsize=10, color=BLUE, alpha=0.4, zorder=3, style="italic",
        va="top")

# ======================== Panel (b) — QGF-guided ======================== #
ax = axes[1]
panel_setup(ax, "(b)  QGF-guided  $\\hat{\\pi}^{\\mathrm{QGF}}_{\\theta}(\\cdot | s)$", BLUE_BG)
draw_q_contours(ax, alpha=0.55)

for j, idx in enumerate(sel_idx):
    bc_traj  = get_bc_traj_2d(steps[idx])
    qgf_traj = get_guided_traj_2d(steps[idx])
    draw_traj(ax, qgf_traj, palette_b[j], lw=2.2)
    # Shared noise start (same as panel a)
    ax.plot(*bc_traj[0], "*", color=MID, markersize=11, zorder=6, alpha=0.8)
    # Guided end point
    ax.plot(*qgf_traj[-1], "o", color=BLUE, markersize=9, zorder=8)
    # Q-gradient arrow at mid-denoising (step 4 of 10)
    mid_bc  = bc_traj[4]
    mid_qgf = qgf_traj[4]
    diff    = mid_qgf - mid_bc
    norm    = np.linalg.norm(diff) + 1e-6
    if norm > 0.05:
        tip = mid_bc + diff / norm * min(norm, 0.45)
        ax.annotate("", xy=tip, xytext=mid_bc,
                    arrowprops=dict(arrowstyle="-|>", color=TEAL,
                                   lw=2.0, mutation_scale=13),
                    zorder=9)

# Label one gradient arrow (on last trajectory)
last_idx = sel_idx[-1]
bc4 = get_bc_traj_2d(steps[last_idx])[4]
ax.text(bc4[0] + 0.1, bc4[1] + 0.15,
        "$\\nabla_{\\hat{a}_t} Q(s, \\hat{a}_t)$",
        fontsize=8.5, color=TEAL, zorder=10, ha="left")

ax.text(0.97, 0.04,
        "Q-gradient steers each step\ntoward high-$Q$ actions",
        transform=ax.transAxes, fontsize=8.5, color=DARK,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", fc=WHITE, ec="#cccccc", alpha=0.85))
ax.text(0.03, 0.97, "$Q(s,a)$", transform=ax.transAxes,
        fontsize=10, color=BLUE, alpha=0.75, zorder=3, style="italic",
        va="top")

# ======================== Cross-panel: shared noise connector =========== #
# Connect the first selected trajectory's noise start across panels
first_start_bc  = get_bc_traj_2d(steps[sel_idx[0]])[0]
con = matplotlib.patches.ConnectionPatch(
    xyA=first_start_bc, xyB=first_start_bc,
    coordsA="data", coordsB="data",
    axesA=axes[0], axesB=axes[1],
    color=MID, lw=1.0, linestyle="--", alpha=0.4, zorder=0,
)
fig.add_artist(con)

# ======================== Titles / equation footer ======================= #
fig.text(0.5, 0.965,
         "Shared initial noise  $z \\sim \\mathcal{N}(0,\\mathbf{I})$"
         "  →  paired denoising trajectories  (route: merger-into-slow-traffic-v2-005)",
         ha="center", va="top", fontsize=9.5, color=MID, style="italic")

eq_text = (
    r"$v_{\mathrm{guided}} = v_{\mathrm{BC}} + \lambda \cdot "
    r"\nabla_{\hat{a}_t} Q(s,\hat{a}_t)$"
    r"     where     "
    r"$\hat{a}_t = \mathrm{clip}(x_t + t \cdot v_{\mathrm{BC}},\ {-1},\ 1)$"
    r"     $(\lambda = 0.1)$"
)
fig.text(0.5, 0.01, eq_text,
         ha="center", va="bottom", fontsize=10, color=DARK,
         bbox=dict(boxstyle="round,pad=0.4", fc="#f0f4ff",
                   ec="#aabbdd", alpha=0.9))

# ======================== Save ========================================== #
out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for ext in ("pdf", "png"):
    out = os.path.join(out_dir, f"qgf_figure1.{ext}")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=WHITE)
    print(f"Saved: {out}")
