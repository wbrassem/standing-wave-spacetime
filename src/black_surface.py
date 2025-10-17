#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Warp Tower + Projection Disk (Mass Density + EH Boundary)
Derived-from-warp implementation: density + EH come from W each frame.
Generates animation + 6 key frame snapshots
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os, re, argparse

# ---------------------- CLI ----------------------
parser = argparse.ArgumentParser(description="Warp Tower + Projection Disk Movie")
parser.add_argument("--frames", type=int, default=500)
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--outfile", type=str, default="black_surface_formation.mp4")
args = parser.parse_args()

# ---------------------- Directories ----------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
BUILD_DIR = os.path.join(BASE_DIR, "build")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(BUILD_DIR, exist_ok=True)

# ---------------------- Parameters ----------------------
total_frames = args.frames
sigma_crit = 1.0

# Enforce centiframe alignment
assert total_frames % 100 == 0, (
    f"total_frames ({total_frames}) must be a multiple of 100 "
    "for phase synchronization."
)

# thresholds derived from W
eh_frac = 0.40            # EH when W >= eh_frac * sigma_crit
ring_low_frac = 0.80      # ring band lower bound (fraction of sigma_crit)
ring_high_frac = 0.90     # ring band upper bound (fraction of sigma_crit)

epsilon = 0.1
A_max = 20.0

# heights for projection surface
Event_H_disk = 1.2   # EH disk (above density disk)

# Mass density colormap: red -> yellow -> orange -> white
colormap = [
    (0.9, 0.0, 0.0),   # red
    (1.0, 0.7, 0.2),   # orange
    (1.0, 0.95, 0.6),  # soft yellow
    (1.0, 1.0, 1.0)    # white
]

# Define warp profile colormap and custom density colormaps
warp_cmap = LinearSegmentedColormap.from_list("warp_cmap", ["white", "purple"])
density_cmap = LinearSegmentedColormap.from_list("white_yellow_orange_red", colormap)

# Camera perspective
elev, azim = 30, 45

# Phase frames for snapshots (percentage of total frames)
phase_frames = {
    0: "Flat spacetime",
    15: "Warp begins",
    30: "EH appears",
    35: "Ring forms inside EH",
    45: "Sigma_crit reached",
    70: "EH grows",
    100: "Final frame",
}

# Compute integer-aligned frame indices
phase_frames = {
    int(round((pct / 100) * total_frames)): label
    for pct, label in phase_frames.items()
}

snapshot_set = set(phase_frames.keys())

# ---------------------- Grid ----------------------
r = np.linspace(0, 4, 200)            # 1D radius array (needed by helpers)
theta = np.linspace(0, 2*np.pi, 200)
R, Theta = np.meshgrid(r, theta)      # R shape: (len(theta), len(r))
X, Y = R*np.cos(Theta), R*np.sin(Theta)

# ---------------------- Helpers ----------------------
def slugify(label: str) -> str:
    return re.sub(r'[^a-z0-9_]', '', label.lower().replace(" ", "_"))

def warp_profile(tt):
    """Return W (shape theta x r) and sigma (scalar here)."""
    A = A_max * (tt ** 1.4)
    sigma = 0.5 + 0.8 * tt
    w_raw = -A * np.exp(-R**2 / (2*sigma**2))
    w = -5.0 + (w_raw + 5.0)/2.0 * (1.0 + np.tanh((w_raw + 5.0)/epsilon))
    # W is positive height = normalized depth
    return sigma_crit * (-w / 5.0), sigma

def radial_profile_of_W(W):
    """Average W over theta -> 1D radial profile same length as r."""
    return np.mean(W, axis=0)   # axis 0 is theta

def draw_eh_disk(ax, radius, z_level, color="black", alpha=0.9, n_segments=200):
    """Draw a filled disk (flat circle) at height z_level."""
    theta = np.linspace(0, 2*np.pi, n_segments)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.full_like(x, z_level)

    # Create the polygon (list of [x, y, z] vertices)
    verts = [list(zip(x, y, z))]
    disk = Poly3DCollection(verts, color=color, alpha=alpha, linewidth=0)
    ax.add_collection3d(disk)
    return disk

def eh_radius_from_W(W, r_array, sigma_crit, eh_frac=0.5):
    """Return outermost radius where W_radial >= eh_frac * sigma_crit (0.0 if none)."""
    thresh = eh_frac * sigma_crit
    W_rad = radial_profile_of_W(W)
    mask = (W_rad >= thresh)
    if not np.any(mask):
        return 0.0
    return float(r_array[np.where(mask)[0].max()])

def ring_density_from_W(W, r_array, sigma_crit,
                        ring_low_frac=0.80, ring_high_frac=0.90,
                        ring_thickness_scale=0.10):
    """
    Build density D from W:
    - If ring-band exists (W_radial in [ring_low, ring_high]) -> Gaussian ring centered on mean radius.
    - Otherwise produce a central Gaussian blob whose brightness scales with warp height.
    Returns (D2, use_ring, r_center)
    D2 has same shape as W (theta x r).
    """
    W_rad = radial_profile_of_W(W)
    ring_low = ring_low_frac * sigma_crit
    ring_high = ring_high_frac * sigma_crit

    idx = np.where((W_rad >= ring_low) & (W_rad <= ring_high))[0]

    if idx.size > 0:
        r_center = float(np.mean(r_array[idx]))
        ring_sigma = max(ring_thickness_scale * max(0.2, r_center), 0.10)  # optimal for 500 frames
        use_ring = True
    else:
        r_center = 0.0
        ring_sigma = 0.13  # optimal for 500 frames
        use_ring = False

    # Build 1D Gaussian radial profile
    radial_profile = np.exp(-((r_array - r_center) / (ring_sigma + 1e-12))**2)
    if radial_profile.max() > 0:
        radial_profile = radial_profile / radial_profile.max()

    # Warp-dependent scaling for central blob
    warp_scale = np.clip(np.max(W) / sigma_crit, 0.0, 1.0)

    if use_ring:
        # Hollow out the center when ring forms
        suppression = 1.0 - 0.95 * np.exp(-(r_array / (0.2 * (r_center + 1e-12)))**2)
        radial_profile = radial_profile * suppression
    else:
        # Brightness tracks warp amplitude before ring formation
        radial_profile *= warp_scale

    # Broadcast to 2D
    D2 = np.tile(radial_profile, (W.shape[0], 1))
    D2[np.abs(D2) < 1e-8] = 0.0
    return D2, use_ring, r_center

# ---------------------- Plot setup ----------------------
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=elev, azim=azim)
ax.set_zlim(0.0, 1.35)

surf, disk, eh_disk = None, None, None

# ---------------------- Update ----------------------
def update(frame):
    global surf, disk, eh_disk
    # clear previous
    if surf: surf.remove()
    if disk: disk.remove()
    if eh_disk: eh_disk.remove()

    # Normalizise frames to start at 1
    tt = frame / total_frames
    W, sigma = warp_profile(tt)           # W shape: (theta, r)

    # derive EH radius from W
    r_eh = eh_radius_from_W(W, r, sigma_crit, eh_frac=eh_frac)

    # derive density from W
    D, ring_exists, r_ring_center = ring_density_from_W(W, r, sigma_crit,
                                                       ring_low_frac, ring_high_frac)

    # Warp tower (pure purple gradient)
    colors = warp_cmap(np.clip(W / sigma_crit, 0.0, 1.0))
    surf = ax.plot_surface(X, Y, W,
                           facecolors=colors,
                           rstride=1, cstride=1,
                           antialiased=True, linewidth=0, alpha=0.95,
                           rasterized=True)

    # --- Projection disk using density as both color and transparency ---
    colors_rgb = density_cmap(D)[..., :3]  # drop default alpha
    rgba = np.dstack((colors_rgb, D))      # use density as alpha channel

    # Projection disk with density map (semi-transparent so tower shows through)
    disk = ax.plot_surface(
        X, Y, Event_H_disk * np.ones_like(R),
        facecolors=rgba,
        rstride=1, cstride=1,
        antialiased=False, linewidth=0,
        # No global alpha; each pixel uses its own from D
        rasterized=True
    )

    # EH disk (filled, drawn with density disk embedded)b   
    if r_eh > 0:
        re = np.linspace(0.0, r_eh, 160)
        te = np.linspace(0.0, 2*np.pi, 160)
        RE, TE = np.meshgrid(re, te)
        XE = RE * np.cos(TE)
        YE = RE * np.sin(TE)
        ZE = Event_H_disk * np.ones_like(RE)
        eh_disk = draw_eh_disk(ax, r_eh, Event_H_disk, color="black", alpha=0.7)

    # Title
    label = phase_frames.get(frame, "")
    ax.set_title(f"Frame {frame}/{total_frames}\n{label}")

    # Snapshots
    if frame in snapshot_set:
        slug = slugify(phase_frames[frame])
        fname = os.path.join(FIGURES_DIR, f"{slug}.pdf")
        current_title = ax.get_title()
        ax.set_title("")
        plt.savefig(fname, bbox_inches="tight")
        ax.set_title(current_title)
        print(f"Saved snapshot: {fname}")

    return surf, disk, eh_disk

# ---------------------- Animate ----------------------
anim = FuncAnimation(fig, update, frames=total_frames+1, blit=False, repeat=False, init_func=lambda: [])  # +1 to include last frame @ 100%
outfile = os.path.join(BUILD_DIR, args.outfile)
writer = FFMpegWriter(fps=args.fps, extra_args=['-vcodec', 'libx264'])
print("Saving animation...")
anim.save(outfile, writer=writer)
print(f"Done. Saved to {outfile}")
