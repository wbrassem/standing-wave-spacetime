#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Black Surface Formation + Density Projection (σ_crit depth axis)

- Warp well (blue) + density projection (red)
- Projection disk of density at z = σ_crit + projection_delta
- Vertical axis runs 0 (flat spacetime) down to σ_crit
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LightSource
from matplotlib.colors import LinearSegmentedColormap
import re
import argparse
import os

# ----------------------
# CLI
# ----------------------
parser = argparse.ArgumentParser(description="Black Surface Formation + Density Projection")
parser.add_argument("--view", choices=["original", "rabbit"], default="original")
parser.add_argument("--frames", type=int, default=200)
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--save-snapshots", action="store_true")
parser.add_argument("--snapshot-frames", type=str, default="1,60,75,105,125,170")
parser.add_argument("--outfile", type=str, default="black_surface_formation.mp4")
args = parser.parse_args()

# ----------------------
# Output directory
# ----------------------
# This makes ../figures relative to the script's own directory
# Base directories relative to this script
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
BUILD_DIR = os.path.join(BASE_DIR, "build")

# Ensure they exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(BUILD_DIR, exist_ok=True)

# ----------------------
# Presets & params
# ----------------------
view_presets = {"original": (30, 45), "rabbit": (75, 45)}
elev, azim = view_presets[args.view]

total_frames = args.frames
tcrit = 75 / 200
sigma_crit = 1.0   # normalized vertical depth (downward)
projection_delta = 0.1 # distance below σ_crit for projection disk
epsilon = 0.1
A_max = 20.0

phase_labels = {
    1: "Flat spacetime",
    60: "Subcritical",
    75: "Critical transition",
    105: "Supercritical",
    125: "Photosphere visible",
    170: "Event horizon"
}

if args.save_snapshots:
    snapshot_set = {int(s) for s in args.snapshot_frames.split(",") if s.strip().isdigit()}
else:
    snapshot_set = set()

# White to blue for warp
warp_cmap = LinearSegmentedColormap.from_list("warp_cmap", ["white", "blue"])

# ----------------------
# Grid
# ----------------------
r = np.linspace(0, 4, 200)
theta = np.linspace(0, 2*np.pi, 200)
R, Theta = np.meshgrid(r, theta)
X, Y = R * np.cos(Theta), R * np.sin(Theta)

# ----------------------
# Figure
# ----------------------
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.view_init(elev=elev, azim=azim)
ax.set_zlim(sigma_crit+projection_delta, 0.0)  # inverted so 0 is top, σcrit is bottom

ls = LightSource(azdeg=45, altdeg=45)

surf, projection = None, None

# ----------------------
# Helpers
# ----------------------
def slugify(label: str) -> str:
    return re.sub(r'[^a-z0-9_]', '', label.lower().replace(" ", "_"))

def smoothstep(x, edge0, edge1):
    x = np.clip((x - edge0) / (edge1 - edge0 + 1e-12), 0.0, 1.0)
    return x * x * (3 - 2 * x)

def warp_profile(tt):
    A = A_max * (tt ** 1.4)
    sigma = 0.5 + 0.8 * tt
    w_raw = -A * np.exp(-R**2 / (2 * sigma**2))
    w = -5.0 + (w_raw + 5.0)/2.0 * (1.0 + np.tanh((w_raw + 5.0)/epsilon))
    return sigma_crit * (-w / 5.0), sigma  # normalized depth

def density_field(tt, sigma):
    r_core = 0.95 * smoothstep(tt, 0.0, tcrit)
    pre_amp = (tt / max(tcrit, 1e-6)) ** 1.2
    pre = pre_amp * np.exp(-(R/(r_core+1e-6))**2)

    s = smoothstep(tt, tcrit, 1.0)
    r_ring = (0.75 + 0.6*s) * sigma  # faster outward growth after crit
    ring_thickness = np.clip(0.35 * sigma * (1.0 - 0.7*s), 0.05, None)
    post = (0.75+0.25*s)*np.exp(-((R-r_ring)/ring_thickness)**2)

    h = smoothstep(tt, tcrit*0.85, tcrit*1.15)
    D = (1-h)*pre + h*post

    if tt > tcrit:
        D *= 1.0 - 0.85*s*np.exp(-(R/(0.3*sigma+1e-6))**2)

    return D/(np.max(D)+1e-9)

# ----------------------
# Update
# ----------------------
def update(frame):
    global surf, projection
    if surf: surf.remove()
    if projection: projection.remove()

    tt = frame / (total_frames - 1)
    W, sigma = warp_profile(tt)
    D = density_field(tt, sigma)

    surface_colors = warp_cmap(np.clip(W / sigma_crit, 0, 1))

    ax.view_init(elev=elev, azim=azim)
    ax.set_zlim(sigma_crit + projection_delta, 0.0)

    fnum = frame + 1
    label = phase_labels.get(fnum, None)

    # --- Video title logic ---
    line1 = f"Formation Sequence: Frame {fnum}/{total_frames}"
    line2 = label if label else " "
    ax.set_title(f"{line1}\n{line2}")

    # Warp surface
    surf = ax.plot_surface(
        X, Y, W,
        facecolors=surface_colors,
        rstride=1, cstride=1,
        antialiased=True, linewidth=0, alpha=0.98,
        rasterized=True
    )

    # Projection disk
    projection = ax.plot_surface(
        X, Y, (sigma_crit + projection_delta) * np.ones_like(R),
        facecolors=plt.cm.Reds(D)[..., :3],
        rstride=1, cstride=1,
        antialiased=False, linewidth=0, alpha=0.85
    )

    # Save snapshots (go to figures/)
    if fnum in snapshot_set:
        if label:
            slug = slugify(label)
            fname = os.path.join(FIGURES_DIR, f"frame_{fnum:03d}_{slug}.pdf")
        else:
            fname = os.path.join(FIGURES_DIR, f"frame_{fnum:03d}.pdf")

        # Temporarily clear title for LaTeX figures
        current_title = ax.get_title()
        ax.set_title("")
        plt.savefig(fname, bbox_inches="tight")
        ax.set_title(current_title)  # restore for video
        print(f"Saved snapshot: {fname}")

    return surf, projection

# ----------------------
# Animate
# ----------------------
anim = FuncAnimation(fig, update, frames=total_frames, blit=False, repeat=False, init_func=lambda: [])

# Save MP4 (goes to build/)
outfile = os.path.join(BUILD_DIR, args.outfile)
writer = FFMpegWriter(fps=args.fps, extra_args=['-vcodec', 'libx264'])
print("Saving animation...")
anim.save(outfile, writer=writer)
print(f"Done. View = {args.view}, saved to {outfile}")
