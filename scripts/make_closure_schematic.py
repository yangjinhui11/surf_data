#!/usr/bin/env python3
"""Figure 15: where the two-timescale surface-layer closure sits in the SURF driver.

Self-contained: no data files needed. Renders the driver + process schematic
used as ``manuscript/figures/modis_closure_schematic.png``.

Run:
    python scripts/make_closure_schematic.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Output anchored to the package root (parent of this script's directory).
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "manuscript" / "figures" / "modis_closure_schematic.png"

fig, ax = plt.subplots(figsize=(11.6, 7.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

NAVY = "#1f3864"
BLUE = "#2e6fb7"
GREY = "#5b6470"
RED = "#c0392b"
LGREY = "#eef1f5"
LBLUE = "#e6eef8"
LRED = "#fbeeec"


def box(x, y, w, h, text, fc, ec, tc, fs=9.4, bold=False, lw=1.5):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=1.2",
                       linewidth=lw, edgecolor=ec, facecolor=fc, mutation_aspect=1.0)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal",
            linespacing=1.4)


def arrow(x0, y0, x1, y1, color=NAVY, lw=1.8, cs="arc3,rad=0", ms=16):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, color=color, connectionstyle=cs,
                        shrinkA=2, shrinkB=2, zorder=3)
    ax.add_patch(a)


def note(x, y, text, color=GREY, fs=8.4, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color, linespacing=1.35)


# ============================  PANEL A: driver step  =========================
ax.text(2, 95.5, "A", fontsize=14, fontweight="bold", color=NAVY)
ax.text(6.5, 95.5, "offline driver step  (where the closure acts)",
        fontsize=11, fontweight="bold", color=NAVY, va="center")

box(3, 76, 22, 12, "CMFD forcing\n$T_a, q_a, R_{SW}, R_{LW},$\n$u_a, v_a, p_s, P$", LGREY, GREY, NAVY)
box(31, 76, 25, 12, "offline surface coupling\nbuild exchange state\n& $C_H, C_E, z_0$", LGREY, GREY, NAVY)
box(61, 74.5, 37, 14, "two-timescale surface-layer closure\n"
    r"$\Delta T_{n+1}=\beta_n\Delta T_n+(1-\beta_n)(-\lambda_s H_n)$" + "\n"
    r"$\tau^{-1}=w/\tau_{\rm day}+(1-w)/\tau_{\rm night}$",
    LRED, RED, RED, fs=7.8, bold=True)
note(80, 71.8, r"$\lambda_s=0.05,\ \tau_{\rm day}=6\,$h, $\tau_{\rm night}=3\,$h",
     color=RED, fs=8.2, ha="center")

# SURF physics entry (full width), panel-A bottom
box(3, 56, 93, 9, "SURF physics  (tiled land surface \u2014 panel B)", LBLUE, BLUE, NAVY, bold=True, fs=10)

# flow arrows
arrow(25, 82, 31, 82)                       # forcing -> coupling
arrow(56, 82, 61, 82)                       # coupling -> closure
arrow(43.5, 76, 43.5, 65)                   # coupling -> SURF
arrow(70, 74.5, 70, 65, color=RED, lw=2.2)  # closure -> SURF (modified T_a)
note(72, 68.4, r"$T_a \leftarrow T_a+\Delta T$", color=RED, fs=8.4, ha="left")

# H feedback: from SURF right edge, up into closure bottom-right corner
arrow(96, 60.5, 94, 76.5, color=GREY, lw=1.5, cs="arc3,rad=-0.3")
note(95, 68.5, "previous-step\n$H_n$", color=GREY, fs=7.8, ha="right")

# ============================  PANEL B: SURF sequence  ======================
ax.text(2, 44, "B", fontsize=14, fontweight="bold", color=NAVY)
ax.text(6.5, 44, "SURF tiled process sequence  (driven by the modified $T_a$)",
        fontsize=11, fontweight="bold", color=NAVY, va="center")

steps = [
    ("tiled turbulent\nexchange\n$H_i=\\rho c_p C_{H,i}|V|(T_{s,i}-T_a)$"),
    ("radiation\n$R_{n,i}=(1-\\alpha_i)R_{SW}$\n$+\\,\\epsilon_i R_{LW}-\\epsilon_i\\sigma T_i^4$"),
    ("surface energy\nbalance\n$C_i\\,\\dot T_{s,i}=R_n-H-L_vE-G-M$"),
    ("soil, snow,\ncanopy &\nhydrology\nupdate"),
    ("grid mean\n$\\bar q=\\sum_i f_i q_i$\nskin temperature"),
]
bw, bh, by = 17.4, 14, 24
xs = [3, 22.6, 42.2, 61.8, 79.6]
for text, x in zip(steps, xs):
    box(x, by, bw, bh, text, LBLUE, BLUE, NAVY, fs=8.0)
for k in range(len(xs) - 1):
    arrow(xs[k] + bw, by + bh / 2, xs[k + 1], by + bh / 2, color=BLUE, lw=1.8)

# arrow from SURF physics entry down into first tile step
arrow(11.7, 56, 11.7, by + bh + 0.5, color=RED, lw=2.2)
note(13.5, 47.5, r"the modified $T_a$ enters the bulk transfer (first box)",
     color=RED, fs=8.4, ha="left")

# footer
note(50, 14, "the closure changes only the air temperature seen by the surface exchange;\n"
     "every land process \u2014 soil, snow, canopy, hydrology \u2014 is left unchanged",
     color=GREY, fs=9.0)

fig.suptitle("Where the surface-layer closure sits: a driver-level correction to the offline "
             "atmosphere\u2013surface boundary", fontsize=12, fontweight="bold", color=NAVY, y=0.995)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
