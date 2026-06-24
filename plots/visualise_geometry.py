#!/usr/bin/env python3
"""
visualize_geometry.py
=====================
Visualises the EXACT geometry from DetectorConstruction.cc
(SuperK-style flush PMT, extended-glass-fix-v4).

Three output figures
--------------------
Fig 1 — geo_3d_overview.png
    Cut-away 3D view of the full detector + one PMT stack exploded.
    Simulated 5 MeV e- track with Cherenkov + scintillation + WLS photons.

Fig 2 — geo_pmt_detail.png
    Left : annotated radial cross-section of a single PMT bore showing every
           layer (notch, extended glass, photocathode, body) to scale.
    Right: 4-panel physics sheet (QE, Cherenkov thresholds, refractive
           index dispersion, geometric coverage vs radius).

Fig 3 — geo_cross_sections.png
    Three orthogonal 2D cross-sections (XY, XZ, YZ) with tracks + photons.

All dimensions mirror DetectorConstruction.cc *exactly* using the same
derived formulae — any discrepancy is a bug to fix in this file, not here.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Wedge, Arc, Circle
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings, os
warnings.filterwarnings("ignore")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "figure.dpi": 150, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})

# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY CONSTANTS — mirror DetectorConstruction.cc exactly
#  All lengths in mm unless noted.
# ══════════════════════════════════════════════════════════════════════════════
R_INNER   = 150.0          # acrylic inner radius
WALL      = 5.0            # acrylic wall thickness
R_OUTER   = R_INNER + WALL # 155.0 mm

# R1307 datasheet
TUBE_R    = 38.0           # tube OD/2  (kTubeRadius)
T_GLASS   = 2.0            # borosilicate window (kWindowThickness)
T_PC      = 0.5            # photocathode disc   (kPhotocathodeThick)
PC_R      = 25.75          # photocathode active radius (kPhotocathodeR)
BODY_L    = 150.0 - T_GLASS - T_PC   # = 147.5 mm

# Derived — same formulae as Construct()
SAGITTA      = R_INNER - np.sqrt(R_INNER**2 - TUBE_R**2)   # ≈ 4.893 mm
NOTCH_DEPTH  = SAGITTA - 0.001                              # deliberate overlap margin
GLASS_T_EFF  = T_GLASS + SAGITTA                            # ≈ 6.893 mm (extended)
GLASS_BACK   = R_INNER - SAGITTA                            # ≈ 145.107 mm
GLASS_FRONT  = R_INNER + T_GLASS                            # = 152.0 mm
GLASS_CENTRE = (GLASS_BACK + GLASS_FRONT) / 2.0             # ≈ 148.553 mm
PC_CENTRE    = GLASS_FRONT + T_PC / 2.0                     # = 152.25 mm
BODY_CENTRE  = GLASS_FRONT + T_PC + BODY_L / 2.0            # = 226.25 mm

COVERAGE_PCT = 6 * np.pi * PC_R**2 / (4 * np.pi * R_INNER**2) * 100.0  # ≈ 4.42 %

PMT_COLORS   = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261", "#a8dadc"]
AXIS_LABELS  = ["+x", "−x", "+y", "−y", "+z", "−z"]

np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
#  TRACK / PHOTON GENERATORS
# ──────────────────────────────────────────────────────────────────────────────
def electron_track(E_MeV=5.0, n_steps=140):
    pos = np.zeros(3); pts = [pos.copy()]
    d = np.array([0.55, 0.25, 0.80]); d /= np.linalg.norm(d)
    step = 0.20; scat = 0.12 / max(E_MeV, 0.5)
    rng = 0.412 * E_MeV**1.27; cum = 0.
    for _ in range(n_steps):
        if cum > rng: break
        p1 = np.cross(d, [0, 1, 0])
        p1 = p1/np.linalg.norm(p1) if np.linalg.norm(p1) > 1e-6 else np.cross(d, [1, 0, 0])
        p1 /= np.linalg.norm(p1); p2 = np.cross(d, p1)
        d += scat * (np.random.randn() * p1 + np.random.randn() * p2)
        d /= np.linalg.norm(d)
        pos = pos + step * d; pts.append(pos.copy()); cum += step
        if np.linalg.norm(pos) > R_INNER * 0.12: break
    return np.array(pts)

def cherenkov_photons(track, n=24, theta_c=41.2):
    photons = []; tc = np.radians(theta_c); step = max(1, len(track) // n)
    for i in range(0, len(track) - step, step):
        d = track[i+1] - track[i]; d /= (np.linalg.norm(d) + 1e-12)
        phi = np.random.uniform(0, 2*np.pi)
        p = np.cross(d, [0, 0, 1])
        p = (p/np.linalg.norm(p)) if np.linalg.norm(p) > 1e-6 else np.cross(d, [1, 0, 0])
        p /= np.linalg.norm(p); v = np.cross(d, p)
        pd = np.cos(tc)*d + np.sin(tc)*(np.cos(phi)*p + np.sin(phi)*v)
        pd /= np.linalg.norm(pd)
        o = track[i]
        b = 2*np.dot(o, pd); c = np.dot(o, o) - R_INNER**2
        disc = b*b - 4*c
        if disc < 0: continue
        t = (-b + np.sqrt(disc)) / 2.
        if t < 0: continue
        photons.append((o, o + t*pd))
    return photons

def scint_photons(track, n=22):
    photons = []
    for i in np.random.choice(len(track), min(n, len(track)), replace=False):
        o = track[i]; d = np.random.randn(3); d /= np.linalg.norm(d)
        b = 2*np.dot(o, d); c = np.dot(o, o) - R_INNER**2
        disc = b*b - 4*c
        if disc < 0: continue
        t = (-b + np.sqrt(disc)) / 2.
        if t < 0: continue
        photons.append((o, o + t*d))
    return photons

def wls_photons(sci, n=8):
    wls = []
    for i in np.random.choice(len(sci), min(n, len(sci)), replace=False):
        o, e = sci[i]; f = np.random.uniform(0.3, 0.6); ab = o + f*(e - o)
        d = np.random.randn(3); d /= np.linalg.norm(d)
        b = 2*np.dot(ab, d); c = np.dot(ab, ab) - R_INNER**2
        disc = b*b - 4*c
        if disc < 0: continue
        t = (-b + np.sqrt(disc)) / 2.
        if t < 0: continue
        wls.append((o, ab, ab + t*d))
    return wls

# ──────────────────────────────────────────────────────────────────────────────
#  3D GEOMETRY HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def disc_verts(centre, normal, radius, n=44):
    nv = np.array(normal) / np.linalg.norm(normal)
    u = np.cross(nv, [1, 0, 0])
    u = u/np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.cross(nv, [0, 1, 0])
    u /= np.linalg.norm(u); v = np.cross(nv, u)
    th = np.linspace(0, 2*np.pi, n)
    pts = np.array(centre) + radius*(np.outer(np.cos(th), u) + np.outer(np.sin(th), v))
    return [list(map(tuple, pts))]

def cylinder_faces(centre, axis, radius, length, n=22):
    ax = np.array(axis)/np.linalg.norm(axis)
    u = np.cross(ax, [1, 0, 0])
    u = u/np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.cross(ax, [0, 1, 0])
    u /= np.linalg.norm(u); v = np.cross(ax, u)
    th = np.linspace(0, 2*np.pi, n, endpoint=True)
    c = np.array(centre); h = ax * length / 2; faces = []
    for i in range(n-1):
        p0 = c - h + radius*(np.cos(th[i])*u + np.sin(th[i])*v)
        p1 = c - h + radius*(np.cos(th[i+1])*u + np.sin(th[i+1])*v)
        p2 = c + h + radius*(np.cos(th[i+1])*u + np.sin(th[i+1])*v)
        p3 = c + h + radius*(np.cos(th[i])*u + np.sin(th[i])*v)
        faces.append([tuple(p0), tuple(p1), tuple(p2), tuple(p3)])
    return faces

def sphere_wire(ax3d, R, n=26, color="steelblue", alpha=0.15, lw=0.5):
    u = np.linspace(0, np.pi, 80)
    for phi in np.linspace(0, 2*np.pi, n, endpoint=False):
        x = R*np.sin(u)*np.cos(phi); y = R*np.sin(u)*np.sin(phi); z = R*np.cos(u)
        ax3d.plot(x, y, z, color=color, alpha=alpha, lw=lw)

def draw_pmt_3d(ax3d, outward_dir, col, alpha_scale=1.0):
    """Draw one complete PMT stack (extended glass + PC + body) in 3D."""
    d = np.array(outward_dir, dtype=float)
    # Extended glass (fills bore including sagitta region)
    gc = d * GLASS_CENTRE
    gv = disc_verts(gc, d, TUBE_R)
    ax3d.add_collection3d(Poly3DCollection(gv, alpha=0.50*alpha_scale,
        facecolor="#aaddff", edgecolor="#88bbdd", lw=0.7))
    gf = cylinder_faces(gc, d, TUBE_R, GLASS_T_EFF, n=18)
    ax3d.add_collection3d(Poly3DCollection(gf, alpha=0.25*alpha_scale,
        facecolor="#aaddff", edgecolor="none"))
    # Photocathode (active area only, gold)
    pc = d * PC_CENTRE
    pv = disc_verts(pc, d, PC_R)
    ax3d.add_collection3d(Poly3DCollection(pv, alpha=0.95*alpha_scale,
        facecolor=col, edgecolor="gold", lw=1.2))
    # PMT body
    bc = d * BODY_CENTRE
    bf = cylinder_faces(bc, d, TUBE_R, BODY_L, n=18)
    ax3d.add_collection3d(Poly3DCollection(bf, alpha=0.45*alpha_scale,
        facecolor="#2a2a3a", edgecolor="#1a1a2a", lw=0.3))
    # Back cap
    bk = disc_verts(d*(BODY_CENTRE + BODY_L/2), d, TUBE_R, n=20)
    ax3d.add_collection3d(Poly3DCollection(bk, alpha=0.55*alpha_scale,
        facecolor="#2a2a3a", edgecolor="#444"))

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — 3D OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def plot_3d_overview(out_dir):
    fig = plt.figure(figsize=(20, 14)); fig.patch.set_facecolor("#080c10")

    # ── Main 3D panel ─────────────────────────────────────────────────────────
    ax = fig.add_axes([0.01, 0.10, 0.58, 0.88], projection='3d')
    ax.set_facecolor("#080c10")
    lim = 310
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel("x [mm]", color="white", labelpad=6)
    ax.set_ylabel("y [mm]", color="white", labelpad=6)
    ax.set_zlabel("z [mm]", color="white", labelpad=6)
    ax.tick_params(colors="white")
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor("#1a1a2e")
    ax.grid(True, color="#111122", alpha=0.3)

    # Acrylic outer shell (ghost)
    sphere_wire(ax, R_OUTER, n=24, color="#88ccff", alpha=0.10, lw=0.5)
    u = np.linspace(0, 2*np.pi, 36); v = np.linspace(0, np.pi, 18)
    xs = R_OUTER*np.outer(np.cos(u), np.sin(v))
    ys = R_OUTER*np.outer(np.sin(u), np.sin(v))
    zs = R_OUTER*np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, alpha=0.030, color="#88ccff", shade=False, linewidth=0)

    # Inner medium sphere (ghost, with PTFE reflector annotation)
    sphere_wire(ax, R_INNER, n=28, color="#4488ff", alpha=0.20, lw=0.6)

    # 6 PMTs — full opacity on all axes
    axis_dirs = [
        np.array([ 1, 0, 0]), np.array([-1, 0, 0]),
        np.array([ 0, 1, 0]), np.array([ 0,-1, 0]),
        np.array([ 0, 0, 1]), np.array([ 0, 0,-1]),
    ]
    for i, (d, col, lbl) in enumerate(zip(axis_dirs, PMT_COLORS, AXIS_LABELS)):
        draw_pmt_3d(ax, d, col, alpha_scale=1.0)
        # Inward arrow (photon direction)
        tip = d * (GLASS_FRONT - 18)
        ax.quiver(*tip, *(-d * 22), color="gold", alpha=0.80, lw=1.5,
                  arrow_length_ratio=0.35, length=1)
        lbl_pos = d * (BODY_CENTRE + BODY_L/2 + 22)
        ax.text(*lbl_pos, f"R1307\n{lbl}", color=col, fontsize=7.5,
                ha='center', va='center', fontweight='bold')

    # Tracks + photons
    track = electron_track(5.0, 140)
    cer   = cherenkov_photons(track, 24)
    sci   = scint_photons(track, 22)
    wls   = wls_photons(sci, 8)

    ax.plot(track[:,0], track[:,1], track[:,2],
            color="#00ff88", lw=2.5, alpha=0.95, zorder=10)
    ax.scatter(0, 0, 0, color="lime", s=90, zorder=15, edgecolors="white", lw=1.2)
    for i, (o, e) in enumerate(cer):
        ax.plot([o[0],e[0]], [o[1],e[1]], [o[2],e[2]],
                color="#60c0ff", alpha=0.38, lw=0.8,
                label="Cherenkov" if i==0 else "")
    for i, (o, e) in enumerate(sci):
        ax.plot([o[0],e[0]], [o[1],e[1]], [o[2],e[2]],
                color="#ffaa00", alpha=0.25, lw=0.7, ls="--",
                label="Scint. (PPO)" if i==0 else "")
    for i, (_, ab, ne) in enumerate(wls):
        ax.plot([ab[0],ne[0]], [ab[1],ne[1]], [ab[2],ne[2]],
                color="#da70d6", alpha=0.60, lw=1.0,
                label="WLS (bis-MSB)" if i==0 else "")
        ax.scatter(*ab, color="violet", s=14, alpha=0.8, zorder=8)

    # Coordinate axes
    for vec, col, lbl in [([1,0,0],"#ff4444","X"),
                           ([0,1,0],"#44ff44","Y"),
                           ([0,0,1],"#4499ff","Z")]:
        L = R_INNER * 1.22
        ax.quiver(0, 0, 0, vec[0]*L, vec[1]*L, vec[2]*L,
                  color=col, lw=2, arrow_length_ratio=0.07)
        ax.text(vec[0]*(L+14), vec[1]*(L+14), vec[2]*(L+14),
                lbl, color=col, fontsize=11, fontweight='bold')

    ax.set_title(
        "SuperK-style Flush PMT Geometry\n"
        f"6 × Hamamatsu R1307 · R_inner={R_INNER:.0f} mm · 5 mm PMMA · 5 MeV e⁻\n"
        f"Extended-glass fix: glass fills bore to r={GLASS_BACK:.2f} mm "
        f"(sagitta={SAGITTA:.3f} mm)",
        color="white", fontsize=10, pad=8)

    handles = [
        Line2D([0],[0], color="#00ff88", lw=2,        label="e⁻ track (5 MeV)"),
        Line2D([0],[0], color="#60c0ff", lw=1,        label="Cherenkov photons"),
        Line2D([0],[0], color="#ffaa00", lw=1, ls="--", label="PPO scintillation"),
        Line2D([0],[0], color="#da70d6", lw=1,        label="bis-MSB WLS re-emit"),
        Line2D([0],[0], color="#aaddff", lw=4,        label=f"Extended glass (Teff={GLASS_T_EFF:.2f} mm)"),
        Line2D([0],[0], color="gold",    lw=4,        label=f"Photocathode Ø{2*PC_R:.1f} mm"),
        Line2D([0],[0], color="#2a2a3a", lw=4,        label="PMT body (vacuum)"),
        Line2D([0],[0], color="#4488ff", lw=1.5, alpha=0.5, label="Detection medium"),
        Line2D([0],[0], color="#88ccff", lw=1.5, alpha=0.4, label="Acrylic PMMA (5 mm)"),
    ]
    ax.legend(handles=handles, loc="lower left", facecolor="#111111",
              edgecolor="#444444", labelcolor="white", fontsize=7.2, framealpha=0.88)
    ax.view_init(elev=20, azim=-48)
    """
    # ── Right top: annotated bore cross-section (radial, 1 PMT) ──────────────
    ax2 = fig.add_axes([0.61, 0.52, 0.37, 0.44])
    ax2.set_facecolor("#080c10"); ax2.set_aspect("equal")
    ax2.tick_params(colors="white"); ax2.spines[:].set_color("#333344")
    ax2.set_title("Bore Cross-Section: +x PMT (radial view)", color="white", fontsize=9.5)
    ax2.set_xlabel("radial position [mm]", color="white", fontsize=8.5)
    ax2.set_ylabel("transverse [mm]", color="white", fontsize=8.5)
    ax2.grid(True, color="#111122", alpha=0.35)

    # ---- Sphere profile (reference arc at R_INNER) ----
    th_arc = np.linspace(-np.arcsin(TUBE_R/R_INNER)*1.15,
                          np.arcsin(TUBE_R/R_INNER)*1.15, 200)
    ax2.plot(R_INNER*np.cos(th_arc), R_INNER*np.sin(th_arc),
             color="#4488ff", lw=1.4, ls=":", alpha=0.5, label="sphere r=R_inner")

    # ---- Acrylic shell (cross-section rectangle) ----
    ax2.add_patch(Rectangle((R_INNER, -TUBE_R-12), WALL, 2*(TUBE_R+12),
                             facecolor="#1a2535", edgecolor="#334455", lw=0.8, zorder=0))
    # Hole in acrylic (blank region)
    ax2.add_patch(Rectangle((R_INNER, -TUBE_R), WALL, 2*TUBE_R,
                             facecolor="#080c10", edgecolor="none", zorder=1))
    ax2.add_patch(Rectangle((R_INNER, -TUBE_R), WALL, 2*TUBE_R,
                             facecolor="none", edgecolor="#88ccff", lw=1.4, ls="--", zorder=2))
    ax2.text(R_INNER + WALL/2, TUBE_R+18, "acrylic PMMA\n(hole through)",
             ha='center', color="#88ccff", fontsize=7.5)

    # ---- Medium region (detection liquid, left of R_INNER) ----
    ax2.add_patch(Rectangle((R_INNER-18, -TUBE_R-12), 18, 2*(TUBE_R+12),
                             facecolor="#0a1828", edgecolor="#2244aa", lw=0.7, zorder=0))
    ax2.text(R_INNER-9, TUBE_R+18, "medium\n(liquid)", ha='center',
             color="#4488ff", fontsize=7.5)

    # ---- Sagitta + notch annotation ----
    th_sag = np.linspace(-np.arcsin(TUBE_R/R_INNER), np.arcsin(TUBE_R/R_INNER), 120)
    xsag = R_INNER*np.cos(th_sag); ysag = R_INNER*np.sin(th_sag)
    ax2.fill_betweenx(ysag, GLASS_BACK, xsag,
                       where=(xsag > GLASS_BACK),
                       color="#ff5566", alpha=0.18, label="sagitta zone (filled by extended glass)")
    ax2.plot(xsag, ysag, color="#ff5566", lw=1.3, ls=":", alpha=0.7)

    # ---- Extended glass (key fix) ----
    ax2.add_patch(Rectangle((GLASS_BACK, -TUBE_R), GLASS_T_EFF, 2*TUBE_R,
                             facecolor="#aaddff", edgecolor="#88bbdd",
                             lw=1.5, alpha=0.70, zorder=5))
    ax2.text(GLASS_CENTRE, -TUBE_R-9,
             f"extended glass\nTeff={GLASS_T_EFF:.2f} mm",
             ha='center', va='top', color="#aaddff", fontsize=7.5)

    # ---- Photocathode ----
    ax2.add_patch(Rectangle((GLASS_FRONT, -PC_R), T_PC, 2*PC_R,
                             facecolor="gold", edgecolor="orange",
                             lw=1.5, alpha=0.95, zorder=6))
    ax2.text(GLASS_FRONT + T_PC/2, PC_R+5,
             f"PC Ø{2*PC_R:.1f}mm", ha='center', color="gold", fontsize=7.2)

    # ---- Body ----
    ax2.add_patch(Rectangle((GLASS_FRONT + T_PC, -TUBE_R), min(BODY_L, 35), 2*TUBE_R,
                             facecolor="#2a2a3a", edgecolor="#444466",
                             lw=0.9, alpha=0.80, zorder=5))
    ax2.text(GLASS_FRONT + T_PC + 10, 0, "vacuum\nbody", ha='center',
             color="#888899", fontsize=7.2)

    # ---- Key dimension lines ----
    for xv, lbl, ypos, col in [
        (GLASS_BACK,  f"glass back\n{GLASS_BACK:.2f} mm", -TUBE_R-18, "#aaddff"),
        (R_INNER,     f"R_inner\n{R_INNER:.0f} mm",       TUBE_R+28,  "white"),
        (GLASS_FRONT, f"glass front\n{GLASS_FRONT:.1f} mm",-TUBE_R-18, "#aaddff"),
    ]:
        ax2.axvline(xv, color=col, lw=0.8, ls=":", alpha=0.55)
        ax2.text(xv, ypos, lbl, ha='center', color=col, fontsize=6.8)

    # Sagitta double-arrow
    ax2.annotate("", xy=(R_INNER, TUBE_R*0.55),
                 xytext=(GLASS_BACK, TUBE_R*0.55),
                 arrowprops=dict(arrowstyle="<->", color="#ff8888", lw=1.3))
    ax2.text((R_INNER+GLASS_BACK)/2, TUBE_R*0.55+4,
             f"sagitta\n{SAGITTA:.3f} mm", ha='center', color="#ff8888", fontsize=7)

    ax2.set_xlim(R_INNER-22, GLASS_FRONT + min(BODY_L, 40) + 4)
    ax2.set_ylim(-(TUBE_R+30), TUBE_R+35)
    ax2.legend(fontsize=6.8, loc="lower right", facecolor="#111111",
               edgecolor="#444444", labelcolor="white")

    # ── Right bottom: geometry + optical-path summary ─────────────────────────
    ax3 = fig.add_axes([0.61, 0.06, 0.37, 0.41])
    ax3.axis("off"); ax3.set_facecolor("#080c10")
    txt = (
       # "Geometry — DetectorConstruction.cc  (extended-glass-fix-v4)\n"
        "────────────────────────────────────────────────────────────\n"
        f"  Acrylic vessel   inner R = {R_INNER:.1f} mm\n"
        f"                   outer R = {R_OUTER:.1f} mm  (wall = {WALL:.0f} mm)\n"
        f"  6 bore holes     Ø = {2*TUBE_R:.0f} mm (R1307 tube OD)\n"
        f"  Sagitta          s = R − √(R²−r²) = {SAGITTA:.4f} mm\n"
        f"  Notch depth      sagitta − 0.001 mm = {NOTCH_DEPTH:.4f} mm\n"
        f"                   → medium overlaps glass by 0.001 mm\n\n"
        "PMT stack (per axis):\n"
        f"  glass back face  r = {GLASS_BACK:.3f} mm  (= R − sagitta)\n"
        f"  glass front face r = {GLASS_FRONT:.1f} mm  (= R + T_glass)\n"
        f"  glass T_eff      = {GLASS_T_EFF:.4f} mm  (= T_glass + sagitta)\n"
        f"  photocathode     Ø{2*PC_R:.1f} mm, centre r={PC_CENTRE:.3f} mm\n"
        f"  body             L={BODY_L:.1f} mm, vacuum, centre r={BODY_CENTRE:.3f} mm\n\n"
        "Optical surfaces:\n"
        "  medium→acrylic   95 % PTFE diffuse  (BorderSurface)\n"
        "  medium→glass     automatic Fresnel  (no Surface object)\n"
        "                   n_med≈1.34 → n_glass≈1.51 → ~0.4 % refl.\n"
        "  photocathode     R1307 QE(λ), SD on fGeomBoundary entry\n\n"
        f"  Geometric coverage  =  {COVERAGE_PCT:.3f} %\n"
        f"  (6 × πr² / 4πR²,  r={PC_R:.2f} mm, R={R_INNER:.0f} mm)"
    )
    ax3.text(0.02, 0.98, txt, transform=ax3.transAxes, color="lightgrey",
             fontsize=7.8, va='top', ha='left', family='monospace',
             bbox=dict(facecolor="#0d1118", edgecolor="#333344",
                       boxstyle="round,pad=0.5", alpha=0.92))
    """
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, "geo_3d_overview.png")
    fig.savefig(fname, facecolor=fig.get_facecolor(), dpi=200)
    plt.close(fig)
    print(f"  [✓] 3D overview          → {fname}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — PMT DETAIL + PHYSICS SHEET
# ══════════════════════════════════════════════════════════════════════════════
def plot_pmt_detail_and_physics(out_dir):
    fig = plt.figure(figsize=(18, 12)); fig.patch.set_facecolor("#080c10")

    # ── Left: full-scale 1D PMT cross-section ─────────────────────────────────
    axL = fig.add_axes([0.04, 0.08, 0.42, 0.88])
    axL.set_facecolor("#0a0f18"); axL.set_aspect("equal")
    axL.tick_params(colors="white"); axL.spines[:].set_color("#333344")
    axL.set_title("Hamamatsu R1307 — Full Cross-Section (to scale)",
                  color="white", fontsize=10.5)
    axL.set_xlabel("radial position r [mm]", color="white")
    axL.set_ylabel("transverse [mm]", color="white")
    axL.grid(True, color="#111122", alpha=0.35)

    draw_x0 = R_INNER - 30   # leftmost extent of the drawing

    # Medium (liquid)
    axL.add_patch(Rectangle((draw_x0, -TUBE_R-14), 30, 2*(TUBE_R+14),
                             facecolor="#0a1828", edgecolor="#224488", lw=0.8, zorder=0))
    axL.text(draw_x0 + 8, TUBE_R+20, "detection medium\n(liquid, n≈1.34)",
             ha='center', color="#4488ff", fontsize=8)

    # Acrylic (between PMTs)
    axL.add_patch(Rectangle((R_INNER, -TUBE_R-14), WALL, 2*(TUBE_R+14),
                             facecolor="#1a2535", edgecolor="#334455", lw=0.8, zorder=0))
    axL.text(R_INNER + WALL/2, TUBE_R+20, "acrylic\n5 mm PMMA",
             ha='center', color="#88ccff", fontsize=8)

    # Hole in acrylic
    axL.add_patch(Rectangle((R_INNER, -TUBE_R), WALL, 2*TUBE_R,
                             facecolor="#0a0f18", edgecolor="none", zorder=2))
    axL.add_patch(Rectangle((R_INNER, -TUBE_R), WALL, 2*TUBE_R,
                             facecolor="none", edgecolor="#6699cc", lw=1.2, ls="--", zorder=3))

    # Sagitta region fill (highlight what old code left as a gap)
    th_s = np.linspace(-np.arcsin(TUBE_R/R_INNER), np.arcsin(TUBE_R/R_INNER), 200)
    xsag = R_INNER*np.cos(th_s); ysag = R_INNER*np.sin(th_s)
    axL.fill_betweenx(ysag, GLASS_BACK, xsag,
                       where=(xsag > GLASS_BACK),
                       color="#ff5566", alpha=0.22, zorder=1,
                       label=f"sagitta zone ({SAGITTA:.3f} mm) — filled by extended glass")
    axL.plot(xsag, ysag, color="#ff5566", lw=1.2, ls=":", alpha=0.65)

    # Extended glass (full thickness)
    axL.add_patch(Rectangle((GLASS_BACK, -TUBE_R), GLASS_T_EFF, 2*TUBE_R,
                             facecolor="#aaddff", edgecolor="#88bbdd",
                             lw=1.8, alpha=0.72, zorder=4,
                             label=f"extended BS glass  Teff={GLASS_T_EFF:.3f} mm"))

    # Photocathode
    axL.add_patch(Rectangle((GLASS_FRONT, -PC_R), T_PC, 2*PC_R,
                             facecolor="#ffd700", edgecolor="orange",
                             lw=1.8, alpha=0.97, zorder=5,
                             label=f"photocathode K₂CsSb  Ø{2*PC_R:.1f} mm"))

    # Body
    body_draw = min(BODY_L, 60)   # truncate for display
    axL.add_patch(Rectangle((GLASS_FRONT+T_PC, -TUBE_R), body_draw, 2*TUBE_R,
                             facecolor="#22223a", edgecolor="#44446a",
                             lw=1.0, alpha=0.85, zorder=4,
                             label=f"PMT body (vacuum)  L={BODY_L:.1f} mm  [truncated]"))
    axL.text(GLASS_FRONT+T_PC+body_draw/2, 0,
             f"vacuum\n{BODY_L:.1f} mm\n[truncated]",
             ha='center', color="#888899", fontsize=7.5)

    # Dimension annotations
    dims = [
        (GLASS_BACK,   GLASS_FRONT,  TUBE_R*0.72,  f"T_glass_eff = {GLASS_T_EFF:.3f} mm", "#aaddff"),
        (R_INNER,      GLASS_FRONT,  TUBE_R*0.45,  f"T_glass = {T_GLASS:.1f} mm (R1307)", "#ddeeff"),
        (GLASS_BACK,   R_INNER,      TUBE_R*0.45,  f"sagitta = {SAGITTA:.3f} mm", "#ff8888"),
        (GLASS_FRONT,  GLASS_FRONT+T_PC, -TUBE_R*0.55, f"T_PC={T_PC:.1f} mm", "#ffd700"),
    ]
    for x0, x1, y, lbl, col in dims:
        axL.annotate("", xy=(x1, y), xytext=(x0, y),
                     arrowprops=dict(arrowstyle="<->", color=col, lw=1.2))
        axL.text((x0+x1)/2, y + 2.8, lbl, ha='center', color=col, fontsize=7.2)

    # Vertical reference lines
    for xv, lbl, col in [
        (GLASS_BACK,   f"glass back\n{GLASS_BACK:.2f}", "#aaddff"),
        (R_INNER,      f"R_inner\n{R_INNER:.0f}",        "white"),
        (GLASS_FRONT,  f"glass front\n{GLASS_FRONT:.1f}", "#ddeeff"),
    ]:
        axL.axvline(xv, color=col, lw=0.9, ls=":", alpha=0.6)
        axL.text(xv, -TUBE_R-20, lbl, ha='center', color=col, fontsize=7)

    # Optical path arrow
    arrow_y = TUBE_R * 0.15
    axL.annotate("", xy=(GLASS_FRONT+T_PC, arrow_y), xytext=(draw_x0+5, arrow_y),
                 arrowprops=dict(arrowstyle="-|>", color="gold", lw=2.0,
                                 mutation_scale=14))
    axL.text((draw_x0+5+GLASS_FRONT+T_PC)/2, arrow_y+4,
             "photon direction → ", ha='center', color="gold", fontsize=8)

    # RINDEX labels
    for xv, n, col in [(draw_x0+8, "n≈1.34", "#4499ff"),
                        (GLASS_CENTRE, "n≈1.51", "#aaddff"),
                        (GLASS_FRONT+0.4, "n=1.50", "#ffd700")]:
        axL.text(xv, -TUBE_R*0.35, n, ha='center', color=col, fontsize=7.5,
                 bbox=dict(facecolor="#111122", edgecolor=col, boxstyle="round,pad=0.2", alpha=0.7))

    axL.set_xlim(draw_x0 - 3, GLASS_FRONT + body_draw + 5)
    axL.set_ylim(-TUBE_R-32, TUBE_R+30)
    axL.legend(fontsize=7.5, loc="upper left", facecolor="#111111",
               edgecolor="#444444", labelcolor="white", framealpha=0.90)

    # ── Right: 4-panel physics sheet ──────────────────────────────────────────
    # (A) R1307 QE spectrum
    axA = fig.add_axes([0.54, 0.56, 0.21, 0.38])
    axA.set_facecolor("#0d1420"); axA.tick_params(colors="white")
    axA.spines[:].set_color("#333344")
    axA.set_title("A — R1307 Quantum Efficiency", color="white", fontsize=8.5)
    axA.set_xlabel("Wavelength [nm]", color="white", fontsize=8)
    axA.set_ylabel("QE [%]", color="white", fontsize=8)
    axA.grid(True, color="#111122", alpha=0.35)

    wl_qe = np.array([270,290,310,330,350,370,390,410,420,430,
                       440,460,480,500,520,540,560,580,600,640])
    qe    = np.array([0.1, 0.5, 1.5, 4.0, 8.0,14.0,23.0,27.0,30.0,30.0,
                      29.0,26.0,22.0,17.0,13.0, 9.0, 6.0, 3.0, 1.0, 0.1])
    wlf = np.linspace(260, 660, 400)
    qef = np.interp(wlf, wl_qe, qe, left=0, right=0)
    axA.fill_between(wlf, qef, alpha=0.35, color="#ffd700")
    axA.plot(wlf, qef, color="#ffd700", lw=2.2, label="R1307 QE")
    axA.axvline(420, color="white", lw=1, ls="--", alpha=0.6)
    axA.text(422, 31, "420 nm\n30%", color="white", fontsize=7.5)
    axA.set_xlim(260, 660)
    axA.legend(fontsize=7.5, facecolor="#111111", edgecolor="#444444", labelcolor="white")

    # (B) Cherenkov thresholds
    axB = fig.add_axes([0.78, 0.56, 0.20, 0.38])
    axB.set_facecolor("#0d1420"); axB.tick_params(colors="white")
    axB.spines[:].set_color("#333344")
    axB.set_title("B — Cherenkov Threshold", color="white", fontsize=8.5)
    axB.set_xlabel("E_thr [MeV]", color="white", fontsize=8)
    axB.grid(True, color="#111122", alpha=0.35)

    media_n = {"H₂O":1.340,"D₂O":1.333,"WbLS":1.345,"GdWbLS":1.346,
               "TeWbLS":1.353,"D₂O+LAB":1.338,"D₂O+GdLAB":1.339,"D₂O+TeLAB":1.345}
    mcols = ["#0077BB","#33BBEE","#EE7733","#CC3311",
             "#EE3377","#009988","#117733","#AA3377"]
    for i, (med, n) in enumerate(media_n.items()):
        E = (1/np.sqrt(1 - 1/n**2) - 1) * 0.511
        axB.barh(i, E, color=mcols[i], alpha=0.85, edgecolor="#333344", lw=0.5)
        axB.text(E + 0.004, i, f"{E:.3f}", va='center', color="white", fontsize=7)
    axB.set_yticks(range(len(media_n)))
    axB.set_yticklabels(list(media_n.keys()), color="white", fontsize=7)

    # (C) Refractive index dispersion
    axC = fig.add_axes([0.54, 0.08, 0.21, 0.38])
    axC.set_facecolor("#0d1420"); axC.tick_params(colors="white")
    axC.spines[:].set_color("#333344")
    axC.set_title("C — Refractive Index n(λ)", color="white", fontsize=8.5)
    axC.set_xlabel("Wavelength [nm]", color="white", fontsize=8)
    axC.set_ylabel("n", color="white", fontsize=8)
    axC.grid(True, color="#111122", alpha=0.35)

    wl_n = np.linspace(270, 620, 300)
    ri_wl = [270,290,310,330,370,390,420,440,480,540,590,620]
    ri_data = {
        "H₂O":     [1.332,1.333,1.334,1.335,1.337,1.339,1.342,1.344,1.348,1.352,1.355,1.358],
        "D₂O":     [1.325,1.326,1.327,1.328,1.330,1.332,1.334,1.336,1.339,1.343,1.347,1.351],
        "WbLS":    [1.334,1.335,1.336,1.337,1.340,1.342,1.344,1.346,1.350,1.355,1.358,1.362],
        "Acrylic": [1.504,1.502,1.500,1.498,1.494,1.492,1.490,1.488,1.486,1.484,1.482,1.480],
        "BS Glass":[1.530,1.527,1.524,1.521,1.517,1.515,1.513,1.511,1.509,1.506,1.503,1.500],
    }
    ri_cols = ["#0077BB","#33BBEE","#EE7733","#88ccff","#ffd700"]
    ri_ls   = ["-","-","-","--","-."]
    for (med, vals), col, ls in zip(ri_data.items(), ri_cols, ri_ls):
        axC.plot(wl_n, np.interp(wl_n, ri_wl, vals), color=col, lw=1.8, ls=ls, label=med)
    axC.axvline(420, color="white", lw=0.7, ls=":", alpha=0.4)
    axC.set_xlim(270, 620)
    axC.legend(fontsize=7, facecolor="#111111", edgecolor="#444444", labelcolor="white")

    # (D) Geometric coverage vs radius
    axD = fig.add_axes([0.78, 0.08, 0.20, 0.38])
    axD.set_facecolor("#0d1420"); axD.tick_params(colors="white")
    axD.spines[:].set_color("#333344")
    axD.set_title("D — PMT Coverage vs R", color="white", fontsize=8.5)
    axD.set_xlabel("R_inner [mm]", color="white", fontsize=8)
    axD.set_ylabel("Coverage [%]", color="white", fontsize=8)
    axD.grid(True, color="#111122", alpha=0.35)

    rr = np.linspace(50, 350, 300)
    for n_pmt, col, lbl in [(6,"#ffd700",f"6 PMTs (this)"),
                              (12,"#ff7733","12 PMTs"),
                              (20,"#44ff88","20 PMTs")]:
        axD.plot(rr, n_pmt*np.pi*PC_R**2/(4*np.pi*rr**2)*100, color=col, lw=1.8, label=lbl)
    axD.axvline(R_INNER, color="white", lw=1.1, ls="--", alpha=0.7)
    axD.scatter([R_INNER], [COVERAGE_PCT], color="#ffd700", s=70, zorder=5)
    axD.text(R_INNER+5, COVERAGE_PCT+0.3, f"{COVERAGE_PCT:.2f}%",
             color="#ffd700", fontsize=8)
    axD.legend(fontsize=7.5, facecolor="#111111", edgecolor="#444444", labelcolor="white")

    fname = os.path.join(out_dir, "geo_pmt_detail.png")
    fig.savefig(fname, facecolor=fig.get_facecolor(), dpi=200)
    plt.close(fig)
    print(f"  [✓] PMT detail + physics → {fname}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — THREE ORTHOGONAL CROSS-SECTIONS
# ══════════════════════════════════════════════════════════════════════════════
def plot_cross_sections(out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor("#080c10")

    titles    = ["XY Plane  (top, z=0)", "XZ Plane  (side, y=0)", "YZ Plane  (front, x=0)"]
    plane_idx = [(0,1), (0,2), (1,2)]
    plane_pmts = [
        [(1,0,0,"#e63946","+x"),(-1,0,0,"#457b9d","−x"),
         (0,1,0,"#2a9d8f","+y"),(0,-1,0,"#e9c46a","−y")],
        [(1,0,0,"#e63946","+x"),(-1,0,0,"#457b9d","−x"),
         (0,0,1,"#a8dadc","+z"),(0,0,-1,"#f4a261","−z")],
        [(0,1,0,"#2a9d8f","+y"),(0,-1,0,"#e9c46a","−y"),
         (0,0,1,"#a8dadc","+z"),(0,0,-1,"#f4a261","−z")],
    ]
    al = ["x [mm]", "y [mm]", "z [mm]"]

    track = electron_track(5.0, 140)
    cer   = cherenkov_photons(track, 22)
    sci   = scint_photons(track, 20)
    wls   = wls_photons(sci, 7)

    for ci, (ax, title, (d0, d1), pmts) in \
            enumerate(zip(axes, titles, plane_idx, plane_pmts)):
        ax.set_facecolor("#080c10"); ax.set_aspect("equal")
        ax.set_xlim(-310, 310); ax.set_ylim(-310, 310)
        ax.tick_params(colors="white"); ax.spines[:].set_color("#333344")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xlabel(al[d0], color="white"); ax.set_ylabel(al[d1], color="white")
        ax.grid(True, color="#111122", alpha=0.35)
        th = np.linspace(0, 2*np.pi, 360)

        # Acrylic outer ring + fill
        ax.plot(R_OUTER*np.cos(th), R_OUTER*np.sin(th), color="#88ccff", lw=2.0, alpha=0.7)
        ax.fill_between(R_OUTER*np.cos(th), R_OUTER*np.sin(th),
                         R_INNER*np.sin(th), alpha=0.10, color="#88ccff")
        # Inner medium
        ax.plot(R_INNER*np.cos(th), R_INNER*np.sin(th), color="#4488ff", lw=1.2,
                alpha=0.45, ls=":")
        ax.fill(R_INNER*np.cos(th), R_INNER*np.sin(th), alpha=0.05, color="#4488ff")

        for px, py, pz, col, lbl in pmts:
            full = np.array([px, py, pz], dtype=float)
            nd2  = full[[d0, d1]]; nm = np.linalg.norm(nd2)
            nd2  = nd2/nm if nm > 0 else nd2
            perp = np.array([-nd2[1], nd2[0]])

            # Erase acrylic ring at hole
            gap = np.array([nd2*(R_INNER-2) + TUBE_R*perp,
                             nd2*(R_OUTER+3) + TUBE_R*perp,
                             nd2*(R_OUTER+3) - TUBE_R*perp,
                             nd2*(R_INNER-2) - TUBE_R*perp])
            ax.fill(gap[:,0], gap[:,1], color="#080c10", zorder=2)

            # Extended glass (back to GLASS_BACK, front to GLASS_FRONT)
            g0 = nd2 * GLASS_BACK;  g1 = nd2 * GLASS_FRONT
            gpts = np.array([g0 + TUBE_R*perp, g0 - TUBE_R*perp,
                              g1 - TUBE_R*perp, g1 + TUBE_R*perp])
            ax.fill(gpts[:,0], gpts[:,1], color="#aaddff", alpha=0.60, zorder=4)
            # front face line (real glass face)
            ax.plot([g1[0]-perp[0]*TUBE_R, g1[0]+perp[0]*TUBE_R],
                    [g1[1]-perp[1]*TUBE_R, g1[1]+perp[1]*TUBE_R],
                    color="#ddeeff", lw=1.5, alpha=0.85, zorder=5)
            # back face line (extended)
            ax.plot([g0[0]-perp[0]*TUBE_R, g0[0]+perp[0]*TUBE_R],
                    [g0[1]-perp[1]*TUBE_R, g0[1]+perp[1]*TUBE_R],
                    color="#aaddff", lw=1.0, alpha=0.60, ls="--", zorder=5)

            # Photocathode (gold line)
            pc0 = nd2 * GLASS_FRONT; pc1 = nd2 * (GLASS_FRONT + T_PC)
            for ri in [pc0, pc1]:
                p1 = ri + PC_R*perp; p2 = ri - PC_R*perp
                ax.plot([p1[0],p2[0]], [p1[1],p2[1]], color=col, lw=5,
                        alpha=0.92, solid_capstyle='round', zorder=6)

            # Body rectangle (truncated at plot edge)
            bc0 = nd2 * (GLASS_FRONT + T_PC)
            bc1 = nd2 * min(GLASS_FRONT + T_PC + BODY_L, 305)
            bpts = np.array([bc0+TUBE_R*perp, bc0-TUBE_R*perp,
                              bc1-TUBE_R*perp, bc1+TUBE_R*perp])
            ax.fill(bpts[:,0], bpts[:,1], color="#22223a", alpha=0.75, zorder=3)
            ax.plot(np.append(bpts[:,0], bpts[0,0]),
                    np.append(bpts[:,1], bpts[0,1]),
                    color="#44446a", lw=0.6, zorder=3)

            # Inward photon arrow
            ax.annotate("", xy=nd2*(R_INNER-18), xytext=nd2*(GLASS_FRONT),
                         arrowprops=dict(arrowstyle="->", color="gold", lw=1.6))

            # Label
            lbl_r = min(GLASS_FRONT + T_PC + BODY_L + 22, 308)
            ax.text(nd2[0]*lbl_r, nd2[1]*lbl_r, lbl, color=col,
                    ha='center', va='center', fontsize=8.5, fontweight='bold')

        # Tracks
        ax.plot(track[:,d0], track[:,d1], color="#00ff88", lw=2.0, alpha=0.9)
        ax.scatter(0, 0, color="lime", s=55, zorder=10)
        for o, e in cer:
            ax.plot([o[d0],e[d0]], [o[d1],e[d1]], color="#60c0ff", alpha=0.35, lw=0.8)
        for o, e in sci[:12]:
            ax.plot([o[d0],e[d0]], [o[d1],e[d1]], color="#ffaa00", alpha=0.22, lw=0.7, ls="--")
        for _, ab, ne in wls:
            ax.plot([ab[d0],ne[d0]], [ab[d1],ne[d1]], color="#da70d6", alpha=0.55, lw=0.9)

        # Dimension arrow
        ax.annotate("", xy=(R_OUTER, -R_OUTER*1.18), xytext=(-R_OUTER, -R_OUTER*1.18),
                     arrowprops=dict(arrowstyle="<->", color="white", lw=1.2))
        ax.text(0, -R_OUTER*1.26, f"2×{R_OUTER:.0f} = {2*R_OUTER:.0f} mm",
                ha='center', color="white", fontsize=7.5, va='top')

        if ci == 0:
            handles = [
                Line2D([0],[0], color="#00ff88", lw=2,         label="e⁻ track (5 MeV)"),
                Line2D([0],[0], color="#60c0ff", lw=1,         label="Cherenkov"),
                Line2D([0],[0], color="#ffaa00", lw=1, ls="--", label="Scintillation"),
                Line2D([0],[0], color="#da70d6", lw=1,         label="WLS (bis-MSB)"),
                Line2D([0],[0], color="#aaddff", lw=4,         label=f"Extended glass (Teff={GLASS_T_EFF:.2f} mm)"),
                Line2D([0],[0], color="gold",    lw=4,         label="Photocathode"),
                Line2D([0],[0], color="#22223a", lw=4,         label="PMT body (vacuum)"),
                Line2D([0],[0], color="#88ccff", lw=2, alpha=0.6, label="Acrylic PMMA"),
            ]
            ax.legend(handles=handles, facecolor="#111111", edgecolor="#444444",
                      labelcolor="white", fontsize=7.2, loc="upper right")

    fig.suptitle(
        "NeutrinoDetSim — Orthogonal Cross-Sections  |  SuperK-style flush PMT  "
        f"(extended-glass-fix-v4)  |  glass back = {GLASS_BACK:.2f} mm, front = {GLASS_FRONT:.1f} mm",
        color="white", fontsize=10, y=1.01)
    fig.tight_layout()

    fname = os.path.join(out_dir, "geo_cross_sections.png")
    fig.savefig(fname, facecolor=fig.get_facecolor(), dpi=200)
    plt.close(fig)
    print(f"  [✓] Cross-sections        → {fname}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NeutrinoDetSim geometry visualiser")
    parser.add_argument("--out-dir", default="./plots/output")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n NeutrinoDetSim — Geometry Visualisation (extended-glass-fix-v4)")
    print(f" R_inner={R_INNER} mm  R_outer={R_OUTER} mm")
    print(f" Sagitta={SAGITTA:.4f} mm  glass_T_eff={GLASS_T_EFF:.4f} mm")
    print(f" Glass back={GLASS_BACK:.4f} mm  front={GLASS_FRONT:.1f} mm")
    print(f" Coverage={COVERAGE_PCT:.3f}%  Output: {args.out_dir}\n")

    plot_3d_overview(args.out_dir)
    plot_pmt_detail_and_physics(args.out_dir)
    plot_cross_sections(args.out_dir)

    print("\n Done.\n")
