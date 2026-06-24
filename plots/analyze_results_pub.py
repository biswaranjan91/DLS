#!/usr/bin/env python3
"""
analyze_results_pub.py  —  NeutrinoDetSim  (Publication Quality)
=============================================================
Reads results/<medium>_<E>MeV/{summary,timing_hist,wavelength_hist,pmt_hits}.csv
and produces 15 publication-ready figures.

New / upgraded plots vs original:
  01  Light yield vs energy          — dual panel, log-log, σ bands
  02  Detection efficiency           — dual panel with residuals
  03  Cherenkov / scint fractions    — stacked area + line overlay
  04  Bar chart @ 5 MeV             — horizontal, error bars, fractional
  05  Timing: 2×4 grid per medium   — log-y, labelled windows
  06  Early vs late light            — stacked, with C/S annotation
  07  Wavelength spectra             — full fluor chain overlay, QE mask
  08  PMT hit map                    — radar/spider + bar panel
  09  D₂O vs H₂O comparison         — ratio panel, shaded bands
  10  Timing overlay                 — log-y, colour bands, inset zoom
  11  PPO/bis-MSB yield scan        — surface + 1D slices
  12  Heatmap matrix                 — diverging colormap, per-cell text
  13  PPO+bis-MSB energy cascade     — refined schematic
  14  Cherenkov photon budget        — ring pie/sankey per medium
  15  Neutrino physics reach         — signal-to-noise model

Usage:
    python3 analyze_results_pub.py [--results-dir ./results] [--out-dir ./plots]
"""

import os, sys, glob, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import Patch 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Wedge, ConnectionPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize, BoundaryNorm
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  PUBLICATION STYLE
# ══════════════════════════════════════════════════════════════════════════════
FONT = "DejaVu Sans"
plt.rcParams.update({
    "font.family":           FONT,
    "font.size":             11,
    "axes.titlesize":        13,
    "axes.titleweight":      "bold",
    "axes.labelsize":        12,
    "axes.labelweight":      "normal",
    "legend.fontsize":       9.5,
    "legend.framealpha":     0.92,
    "legend.edgecolor":      "#cccccc",
    "legend.fancybox":       True,
    "xtick.labelsize":       10,
    "ytick.labelsize":       10,
    "xtick.direction":       "in",
    "ytick.direction":       "in",
    "xtick.top":             True,
    "ytick.right":           True,
    "xtick.minor.visible":   True,
    "ytick.minor.visible":   True,
    "axes.linewidth":        1.2,
    "axes.grid":             True,
    "grid.alpha":            0.25,
    "grid.linestyle":        "--",
    "grid.linewidth":        0.6,
    "figure.dpi":            150,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.facecolor":     "white",
})

# ══════════════════════════════════════════════════════════════════════════════
#  PALETTE  (Wong 2011 colourblind-safe + extensions)
# ══════════════════════════════════════════════════════════════════════════════
MEDIA    = ["H2O","D2O","WbLS","GdWbLS","TeWbLS","D2OLAB","D2OGdLAB","D2OTeLAB"]
ENERGIES = [0.5, 1.0, 3.0, 5.0, 10.0]

MEDIA_LABELS = {
    "H2O":      r"H$_2$O",
    "D2O":      r"D$_2$O",
    "WbLS":     r"WbLS",
    "GdWbLS":   r"Gd-WbLS",
    "TeWbLS":   r"Te-WbLS",
    "D2OLAB":   r"D$_2$O+LAB$^\star$",
    "D2OGdLAB": r"D$_2$O+Gd-LAB$^\star$",
    "D2OTeLAB": r"D$_2$O+Te-LAB$^\star$",
}
COLORS = {
    "H2O":      "#0072B2",
    "D2O":      "#56B4E9",
    "WbLS":     "#E69F00",
    "GdWbLS":   "#F0E442",
    "TeWbLS":   "#D55E00",
    "D2OLAB":   "#009E73",
    "D2OGdLAB": "#CC79A7",
    "D2OTeLAB": "#000000",
}
MARKERS = {"H2O":"o","D2O":"s","WbLS":"^","GdWbLS":"v",
           "TeWbLS":"D","D2OLAB":"P","D2OGdLAB":"X","D2OTeLAB":"*"}
LINESTYLS = {"H2O":"-","D2O":"--","WbLS":"-","GdWbLS":"-.","TeWbLS":":",
             "D2OLAB":"-","D2OGdLAB":"-.","D2OTeLAB":"--"}

# Colours for photon types
C_CER   = "#0072B2"   # blue
C_SCINT = "#E69F00"   # amber
C_WLS   = "#009E73"   # green

def _mstyle(med, **kw):
    d = {
        "color": COLORS[med],
        "marker": MARKERS[med],
        "ls": LINESTYLS[med],
        "lw": 2.0,
        "ms": 6,
        "mew": 1.2,
    }
    d.update(kw)   # overrides any duplicates
    return d
"""
def _mstyle(med, **kw):
    return dict(color=COLORS[med], marker=MARKERS[med], ls=LINESTYLS[med],
                lw=2.0, ms=6, mew=1.2, **kw)
"""
# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════
def load_summary(rd, med, e):
    p = os.path.join(rd, f"{med}_{e:.1f}MeV", "summary.csv")
    if not os.path.exists(p): return None
    d = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or "," not in line: continue
            k, v = line.split(",", 1)
            try:    d[k.strip()] = float(v.strip())
            except: d[k.strip()] = v.strip()
    return d

def load_timing(rd, med, e):
    p = os.path.join(rd, f"{med}_{e:.1f}MeV", "timing_hist.csv")
    if not os.path.exists(p): return None, None
    df = pd.read_csv(p, comment="#", names=["t","c"])
    return df["t"].values, df["c"].values.astype(float)

def load_wavelength(rd, med, e):
    p = os.path.join(rd, f"{med}_{e:.1f}MeV", "wavelength_hist.csv")
    if not os.path.exists(p): return None, None
    df = pd.read_csv(p, comment="#", names=["wl","c"])
    return df["wl"].values, df["c"].values.astype(float)

def load_pmt_hits(rd, med, e):
    p = os.path.join(rd, f"{med}_{e:.1f}MeV", "pmt_hits.csv")
    if not os.path.exists(p): return None
    return pd.read_csv(p, comment="#", names=["id","axis","hits"])

def load_all(rd):
    d = {m: {} for m in MEDIA}
    for m in MEDIA:
        for e in ENERGIES:
            s = load_summary(rd, m, e)
            if s: d[m][e] = s
    return d

def _get(d, med, e, key, default=0.):
    try: return float(d[med][e].get(key, default))
    except: return default

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def label_panel(ax, letter, x=0.035, y=0.96, fs=14):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=fs,
            fontweight="bold", va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.0))

def despine(ax, which=("top","right")):
    for s in which: ax.spines[s].set_visible(False)

def savefig(fig, path):
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  [✓] {os.path.basename(path)}")

# ══════════════════════════════════════════════════════════════════════════════
#  THEORETICAL SPECTRA  (used by several plots)
# ══════════════════════════════════════════════════════════════════════════════
WL = np.linspace(240, 700, 1000)

def _gauss(wl, mu, sig, A=1.): return A*np.exp(-0.5*((wl-mu)/sig)**2)

PPO_ABS  = _gauss(WL,305,15)+0.35*_gauss(WL,330,18,0.5)
PPO_EMIT = _gauss(WL,360,18)+0.45*_gauss(WL,382,14,0.5)
BIS_ABS  = _gauss(WL,368,16)+0.2*_gauss(WL,348,12,0.3)
BIS_EMIT = _gauss(WL,421,22)+0.3*_gauss(WL,445,16,0.4)
PMT_QE   = np.interp(WL,
    [260,270,290,310,330,350,370,390,410,420,440,460,480,500,520,540,560,580,600,640,680],
    [0.00,0.05,0.10,0.17,0.23,0.27,0.29,0.30,0.30,0.30,0.29,0.26,0.22,0.17,0.13,0.09,0.06,0.04,0.02,0.01,0.00])

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 01  —  Light yield vs energy  (publication dual-panel)
# ══════════════════════════════════════════════════════════════════════════════
def plot01_light_yield(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Optical Photon Production vs Primary Electron Energy",
                 fontsize=14, fontweight="bold", y=1.01)

    ax1, ax2 = axes
    ax1.set_title("Total Photons Produced  (1000 events)")
    ax2.set_title("Mean Photons per Event")

    for med in MEDIA:
        Es, ytot, yevt = [], [], []
        for e in ENERGIES:
            if e not in data[med]: continue
            Es.append(e)
            ytot.append(_get(data,med,e,"total_produced"))
            yevt.append(_get(data,med,e,"mean_produced_per_event"))
        if not Es: continue
        ax1.plot(Es, ytot, **_mstyle(med), label=MEDIA_LABELS[med])
        ax2.plot(Es, yevt, **_mstyle(med), label=MEDIA_LABELS[med])

    for ax, ylabel in zip(axes, ["Total photons produced", "Photons event$^{-1}$"]):
        ax.set_xlabel("Primary electron energy  [MeV]")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks(ENERGIES); ax.set_xticklabels([str(e) for e in ENERGIES])
        ax.legend(ncol=1, fontsize=8.5)
        despine(ax, [])

    label_panel(ax1, "(a)"); label_panel(ax2, "(b)")

    # Highlight Cherenkov threshold band
    for ax in axes:
        ax.axvspan(0.5, 0.78, alpha=0.07, color="navy", lw=0)
        ax.text(0.52, ax.get_ylim()[1]*0.55, "Č\nthresh.", fontsize=7.5,
                color="navy", va="top")

    fig.tight_layout()
    savefig(fig, os.path.join(out, "01_light_yield_vs_energy.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 02  —  Detected photons + efficiency  (with uncertainty bands)
# ══════════════════════════════════════════════════════════════════════════════
def plot02_detected_efficiency(data, out):
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 2, height_ratios=[3, 1, 3], hspace=0.08, wspace=0.32)

    ax_det  = fig.add_subplot(gs[0, 0])
    ax_res  = fig.add_subplot(gs[1, 0], sharex=ax_det)
    ax_eff  = fig.add_subplot(gs[0, 1])
    ax_res2 = fig.add_subplot(gs[1, 1], sharex=ax_eff)
    ax_note = fig.add_subplot(gs[2, :])

  #  fig.suptitle("Detection Performance vs Primary Electron Energy",
   #              fontsize=14, fontweight="bold", y=1.01)

#    ax_det.set_title("Detected Photons per Event")
 #   ax_eff.set_title("Detection Efficiency")

    det_ref = {}  # H2O as reference
    for med in MEDIA:
        Es, det, eff = [], [], []
        for e in ENERGIES:
            if e not in data[med]: continue
            Es.append(e)
            det.append(_get(data,med,e,"mean_detected_per_event"))
            eff.append(_get(data,med,e,"detection_efficiency")*100)
        if not Es: continue
        Es = np.array(Es); det = np.array(det); eff = np.array(eff)
        # Poisson uncertainty band: σ = √N/N_ev
        N_ev = 1000
        det_err = np.sqrt(det * N_ev) / N_ev
        ax_det.plot(Es, det, **_mstyle(med), label=MEDIA_LABELS[med])
        ax_det.fill_between(Es, det-det_err, det+det_err,
                            color=COLORS[med], alpha=0.12)
        ax_eff.plot(Es, eff, **_mstyle(med), label=MEDIA_LABELS[med])

        if med == "H2O": det_ref = dict(zip(Es, det))

    # Residual: ratio to H2O
    for med in MEDIA:
        if med == "H2O": continue
        Es_r, ratio = [], []
        for e in ENERGIES:
            if e not in data[med] or e not in det_ref: continue
            d = _get(data,med,e,"mean_detected_per_event")
            Es_r.append(e); ratio.append(d / max(det_ref.get(e,1e-9),1e-9))
        if Es_r:
            ax_res.plot(Es_r, ratio, color=COLORS[med], marker=MARKERS[med],
                        ls=LINESTYLS[med], lw=1.5, ms=5)
    ax_res.axhline(1.0, color="black", lw=1.0, ls="--")
    ax_res.set_ylabel("Ratio\nto H$_2$O", fontsize=9)

    for med in MEDIA:
        if med == "H2O": continue
        Es_r, eff_vals = [], []
        for e in ENERGIES:
            if e not in data[med]: continue
            Es_r.append(e); eff_vals.append(_get(data,med,e,"detection_efficiency")*100)
        h2o_effs = [_get(data,"H2O",e,"detection_efficiency")*100 for e in ENERGIES if e in data["H2O"]]
        h2o_Es   = [e for e in ENERGIES if e in data["H2O"]]
        if Es_r:
            ref_interp = np.interp(Es_r, h2o_Es, h2o_effs)
            ratio2 = [a/max(b,1e-9) for a,b in zip(eff_vals, ref_interp)]
            ax_res2.plot(Es_r, ratio2, color=COLORS[med], marker=MARKERS[med],
                         ls=LINESTYLS[med], lw=1.5, ms=5)
    ax_res2.axhline(1.0, color="black", lw=1.0, ls="--")
    ax_res2.set_ylabel("Ratio\nto H$_2$O", fontsize=9)

    for ax in [ax_det, ax_eff, ax_res, ax_res2]:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks(ENERGIES)
    for ax in [ax_det, ax_eff]:
        plt.setp(ax.get_xticklabels(), visible=False)
    for ax in [ax_res, ax_res2]:
        ax.set_xlabel("Primary electron energy  [MeV]")

    ax_det.set_ylabel("Detected photons / event")
    ax_eff.set_ylabel("Detection efficiency  [%]")
    ax_det.legend(ncol=2, fontsize=8)
    ax_eff.legend(ncol=2, fontsize=8)

    label_panel(ax_det, "(a)"); label_panel(ax_eff, "(b)")

    """
    # Physics note panel
    ax_note.axis("off")
    note = (
        r"$\bf{Detector\ geometry}$: 6 × Hamamatsu R1307 (bialkali, Ø51.5 mm photocathode), flush-mount in acrylic vessel  ($R_\mathrm{inner}=150$ mm, wall $=5$ mm)."
        "\n"
        r"$\bf{Optical\ path}$: detection medium $\to$ borosilicate glass ($n\approx1.51$, Fresnel reflection $\approx$0.4%) $\to$ photocathode ($\mathrm{QE_{peak}}\approx30\%$ at 420 nm)."
        "  "
        r"$\bf{Reflector}$: 95% PTFE Lambertian on inner acrylic wall (border surface)."
        "\n"
        r"$\star$ D$_2$O-based hybrid media use heavy water as Cherenkov radiator with dissolved LAB+PPO+bis-MSB scintillator."
        "  PMT geometric coverage: 4.42%.  Effective detection efficiency includes transport × QE × geometry."
    )
    ax_note.text(0.01, 0.80, note, transform=ax_note.transAxes, fontsize=9,
                 va="top", ha="left", wrap=True,
                 bbox=dict(facecolor="#f7f7f7", edgecolor="#cccccc",
                           boxstyle="round,pad=0.5"))
    """
    fig.tight_layout()
    savefig(fig, os.path.join(out, "02_detected_and_efficiency.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 03  —  Cherenkov / Scintillation fractions  (stacked area)
# ══════════════════════════════════════════════════════════════════════════════
def plot03_fractions(data, out):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
    axes = axes.flatten()
    fig.suptitle("Cherenkov and Scintillation Photon Fractions vs Energy",
                 fontsize=14, fontweight="bold", y=1.01)

    for idx, med in enumerate(MEDIA):
        ax = axes[idx]
        Es, cf, sf = [], [], []
        for e in ENERGIES:
            if e not in data[med]: continue
            Es.append(e)
            cf.append(_get(data,med,e,"cerenkov_fraction_produced")*100)
            sf.append(_get(data,med,e,"scint_fraction_produced")*100)
        if not Es:
            ax.text(0.5,0.5,"No data",transform=ax.transAxes,ha="center")
            ax.set_title(MEDIA_LABELS[med]); continue

        Es = np.array(Es); cf = np.array(cf); sf = np.array(sf)
        ax.stackplot(Es, cf, sf,
                     labels=["Cherenkov","Scintillation"],
                     colors=[C_CER, C_SCINT], alpha=0.72)
        ax.plot(Es, cf, color=C_CER,   lw=2.0, ls="--")
        ax.plot(Es, sf, color=C_SCINT, lw=2.0, ls="-")
        ax.set_xscale("log")
        ax.set_ylim(0, 105)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks(ENERGIES)
        ax.set_xticklabels([str(e) for e in ENERGIES], fontsize=8)
        ax.set_title(MEDIA_LABELS[med], fontsize=11)
        ax.axhline(50, color="grey", lw=0.8, ls=":", alpha=0.6)
        despine(ax, [])

        if idx >= 4: ax.set_xlabel("Energy [MeV]")
        if idx % 4 == 0: ax.set_ylabel("Photon fraction  [%]")
        if idx == 0:
            ax.legend(loc="center right", fontsize=8.5)

    label_panel(axes[0], "(a)")
    fig.tight_layout()
    savefig(fig, os.path.join(out, "03_cerenkov_scint_fraction.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 04  —  Horizontal bar chart @ 5 MeV  (stacked, annotated)
# ══════════════════════════════════════════════════════════════════════════════
def plot04_bar_5MeV(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(r"Detected Photon Budget @ 5 MeV Electron — All Media",
                 fontsize=14, fontweight="bold", y=1.01)

    E_ref = 5.0
    labels, cer_arr, scint_arr, wls_arr, eff_arr = [], [], [], [], []

    for med in MEDIA:
        if E_ref not in data[med]:
            labels.append(MEDIA_LABELS[med])
            cer_arr.append(0); scint_arr.append(0); wls_arr.append(0); eff_arr.append(0)
            continue
        s = data[med][E_ref]
        labels.append(MEDIA_LABELS[med])
        tot = float(s.get("mean_detected_per_event",0))
        n_tot = float(s.get("total_detected",1))
        n_cer = float(s.get("cerenkov_detected",0))
        n_sc  = float(s.get("scint_detected",0))
        n_wls = float(s.get("wls_detected",0))
        cer_arr.append(n_cer/max(n_tot,1)*tot)
        wls_arr.append(n_wls/max(n_tot,1)*tot)
        scint_arr.append(max(0, n_sc/max(n_tot,1)*tot - n_wls/max(n_tot,1)*tot))
        eff_arr.append(float(s.get("detection_efficiency",0))*100)

    cer_arr = np.array(cer_arr); scint_arr = np.array(scint_arr)
    wls_arr = np.array(wls_arr)
    y = np.arange(len(labels))

    ax1, ax2 = axes
    h = 0.55
    b1 = ax1.barh(y, cer_arr, h, label="Cherenkov",          color=C_CER,   alpha=0.88, edgecolor="white", lw=0.7)
    b2 = ax1.barh(y, scint_arr, h, left=cer_arr,             label="PPO scintillation", color=C_SCINT, alpha=0.88, edgecolor="white", lw=0.7)
    b3 = ax1.barh(y, wls_arr, h, left=cer_arr+scint_arr,     label="bis-MSB WLS",       color=C_WLS,   alpha=0.88, edgecolor="white", lw=0.7)

    # Value labels
    total_arr = cer_arr + scint_arr + wls_arr
    for i, tot in enumerate(total_arr):
        ax1.text(tot + 0.15, i, f"{tot:.1f}", va="center", fontsize=9, color="#333333")

    ax1.set_yticks(y); ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlabel("Mean detected photons / event")
    #ax1.set_title("Photon counts by component")
    ax1.legend(loc="lower right", fontsize=9)
    despine(ax1, ["top","right"])
    label_panel(ax1, "(a)")

    # Right: efficiency + normalised fractions as 100% stacked
    cer_frac   = np.where(total_arr>0, cer_arr/total_arr*100,   0)
    scint_frac = np.where(total_arr>0, scint_arr/total_arr*100, 0)
    wls_frac   = np.where(total_arr>0, wls_arr/total_arr*100,   0)

    ax2.barh(y, cer_frac,   h, label="Cherenkov %",      color=C_CER,   alpha=0.88, edgecolor="white", lw=0.7)
    ax2.barh(y, scint_frac, h, left=cer_frac,            label="PPO %",             color=C_SCINT, alpha=0.88, edgecolor="white", lw=0.7)
    ax2.barh(y, wls_frac,   h, left=cer_frac+scint_frac, label="WLS %",             color=C_WLS,   alpha=0.88, edgecolor="white", lw=0.7)
    ax2.axvline(50, color="grey", lw=1.0, ls=":", alpha=0.7)
    ax2.set_yticks(y); ax2.set_yticklabels(["" for _ in labels])
    ax2.set_xlabel("Fraction of detected photons  [%]")
#    ax2.set_title("Photon component fractions")
    ax2.set_xlim(0, 103)
    ax2.legend(loc="lower right", fontsize=9)

    # Secondary axis: detection efficiency
    ax2b = ax2.twiny()
    ax2b.plot(eff_arr, y, "kD", ms=7, label="Det. efficiency", mew=1.5, mfc="none")
    ax2b.set_xlabel("Detection efficiency  [%]", color="#555555", fontsize=10)
    ax2b.tick_params(axis="x", labelcolor="#555555")
    ax2b.legend(loc="upper right", fontsize=9)
    despine(ax2, ["top","right"])
    label_panel(ax2, "(b)")

    fig.tight_layout()
    savefig(fig, os.path.join(out, "04_bar_detected_5MeV.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 05  —  Timing distributions  (2×4 grid, log-y, window bands)
# ══════════════════════════════════════════════════════════════════════════════
def plot05_timing_grid(rd, out):
    E_ref = 5.0
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharey=False)
    axes = axes.flatten()
    fig.suptitle(r"Photon Arrival Time Distributions @ 5 MeV  —  PPO+bis-MSB Model",
                 fontsize=14, fontweight="bold", y=1.01)

    windows = [(0, 5,  "#0072B2", 0.10, "Cherenkov\nwindow"),
               (5, 30, "#E69F00", 0.10, "PPO fast"),
               (30,150,"#009E73", 0.07, "bis-MSB\nslow")]

    for idx, med in enumerate(MEDIA):
        ax = axes[idx]
        t, c = load_timing(rd, med, E_ref)

        if t is None:
            ax.text(0.5,0.5,"No simulation\ndata",transform=ax.transAxes,
                    ha="center",va="center",fontsize=10,color="#888888")
            ax.set_title(MEDIA_LABELS[med]); continue

        c = gaussian_filter1d(c.astype(float), sigma=1.2)
        c = np.maximum(c, 0)

        for t0, t1, col, al, lbl in windows:
            ax.axvspan(t0, t1, color=col, alpha=al, lw=0)
            if idx == 0:
                ax.text((t0+t1)/2, c.max()*0.7, lbl, ha="center",
                        fontsize=7.5, color=col, alpha=0.9)

        mask = t < 150
        ax.fill_between(t[mask], c[mask], alpha=0.30, color=COLORS[med])
        ax.plot(t[mask], c[mask], color=COLORS[med], lw=1.8)

        # Cherenkov peak marker
        if c[t<3].sum() > 0:
            ipk = np.argmax(c[t<3])
            ax.axvline(t[t<3][ipk], color="navy", lw=1.0, ls="--", alpha=0.7)

        ax.set_yscale("log")
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0.3)
        ax.set_title(MEDIA_LABELS[med], fontsize=11)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
        despine(ax, [])

        if idx >= 4: ax.set_xlabel("Time  [ns]")
        if idx % 4 == 0: ax.set_ylabel("Photon counts  [log scale]")

    fig.tight_layout()
    savefig(fig, os.path.join(out, "05_timing_distributions.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 06  —  Early vs late light  (stacked bar + ratio)
# ══════════════════════════════════════════════════════════════════════════════
def plot06_early_late(rd, out):
    E_ref = 5.0
    t_cut = [0, 3, 15, 150]   # 3 windows: prompt / fast / slow

    media_ok, w_prompt, w_fast, w_slow = [], [], [], []
    for med in MEDIA:
        t, c = load_timing(rd, med, E_ref)
        if t is None: continue
        tot = c.sum(); 
        if tot == 0: continue
        media_ok.append(med)
        w_prompt.append(c[(t>=t_cut[0])&(t<t_cut[1])].sum()/tot*100)
        w_fast.append(  c[(t>=t_cut[1])&(t<t_cut[2])].sum()/tot*100)
        w_slow.append(  c[(t>=t_cut[2])&(t<t_cut[3])].sum()/tot*100)

    if not media_ok:
        print("  [!] Plot 06: no timing data"); return

    x = np.arange(len(media_ok)); w = 0.55
    fig, axes = plt.subplots(2, 1, figsize=(12, 9),
                              gridspec_kw={"height_ratios":[3,1.2],"hspace":0.08})
    ax, ax_r = axes
    fig.suptitle(r"Early/Fast/Slow Light Partition @ 5 MeV",
                 fontsize=14, fontweight="bold", y=1.01)

    b1 = ax.bar(x, w_prompt, w, label=f"Prompt  (0–{t_cut[1]} ns)  Cherenkov",
                color=C_CER,   alpha=0.88, edgecolor="white", lw=0.6)
    b2 = ax.bar(x, w_fast,   w, bottom=np.array(w_prompt),
                label=f"PPO fast  ({t_cut[1]}–{t_cut[2]} ns)",
                color=C_SCINT, alpha=0.88, edgecolor="white", lw=0.6)
    b3 = ax.bar(x, w_slow,   w, bottom=np.array(w_prompt)+np.array(w_fast),
                label=f"bis-MSB slow  ({t_cut[2]}–{t_cut[3]} ns)",
                color=C_WLS,   alpha=0.88, edgecolor="white", lw=0.6)

    for i, (p,f,s) in enumerate(zip(w_prompt,w_fast,w_slow)):
        ax.text(i, p/2,     f"{p:.1f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        if f>5: ax.text(i, p+f/2,  f"{f:.1f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        if s>5: ax.text(i, p+f+s/2,f"{s:.1f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([MEDIA_LABELS[m] for m in media_ok], fontsize=10)
    ax.set_ylabel("Fraction of total detected light  [%]")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9, loc="upper right")
    plt.setp(ax.get_xticklabels(), visible=False)
    despine(ax, [])
    label_panel(ax, "(a)")

    # Ratio: prompt / (fast+slow) — Cherenkov discrimination power
    ratio = [p/(f+s+1e-9) for p,f,s in zip(w_prompt,w_fast,w_slow)]
    ax_r.bar(x, ratio, w, color=C_CER, alpha=0.70, edgecolor="white", lw=0.6)
    ax_r.axhline(1.0, color="grey", lw=1, ls="--")
    ax_r.set_xticks(x)
    ax_r.set_xticklabels([MEDIA_LABELS[m] for m in media_ok], fontsize=10, rotation=15, ha="right")
    ax_r.set_ylabel("Prompt/Delayed\nratio", fontsize=9)
    ax_r.set_xlabel("Detection medium")
    despine(ax_r, [])
    label_panel(ax_r, "(b)")

    fig.tight_layout()
    savefig(fig, os.path.join(out, "06_early_late_light.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 07  —  Wavelength spectra  (with fluor chain + QE overlay)
# ══════════════════════════════════════════════════════════════════════════════
def plot07_wavelength(rd, out):
    fig = plt.figure(figsize=(16, 6))
    gs  = gridspec.GridSpec(1, 3, wspace=0.35, width_ratios=[1.3, 1.3, 1])

    ax1 = fig.add_subplot(gs[0])   # PPO+bis-MSB theoretical
    ax2 = fig.add_subplot(gs[1])   # sim spectra
    ax3 = fig.add_subplot(gs[2])   # QE-weighted convolution

    fig.suptitle("Optical Photon Wavelength Spectra and Fluorophore Chain",
                 fontsize=14, fontweight="bold", y=1.01)

    # Panel A — theoretical spectra
    ax1.fill_between(WL, PPO_ABS,   alpha=0.20, color="#0072B2")
    ax1.plot(WL, PPO_ABS,  color="#0072B2", lw=2.2, ls="--", label="PPO absorption")
    ax1.fill_between(WL, PPO_EMIT,  alpha=0.20, color="#56B4E9")
    ax1.plot(WL, PPO_EMIT, color="#56B4E9", lw=2.2,         label=r"PPO emission  ($\tau\approx2$ ns)")
    ax1.fill_between(WL, BIS_ABS,   alpha=0.20, color="#E69F00")
    ax1.plot(WL, BIS_ABS,  color="#E69F00", lw=2.2, ls="--", label="bis-MSB absorption (WLS)")
    ax1.fill_between(WL, BIS_EMIT,  alpha=0.20, color="#D55E00")
    ax1.plot(WL, BIS_EMIT, color="#D55E00", lw=2.2,         label=r"bis-MSB emission  ($\tau\approx1.5$ ns)")

    # QE (R1307) as shaded right-axis proxy
    ax1b = ax1.twinx()
    ax1b.fill_between(WL, PMT_QE*100, alpha=0.08, color="black")
    ax1b.plot(WL, PMT_QE*100, color="black", lw=1.4, ls=":", label="R1307 QE")
    ax1b.set_ylabel("QE  [%]", fontsize=10, color="grey")
    ax1b.tick_params(axis="y", labelcolor="grey")
    ax1b.set_ylim(0, 55)

    # WLS window
    ax1.axvspan(330, 410, alpha=0.06, color="#56B4E9", lw=0,
                label="PPO→bis-MSB\nWLS window")
    ax1.set_xlim(250, 680)
    ax1.set_xlabel("Wavelength  [nm]")
    ax1.set_ylabel("Normalised intensity")
    ax1.set_title("Fluorophore Spectra + PMT QE")
    handles1  = [Line2D([0],[0],color="#0072B2",lw=2,ls="--",label="PPO absorption"),
                 Line2D([0],[0],color="#56B4E9",lw=2,label="PPO emission"),
                 Line2D([0],[0],color="#E69F00",lw=2,ls="--",label="bis-MSB absorption"),
                 Line2D([0],[0],color="#D55E00",lw=2,label="bis-MSB emission"),
                 Line2D([0],[0],color="black",lw=1.4,ls=":",label="R1307 QE")]
    ax1.legend(handles=handles1, fontsize=8.0, loc="upper right")
    despine(ax1, []); label_panel(ax1, "(a)")

    # Panel B — simulation spectra
    for med in MEDIA:
        wl, c = load_wavelength(rd, med, 5.0)
        if wl is None: continue
        norm = c.max() if c.max()>0 else 1
        s = gaussian_filter1d(c/norm, sigma=1.5)
        ax2.plot(wl, s, **_mstyle(med, marker=''), label=MEDIA_LABELS[med])
        #ax2.plot(wl, s, **_mstyle(med), marker='', label=MEDIA_LABELS[med])
#        ax2.plot(wl, s, **_mstyle(med), ms=0, label=MEDIA_LABELS[med])

    # Shade QE region (where detection is efficient)
    ax2.axvspan(330, 520, alpha=0.07, color="green", lw=0, label="PMT sensitive region")
    ax2.set_xlim(250, 680)
    ax2.set_xlabel("Wavelength  [nm]")
    ax2.set_ylabel("Normalised detected photon flux")
    ax2.set_title("Simulated Spectra @ 5 MeV")
    ax2.legend(ncol=1, fontsize=8)
    despine(ax2, []); label_panel(ax2, "(b)")

    # Panel C — QE-weighted integrated yield per medium
    media_ok, qe_yield = [], []
    for med in MEDIA:
        wl, c = load_wavelength(rd, med, 5.0)
        if wl is None: continue
        qe_i = np.interp(wl, WL, PMT_QE)
        media_ok.append(MEDIA_LABELS[med])
        qe_yield.append((c * qe_i).sum())

    if qe_yield:
        qe_yield = np.array(qe_yield)
        qe_yield /= qe_yield.max()
        y = np.arange(len(media_ok))
        colors = [COLORS[m] for m in MEDIA if _get({m:{}},m,0,"") is not None and
                  any(load_wavelength(rd,m,5.0)[0] is not None for _ in [1])]
        col_list = [COLORS[m] for m in MEDIA if load_wavelength(rd,m,5.0)[0] is not None]
        ax3.barh(y, qe_yield, 0.6, color=col_list[:len(y)], alpha=0.85,
                 edgecolor="white", lw=0.7)
        ax3.set_yticks(y)
        ax3.set_yticklabels(media_ok, fontsize=9)
        ax3.set_xlabel("QE-weighted photon yield\n(normalised)")
        ax3.set_title("QE-Weighted Yield")
        ax3.set_xlim(0, 1.15)
        for i, v in enumerate(qe_yield):
            ax3.text(v+0.02, i, f"{v:.2f}", va="center", fontsize=9)
        despine(ax3, ["top","right"])

    label_panel(ax3, "(c)")
    fig.tight_layout()
    savefig(fig, os.path.join(out, "07_wavelength_spectra.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 08  —  PMT hit maps  (radar + bar, 2×4)
# ══════════════════════════════════════════════════════════════════════════════
def plot08_pmt_hits(rd, out):
    E_ref = 5.0
    AXES_LABELS = ["+x","−x","+y","−y","+z","−z"]
    N = 6
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    angles_closed = np.append(angles, angles[0])

    fig, axes = plt.subplots(2, 4, figsize=(17, 8),
                              subplot_kw=dict(polar=False))
    fig.suptitle(r"PMT Hit Distribution @ 5 MeV  (6 PMTs: $\pm x,\pm y,\pm z$)",
                 fontsize=14, fontweight="bold", y=1.01)

    for idx, med in enumerate(MEDIA):
        ax = axes.flatten()[idx]
        df = load_pmt_hits(rd, med, E_ref)

        if df is None:
            ax.text(0.5,0.5,"No data",transform=ax.transAxes,ha="center")
            ax.set_title(MEDIA_LABELS[med]); continue

        hits = df["hits"].values[:6].astype(float)
        mean_h = hits.mean(); std_h = hits.std()
        x = np.arange(N)

        bars = ax.bar(x, hits, 0.6, color=COLORS[med], alpha=0.80,
                      edgecolor="white", lw=0.7)
        ax.axhline(mean_h, color="crimson", lw=1.5, ls="--",
                   label=f"Mean = {mean_h:.0f}")
        ax.axhspan(mean_h-std_h, mean_h+std_h, alpha=0.10, color="crimson")

        # Isotropy χ²
        chi2 = np.sum((hits-mean_h)**2/max(mean_h,1))
        ax.text(0.98, 0.97, f"$\\chi^2/N = {chi2/N:.2f}$",
                transform=ax.transAxes, fontsize=8, ha="right", va="top",
                color="grey")
        ax.set_xticks(x); ax.set_xticklabels(AXES_LABELS, fontsize=9)
        ax.set_title(MEDIA_LABELS[med], fontsize=11)
        if idx % 4 == 0: ax.set_ylabel("PMT hit counts")
        if idx >= 4: ax.set_xlabel("PMT axis")
        ax.legend(fontsize=8, loc="upper left")
        despine(ax, [])

    fig.tight_layout()
    savefig(fig, os.path.join(out, "08_pmt_hit_map.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 09  —  D₂O hybrid vs H₂O-based comparison  (3-panel + shaded band)
# ══════════════════════════════════════════════════════════════════════════════
def plot09_d2o_comparison(data, out):
    pairs = [
        ("WbLS",   "D2OLAB",   r"WbLS  vs  D$_2$O+LAB"),
        ("GdWbLS", "D2OGdLAB", r"Gd-WbLS  vs  D$_2$O+Gd-LAB"),
        ("TeWbLS", "D2OTeLAB", r"Te-WbLS  vs  D$_2$O+Te-LAB"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9),
                              gridspec_kw={"height_ratios":[3,1.2],"hspace":0.10})
    fig.suptitle(r"D$_2$O Hybrid Media vs H$_2$O-Based WbLS: Detected Photons / Event",
                 fontsize=14, fontweight="bold", y=1.01)

    for col, (m1, m2, title) in enumerate(pairs):
        ax  = axes[0, col]
        axr = axes[1, col]
        ax.set_title(title, fontsize=11)

        def series(med):
            Es, det = [], []
            for e in ENERGIES:
                if e not in data[med]: continue
                Es.append(e); det.append(_get(data,med,e,"mean_detected_per_event"))
            return np.array(Es), np.array(det)

        Es1, d1 = series(m1); Es2, d2 = series(m2)
        N_ev = 1000
        err1 = np.sqrt(d1*N_ev)/N_ev; err2 = np.sqrt(d2*N_ev)/N_ev

        ax.plot(Es1, d1, **_mstyle(m1), label=MEDIA_LABELS[m1])
        ax.fill_between(Es1, d1-err1, d1+err1, color=COLORS[m1], alpha=0.13)
        ax.plot(Es2, d2, **_mstyle(m2), label=MEDIA_LABELS[m2])
        ax.fill_between(Es2, d2-err2, d2+err2, color=COLORS[m2], alpha=0.13)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks(ENERGIES)
        plt.setp(ax.get_xticklabels(), visible=False)
        if col == 0: ax.set_ylabel("Detected photons event$^{-1}$")
        ax.legend(fontsize=8.5)
        despine(ax, [])
        label_panel(ax, ["(a)","(b)","(c)"][col])

        # Ratio panel
        if len(Es1)==len(Es2) and len(Es1)>0:
            ratio = d2 / np.maximum(d1, 1e-9)
            ratio_err = ratio * np.sqrt((err2/np.maximum(d2,1e-9))**2 + (err1/np.maximum(d1,1e-9))**2)
            axr.plot(Es1, ratio, color=COLORS[m2], marker=MARKERS[m2],
                     ls=LINESTYLS[m2], lw=1.8, ms=5)
            axr.fill_between(Es1, ratio-ratio_err, ratio+ratio_err,
                             color=COLORS[m2], alpha=0.15)
        axr.axhline(1.0, color="grey", lw=1.0, ls="--")
        axr.set_xscale("log")
        axr.xaxis.set_major_formatter(mticker.ScalarFormatter())
        axr.set_xticks(ENERGIES); axr.set_xticklabels([str(e) for e in ENERGIES])
        axr.set_xlabel("Energy  [MeV]")
        if col == 0: axr.set_ylabel("D$_2$O/H$_2$O\nratio", fontsize=9)
        despine(axr, [])

    fig.tight_layout()
    savefig(fig, os.path.join(out, "09_d2o_vs_wbls_comparison.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 10  —  Timing overlay  (log-y, inset zoom)
# ══════════════════════════════════════════════════════════════════════════════
def plot10_timing_overlay(rd, out):
    E_ref    = 5.0
    selected = ["H2O","D2O","WbLS","D2OLAB","D2OTeLAB"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1, ax2  = axes
    fig.suptitle(r"Photon Timing Overlay @ 5 MeV — Key Media Comparison",
                 fontsize=14, fontweight="bold", y=1.01)

    for med in selected:
        t, c = load_timing(rd, med, E_ref)
        if t is None: continue
        c = gaussian_filter1d(c.astype(float), sigma=1.0)
        tot = c.sum(); 
        if tot==0: continue

        # Full window
        m = t < 120
        ax1.plot(t[m], c[m]/tot, **_mstyle(med, marker=''), label=MEDIA_LABELS[med])
        #ax1.plot(t[m], c[m]/tot, **_mstyle(med), ms=0, label=MEDIA_LABELS[med])
        # Early window
        m2 = t < 25
        ax2.plot(t[m2], c[m2]/tot, **_mstyle(med, marker=''), label=MEDIA_LABELS[med])

    for ax in [ax1, ax2]:
        ax.set_yscale("log")
        ax.set_xlabel("Time  [ns]")
        ax.set_ylabel("Normalised photon rate")
        ax.legend(fontsize=9)
        ax.axvspan(0,  3,  alpha=0.07, color=C_CER,   lw=0)
        ax.axvspan(3,  30, alpha=0.06, color=C_SCINT, lw=0)
        ax.axvspan(30, 120,alpha=0.04, color=C_WLS,   lw=0)
        despine(ax, [])

    ax1.set_title("Full timing window  (0–120 ns)")
    ax2.set_title(r"Early window  (0–25 ns)  — Cherenkov isolation")

    # Window labels on ax1
    for t_c, lbl, col in [(1.5,"Cherenkov\nprompt", C_CER),
                           (16, "PPO fast\n(τ≈5 ns)", C_SCINT),
                           (75, "bis-MSB\n(τ≈50 ns)", C_WLS)]:
        ax1.text(t_c, ax1.get_ylim()[0]*2.5, lbl, ha="center", fontsize=8,
                 color=col, style="italic")

    label_panel(ax1, "(a)"); label_panel(ax2, "(b)")
    fig.tight_layout()
    savefig(fig, os.path.join(out, "10_timing_overlay.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 11  —  PPO/bis-MSB yield scan  (2D surface + 1D slices)
# ══════════════════════════════════════════════════════════════════════════════
def plot11_yield_scan(out):
    ppo_v   = np.linspace(0.1, 5.0, 120)
    bism_v  = np.linspace(0.005, 0.5, 100)
    PPO, BIS = np.meshgrid(ppo_v, bism_v)

    LAB = 0.0116

    def yield_model(ppo, bis, lab=LAB, boost=1.0):
        eta_ppo = ppo / (ppo + 0.4)
        eta_wls = 0.95 * (1 - np.exp(-bis / 0.02))
        return 15000. * lab * eta_ppo * eta_wls * boost

    Y_wbls   = yield_model(PPO, BIS, boost=1.00)
    Y_d2olab = yield_model(PPO, BIS, boost=1.12)

    fig = plt.figure(figsize=(16, 6))
    gs  = gridspec.GridSpec(1, 3, wspace=0.35)

    # Panel A: 2D contour for WbLS
    ax1 = fig.add_subplot(gs[0])
    lvls = np.linspace(0, Y_wbls.max(), 18)
    cs = ax1.contourf(PPO, BIS, Y_wbls, levels=lvls, cmap="YlOrRd")
    ax1.contour(PPO, BIS, Y_wbls, levels=lvls[::3], colors="white", alpha=0.35, linewidths=0.7)
    plt.colorbar(cs, ax=ax1, label="Scintillation yield  [ph MeV$^{-1}$]")
    ax1.axvline(1.0,  color="white", lw=1.5, ls="--", alpha=0.8, label="Default PPO")
    ax1.axhline(0.05, color="white", lw=1.5, ls=":",  alpha=0.8, label="Default bis-MSB")
    ax1.scatter([1.0],[0.05], color="white", s=80, zorder=5, edgecolors="black")
    ax1.set_xlabel("PPO concentration  [g L$^{-1}$]")
    ax1.set_ylabel("bis-MSB concentration  [g L$^{-1}$]")
    ax1.set_title("WbLS scintillation yield\nvs concentrations")
    ax1.legend(fontsize=8.5, facecolor="grey", labelcolor="white")
    despine(ax1, []); label_panel(ax1, "(a)")

    # Panel B: 1D PPO scan at fixed bis-MSB
    ax2 = fig.add_subplot(gs[1])
    for bv, al in [(0.02, 0.5), (0.05, 1.0), (0.20, 0.7)]:
        ax2.plot(ppo_v, yield_model(ppo_v, bv, boost=1.00),
                 color=COLORS["WbLS"],   lw=2, alpha=al,
                 label=f"WbLS  bis-MSB={bv} g/L" if al==1. else f"WbLS  bis={bv}")
        ax2.plot(ppo_v, yield_model(ppo_v, bv, boost=1.12),
                 color=COLORS["D2OLAB"], lw=2, alpha=al, ls="--",
                 label=f"D₂O+LAB  bis-MSB={bv}" if al==1. else "")
    ax2.axvline(1.0, color="grey", lw=1, ls=":", label="Default PPO = 1 g/L")
    ax2.set_xlabel("PPO concentration  [g L$^{-1}$]")
    ax2.set_ylabel("Effective yield  [ph MeV$^{-1}$]")
    ax2.set_title("Yield vs PPO\n(bis-MSB varied)")
    ax2.legend(fontsize=8, ncol=1)
    despine(ax2, []); label_panel(ax2, "(b)")

    # Panel C: 1D bis-MSB scan at fixed PPO=1
    ax3 = fig.add_subplot(gs[2])
    for pv, al in [(0.5, 0.5), (1.0, 1.0), (3.0, 0.7)]:
        ax3.plot(bism_v, yield_model(pv, bism_v, boost=1.00),
                 color=COLORS["WbLS"], lw=2, alpha=al, label=f"WbLS  PPO={pv}")
        ax3.plot(bism_v, yield_model(pv, bism_v, boost=1.12),
                 color=COLORS["D2OLAB"], lw=2, alpha=al, ls="--",
                 label=f"D₂O+LAB  PPO={pv}")
    ax3.axvline(0.05, color="grey", lw=1, ls=":", label="Default bis-MSB = 0.05 g/L")
    ax3.set_xlabel("bis-MSB concentration  [g L$^{-1}$]")
    ax3.set_ylabel("Effective yield  [ph MeV$^{-1}$]")
    ax3.set_title("Yield vs bis-MSB\n(PPO varied)")
    ax3.legend(fontsize=8)
    despine(ax3, []); label_panel(ax3, "(c)")

    fig.suptitle("PPO + bis-MSB Scintillation Yield Model  (MaterialsWithPPO)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, os.path.join(out, "11_yield_model_ppo_scan.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 12  —  Summary heatmap  (diverging, annotated)
# ══════════════════════════════════════════════════════════════════════════════
def plot12_heatmap(data, out):
    metrics = [
        ("mean_detected_per_event",    "Detected  [ph evt$^{-1}$]",   "YlOrRd",  None),
        ("detection_efficiency",       "Det. efficiency  [%]",         "Blues",   100.),
        ("cerenkov_fraction_produced", "Cherenkov fraction  [%]",      "PuBu",    100.),
        ("scint_fraction_produced",    "Scint. fraction  [%]",         "YlOrBr",  100.),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Performance Matrix: All Media × All Energies",
                 fontsize=14, fontweight="bold", y=1.01)

    for ax, (metric, mlabel, cmap, scale) in zip(axes.flatten(), metrics):
        mat = np.zeros((len(MEDIA), len(ENERGIES)))
        for i, med in enumerate(MEDIA):
            for j, e in enumerate(ENERGIES):
                v = _get(data, med, e, metric, 0.)
                mat[i, j] = v * (scale if scale else 1.)

        im = ax.imshow(mat, aspect="auto", cmap=cmap,
                       vmin=0, vmax=mat.max()+1e-9)
        ax.set_xticks(range(len(ENERGIES)))
        ax.set_xticklabels([f"{e} MeV" for e in ENERGIES], fontsize=9.5)
        ax.set_yticks(range(len(MEDIA)))
        ax.set_yticklabels([MEDIA_LABELS[m] for m in MEDIA], fontsize=10)
        ax.set_title(mlabel, fontsize=12)

        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.ax.tick_params(labelsize=8.5)

        thresh = mat.max() * 0.60
        for i in range(len(MEDIA)):
            for j in range(len(ENERGIES)):
                v = mat[i,j]
                fmt = f"{v:.1f}" if v >= 0.5 else f"{v:.3f}"
                ax.text(j, i, fmt, ha="center", va="center", fontsize=8,
                        color="white" if v > thresh else "#222222",
                        fontweight="bold" if v > thresh else "normal")

    fig.tight_layout()
    savefig(fig, os.path.join(out, "12_summary_heatmap.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 13  —  PPO + bis-MSB energy cascade  (publication schematic)
# ══════════════════════════════════════════════════════════════════════════════
def plot13_cascade(out):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6)
    ax.axis("off")
    fig.suptitle("PPO + bis-MSB Wavelength-Shifting Energy Transfer Cascade in WbLS / D$_2$O+LAB",
                 fontsize=13, fontweight="bold", y=1.01)

    BOX_H, BOX_W = 1.20, 1.55
    boxes = [
        (0.7,  2.9, r"Primary track" "\n" r"$e^-$, $\gamma$", "#d0d8f0"),
        (2.8,  2.9, r"LAB solvent" "\n" r"excited",           "#c8e8f8"),
        (5.0,  4.2, r"PPO" "\n" r"S$_1$ excited",            "#b8f0b8"),
        (5.0,  1.6, r"PPO emission" "\n" r"360 nm" "\n" r"$\tau \approx 2$ ns  FAST","#80e080"),
        (8.5,  4.2, r"bis-MSB" "\n" r"absorbs" "\n" r"350–410 nm",      "#fff0a0"),
        (8.5,  1.6, r"bis-MSB" "\n" r"emission" "\n" r"420 nm" "\n" r"$\tau \approx 1.5$ ns","#ffd040"),
        (11.8, 3.0, r"Hamamatsu" "\n" r"R1307" "\n" r"QE≈30%@420nm",   "#e8e8e8"),
    ]
    for bx, by, lbl, col in boxes:
        rect = FancyBboxPatch((bx-BOX_W/2, by-BOX_H/2), BOX_W, BOX_H,
                               boxstyle="round,pad=0.10",
                               facecolor=col, edgecolor="#555555", lw=1.4)
        ax.add_patch(rect)
        ax.text(bx, by, lbl, ha="center", va="center", fontsize=9, color="#111111")

    def arrow(x0,y0,x1,y1,lbl="",col="black",rad=0.0):
        ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8,
                                    connectionstyle=f"arc3,rad={rad}",
                                    mutation_scale=16))
        if lbl:
            mx,my = (x0+x1)/2, (y0+y1)/2
            ax.text(mx, my+0.18, lbl, ha="center", fontsize=8.5, color=col)

    arrow(0.7+BOX_W/2, 2.9, 2.8-BOX_W/2, 2.9, "Förster\ntransfer")
    arrow(2.8+BOX_W/2, 3.2, 5.0-BOX_W/2, 4.2, "excitation",   col="#009E73", rad=-0.2)
    arrow(5.0,          4.2-BOX_H/2, 5.0, 1.6+BOX_H/2, r"hν PPO",  col="#009E73")
    arrow(5.0+BOX_W/2, 1.6, 8.5-BOX_W/2, 1.6, "WLS\nabsorption", col="#E69F00")
    arrow(8.5,          4.2-BOX_H/2, 8.5, 1.6+BOX_H/2, r"",        col="#E69F00")
    arrow(8.5-BOX_W/2, 4.2, 5.0+BOX_W/2, 4.2, "re-absorb\n368 nm", col="#888888", rad=0.0)
    arrow(8.5+BOX_W/2, 1.6, 11.8-BOX_W/2, 1.6+0.5, "hν  420 nm\nto PMT", col="#D55E00")
    arrow(2.8+BOX_W/2, 2.6, 5.0-BOX_W/2, 1.6, "Cherenkov\nphoton direct", col="#0072B2", rad=0.25)
    arrow(0.7+BOX_W/2, 2.6, 11.8-BOX_W/2, 2.6, "Cherenkov direct (H₂O/D₂O)", col="#0072B2", rad=0.30)

    # Wavelength ruler
    ax.annotate("", xy=(11.0,0.45), xytext=(4.5,0.45),
                arrowprops=dict(arrowstyle="<->",color="#555555",lw=1.5))
    ax.text(7.8, 0.18, "Spectral shift: 360 nm (PPO) → 420 nm (bis-MSB) → R1307 peak sensitivity",
            ha="center", fontsize=8.5, color="#444444", style="italic")

    fig.tight_layout()
    savefig(fig, os.path.join(out, "13_ppo_bismsb_chain.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 14  —  Photon budget: pie chart + Sankey-style per medium (NEW)
# ══════════════════════════════════════════════════════════════════════════════
def plot14_photon_budget(data, out):
    E_ref = 5.0
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    axes = axes.flatten()
    fig.suptitle(r"Photon Fate Budget @ 5 MeV — Production vs Detection by Channel",
                 fontsize=14, fontweight="bold", y=1.01)

    for idx, med in enumerate(MEDIA):
        ax = axes[idx]
        if E_ref not in data[med]:
            ax.text(0.5,0.5,"No data",transform=ax.transAxes,ha="center")
            ax.set_title(MEDIA_LABELS[med]); ax.axis("off"); continue

        s = data[med][E_ref]
        tot_prod = float(s.get("total_produced",1))
        cer_prod = float(s.get("cerenkov_produced",0))
        sci_prod = float(s.get("scint_produced",0))
        tot_det  = float(s.get("total_detected",0))
        cer_det  = float(s.get("cerenkov_detected",0))
        sci_det  = float(s.get("scint_detected",0))
        wls_det  = float(s.get("wls_detected",0))

        eff_c = cer_det/max(cer_prod,1)*100
        eff_s = sci_det/max(sci_prod,1)*100
        absorbed = tot_prod - tot_det
        cer_abs  = cer_prod - cer_det
        sci_abs  = sci_prod - sci_det

        # Nested pie: outer=fate, inner=source
        outer_sizes = [cer_det, sci_det, absorbed]
        outer_colors= [C_CER, C_SCINT, "#cccccc"]
        outer_labels= [f"Cer det\n{cer_det:.0f}", f"Sci det\n{sci_det:.0f}", f"Absorbed\n{absorbed:.0f}"]

        inner_sizes = [cer_prod, sci_prod]
        inner_colors= [C_CER+"88", C_SCINT+"88"]
        
        wedge_kw = dict(wedgeprops=dict(edgecolor="white", linewidth=1.2))
        #wedge_kw = dict(edgecolor="white", linewidth=1.2)
        ax.pie(outer_sizes, radius=1.0, colors=outer_colors,
               startangle=90, **wedge_kw,
               autopct=lambda p: f"{p:.0f}%" if p>4 else "",
               pctdistance=0.78, textprops={"fontsize":8})
        ax.pie(inner_sizes, radius=0.58, colors=inner_colors,
               startangle=90, **wedge_kw)

        ax.set_title(MEDIA_LABELS[med], fontsize=11)
        ax.text(0, -1.28, f"Produced: {tot_prod:.0f}  |  Detected: {tot_det:.0f}\n"
                           f"Eff(Č)={eff_c:.1f}%  Eff(S)={eff_s:.1f}%",
                ha="center", fontsize=7.8, color="#333333")

        if idx == 0:
            handles = [
                Patch(facecolor=C_CER,     label="Cherenkov detected"),
                Patch(facecolor=C_SCINT,   label="Scint. detected"),
                Patch(facecolor="#cccccc", label="Absorbed / lost"),
                Patch(facecolor=C_CER+"88",label="Cherenkov produced (inner)"),
                Patch(facecolor=C_SCINT+"88",label="Scint. produced (inner)"),
            ]
            ax.legend(handles=handles, loc="upper left",
                      bbox_to_anchor=(-0.05,1.55), fontsize=7.5, framealpha=0.9)

    fig.tight_layout()
    savefig(fig, os.path.join(out, "14_photon_budget.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 15  —  Neutrino physics reach: S/√B vs energy  (NEW, physics model)
# ══════════════════════════════════════════════════════════════════════════════
def plot15_physics_reach(data, out):
    """
    Simplified figure-of-merit: signal/√background for three hypothetical
    neutrino channels, using the detected photon counts as proxy for
    energy resolution and vertex reconstruction quality.
    """
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.35)

    channels = [
        ("IBD:  $\\bar{\\nu}_e + p \\to e^+ + n$",
         "Reactor / geo-neutrino signal",  0.95,  1.8),
        ("ES:  $\\nu + e^- \\to \\nu + e^-$",
         "Solar $^8$B / HEP",              0.50,  2.5),
        (r"NC:  $\nu + ^{12}$C $\to \nu + ^{12}$C$^*$ (15.1 MeV $\gamma$)",
         "Supernova / atmospheric",        0.70,  5.0),
    ]

    for ci, (chan_title, sub, signal_eff, E_ref_ch) in enumerate(channels):
        ax  = fig.add_subplot(gs[0, ci])
        axb = fig.add_subplot(gs[1, ci])

        ax.set_title(f"{chan_title}\n({sub})", fontsize=9.5, pad=4)
        axb.set_title("Energy resolution proxy", fontsize=9.5, pad=4)

        for med in MEDIA:
            Es_ok, snr, dE_E = [], [], []
            for e in ENERGIES:
                if e not in data[med]: continue
                n_det = _get(data,med,e,"mean_detected_per_event")
                if n_det < 0.5: continue
                Es_ok.append(e)
                # S/√B model: signal_eff * √(N_det) / scale
                sig   = signal_eff * np.sqrt(n_det * 1000)
                bg    = np.sqrt(max(n_det * 1000 * 0.1, 1))
                snr.append(sig / bg)
                # Energy resolution: σ/E ≈ a/√N_det
                dE_E.append(0.03 / np.sqrt(n_det) * 100)  # in %

            if not Es_ok: continue
            ax.plot(Es_ok, snr, **_mstyle(med), label=MEDIA_LABELS[med])
            axb.plot(Es_ok, dE_E, **_mstyle(med), label=MEDIA_LABELS[med])

        ax.set_xlabel("Electron energy  [MeV]")
        ax.set_ylabel("S/√B  [arbitrary units]")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks(ENERGIES)
        ax.legend(fontsize=7.5, ncol=1)
        despine(ax, [])
        label_panel(ax, ["(a)","(b)","(c)"][ci])

        axb.set_xlabel("Electron energy  [MeV]")
        axb.set_ylabel(r"$\sigma_E/E$  [%]")
        axb.set_xscale("log")
        axb.xaxis.set_major_formatter(mticker.ScalarFormatter())
        axb.set_xticks(ENERGIES)
        axb.set_yscale("log")
        axb.legend(fontsize=7.5)
        despine(axb, [])
        label_panel(axb, ["(d)","(e)","(f)"][ci])

    fig.suptitle("Neutrino Physics Reach: Signal-to-Noise and Energy Resolution Proxy\n"
                 r"(Model: $S/\sqrt{B} \propto \epsilon_{\rm sig}\sqrt{N_{\rm det}}$, "
                 r"$\sigma_E/E \propto N_{\rm det}^{-1/2}$)",
                 fontsize=13, fontweight="bold", y=1.01)
    savefig(fig, os.path.join(out, "15_neutrino_physics_reach.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="NeutrinoDetSim — publication plots")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--out-dir",     default="./plots")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n NeutrinoDetSim — Publication Analysis")
    print(f" Results : {args.results_dir}")
    print(f" Output  : {args.out_dir}\n")

    data    = load_all(args.results_dir)
    n_loaded = sum(len(v) for v in data.values())
    print(f" Loaded {n_loaded} run summaries ({len(MEDIA)} media × {len(ENERGIES)} energies)\n")

    # Always-run (model / theory)
    plot11_yield_scan(args.out_dir)
    plot13_cascade(args.out_dir)

    if n_loaded == 0:
        print(" [!] No simulation data — only model plots produced.\n"); return

    plot01_light_yield(data, args.out_dir)
    plot02_detected_efficiency(data, args.out_dir)
    plot03_fractions(data, args.out_dir)
    plot04_bar_5MeV(data, args.out_dir)
    plot05_timing_grid(args.results_dir, args.out_dir)
    plot06_early_late(args.results_dir, args.out_dir)
    plot07_wavelength(args.results_dir, args.out_dir)
    plot08_pmt_hits(args.results_dir, args.out_dir)
    plot09_d2o_comparison(data, args.out_dir)
    plot10_timing_overlay(args.results_dir, args.out_dir)
    plot12_heatmap(data, args.out_dir)
    plot14_photon_budget(data, args.out_dir)
    plot15_physics_reach(data, args.out_dir)

    print(f"\n All 15 plots written to: {args.out_dir}/\n")

if __name__ == "__main__":
    main()
