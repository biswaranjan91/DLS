#!/usr/bin/env python3
"""
analyze_results.py
==================
Comprehensive analysis and plotting for NeutrinoDetSim output.

Reads CSV files from results/<medium>_<energy>MeV/ directories and produces:
  1.  Light yield vs energy (all media)
  2.  Detected photons vs energy
  3.  Detection efficiency vs energy
  4.  Cherenkov fraction vs scintillation fraction
  5.  Photon arrival time distributions (per medium, per energy)
  6.  Early vs late light separation
  7.  Wavelength spectra of detected photons
  8.  Bar plots: detected photons at 5 MeV (all media)
  9.  PMT multiplicity and hit map
  10. PPO + bis-MSB emission spectra overlay (theoretical)
  11. D2O vs H2O hybrid comparison matrix
  12. Scintillator fraction scan (yield model)

Usage:
    python3 analyze_results.py [--results-dir ./results] [--out-dir ./plots]
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d

# ── Aesthetics ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})


# ── Media definitions ─────────────────────────────────────────────────────────
MEDIA = ["H2O", "D2O", "WbLS", "GdWbLS", "TeWbLS", "D2OLAB", "D2OGdLAB", "D2OTeLAB"]
ENERGIES = [0.5, 1.0, 3.0, 5.0, 10.0]

MEDIA_LABELS = {
    "H2O":      r"H$_2$O",
    "D2O":      r"D$_2$O",
    "WbLS":     r"WbLS (H$_2$O+LAB)",
    "GdWbLS":   r"Gd-WbLS",
    "TeWbLS":   r"Te-WbLS",
    "D2OLAB":   r"D$_2$O+LAB ★",
    "D2OGdLAB": r"D$_2$O+Gd-LAB ★",
    "D2OTeLAB": r"D$_2$O+Te-LAB ★",
}

# Colour palette (colourblind-friendly)
COLORS = {
    "H2O":      "#1f77b4",
    "D2O":      "#aec7e8",
    "WbLS":     "#ff7f0e",
    "GdWbLS":   "#ffbb78",
    "TeWbLS":   "#d62728",
    "D2OLAB":   "#2ca02c",
    "D2OGdLAB": "#98df8a",
    "D2OTeLAB": "#9467bd",
}

MARKERS = {
    "H2O":      "o",  "D2O":      "s",
    "WbLS":     "^",  "GdWbLS":   "v",
    "TeWbLS":   "D",  "D2OLAB":   "P",
    "D2OGdLAB": "X",  "D2OTeLAB": "*",
}

LINESTYLES = {
    "H2O":      "-",  "D2O":      "--",
    "WbLS":     "-",  "GdWbLS":   "-.",
    "TeWbLS":   ":",  "D2OLAB":   "-",
    "D2OGdLAB": "-.", "D2OTeLAB": "--",
}

# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_summary(results_dir, medium, energy):
    """Load summary.csv for one (medium, energy) pair."""
    path = os.path.join(results_dir, f"{medium}_{energy:.1f}MeV", "summary.csv")
    if not os.path.exists(path):
        return None
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or "," not in line:
                continue
            k, v = line.split(",", 1)
            try:
                data[k.strip()] = float(v.strip())
            except ValueError:
                data[k.strip()] = v.strip()
    return data


def load_timing(results_dir, medium, energy):
    path = os.path.join(results_dir, f"{medium}_{energy:.1f}MeV", "timing_hist.csv")
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path, comment="#", names=["time_ns", "counts"])
    return df["time_ns"].values, df["counts"].values


def load_wavelength(results_dir, medium, energy):
    path = os.path.join(results_dir, f"{medium}_{energy:.1f}MeV", "wavelength_hist.csv")
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path, comment="#", names=["wl_nm", "counts"])
    return df["wl_nm"].values, df["counts"].values


def load_pmt_hits(results_dir, medium, energy):
    path = os.path.join(results_dir, f"{medium}_{energy:.1f}MeV", "pmt_hits.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, comment="#", names=["pmt_id", "axis", "hits"])
    return df


def load_all_summaries(results_dir):
    """Load all (medium, energy) summaries into a nested dict."""
    data = {m: {} for m in MEDIA}
    for m in MEDIA:
        for e in ENERGIES:
            s = load_summary(results_dir, m, e)
            if s:
                data[m][e] = s
    return data


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 1 – Total photons produced vs energy
# ─────────────────────────────────────────────────────────────────────────────
def plot_light_yield(data, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax1, ax2 = axes
    ax1.set_title("Total Optical Photons Produced vs Electron Energy")
    ax2.set_title("Mean Photons / Event vs Energy")

    for med in MEDIA:
        Es, totProd, perEvt = [], [], []
        for e in ENERGIES:
            if e in data[med]:
                s = data[med][e]
                Es.append(e)
                totProd.append(float(s.get("total_produced", 0)))
                perEvt.append(float(s.get("mean_produced_per_event", 0)))
        if not Es:
            continue
        ax1.plot(Es, totProd, color=COLORS[med], marker=MARKERS[med],
                 ls=LINESTYLES[med], label=MEDIA_LABELS[med], lw=1.8)
        ax2.plot(Es, perEvt, color=COLORS[med], marker=MARKERS[med],
                 ls=LINESTYLES[med], label=MEDIA_LABELS[med], lw=1.8)

    for ax in axes:
        ax.set_xlabel("Electron Energy [MeV]")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(loc="upper left", fontsize=8)
    ax1.set_ylabel("Total Photons (all events)")
    ax2.set_ylabel("Mean Photons / Event")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_light_yield_vs_energy.png"))
    plt.close(fig)
    print("  [✓] Plot 01: light yield vs energy")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 2 – Detected photons vs energy + detection efficiency
# ─────────────────────────────────────────────────────────────────────────────
def plot_detected_and_efficiency(data, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1, ax2 = axes
    ax1.set_title("Detected Photons / Event vs Energy")
    ax2.set_title("Detection Efficiency vs Energy")

    for med in MEDIA:
        Es, det, eff = [], [], []
        for e in ENERGIES:
            if e in data[med]:
                s = data[med][e]
                Es.append(e)
                det.append(float(s.get("mean_detected_per_event", 0)))
                eff.append(float(s.get("detection_efficiency", 0)) * 100.)
        if not Es:
            continue
        ax1.plot(Es, det, color=COLORS[med], marker=MARKERS[med],
                 ls=LINESTYLES[med], label=MEDIA_LABELS[med], lw=1.8)
        ax2.plot(Es, eff, color=COLORS[med], marker=MARKERS[med],
                 ls=LINESTYLES[med], label=MEDIA_LABELS[med], lw=1.8)

    ax1.set_xlabel("Electron Energy [MeV]")
    ax1.set_ylabel("Detected Photons / Event")
    ax1.set_xscale("log")
    ax1.legend(loc="upper left", fontsize=8)

    ax2.set_xlabel("Electron Energy [MeV]")
    ax2.set_ylabel("Detection Efficiency [%]")
    ax2.set_xscale("log")
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_detected_and_efficiency.png"))
    plt.close(fig)
    print("  [✓] Plot 02: detected photons + efficiency")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 3 – Cherenkov vs Scintillation fraction
# ─────────────────────────────────────────────────────────────────────────────
def plot_cerenkov_scint_fraction(data, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1, ax2 = axes
    ax1.set_title("Cherenkov Fraction of Produced Photons")
    ax2.set_title("Scintillation Fraction of Produced Photons")

    for med in MEDIA:
        Es, cf, sf = [], [], []
        for e in ENERGIES:
            if e in data[med]:
                s = data[med][e]
                Es.append(e)
                cf.append(float(s.get("cerenkov_fraction_produced", 0)) * 100.)
                sf.append(float(s.get("scint_fraction_produced", 0)) * 100.)
        if not Es:
            continue
        ax1.plot(Es, cf, color=COLORS[med], marker=MARKERS[med],
                 ls=LINESTYLES[med], label=MEDIA_LABELS[med], lw=1.8)
        ax2.plot(Es, sf, color=COLORS[med], marker=MARKERS[med],
                 ls=LINESTYLES[med], label=MEDIA_LABELS[med], lw=1.8)

    for ax, title in zip(axes, ["Cherenkov fraction [%]", "Scintillation fraction [%]"]):
        ax.set_xlabel("Electron Energy [MeV]")
        ax.set_ylabel(title)
        ax.set_xscale("log")
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_cerenkov_scint_fraction.png"))
    plt.close(fig)
    print("  [✓] Plot 03: Cherenkov vs scintillation fraction")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 4 – Bar chart: detected photons at 5 MeV, all media
# ─────────────────────────────────────────────────────────────────────────────
def plot_bar_at_5MeV(data, out_dir):
    E_ref = 5.0
    labels, total, scint, cerenkov = [], [], [], []
    for med in MEDIA:
        if E_ref in data[med]:
            s = data[med][E_ref]
            labels.append(MEDIA_LABELS[med])
            total.append(float(s.get("mean_detected_per_event", 0)))
            n_tot  = float(s.get("total_detected", 1))
            n_scin = float(s.get("scint_detected", 0))
            n_cer  = float(s.get("cerenkov_detected", 0))
            scint.append(n_scin / max(n_tot, 1) * total[-1])
            cerenkov.append(n_cer / max(n_tot, 1) * total[-1])
        else:
            labels.append(MEDIA_LABELS[med])
            total.append(0); scint.append(0); cerenkov.append(0)

    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(12, 6))
    cer_arr   = np.array(cerenkov)
    scin_arr  = np.array(scint)
    wls_arr   = np.array(total) - cer_arr - scin_arr
    wls_arr   = np.clip(wls_arr, 0, None)

    ax.bar(x, cer_arr,  width, label="Cherenkov",     color="#1f77b4")
    ax.bar(x, scin_arr, width, bottom=cer_arr,         label="Scintillation (PPO fast)", color="#ff7f0e")
    ax.bar(x, wls_arr,  width, bottom=cer_arr+scin_arr, label="WLS / bis-MSB slow",     color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean Detected Photons / Event")
    ax.set_title("Detected Photons by Component @ 5 MeV Electron — All Media")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_bar_detected_5MeV.png"))
    plt.close(fig)
    print("  [✓] Plot 04: bar chart at 5 MeV")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 5 – Timing distributions (per medium at 5 MeV)
# ─────────────────────────────────────────────────────────────────────────────
def plot_timing_distributions(results_dir, out_dir):
    E_ref = 5.0
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=False)
    axes = axes.flatten()
    fig.suptitle(r"Photon Arrival Time Distributions @ 5 MeV — PPO+bis-MSB Model", y=1.01)

    for idx, med in enumerate(MEDIA):
        ax = axes[idx]
        t, counts = load_timing(results_dir, med, E_ref)
        if t is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            ax.set_title(MEDIA_LABELS[med])
            continue

        # Smooth slightly for display
        smooth = gaussian_filter1d(counts.astype(float), sigma=1.5)
        ax.fill_between(t, smooth, alpha=0.35, color=COLORS[med])
        ax.plot(t, smooth, color=COLORS[med], lw=1.5)

        # Mark PPO fast peak (~2-3 ns) and bis-MSB slow peak (~12 ns)
        ax.axvline(x=2.5,  color="navy",   lw=1, ls="--", alpha=0.7, label="PPO fast peak")
        ax.axvline(x=12.0, color="darkorange", lw=1, ls=":", alpha=0.7, label="bis-MSB peak")

        ax.set_xlim(0, 80)
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Photon counts")
        ax.set_title(MEDIA_LABELS[med], fontsize=10)
        if idx == 0:
            ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_timing_distributions.png"))
    plt.close(fig)
    print("  [✓] Plot 05: timing distributions")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 6 – Early (Cherenkov) vs late (scintillation) window comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_early_late_light(results_dir, out_dir):
    """Integrate timing histogram in early [0-5 ns] and late [5-100 ns] windows."""
    E_ref    = 5.0
    t_split  = 5.0   # ns
    t_max    = 100.0  # ns

    media_list, early_frac, late_frac = [], [], []
    for med in MEDIA:
        t, counts = load_timing(results_dir, med, E_ref)
        if t is None:
            continue
        mask_e = t < t_split
        mask_l = (t >= t_split) & (t < t_max)
        total  = counts.sum()
        if total == 0:
            continue
        media_list.append(MEDIA_LABELS[med])
        early_frac.append(counts[mask_e].sum() / total * 100.)
        late_frac.append(counts[mask_l].sum()  / total * 100.)

    x = np.arange(len(media_list))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.2, early_frac, 0.4,
           label=f"Early light (0–{t_split:.0f} ns) — Cherenkov dominated",
           color="#1f77b4", alpha=0.85)
    ax.bar(x + 0.2, late_frac, 0.4,
           label=f"Late light ({t_split:.0f}–{t_max:.0f} ns) — PPO+bis-MSB",
           color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(media_list, rotation=30, ha="right")
    ax.set_ylabel("Fraction of Total Detected Light [%]")
    ax.set_title(r"Early vs Late Light Separation @ 5 MeV — PPO+bis-MSB Timing Model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_early_late_light.png"))
    plt.close(fig)
    print("  [✓] Plot 06: early vs late light")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 7 – Wavelength spectra (PPO + bis-MSB theoretical + simulation)
# ─────────────────────────────────────────────────────────────────────────────
def plot_wavelength_spectra(results_dir, out_dir):
    # PPO and bis-MSB theoretical spectra (from MaterialsWithPPO)
    wl_grid = np.linspace(250, 650, 400)

    def gaussian_peak(wl, peak, sigma, amp=1.0):
        return amp * np.exp(-0.5 * ((wl - peak) / sigma)**2)

    # PPO: absorption ~305 nm, emission ~360 nm
    ppo_abs  = gaussian_peak(wl_grid, 305, 15, 1.0) + 0.3*gaussian_peak(wl_grid, 330, 20, 0.5)
    ppo_emit = gaussian_peak(wl_grid, 360, 20, 1.0) + 0.4*gaussian_peak(wl_grid, 380, 15, 0.5)
    # bis-MSB: absorption ~368 nm, emission ~420 nm
    bisms_abs  = gaussian_peak(wl_grid, 368, 18, 1.0)
    bisms_emit = gaussian_peak(wl_grid, 420, 22, 1.0) + 0.3*gaussian_peak(wl_grid, 445, 18, 0.4)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: theoretical fluor spectra
    ax = axes[0]
    ax.fill_between(wl_grid, ppo_abs,   alpha=0.25, color="blue",   label="PPO absorption")
    ax.plot(wl_grid, ppo_abs,   color="blue",   lw=2, ls="--")
    ax.fill_between(wl_grid, ppo_emit,  alpha=0.25, color="cyan",   label="PPO emission (fast, τ≈2 ns)")
    ax.plot(wl_grid, ppo_emit,  color="cyan",   lw=2)
    ax.fill_between(wl_grid, bisms_abs, alpha=0.25, color="orange", label="bis-MSB absorption (WLS)")
    ax.plot(wl_grid, bisms_abs, color="orange", lw=2, ls="--")
    ax.fill_between(wl_grid, bisms_emit,alpha=0.25, color="red",    label="bis-MSB emission (slow, τ≈1.5 ns)")
    ax.plot(wl_grid, bisms_emit,color="red",    lw=2)

    # PMT QE overlay (normalised)
    pmt_qe = np.interp(wl_grid,
                       [270,290,310,330,350,370,390,410,420,440,460,480,500,520,540,560,580,600,620,640],
                       [0.05,0.10,0.17,0.23,0.27,0.29,0.30,0.30,0.30,0.29,0.26,0.22,0.17,0.13,0.09,0.06,0.04,0.02,0.01,0.00])
    ax.plot(wl_grid, pmt_qe, color="black", lw=1.5, ls=":", label="PMT QE (Hamamatsu bialkali)")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Normalised intensity / QE")
    ax.set_title("PPO + bis-MSB Fluor Chain — Energy Transfer Cascade")
    ax.axvspan(330, 410, alpha=0.07, color="cyan",  label="PPO→bis-MSB WLS window")
    ax.legend(fontsize=8)

    # Right: simulation wavelength spectra for selected media at 5 MeV
    ax = axes[1]
    selected = ["H2O", "D2O", "WbLS", "D2OLAB", "D2OTeLAB"]
    for med in selected:
        wl, counts = load_wavelength(results_dir, med, 5.0)
        if wl is None:
            continue
        norm = counts.max() if counts.max() > 0 else 1
        smooth = gaussian_filter1d(counts / norm, sigma=1.0)
        ax.plot(wl, smooth, color=COLORS[med], lw=2,
                ls=LINESTYLES[med], label=MEDIA_LABELS[med])

    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Normalised detected photon spectrum")
    ax.set_title("Detected Photon Wavelength Spectra @ 5 MeV")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "07_wavelength_spectra.png"))
    plt.close(fig)
    print("  [✓] Plot 07: wavelength spectra + PPO/bis-MSB chain")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 8 – PMT hit maps
# ─────────────────────────────────────────────────────────────────────────────
def plot_pmt_hits(results_dir, out_dir):
    E_ref = 5.0
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()
    fig.suptitle("PMT Hit Distribution @ 5 MeV (6 PMTs: ±x,±y,±z axes)")

    axes_labels = ["+x", "−x", "+y", "−y", "+z", "−z"]
    x = np.arange(6)

    for idx, med in enumerate(MEDIA):
        ax = axes[idx]
        df = load_pmt_hits(results_dir, med, E_ref)
        if df is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            ax.set_title(MEDIA_LABELS[med])
            continue
        hits = df["hits"].values[:6]
        ax.bar(x, hits, color=COLORS[med], alpha=0.85, edgecolor="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(axes_labels)
        ax.set_title(MEDIA_LABELS[med], fontsize=10)
        ax.set_xlabel("PMT axis")
        if idx % 4 == 0:
            ax.set_ylabel("Hit counts")
        # Isotropy line
        if hits.sum() > 0:
            mean_hits = hits.mean()
            ax.axhline(mean_hits, color="red", ls="--", lw=1, alpha=0.7, label="Mean")
            ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "08_pmt_hit_map.png"))
    plt.close(fig)
    print("  [✓] Plot 08: PMT hit maps")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 9 – D2O hybrid vs WbLS comparison matrix
# ─────────────────────────────────────────────────────────────────────────────
def plot_d2o_vs_wbls_comparison(data, out_dir):
    pairs = [
        ("WbLS",   "D2OLAB",   r"WbLS vs D$_2$O+LAB"),
        ("GdWbLS", "D2OGdLAB", r"Gd-WbLS vs D$_2$O+Gd-LAB"),
        ("TeWbLS", "D2OTeLAB", r"Te-WbLS vs D$_2$O+Te-LAB"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(r"D$_2$O Hybrid Media vs H$_2$O-based WbLS: Detected Photons / Event")

    for ax, (m1, m2, title) in zip(axes, pairs):
        Es1, det1 = [], []
        Es2, det2 = [], []
        for e in ENERGIES:
            if e in data[m1]:
                Es1.append(e)
                det1.append(float(data[m1][e].get("mean_detected_per_event", 0)))
            if e in data[m2]:
                Es2.append(e)
                det2.append(float(data[m2][e].get("mean_detected_per_event", 0)))

        ax.plot(Es1, det1, color=COLORS[m1], marker=MARKERS[m1],
                ls="-",  lw=2, label=MEDIA_LABELS[m1])
        ax.plot(Es2, det2, color=COLORS[m2], marker=MARKERS[m2],
                ls="--", lw=2, label=MEDIA_LABELS[m2])

        # Ratio secondary y-axis
        if Es1 and Es2 and len(det1) == len(det2):
            ax2 = ax.twinx()
            ratio = [d2/max(d1, 1e-9) for d1, d2 in zip(det1, det2)]
            ax2.plot(Es1, ratio, color="grey", ls=":", lw=1.5, label="Ratio D₂O/H₂O")
            ax2.axhline(1.0, color="grey", lw=0.8, ls="-")
            ax2.set_ylabel("Ratio (D₂O / H₂O)", color="grey", fontsize=9)
            ax2.tick_params(axis="y", labelcolor="grey")

        ax.set_xscale("log")
        ax.set_xlabel("Electron Energy [MeV]")
        if ax == axes[0]:
            ax.set_ylabel("Detected Photons / Event")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "09_d2o_vs_wbls_comparison.png"))
    plt.close(fig)
    print("  [✓] Plot 09: D2O hybrid vs WbLS comparison")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 10 – Timing overlay: Cherenkov window (H2O vs WbLS vs D2O+LAB)
# ─────────────────────────────────────────────────────────────────────────────
def plot_timing_overlay(results_dir, out_dir):
    E_ref    = 5.0
    selected = ["H2O", "D2O", "WbLS", "D2OLAB", "D2OTeLAB"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax1, ax2 = axes
    ax1.set_title("Full Timing Window (0–100 ns)")
    ax2.set_title("Early Window: Cherenkov vs PPO fast (0–20 ns)")

    for med in selected:
        t, counts = load_timing(results_dir, med, E_ref)
        if t is None:
            continue
        smooth = gaussian_filter1d(counts.astype(float), sigma=1.0)
        total  = smooth.sum() if smooth.sum() > 0 else 1.

        mask_full  = t < 100
        mask_early = t < 20

        ax1.plot(t[mask_full],  smooth[mask_full]  / total, color=COLORS[med],
                 lw=1.8, ls=LINESTYLES[med], label=MEDIA_LABELS[med])
        ax2.plot(t[mask_early], smooth[mask_early] / total, color=COLORS[med],
                 lw=1.8, ls=LINESTYLES[med], label=MEDIA_LABELS[med])

    for ax in axes:
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Normalised photon rate")
        ax.legend(fontsize=9)
        ax.axvline(x=2.5,  color="navy",   lw=0.8, ls="--", alpha=0.5)
        ax.axvline(x=12.0, color="darkorange", lw=0.8, ls="--", alpha=0.5)

    ax1.text(2.7,  ax1.get_ylim()[1]*0.85, "PPO\nfast",   fontsize=8, color="navy")
    ax1.text(12.3, ax1.get_ylim()[1]*0.70, "bis-MSB\nWLS", fontsize=8, color="darkorange")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "10_timing_overlay.png"))
    plt.close(fig)
    print("  [✓] Plot 10: timing overlay")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 11 – Scintillator fraction yield scan (analytical model from
#             MaterialsWithPPO::ComputeEffectiveYield)
# ─────────────────────────────────────────────────────────────────────────────
def plot_yield_model_scan(out_dir):
    ppo_conc = np.linspace(0.1, 5.0, 200)
    bisms_conc = 0.05  # fixed

    def compute_yield(ppo, bisms, lab_frac, boost=1.0):
        kBase = 15000.
        eta_ppo  = ppo / (ppo + 0.4)
        eta_wls  = 0.95 * (1 - np.exp(-bisms / 0.02))
        return kBase * lab_frac * eta_ppo * eta_wls * boost

    lab_frac = 0.0116
    y_wbls   = compute_yield(ppo_conc, bisms_conc, lab_frac, boost=1.00)
    y_d2olab = compute_yield(ppo_conc, bisms_conc, lab_frac, boost=1.12)
    y_d2otel = compute_yield(ppo_conc, bisms_conc, lab_frac, boost=1.12*1.08)
    y_gdwbls = compute_yield(ppo_conc, bisms_conc, lab_frac, boost=0.95)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax1 = axes[0]
    ax1.plot(ppo_conc, y_wbls,   color=COLORS["WbLS"],     lw=2, label=MEDIA_LABELS["WbLS"])
    ax1.plot(ppo_conc, y_gdwbls, color=COLORS["GdWbLS"],   lw=2, ls="--", label=MEDIA_LABELS["GdWbLS"])
    ax1.plot(ppo_conc, y_d2olab, color=COLORS["D2OLAB"],   lw=2.5, label=MEDIA_LABELS["D2OLAB"])
    ax1.plot(ppo_conc, y_d2otel, color=COLORS["D2OTeLAB"], lw=2.5, ls="--", label=MEDIA_LABELS["D2OTeLAB"])
    ax1.axvline(1.0, ls=":", color="grey", lw=1, label="Default 1 g/L")
    ax1.set_xlabel("PPO concentration [g/L]")
    ax1.set_ylabel("Effective scintillation yield [ph/MeV]")
    ax1.set_title("Yield vs PPO Concentration (bis-MSB = 0.05 g/L fixed)")
    ax1.legend(fontsize=9)

    # Scan bis-MSB at fixed PPO=1 g/L
    bisms_scan = np.linspace(0.005, 0.5, 200)
    ppo_fixed  = 1.0
    yw2  = compute_yield(ppo_fixed, bisms_scan, lab_frac, boost=1.00)
    yd2o = compute_yield(ppo_fixed, bisms_scan, lab_frac, boost=1.12)

    ax2 = axes[1]
    ax2.plot(bisms_scan, yw2,  color=COLORS["WbLS"],   lw=2, label=r"WbLS (H$_2$O)")
    ax2.plot(bisms_scan, yd2o, color=COLORS["D2OLAB"], lw=2, label=r"D$_2$O+LAB")
    ax2.axvline(0.05, ls=":", color="grey", lw=1, label="Default 0.05 g/L")
    ax2.set_xlabel("bis-MSB concentration [g/L]")
    ax2.set_ylabel("Effective scintillation yield [ph/MeV]")
    ax2.set_title("Yield vs bis-MSB Concentration (PPO = 1 g/L fixed)")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "11_yield_model_ppo_scan.png"))
    plt.close(fig)
    print("  [✓] Plot 11: PPO/bis-MSB yield scan (analytical model)")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 12 – Summary heat map: all media × all energies
# ─────────────────────────────────────────────────────────────────────────────
def plot_summary_heatmap(data, out_dir):
    metrics = ["mean_detected_per_event", "detection_efficiency",
               "cerenkov_fraction_produced", "scint_fraction_produced"]
    metric_labels = ["Detected ph/evt", "Det. Efficiency", "Cherenkov frac.", "Scint. frac."]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        matrix = np.zeros((len(MEDIA), len(ENERGIES)))
        for i, med in enumerate(MEDIA):
            for j, e in enumerate(ENERGIES):
                if e in data[med] and metric in data[med][e]:
                    matrix[i, j] = float(data[med][e][metric])
                    if metric == "detection_efficiency":
                        matrix[i, j] *= 100.

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(ENERGIES)))
        ax.set_xticklabels([f"{e} MeV" for e in ENERGIES])
        ax.set_yticks(range(len(MEDIA)))
        ax.set_yticklabels([MEDIA_LABELS[m] for m in MEDIA], fontsize=9)
        ax.set_title(mlabel)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate cells
        for i in range(len(MEDIA)):
            for j in range(len(ENERGIES)):
                v = matrix[i, j]
                ax.text(j, i, f"{v:.1f}" if v >= 0.1 else f"{v:.3f}",
                        ha="center", va="center", fontsize=7,
                        color="black" if matrix[i, j] < matrix.max()*0.7 else "white")

    fig.suptitle("Performance Matrix: All Media × All Energies", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "12_summary_heatmap.png"))
    plt.close(fig)
    print("  [✓] Plot 12: summary heatmap")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 13 – PPO + bis-MSB WLS energy transfer chain (schematic)
# ─────────────────────────────────────────────────────────────────────────────
def plot_ppo_bismsb_chain(out_dir):
    """Visualise the PPO → bis-MSB Förster transfer energy cascade."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("PPO + bis-MSB (b-MIS) Energy Transfer Cascade in WbLS/D₂O+LAB",
                 fontsize=12, fontweight="bold", pad=12)

    # Boxes
    boxes = [
        (1.0, 2.5, "Primary\nionisation\n(e⁻ track)", "#ccccff"),
        (3.0, 2.5, "LAB\nsolvent\nexcited", "#cceeff"),
        (5.0, 2.5, "PPO\nexcited\n(fast fluor)", "#ccffcc"),
        (7.5, 3.8, "PPO\nemission\n~360 nm\nτ ≈ 2 ns\n(FAST)", "#aaffaa"),
        (7.5, 1.2, "bis-MSB\nexcited\n(WLS)", "#ffeeaa"),
        (10.0, 1.2, "bis-MSB\nemission\n~420 nm\nτ ≈ 1.5 ns\n(SLOW)", "#ffcc44"),
    ]

    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    for bx, by, label, color in boxes:
        box = FancyBboxPatch((bx-0.75, by-0.70), 1.5, 1.4,
                              boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor="black", lw=1.2)
        ax.add_patch(box)
        ax.text(bx, by, label, ha="center", va="center", fontsize=8)

    # Arrows
    arrow_kw = dict(arrowstyle="->", color="black", lw=1.5)
    arrows = [
        ((1.75, 2.5), (2.2, 2.5), "Förster\ntransfer"),
        ((3.75, 2.5), (4.2, 2.5), "Absorption"),
        ((5.75, 2.5), (6.0, 3.4), "hν PPO"),
        ((5.75, 2.5), (6.0, 1.6), "WLS abs."),
        ((6.75, 3.8), (7.15, 3.8), ""),
        ((6.75, 1.2), (7.15, 1.2), ""),
        ((8.3,  1.2), (9.2,  1.2), "re-emit"),
    ]
    for start, end, label in arrows:
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        if label:
            mx, my = (start[0]+end[0])/2, (start[1]+end[1])/2 + 0.2
            ax.text(mx, my, label, ha="center", fontsize=7, color="darkblue")

    # PMT icon
    ax.add_patch(plt.Rectangle((11.0, 0.3), 0.5, 4.4,
                                facecolor="#dddddd", edgecolor="black", lw=2))
    ax.text(11.25, 2.5, "PMT\nQE≈30%\n@420nm", ha="center", va="center", fontsize=8)

    # Wavelength ruler
    ax.annotate("", xy=(10.5, 0.3), xytext=(7.5, 0.3),
                arrowprops=dict(arrowstyle="<->", color="purple", lw=1.5))
    ax.text(9.0, 0.05, "420 nm peak → bialkali PMT peak sensitivity",
            ha="center", fontsize=8, color="purple")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "13_ppo_bismsb_chain.png"))
    plt.close(fig)
    print("  [✓] Plot 13: PPO+bis-MSB chain diagram")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NeutrinoDetSim analysis")
    parser.add_argument("--results-dir", default="./results",
                        help="Directory containing simulation output CSVs")
    parser.add_argument("--out-dir",     default="./plots",
                        help="Directory to write plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n NeutrinoDetSim Analysis")
    print(f" Results : {args.results_dir}")
    print(f" Output  : {args.out_dir}\n")

    # Load all summaries
    data = load_all_summaries(args.results_dir)
    n_loaded = sum(len(v) for v in data.values())
    print(f" Loaded {n_loaded} run summaries across {len(MEDIA)} media.\n")

    # Always generate theoretical / model plots (no data needed)
    plot_ppo_bismsb_chain(args.out_dir)
    plot_yield_model_scan(args.out_dir)

    if n_loaded == 0:
        print(" [!] No simulation data found – only model plots generated.")
        print("     Run the simulation first, then re-run this script.\n")
        return

    # Data-dependent plots
    plot_light_yield               (data,             args.out_dir)
    plot_detected_and_efficiency   (data,             args.out_dir)
    plot_cerenkov_scint_fraction   (data,             args.out_dir)
    plot_bar_at_5MeV               (data,             args.out_dir)
    plot_timing_distributions      (args.results_dir, args.out_dir)
    plot_early_late_light          (args.results_dir, args.out_dir)
    plot_wavelength_spectra        (args.results_dir, args.out_dir)
    plot_pmt_hits                  (args.results_dir, args.out_dir)
    plot_d2o_vs_wbls_comparison    (data,             args.out_dir)
    plot_timing_overlay            (args.results_dir, args.out_dir)
    plot_summary_heatmap           (data,             args.out_dir)

    print(f"\n All plots written to: {args.out_dir}/\n")


if __name__ == "__main__":
    main()
