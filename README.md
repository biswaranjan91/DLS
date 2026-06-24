# NeutrinoDetSim — Build & Run Instructions
## Hybrid Neutrino Media: Water, WbLS, D₂O Systems with PPO+bis-MSB

---

## 1. Prerequisites

### Required Software
| Package | Version | Notes |
|---------|---------|-------|
| Geant4 | ≥ 11.1 | Built with OpenGL vis, Qt UI, multithreading |
| ROOT | ≥ 6.26 | For analysis macros |
| CMake | ≥ 3.16 | Build system |
| GCC/Clang | ≥ C++17 | Standard library support |
| Python | ≥ 3.9 | Analysis and geometry plots |
| Python packages | — | numpy, matplotlib, pandas, scipy |

### Install Python dependencies
```bash
pip install numpy matplotlib pandas scipy
```

### Verify Geant4 installation
```bash
source /path/to/geant4/install/bin/geant4.sh   # or geant4.csh
geant4-config --version   # should print 11.x.x
```

### Verify ROOT installation
```bash
source /path/to/root/bin/thisroot.sh
root --version   # should print 6.x
```

---

## 2. Project Structure

```
NeutrinoDetSim/
├── NeutrinoDetSim.cc          # Main driver
├── CMakeLists.txt             # CMake build file
├── include/
│   ├── DetectorConstruction.hh
│   ├── DetectorMessenger.hh
│   ├── MaterialsWithPPO.hh    # PPO + bis-MSB optical model
│   ├── PPOIntegratorPatch.hh  # Applies fluor chain to media
│   ├── PhysicsList.hh
│   ├── ActionInitialization.hh
│   ├── PrimaryGeneratorAction.hh
│   ├── RunAction.hh
│   ├── EventAction.hh
│   ├── SteppingAction.hh
│   ├── TrackingAction.hh
│   ├── PMTSD.hh
│   ├── PMTHit.hh
│   └── RunManager.hh
├── src/
│   ├── DetectorConstruction.cc
│   ├── DetectorMessenger.cc
│   ├── MaterialsWithPPO.cc    # Full PPO/bis-MSB spectra + WLS chain
│   ├── PPOIntegratorPatch.cc
│   ├── PhysicsList.cc
│   ├── ActionInitialization.cc
│   ├── PrimaryGeneratorAction.cc
│   ├── RunAction.cc
│   ├── EventAction.cc
│   ├── SteppingAction.cc
│   ├── TrackingAction.cc
│   ├── PMTSD.cc
│   └── PMTHit.cc
├── macros/
│   ├── run_all.mac            # Master macro: all 40 configs
│   ├── run_medium.mac         # Per-medium template
│   ├── ppo_scan.mac           # PPO concentration scan
│   ├── init_vis.mac           # Interactive visualisation
│   └── generate_macros.sh     # Script to generate individual .mac files
├── analysis/
│   └── AnalyzeRun.C           # ROOT analysis macro
└── plots/
    ├── analyze_results.py     # Python: all physics plots
    └── visualize_geometry.py  # Python: detector geometry + tracks
```

---

## 3. Build

```bash
# ── Step 1: Create build directory (outside source) ──────────────────────────
cd NeutrinoDetSim
mkdir build && cd build

# ── Step 2: Configure with CMake ─────────────────────────────────────────────
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGeant4_DIR=/path/to/geant4/install/lib/Geant4-11.x \
  -DROOT_DIR=/path/to/root/cmake

# If Geant4 and ROOT are already on your PATH/cmake path:
cmake .. -DCMAKE_BUILD_TYPE=Release

# ── Step 3: Compile ───────────────────────────────────────────────────────────
make -j$(nproc)          # Linux / macOS
# or:  make -j8
# Expected output: [ 100%] Built target NeutrinoDetSim

# ── Step 4: Verify executable ────────────────────────────────────────────────
ls -lh NeutrinoDetSim
./NeutrinoDetSim --help   # should print Geant4 help (no crash = good)
```

---

## 4. Generate Individual Run Macros

```bash
# From build/ directory:
bash macros/generate_macros.sh
# Creates: macros/runs/H2O_0.5MeV.mac, WbLS_5.0MeV.mac, D2OTeLAB_10.0MeV.mac, ...
ls macros/runs/ | wc -l   # should show 40
```

---

## 5. Run the Simulation

### Option A — Single medium, single energy (quickest test)
```bash
# From build/ directory:
./NeutrinoDetSim macros/runs/WbLS_5.0MeV.mac

# Output written to: results/WbLS_5.0MeV/
#   summary.csv, timing_hist.csv, wavelength_hist.csv, energy_hist.csv, pmt_hits.csv
```

### Option B — All media, all energies (full study)
```bash
# All 40 configurations sequentially (~30 min at 1000 events each):
./NeutrinoDetSim macros/run_all.mac

# For faster runs, edit run_all.mac or individual .mac files:
# Change:  /run/beamOn 1000
# To:      /run/beamOn 10000   (better statistics, ~10× slower)
```

### Option C — Run specific media in parallel
```bash
# Use GNU parallel or background processes:
./NeutrinoDetSim macros/runs/H2O_5.0MeV.mac    &
./NeutrinoDetSim macros/runs/WbLS_5.0MeV.mac   &
./NeutrinoDetSim macros/runs/D2OLAB_5.0MeV.mac &
./NeutrinoDetSim macros/runs/D2OTeLAB_5.0MeV.mac &
wait

# Or with GNU parallel:
ls macros/runs/*.mac | parallel -j4 ./NeutrinoDetSim {}
```

### Option D — PPO concentration scan
```bash
./NeutrinoDetSim macros/ppo_scan.mac
```

### Option E — Interactive visualisation (requires OpenGL)
```bash
# Launch without arguments → interactive mode
./NeutrinoDetSim
# In Geant4 session prompt:
Idle> /control/execute macros/init_vis.mac
Idle> /det/medium D2OTeLAB
Idle> /run/initialize
Idle> /gun/energy 5 MeV
Idle> /run/beamOn 50
Idle> /vis/viewer/flush
```

---

## 6. Check Output

After running, verify output structure:
```bash
find results/ -name "summary.csv" | head -5
cat results/WbLS_5.0MeV/summary.csv
# Expected lines:
#   medium,WbLS
#   primary_energy_MeV,5.000
#   total_produced,XXXXXX
#   scint_produced,XXXXXX
#   cerenkov_produced,XXXXXX
#   total_detected,XXXX
#   detection_efficiency,0.XXXXX
#   ...

cat results/WbLS_5.0MeV/timing_hist.csv
# time_ns,counts
# 0,XXX
# 1,XXX
# ...
```

---

## 7. Generate Geometry Plots (no simulation needed)

```bash
# From build/ or project root:
python3 plots/visualize_geometry.py --out-dir plots/output

# Generates 3 figures:
#   plots/output/geo_3d_detector.png      — 3D view + tracks
#   plots/output/geo_physics_detail.png   — Cherenkov cone, spectra, timing
#   plots/output/geo_cross_sections.png   — XY, XZ, YZ cross-sections
```

---

## 8. Run Python Analysis (after simulation)

```bash
python3 plots/analyze_results.py \
  --results-dir ./results \
  --out-dir     ./plots/output

# Generates 13 plots:
#   01_light_yield_vs_energy.png
#   02_detected_and_efficiency.png
#   03_cerenkov_scint_fraction.png
#   04_bar_detected_5MeV.png
#   05_timing_distributions.png
#   06_early_late_light.png
#   07_wavelength_spectra.png
#   08_pmt_hit_map.png
#   09_d2o_vs_wbls_comparison.png
#   10_timing_overlay.png
#   11_yield_model_ppo_scan.png
#   12_summary_heatmap.png
#   13_ppo_bismsb_chain.png
```

---

## 9. Run ROOT Analysis

```bash
# From build/ directory:
root -l -b -q 'analysis/AnalyzeRun.C("./results")'

# Generates in ./root_plots/:
#   C1_LightYield.pdf/.png
#   C2_Efficiency.pdf/.png
#   C3_Timing.pdf/.png
#   C4_Wavelength.pdf/.png
#   C5_Bar5MeV.pdf/.png
```

---

## 10. Complete Workflow (one-shot)

```bash
#!/bin/bash
# run_full_study.sh — Full end-to-end workflow

set -e

echo "=== NeutrinoDetSim Full Study ==="

# 1. Build
cd /path/to/NeutrinoDetSim
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 2. Generate macros
bash macros/generate_macros.sh

# 3. Run geometry visualisation (no simulation needed)
python3 ../plots/visualize_geometry.py --out-dir plots/output
echo "[✓] Geometry plots done"

# 4. Run simulation (all 40 configurations)
echo "Starting simulation runs..."
./NeutrinoDetSim macros/run_all.mac
echo "[✓] Simulation complete"

# 5. Python analysis
python3 ../plots/analyze_results.py \
  --results-dir ./results \
  --out-dir     ./plots/output
echo "[✓] Python analysis done"

# 6. ROOT analysis
root -l -b -q "../analysis/AnalyzeRun.C(\"./results\")"
echo "[✓] ROOT analysis done"

echo ""
echo "=== All outputs in build/plots/output/ and build/root_plots/ ==="
ls plots/output/*.png | wc -l
ls root_plots/*.png   | wc -l
```

```bash
chmod +x run_full_study.sh
./run_full_study.sh
```

---

## 11. Macro Command Reference

```
# Detector commands (from DetectorMessenger):
/det/medium  H2O          # Set medium (H2O|D2O|WbLS|GdWbLS|TeWbLS|D2OLAB|D2OGdLAB|D2OTeLAB)
/det/radius  15 cm        # Sphere radius (default 15 cm)

# Gun commands (Geant4 built-in):
/gun/particle  e-         # Particle type (e-|gamma|neutron|proton)
/gun/energy    5.0 MeV    # Kinetic energy

# Run:
/run/initialize           # Must call after /det/medium change
/run/beamOn   1000        # Number of primary events

# Verbosity (keep at 0 for batch):
/run/verbose     0
/event/verbose   0
/tracking/verbose 0

# Optical physics parameters (G4OpticalParameters):
/process/optical/cerenkov/setMaxPhotons      300
/process/optical/scintillation/setYieldFactor 1.0
```

---

## 12. Expected Physics Results (reference)

| Medium | 5 MeV e⁻ | Cher. fraction | Scint. yield | Notes |
|--------|-----------|----------------|--------------|-------|
| H₂O | ~4500 phot/evt | ~100% | 0 | Pure Cherenkov |
| D₂O | ~4300 phot/evt | ~100% | 0 | Slightly lower n |
| WbLS | ~4700 phot/evt | ~85% | ~150 ph/MeV | PPO+bis-MSB |
| Gd-WbLS | ~4600 phot/evt | ~85% | ~143 ph/MeV | 5% Gd quenching |
| Te-WbLS | ~4750 phot/evt | ~82% | ~170 ph/MeV | Te boost |
| **D₂O+LAB** | **~4850 phot/evt** | **~80%** | **~168 ph/MeV** | **D₂O boost x1.12** |
| D₂O+Gd-LAB | ~4750 phot/evt | ~80% | ~160 ph/MeV | Combined |
| **D₂O+Te-LAB** | **~5100 phot/evt** | **~78%** | **~195 ph/MeV** | **Best overall** |

Detection efficiency: ~5–8% (6 × 3-inch PMTs, 30% QE, 30-cm sphere)

---

## 13. Troubleshooting

```bash
# Error: Geant4 datasets not found
export G4NEUTRONHPDATA=/path/to/G4NDL4.7
export G4LEDATA=/path/to/G4EMLOW8.3
export G4LEVELGAMMADATA=/path/to/PhotonEvaporation5.7
# (or source geant4.sh which sets all of these)

# Error: libG4*.so not found
export LD_LIBRARY_PATH=/path/to/geant4/install/lib:$LD_LIBRARY_PATH

# Error: ROOT libraries not found
source /path/to/root/bin/thisroot.sh

# Slow simulation: reduce events for testing
# Edit macros/run_all.mac: /run/beamOn 100   (instead of 1000)

# No optical photons produced:
# Ensure PhysicsList registers G4OpticalPhysics
# Check material has RINDEX property defined
# Verify /process/optical is active: /process/list

# All PMT hits zero:
# Check PMT volume is registered as sensitive detector
# Verify G4SDManager has "PMTSD" registered
# Check PMTSD::ProcessHits returns true

# CMake cannot find Geant4:
cmake .. -DGeant4_DIR=$(geant4-config --prefix)/lib/Geant4-$(geant4-config --version)
```

---

## 14. Extending the Simulation

### Add a new medium
1. Add `BuildMyMedium()` in `DetectorConstruction.cc`
2. Add `AddPPOBisMSB_MyMedium()` in `MaterialsWithPPO.cc`
3. Add case in `PPOIntegratorPatch::Upgrade()`
4. Add entry in `MEDIA` list in analysis scripts
5. Create macro: `macros/runs/MyMedium_5.0MeV.mac`

### Change PPO/bis-MSB concentration
```
# In macro:
# (Currently set via C++ defaults — add messenger commands to expose)
# Default: PPO = 1.0 g/L, bis-MSB = 0.05 g/L

# Or run the analytical scan:
python3 plots/analyze_results.py   # generates Plot 11: yield vs concentration
```

### Add gamma/neutron sources
```
/gun/particle gamma
/gun/energy   2.614 MeV   # Tl-208 line
/run/beamOn 1000

/gun/particle neutron
/gun/energy   1.0 MeV
/run/beamOn 1000
```
