#!/bin/bash
# =============================================================================
#  run_all_sequential.sh
#  Runs all 40 medium×energy configurations as separate processes.
#  Each invocation is a fresh Geant4 session — zero state-machine issues.
#
#  Usage (from NeutrinoDetSim/ project root):
#    bash run_all_sequential.sh
#    bash run_all_sequential.sh 500    # custom event count
# =============================================================================

# ── Config ────────────────────────────────────────────────────────────────────
NEVENTS=${1:-1000}     # events per run (pass as argument, default 1000)
BINARY="./build/NeutrinoDetSim"
MACRO_DIR="./macros/single"
LOG_DIR="./logs"
RESULTS_DIR="./results"

MEDIA=("H2O" "D2O" "WbLS" "GdWbLS" "TeWbLS" "D2OLAB" "D2OGdLAB" "D2OTeLAB")
ENERGIES=("0.5" "1.0" "3.0" "5.0" "10.0")

# ── Checks ────────────────────────────────────────────────────────────────────
if [ ! -f "$BINARY" ]; then
  echo "ERROR: Binary not found at $BINARY"
  echo "       Run:  cd build && make -j\$(sysctl -n hw.logicalcpu)"
  exit 1
fi

mkdir -p "$MACRO_DIR" "$LOG_DIR" "$RESULTS_DIR"

# ── Generate one self-contained macro per config ──────────────────────────────
echo "[1/3] Generating macros..."
for MED in "${MEDIA[@]}"; do
  for E in "${ENERGIES[@]}"; do
    cat > "$MACRO_DIR/${MED}_${E}MeV.mac" << MAC
/run/verbose 0
/event/verbose 0
/tracking/verbose 0
/det/medium ${MED}
/run/initialize
/gun/particle e-
/gun/energy ${E} MeV
/run/beamOn ${NEVENTS}
MAC
  done
done
echo "    Generated $((${#MEDIA[@]} * ${#ENERGIES[@]})) macros in $MACRO_DIR/"

# ── Run all configs sequentially ──────────────────────────────────────────────
TOTAL=$((${#MEDIA[@]} * ${#ENERGIES[@]}))
DONE=0
FAILED=0
START_TIME=$SECONDS

echo "[2/3] Running $TOTAL configurations ($NEVENTS events each)..."
echo "      Estimated time: ~$((TOTAL * NEVENTS / 2000)) minutes"
echo ""

for MED in "${MEDIA[@]}"; do
  for E in "${ENERGIES[@]}"; do
    DONE=$((DONE + 1))
    TAG="${MED}_${E}MeV"
    LOG="$LOG_DIR/${TAG}.log"
    ELAPSED=$((SECONDS - START_TIME))

    # Progress bar
    PBAR=$(printf "%3d/%3d" $DONE $TOTAL)
    printf "\r  [%s] Running %-18s  elapsed: %dm%02ds  " \
      "$PBAR" "$TAG" $((ELAPSED/60)) $((ELAPSED%60))

    # Run as fresh process, capture output to log
    "$BINARY" "$MACRO_DIR/${TAG}.mac" > "$LOG" 2>&1
    EXIT=$?

    if [ $EXIT -ne 0 ]; then
      FAILED=$((FAILED + 1))
      echo ""
      echo "  WARNING: $TAG exited with code $EXIT (see $LOG)"
    fi

    # Quick check: did it produce output?
    if [ ! -f "$RESULTS_DIR/$TAG/summary.csv" ]; then
      echo ""
      echo "  WARNING: No output for $TAG — check $LOG"
    fi
  done
done

# ── Summary ───────────────────────────────────────────────────────────────────
ELAPSED=$((SECONDS - START_TIME))
N_OUTPUT=$(ls "$RESULTS_DIR"/*/summary.csv 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo ""
echo "[3/3] Complete!"
echo "  ┌─────────────────────────────────────────────┐"
echo "  │  Total runs   : $TOTAL"
echo "  │  Succeeded    : $N_OUTPUT output files found"
echo "  │  Failed       : $FAILED"
printf "  │  Total time   : %dm%02ds\n" $((ELAPSED/60)) $((ELAPSED%60))
echo "  │  Results in   : $RESULTS_DIR/"
echo "  │  Logs in      : $LOG_DIR/"
echo "  └─────────────────────────────────────────────┘"
echo ""
echo "  Run analysis:"
echo "    python3 plots/analyze_results.py --results-dir ./results --out-dir ./plots/output"
echo "    python3 plots/publication_physics_plots.py"
