#!/bin/bash
# =============================================================================
#  run_all_parallel.sh
#  Runs all 40 configs in parallel using N background jobs.
#  Faster than sequential but uses more CPU/RAM.
#
#  Usage:
#    bash run_all_parallel.sh        # auto-detect CPU cores
#    bash run_all_parallel.sh 4      # use 4 parallel jobs
#    bash run_all_parallel.sh 4 500  # 4 jobs, 500 events each
# =============================================================================

N_JOBS=${1:-$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)}
NEVENTS=${2:-1000}
BINARY="./build/NeutrinoDetSim"
MACRO_DIR="./macros/single"
LOG_DIR="./logs"
RESULTS_DIR="./results"

MEDIA=("H2O" "D2O" "WbLS" "GdWbLS" "TeWbLS" "D2OLAB" "D2OGdLAB" "D2OTeLAB")
ENERGIES=("0.5" "1.0" "3.0" "5.0" "10.0")

if [ ! -f "$BINARY" ]; then
  echo "ERROR: Binary not found: $BINARY"; exit 1
fi

mkdir -p "$MACRO_DIR" "$LOG_DIR" "$RESULTS_DIR"

# Generate macros
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

TOTAL=$((${#MEDIA[@]} * ${#ENERGIES[@]}))
echo "Running $TOTAL jobs with $N_JOBS parallel workers ($NEVENTS events each)..."
START=$SECONDS
PIDS=()
TAGS=()
RUNNING=0

run_one() {
  local TAG="$1"
  "$BINARY" "$MACRO_DIR/${TAG}.mac" > "$LOG_DIR/${TAG}.log" 2>&1
}

for MED in "${MEDIA[@]}"; do
  for E in "${ENERGIES[@]}"; do
    TAG="${MED}_${E}MeV"

    # Wait if at capacity
    while [ $RUNNING -ge $N_JOBS ]; do
      for i in "${!PIDS[@]}"; do
        if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
          unset PIDS[$i]; unset TAGS[$i]
          RUNNING=$((RUNNING - 1))
        fi
      done
      sleep 0.5
    done

    run_one "$TAG" &
    PIDS+=($!)
    TAGS+=("$TAG")
    RUNNING=$((RUNNING + 1))
    echo "  Started: $TAG (PID $!)"
  done
done

# Wait for all to finish
echo "  Waiting for all jobs to complete..."
wait

ELAPSED=$((SECONDS - START))
N_OUTPUT=$(ls "$RESULTS_DIR"/*/summary.csv 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "Done! $N_OUTPUT/$TOTAL results in ${ELAPSED}s"
echo "Results: $RESULTS_DIR/"
echo ""
echo "Run analysis:"
echo "  python3 plots/analyze_results.py --results-dir ./results --out-dir ./plots/output"
