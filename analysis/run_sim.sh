#!/bin/bash
# =============================================================================
#  run_sim.sh  — Run NeutrinoDetSim from project root automatically
#  Usage (from anywhere):
#    bash run_sim.sh                          # runs all 40 configs
#    bash run_sim.sh macros/runs/WbLS_5.0MeV.mac  # runs one config
# =============================================================================

# Resolve project root = directory this script lives in
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/build/NeutrinoDetSim"

# Check binary exists
if [ ! -f "$BINARY" ]; then
  echo "ERROR: Binary not found at $BINARY"
  echo "       Please build first:  cd build && make -j\$(sysctl -n hw.logicalcpu)"
  exit 1
fi

# Always cd to project root so macro paths resolve correctly
cd "$SCRIPT_DIR"
echo "Working directory: $(pwd)"
echo "Binary: $BINARY"
echo ""

# Determine which macro to run
if [ $# -eq 0 ]; then
  MACRO="macros/run_all.mac"
  echo "No macro specified — running full study: $MACRO"
else
  MACRO="$1"
fi

echo "Running: $MACRO"
echo "========================================"
"$BINARY" "$MACRO"
