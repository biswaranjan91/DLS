#!/bin/bash
# ============================================================================
#  generate_macros.sh
#  Generates individual .mac files for all 40 medium×energy combinations.
#  Run once from the build directory: bash ../macros/generate_macros.sh
# ============================================================================

MEDIA="H2O D2O WbLS GdWbLS TeWbLS D2OLAB D2OGdLAB D2OTeLAB"
ENERGIES="0.5 1.0 3.0 5.0 10.0"
NEVENTS=1000

mkdir -p macros/runs

for MED in $MEDIA; do
  for E in $ENERGIES; do
    FNAME="macros/runs/${MED}_${E}MeV.mac"
    cat > "$FNAME" << EOF
# Auto-generated macro: ${MED} @ ${E} MeV
/run/verbose 0
/event/verbose 0
/tracking/verbose 0

/det/medium ${MED}
/run/initialize

/gun/particle e-
/gun/energy ${E} MeV

/run/beamOn ${NEVENTS}
EOF
    echo "  Created: $FNAME"
  done
done

echo "Done. Generated $(echo $MEDIA | wc -w) × $(echo $ENERGIES | wc -w) = 40 macros."
