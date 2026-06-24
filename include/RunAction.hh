#ifndef RUNACTION_HH
#define RUNACTION_HH

#include "G4UserRunAction.hh"
#include "G4Run.hh"
#include "globals.hh"
#include <map>
#include <string>

class DetectorConstruction;

// ============================================================================
//  RunAction – Creates and manages ROOT output file + histograms
// ============================================================================
class RunAction : public G4UserRunAction
{
public:
  RunAction(DetectorConstruction* det);
  ~RunAction() override;

  void BeginOfRunAction(const G4Run*) override;
  void EndOfRunAction  (const G4Run*) override;

  // Accumulation methods called from EventAction
  void AddPhotonProduced  (G4bool isScint, G4bool isCerenkov);
  void AddPhotonDetected  (G4bool isScint, G4bool isCerenkov, G4bool isWLS,
                           G4double time_ns, G4double wl_nm,
                           G4double energy_eV, G4int pmtNum);
  void SetEventEnergy     (G4double E_MeV);

private:
  void WriteResults();

  DetectorConstruction* fDetector = nullptr;

  // Per-run accumulators
  G4long fTotalProduced     = 0;
  G4long fScintProduced     = 0;
  G4long fCerenkovProduced  = 0;
  G4long fTotalDetected     = 0;
  G4long fScintDetected     = 0;
  G4long fCerenkovDetected  = 0;
  G4long fWLSDetected       = 0;
  G4int  fNEvents           = 0;
  G4double fPrimaryEnergy   = 0.;

  // Per-PMT counters [6]
  G4long fPMTCounts[6]      = {0};

  // Timing histogram (binned in RunAction, written to ROOT)
  static const int kTimeBins = 500;
  G4int fTimeHist[kTimeBins] = {0}; // 0–500 ns in 1 ns bins

  // Wavelength histogram
  static const int kWlBins = 100;
  G4int fWlHist[kWlBins] = {0}; // 250–650 nm in 4 nm bins

  // Energy histogram for detected photons
  static const int kEBins = 100;
  G4int fEHist[kEBins] = {0}; // 2.0–4.7 eV in 0.027 eV bins

  G4String fOutputDir;
};

#endif
