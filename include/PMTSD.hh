#ifndef PMTSD_HH
#define PMTSD_HH

#include "G4VSensitiveDetector.hh"
#include "PMTHit.hh"
#include "globals.hh"

class G4Step;
class G4HCofThisEvent;
class G4TouchableHistory;

// ============================================================================
//  PMTSD – Sensitive detector for PMT volumes
//  Records optical photon hits, tags Cherenkov/Scintillation/WLS origin,
//  applies wavelength-dependent QE (30% flat, can be upgraded to spectral QE)
// ============================================================================
class PMTSD : public G4VSensitiveDetector
{
public:
  PMTSD(const G4String& name, const G4String& hcName);
  ~PMTSD() override = default;

  void   Initialize(G4HCofThisEvent* hce) override;
  G4bool ProcessHits(G4Step* step, G4TouchableHistory* hist) override;
  void   EndOfEvent(G4HCofThisEvent* hce) override;

private:
  PMTHitsCollection* fHitsCollection = nullptr;
  G4int              fHCID           = -1;

  // Wavelength-dependent QE (PMT spectral response)
  G4double GetQE(G4double photonEnergy_eV) const;
};

#endif
