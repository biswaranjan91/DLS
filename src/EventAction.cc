// ============================================================================
//  EventAction.cc  –  Reads PMT hits at end of event, passes to RunAction
// ============================================================================

#include "EventAction.hh"
#include "RunAction.hh"
#include "PMTHit.hh"

#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4HCofThisEvent.hh"
#include "G4SystemOfUnits.hh"

EventAction::EventAction(RunAction* ra)
  : G4UserEventAction(), fRunAction(ra) {}

void EventAction::BeginOfEventAction(const G4Event*)
{}

void EventAction::EndOfEventAction(const G4Event* event)
{
  // ── Retrieve PMT hits collection ──────────────────────────────────────────
  auto* sdm  = G4SDManager::GetSDMpointer();

  // Try both naming conventions Geant4 uses: "SD/Collection" and "Collection"
  if (fHCID < 0) {
    fHCID = sdm->GetCollectionID("PMTSD/PMTHitsCollection");
    if (fHCID < 0)
      fHCID = sdm->GetCollectionID("PMTHitsCollection");
  }
  
  //G4int hcID = sdm->GetCollectionID("PMTHitsCollection");
  //if (hcID < 0) return;

  auto* hce = event->GetHCofThisEvent();
  if (!hce)   return;

  auto* hc = static_cast<PMTHitsCollection*>(hce->GetHC(fHCID));
  if (!hc)   return;

  // ── Pass each hit to RunAction accumulators ────────────────────────────────
  G4int nHits = hc->entries();
  for (G4int i = 0; i < nHits; ++i) {
    PMTHit* hit = (*hc)[i];
    fRunAction->AddPhotonDetected(
      hit->IsScint(),
      hit->IsCherenkov(),
      hit->IsWLS(),
      hit->GetTime() / ns,
      hit->GetWavelength(),
      hit->GetEnergy() / eV,
      hit->GetPMTNumber()
    );
  }
}
