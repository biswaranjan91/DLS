// ============================================================================
//  TrackingAction.cc
//  Captures primary electron energy and passes it to RunAction.
//  fFirstTrack resets at BeginOfEventAction, not here.
// ============================================================================
#include "TrackingAction.hh"
#include "RunAction.hh"
#include "G4Track.hh"
#include "G4Electron.hh"
#include "G4SystemOfUnits.hh"
 
TrackingAction::TrackingAction(RunAction* ra)
  : G4UserTrackingAction(), fRunAction(ra) {}
 
void TrackingAction::PreUserTrackingAction(const G4Track* track)
{
  // Only capture the primary electron (parentID==0) kinetic energy
  if (track->GetParentID() == 0 &&
      track->GetDefinition() == G4Electron::ElectronDefinition())
  {
    fRunAction->SetEventEnergy(track->GetKineticEnergy() / MeV);
  }
}

/*

// ============================================================================
//  TrackingAction.cc
// ============================================================================
#include "TrackingAction.hh"
#include "RunAction.hh"
#include "G4Track.hh"
#include "G4Electron.hh"
#include "G4SystemOfUnits.hh"

TrackingAction::TrackingAction(RunAction* ra)
  : G4UserTrackingAction(), fRunAction(ra), fFirstTrack(true) {}

void TrackingAction::PreUserTrackingAction(const G4Track* track)
{
  // Capture primary electron energy at start of each event
  if (fFirstTrack &&
      track->GetParentID() == 0 &&
      track->GetDefinition() == G4Electron::ElectronDefinition())
  {
    fRunAction->SetEventEnergy(track->GetKineticEnergy() / MeV);
    fFirstTrack = false;
  }
  // Reset for next event (TrackingAction is per-event)
  if (track->GetParentID() == 0)
    fFirstTrack = false;
}
*/
