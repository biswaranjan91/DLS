// ============================================================================
//  ActionInitialization.cc
// ============================================================================
#include "ActionInitialization.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "EventAction.hh"
#include "SteppingAction.hh"
#include "TrackingAction.hh"
#include "DetectorConstruction.hh"

ActionInitialization::ActionInitialization(DetectorConstruction* det)
  : G4VUserActionInitialization(), fDetector(det) {}

void ActionInitialization::BuildForMaster() const
{
  auto* ra = new RunAction(fDetector);
  SetUserAction(ra);
}

void ActionInitialization::Build() const
{
  SetUserAction(new PrimaryGeneratorAction());
  auto* ra = new RunAction(fDetector);
  SetUserAction(ra);
  auto* ea = new EventAction(ra);
  SetUserAction(ea);
  SetUserAction(new TrackingAction(ra));
  SetUserAction(new SteppingAction(ra, fDetector));
}
