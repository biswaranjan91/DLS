#ifndef STEPPINGACTION_HH
#define STEPPINGACTION_HH
#include "G4UserSteppingAction.hh"
#include "globals.hh"
class RunAction;
class DetectorConstruction;

class SteppingAction : public G4UserSteppingAction {
public:
  SteppingAction(RunAction* ra, DetectorConstruction* det);
  void UserSteppingAction(const G4Step*) override;
private:
  RunAction*            fRunAction;
  DetectorConstruction* fDetector;
};
#endif
