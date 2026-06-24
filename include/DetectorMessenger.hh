#ifndef DETECTORMESSENGER_HH
#define DETECTORMESSENGER_HH

#include "G4UImessenger.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "globals.hh"

class DetectorConstruction;

class DetectorMessenger : public G4UImessenger
{
public:
  DetectorMessenger(DetectorConstruction* det);
  ~DetectorMessenger() override;

  void SetNewValue(G4UIcommand* cmd, G4String newVal) override;

private:
  DetectorConstruction*       fDetector;
  G4UIdirectory*              fDetDir;
  G4UIcmdWithAString*         fMediumCmd;
  G4UIcmdWithADoubleAndUnit*  fRadiusCmd;
};

#endif
