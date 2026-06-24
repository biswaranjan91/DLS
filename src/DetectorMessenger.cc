#include "DetectorMessenger.hh"
#include "DetectorConstruction.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"

DetectorMessenger::DetectorMessenger(DetectorConstruction* det)
  : G4UImessenger(), fDetector(det)
{
  fDetDir = new G4UIdirectory("/det/");
  fDetDir->SetGuidance("Detector configuration commands");

  fMediumCmd = new G4UIcmdWithAString("/det/medium", this);
  fMediumCmd->SetGuidance("Set detection medium");
  fMediumCmd->SetGuidance("Choices: H2O D2O WbLS GdWbLS TeWbLS D2OLAB D2OGdLAB D2OTeLAB");
  fMediumCmd->SetParameterName("medium", false);
  //fMediumCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
  // Allow in PreInit, Idle, and GeomClosed states
  fMediumCmd->AvailableForStates(G4State_PreInit, G4State_Idle, G4State_GeomClosed);
  
  fRadiusCmd = new G4UIcmdWithADoubleAndUnit("/det/radius", this);
  fRadiusCmd->SetGuidance("Set spherical detector radius");
  fRadiusCmd->SetParameterName("radius", false);
  fRadiusCmd->SetUnitCategory("Length");
  fRadiusCmd->SetDefaultUnit("cm");
  fRadiusCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
}

DetectorMessenger::~DetectorMessenger()
{
  delete fMediumCmd;
  delete fRadiusCmd;
  delete fDetDir;
}

void DetectorMessenger::SetNewValue(G4UIcommand* cmd, G4String val)
{
  if (cmd == fMediumCmd) {
    fDetector->SetMediumName(val);
    // Mark geometry as modified so /run/reinitializeGeometry rebuilds it
    G4RunManager::GetRunManager()->GeometryHasBeenModified();
  }
  else if (cmd == fRadiusCmd) {
    fDetector->SetDetectorRadius(fRadiusCmd->GetNewDoubleValue(val));
    G4RunManager::GetRunManager()->GeometryHasBeenModified();
  }
}

/*
void DetectorMessenger::SetNewValue(G4UIcommand* cmd, G4String val)
{
  if (cmd == fMediumCmd) {
    fDetector->SetMediumName(val);
    G4RunManager::GetRunManager()->ReinitializeGeometry();
  }
  else if (cmd == fRadiusCmd) {
    fDetector->SetDetectorRadius(fRadiusCmd->GetNewDoubleValue(val));
    G4RunManager::GetRunManager()->ReinitializeGeometry();
  }
}
*/
