#ifndef ACTIONINITIALIZATION_HH
#define ACTIONINITIALIZATION_HH
#include "G4VUserActionInitialization.hh"
class DetectorConstruction;
class ActionInitialization : public G4VUserActionInitialization {
public:
  ActionInitialization(DetectorConstruction* det);
  void BuildForMaster() const override;
  void Build() const override;
private:
  DetectorConstruction* fDetector;
};
#endif
