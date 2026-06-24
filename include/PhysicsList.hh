#ifndef PHYSICSLIST_HH
#define PHYSICSLIST_HH

#include "G4VModularPhysicsList.hh"
#include "globals.hh"

class G4VPhysicsConstructor;

// ============================================================================
//  PhysicsList
//  – Standard EM (option4) for accurate electron/gamma transport
//  – G4OpticalPhysics for Cherenkov + Scintillation + absorption/scattering
//  – Hadronic physics stubs (QGSP_BERT) for optional neutron/gamma runs
// ============================================================================
class PhysicsList : public G4VModularPhysicsList
{
public:
  PhysicsList();
  ~PhysicsList() override;

  void ConstructParticle() override;
  void ConstructProcess()  override;
  void SetCuts()           override;

private:
  G4double fCutValue;
};

#endif
