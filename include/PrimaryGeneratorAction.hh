#ifndef PRIMARYGENERATORACTION_HH
#define PRIMARYGENERATORACTION_HH
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "globals.hh"
class G4Event;
class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction {
public:
  PrimaryGeneratorAction();
  ~PrimaryGeneratorAction() override;
  void GeneratePrimaries(G4Event*) override;
  void SetParticleName(const G4String& n);
  void SetParticleEnergy(G4double E);
  // Read back the currently set gun energy [Geant4 internal units]
  G4double GetGunEnergy() const { return fParticleGun->GetParticleEnergy(); }
private:
  G4ParticleGun* fParticleGun;
};
#endif

