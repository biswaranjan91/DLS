// ============================================================================
//  PrimaryGeneratorAction.cc
//  Default: 1 MeV electron at centre, isotropic direction
// ============================================================================
#include "PrimaryGeneratorAction.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"
#include <cmath>

PrimaryGeneratorAction::PrimaryGeneratorAction()
  : G4VUserPrimaryGeneratorAction()
{
  fParticleGun = new G4ParticleGun(1);
  auto* table = G4ParticleTable::GetParticleTable();
  fParticleGun->SetParticleDefinition(table->FindParticle("e-"));
  fParticleGun->SetParticleEnergy(1.0*MeV);
  fParticleGun->SetParticlePosition(G4ThreeVector(0,0,0));
  // Isotropic initial direction
  fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0,0,1));
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
  delete fParticleGun;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
  // Random isotropic direction each event
  G4double cosTheta = 2.*G4UniformRand() - 1.;
  G4double sinTheta = std::sqrt(1. - cosTheta*cosTheta);
  G4double phi      = CLHEP::twopi * G4UniformRand();
  G4ThreeVector dir(sinTheta*std::cos(phi),
                    sinTheta*std::sin(phi),
                    cosTheta);
  fParticleGun->SetParticleMomentumDirection(dir);
  fParticleGun->GeneratePrimaryVertex(event);
}

void PrimaryGeneratorAction::SetParticleName(const G4String& n)
{
  auto* part = G4ParticleTable::GetParticleTable()->FindParticle(n);
  if (part) fParticleGun->SetParticleDefinition(part);
}

void PrimaryGeneratorAction::SetParticleEnergy(G4double E)
{
  fParticleGun->SetParticleEnergy(E);
}
