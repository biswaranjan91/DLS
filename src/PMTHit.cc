// PMTHit.cc
#include "PMTHit.hh"
#include "G4SystemOfUnits.hh"
#include <iomanip>

G4ThreadLocal G4Allocator<PMTHit>* PMTHitAllocator = nullptr;

PMTHit::PMTHit() : G4VHit() {}

G4bool PMTHit::operator==(const PMTHit& right) const
{
  return (this == &right);
}

void PMTHit::Print()
{
  G4cout << std::fixed << std::setprecision(3)
         << "  PMT#" << fPMTNumber
         << "  t="   << fTime/ns  << " ns"
         << "  E="   << fEnergy/eV << " eV"
         << "  λ="   << fWavelength << " nm"
         << "  ["    << (fIsScint ? "SCINT" : fIsCerenkov ? "CERENKOV" : fIsWLS ? "WLS" : "?") << "]"
         << G4endl;
}
