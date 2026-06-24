#ifndef PMTHIT_HH
#define PMTHIT_HH

#include "G4VHit.hh"
#include "G4THitsCollection.hh"
#include "G4ThreeVector.hh"
#include "G4Allocator.hh"
#include "globals.hh"

// ============================================================================
//  PMTHit – records each optical photon detected by a PMT
// ============================================================================
class PMTHit : public G4VHit
{
public:
  PMTHit();
  PMTHit(const PMTHit&) = default;
  ~PMTHit() override   = default;

  PMTHit& operator=(const PMTHit&) = default;
  G4bool operator==(const PMTHit&) const;

  inline void* operator new(size_t);
  inline void  operator delete(void*);

  void Draw()  override {}
  void Print() override;

  // ── Setters ────────────────────────────────────────────────────────────────
  void SetPMTNumber    (G4int    n) { fPMTNumber   = n; }
  void SetTime         (G4double t) { fTime        = t; }
  void SetEnergy       (G4double e) { fEnergy      = e; }
  void SetPosition     (G4ThreeVector p) { fPosition = p; }
  void SetProcessName  (G4String s) { fProcess     = s; }
  void SetParentProcess(G4String s) { fParentProc  = s; }
  void SetIsScint      (G4bool   b) { fIsScint     = b; }
  void SetIsCherenkov  (G4bool   b) { fIsCerenkov  = b; }
  void SetIsWLS        (G4bool   b) { fIsWLS       = b; }
  void SetTrackID      (G4int    i) { fTrackID     = i; }
  void SetWavelength   (G4double w) { fWavelength  = w; }

  // ── Getters ────────────────────────────────────────────────────────────────
  G4int    GetPMTNumber()     const { return fPMTNumber; }
  G4double GetTime()          const { return fTime;      }
  G4double GetEnergy()        const { return fEnergy;    }
  G4ThreeVector GetPosition() const { return fPosition;  }
  G4String GetProcessName()   const { return fProcess;   }
  G4String GetParentProcess() const { return fParentProc;}
  G4bool   IsScint()          const { return fIsScint;   }
  G4bool   IsCherenkov()      const { return fIsCerenkov;}
  G4bool   IsWLS()            const { return fIsWLS;     }
  G4int    GetTrackID()       const { return fTrackID;   }
  G4double GetWavelength()    const { return fWavelength;}

private:
  G4int         fPMTNumber  = -1;
  G4double      fTime       =  0.;
  G4double      fEnergy     =  0.;
  G4ThreeVector fPosition;
  G4String      fProcess    = "";
  G4String      fParentProc = "";
  G4bool        fIsScint    = false;
  G4bool        fIsCerenkov = false;
  G4bool        fIsWLS      = false;
  G4int         fTrackID    = -1;
  G4double      fWavelength =  0.; // nm
};

using PMTHitsCollection = G4THitsCollection<PMTHit>;

extern G4ThreadLocal G4Allocator<PMTHit>* PMTHitAllocator;

inline void* PMTHit::operator new(size_t)
{
  if (!PMTHitAllocator) PMTHitAllocator = new G4Allocator<PMTHit>;
  return PMTHitAllocator->MallocSingle();
}

inline void PMTHit::operator delete(void* hit)
{
  PMTHitAllocator->FreeSingle(static_cast<PMTHit*>(hit));
}

#endif
