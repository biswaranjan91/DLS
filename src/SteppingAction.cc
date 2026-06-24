// ============================================================================
//  SteppingAction.cc  —  Fixed for Geant4 11.x const secondary track vector
// ============================================================================

#include "SteppingAction.hh"
#include "RunAction.hh"
#include "DetectorConstruction.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4TrackStatus.hh"
#include "G4OpticalPhoton.hh"
#include "G4VProcess.hh"          // ← required for GetProcessName()
#include "G4VPhysicalVolume.hh"
#include "G4SystemOfUnits.hh"
#include <cmath>
#include <string>

SteppingAction::SteppingAction(RunAction* ra, DetectorConstruction* det)
  : G4UserSteppingAction(), fRunAction(ra), fDetector(det) {}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
  // ── TEMPORARY DEBUG ──────────────────────────────────────────────────────
  // Track where optical photons are actually dying, independent of the SD.
  // This will print the volume name and process at the END of every dead
  // optical photon's last step, for the first N occurrences, so we can see
  // empirically whether photons reach the PMT region at all.
  {
    const G4Track* tr = step->GetTrack();
    if (tr->GetDefinition() == G4OpticalPhoton::OpticalPhotonDefinition()
        && tr->GetTrackStatus() != fAlive)
    {
      static G4int deathCount = 0;
      ++deathCount;
      if (deathCount <= 30 || deathCount % 10000 == 0) {
        const G4StepPoint* post = step->GetPostStepPoint();
        const G4VProcess* proc  = post->GetProcessDefinedStep();
        G4String procName = proc ? proc->GetProcessName() : "none";
        G4String volName  = post->GetPhysicalVolume()
                           ? post->GetPhysicalVolume()->GetName() : "OutOfWorld";
        G4String preVolName = step->GetPreStepPoint()->GetPhysicalVolume()
                           ? step->GetPreStepPoint()->GetPhysicalVolume()->GetName() : "?";
        G4ThreeVector pos = post->GetPosition();
        // For World-death photons specifically: compute which axis this
        // point is closest to, and its perpendicular distance (rho) from
        // that axis -- to directly check if it falls inside the glass
        // cylinder's expected bounding region (rho<38mm, axial 145.1-152mm).
        G4String extra = "";
        if (volName == "World") {
          G4double ax = std::fabs(pos.x()), ay = std::fabs(pos.y()), az = std::fabs(pos.z());
          G4double axial, rho;
          char axisName;
          if (ax >= ay && ax >= az) { axial = ax; rho = std::sqrt(pos.y()*pos.y()+pos.z()*pos.z()); axisName='x'; }
          else if (ay >= ax && ay >= az) { axial = ay; rho = std::sqrt(pos.x()*pos.x()+pos.z()*pos.z()); axisName='y'; }
          else { axial = az; rho = std::sqrt(pos.x()*pos.x()+pos.y()*pos.y()); axisName='z'; }
          extra = "  [axis=" + std::string(1,axisName)
                + " axial=" + std::to_string(axial/mm)
                + "mm rho=" + std::to_string(rho/mm) + "mm]";
        }
        G4cout << "[STEP DEBUG] photon death #" << deathCount
               << "  preVol=" << preVolName
               << "  postVol=" << volName
               << "  killedBy=" << procName
               << "  r=" << pos.mag()/mm << " mm"
               << "  pos=(" << pos.x()/mm << "," << pos.y()/mm << "," << pos.z()/mm << ")mm"
               << extra
               << "  status=" << tr->GetTrackStatus()
               << G4endl;
      }
    }
  }
  // ── END TEMPORARY DEBUG ──────────────────────────────────────────────────

  // Geant4 11.x: GetSecondaryInCurrentStep() returns
  //   const std::vector<const G4Track*>*   (not G4TrackVector*)
  const std::vector<const G4Track*>* secondaries =
      step->GetSecondaryInCurrentStep();
  if (!secondaries) return;

  for (const G4Track* sec : *secondaries) {
    if (sec->GetDefinition() != G4OpticalPhoton::OpticalPhotonDefinition())
      continue;

    const G4VProcess* proc = sec->GetCreatorProcess();
    G4String procName = proc ? proc->GetProcessName() : "unknown";

    G4bool isScint    = (procName == "Scintillation");
    G4bool isCerenkov = (procName == "Cerenkov");
    G4bool isWLS      = (procName.find("OpWLS") != G4String::npos);

    fRunAction->AddPhotonProduced(isScint || isWLS, isCerenkov);
  }
}
