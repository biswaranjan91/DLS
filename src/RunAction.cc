// ============================================================================
//  RunAction.cc
//  Manages per-run accumulators and writes ROOT + CSV output.
//
//  Output files per run:
//    results/<medium>_<energy>MeV/
//      summary.csv          – scalar totals (yield, efficiency, fractions)
//      timing_hist.csv      – photon arrival time histogram [0–500 ns]
//      wavelength_hist.csv  – detected photon wavelength histogram
//      energy_hist.csv      – detected photon energy histogram
//      pmt_hits.csv         – per-PMT hit counts
// ============================================================================

#include "RunAction.hh"
#include "DetectorConstruction.hh"
#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4VUserPrimaryGeneratorAction.hh"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <filesystem>
#include <numeric>

// Access the particle gun to read the energy set by /gun/energy macro command
#include "PrimaryGeneratorAction.hh"

RunAction::RunAction(DetectorConstruction* det)
  : G4UserRunAction(), fDetector(det) {}

RunAction::~RunAction() = default;

// ─────────────────────────────────────────────────────────────────────────────
void RunAction::BeginOfRunAction(const G4Run* run)
{
  fTotalProduced    = 0;
  fScintProduced    = 0;
  fCerenkovProduced = 0;
  fTotalDetected    = 0;
  fScintDetected    = 0;
  fCerenkovDetected = 0;
  fWLSDetected      = 0;
  fNEvents          = 0;
  fPrimaryEnergy    = 0.;
  for (int i = 0; i < 6;         ++i) fPMTCounts[i] = 0;
  for (int i = 0; i < kTimeBins; ++i) fTimeHist[i]  = 0;
  for (int i = 0; i < kWlBins;   ++i) fWlHist[i]    = 0;
  for (int i = 0; i < kEBins;    ++i) fEHist[i]      = 0;

  // ── Read primary energy directly from the particle gun ─────────────────────
  // This is reliable at run start — the macro has already set /gun/energy
  // before /run/beamOn, so the gun holds the correct energy now.
  const auto* genAction = static_cast<const PrimaryGeneratorAction*>(
      G4RunManager::GetRunManager()->GetUserPrimaryGeneratorAction());
  if (genAction) {
    fPrimaryEnergy = genAction->GetGunEnergy() / MeV;
  }
 
  G4cout << "[RunAction] Begin run #" << run->GetRunID()
         << "  medium=" << fDetector->GetMediumName()
         << "  energy=" << fPrimaryEnergy << " MeV" << G4endl;
}
 
// ─────────────────────────────────────────────────────────────────────────────
void RunAction::SetEventEnergy(G4double E_MeV)
{
  // Kept for backward compatibility but no longer used for directory naming
  if (fPrimaryEnergy <= 0.) fPrimaryEnergy = E_MeV;
}
 

void RunAction::AddPhotonProduced(G4bool isScint, G4bool isCerenkov)
{
  ++fTotalProduced;
  if (isScint)    ++fScintProduced;
  if (isCerenkov) ++fCerenkovProduced;
}

void RunAction::AddPhotonDetected(G4bool isScint, G4bool isCerenkov,
                                   G4bool isWLS, G4double time_ns,
                                   G4double wl_nm, G4double energy_eV,
                                   G4int pmtNum)
{
  ++fTotalDetected;
  if (isScint)    ++fScintDetected;
  if (isCerenkov) ++fCerenkovDetected;
  if (isWLS)      ++fWLSDetected;

  // PMT counter
  if (pmtNum >= 0 && pmtNum < 6) ++fPMTCounts[pmtNum];

  // Timing histogram: 0–500 ns, 1 ns/bin
  int tBin = static_cast<int>(time_ns);
  if (tBin >= 0 && tBin < kTimeBins) ++fTimeHist[tBin];

  // Wavelength histogram: 250–650 nm, 4 nm/bin
  int wBin = static_cast<int>((wl_nm - 250.) / 4.);
  if (wBin >= 0 && wBin < kWlBins) ++fWlHist[wBin];

  // Energy histogram: 2.0–4.7 eV, 0.027 eV/bin
  int eBin = static_cast<int>((energy_eV - 2.0) / 0.027);
  if (eBin >= 0 && eBin < kEBins) ++fEHist[eBin];
}

// ─────────────────────────────────────────────────────────────────────────────
void RunAction::EndOfRunAction(const G4Run* run)
{
  fNEvents = run->GetNumberOfEvent();
  if (fNEvents == 0) return;
  WriteResults();
}

// ─────────────────────────────────────────────────────────────────────────────
void RunAction::WriteResults()
{
  G4String medium = fDetector->GetMediumName();
  G4double E_MeV  = fPrimaryEnergy;

  // Build output directory name
  std::ostringstream dirSS;
  dirSS << "results/" << medium << "_"
        << std::fixed << std::setprecision(1) << E_MeV << "MeV";
  G4String outDir = dirSS.str();

  // Create directory
  std::filesystem::create_directories(outDir.c_str());

  G4double nEvts  = static_cast<G4double>(fNEvents);
  G4double eff    = (fTotalProduced > 0)
                  ? static_cast<G4double>(fTotalDetected) / fTotalProduced
                  : 0.;
  G4double scintF = (fTotalProduced > 0)
                  ? static_cast<G4double>(fScintProduced)    / fTotalProduced
                  : 0.;
  G4double cerenF = (fTotalProduced > 0)
                  ? static_cast<G4double>(fCerenkovProduced) / fTotalProduced
                  : 0.;

  // ── summary.csv ────────────────────────────────────────────────────────────
  {
    std::ofstream f(std::string(outDir) + "/summary.csv");
    f << "# NeutrinoDetSim run summary\n";
    f << "medium,"            << medium                    << "\n";
    f << "primary_energy_MeV," << std::fixed << std::setprecision(3) << E_MeV << "\n";
    f << "n_events,"          << fNEvents                  << "\n";
    f << "total_produced,"    << fTotalProduced             << "\n";
    f << "scint_produced,"    << fScintProduced             << "\n";
    f << "cerenkov_produced," << fCerenkovProduced          << "\n";
    f << "total_detected,"    << fTotalDetected             << "\n";
    f << "scint_detected,"    << fScintDetected             << "\n";
    f << "cerenkov_detected," << fCerenkovDetected          << "\n";
    f << "wls_detected,"      << fWLSDetected               << "\n";
    f << "detection_efficiency," << std::setprecision(6) << eff << "\n";
    f << "scint_fraction_produced,"  << scintF             << "\n";
    f << "cerenkov_fraction_produced," << cerenF           << "\n";
    f << "mean_produced_per_event,"
      << static_cast<G4double>(fTotalProduced)/nEvts       << "\n";
    f << "mean_detected_per_event,"
      << static_cast<G4double>(fTotalDetected)/nEvts       << "\n";
  }

  // ── timing_hist.csv ────────────────────────────────────────────────────────
  {
    std::ofstream f(std::string(outDir) + "/timing_hist.csv");
    f << "# time_ns,counts\n";
    for (int i = 0; i < kTimeBins; ++i)
      if (fTimeHist[i] > 0)
        f << i << "," << fTimeHist[i] << "\n";
  }

  // ── wavelength_hist.csv ────────────────────────────────────────────────────
  {
    std::ofstream f(std::string(outDir) + "/wavelength_hist.csv");
    f << "# wavelength_nm,counts\n";
    for (int i = 0; i < kWlBins; ++i)
      if (fWlHist[i] > 0)
        f << (250. + i*4. + 2.) << "," << fWlHist[i] << "\n";
  }

  // ── energy_hist.csv ────────────────────────────────────────────────────────
  {
    std::ofstream f(std::string(outDir) + "/energy_hist.csv");
    f << "# energy_eV,counts\n";
    for (int i = 0; i < kEBins; ++i)
      if (fEHist[i] > 0)
        f << (2.0 + i*0.027 + 0.0135) << "," << fEHist[i] << "\n";
  }

  // ── pmt_hits.csv ───────────────────────────────────────────────────────────
  {
    std::ofstream f(std::string(outDir) + "/pmt_hits.csv");
    f << "# pmt_id,axis,hits\n";
    const char* axes[6] = {"+x","-x","+y","-y","+z","-z"};
    for (int i = 0; i < 6; ++i)
      f << i << "," << axes[i] << "," << fPMTCounts[i] << "\n";
  }

  // ── Console summary ────────────────────────────────────────────────────────
  G4cout << "\n  ╔══ Run Summary: " << medium << " @ " << E_MeV << " MeV ══\n"
         << "  ║  Events          : " << fNEvents          << "\n"
         << "  ║  Photons produced: " << fTotalProduced     << " ("
         << std::fixed << std::setprecision(0)
         << static_cast<G4double>(fTotalProduced)/nEvts << "/evt)\n"
         << "  ║    Scint         : " << fScintProduced     << " ("
         << std::setprecision(1) << scintF*100. << "%)\n"
         << "  ║    Cherenkov     : " << fCerenkovProduced  << " ("
         << cerenF*100. << "%)\n"
         << "  ║  Photons detected: " << fTotalDetected     << "\n"
         << "  ║  Detection eff   : " << std::setprecision(3) << eff*100. << "%\n"
         << "  ║  Output          : " << outDir            << "/\n"
         << "  ╚═══════════════════════════════════════\n" << G4endl;
}
