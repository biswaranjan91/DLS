#include "PPOIntegratorPatch.hh"
#include "MaterialsWithPPO.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4SystemOfUnits.hh"
#include <iomanip>

void PPOIntegratorPatch::Upgrade(G4Material* mat,
                                  const G4String& tag,
                                  G4double ppoConcGperL,
                                  G4double bisMSBConcGperL)
{
  if (!mat) {
    G4cerr << "[PPOIntegratorPatch] ERROR: null material pointer" << G4endl;
    return;
  }

  G4MaterialPropertiesTable* mpt = mat->GetMaterialPropertiesTable();
  if (!mpt) {
    G4cerr << "[PPOIntegratorPatch] ERROR: no MPT on " << mat->GetName() << G4endl;
    return;
  }

  G4cout << "[PPOIntegratorPatch] Upgrading " << tag
         << " with PPO=" << ppoConcGperL << " g/L"
         << " + bis-MSB=" << bisMSBConcGperL << " g/L" << G4endl;

  if      (tag == "WbLS")     MaterialsWithPPO::AddPPOBisMSB_WbLS    (mpt, ppoConcGperL, bisMSBConcGperL);
  else if (tag == "GdWbLS")   MaterialsWithPPO::AddPPOBisMSB_GdWbLS  (mpt, ppoConcGperL, bisMSBConcGperL);
  else if (tag == "TeWbLS")   MaterialsWithPPO::AddPPOBisMSB_TeWbLS  (mpt, std::max(ppoConcGperL, 1.5), bisMSBConcGperL);
  else if (tag == "D2OLAB")   MaterialsWithPPO::AddPPOBisMSB_D2OLAB  (mpt, ppoConcGperL, bisMSBConcGperL);
  else if (tag == "D2OGdLAB") MaterialsWithPPO::AddPPOBisMSB_D2OGdLAB(mpt, ppoConcGperL, bisMSBConcGperL);
  else if (tag == "D2OTeLAB") MaterialsWithPPO::AddPPOBisMSB_D2OTeLAB(mpt, std::max(ppoConcGperL, 1.5), bisMSBConcGperL);
  else {
    G4cout << "[PPOIntegratorPatch] " << tag
           << " has no scintillation (pure Cherenkov medium) – skipping." << G4endl;
    return;
  }

  PrintSummary(mat);
}

void PPOIntegratorPatch::PrintSummary(G4Material* mat)
{
  auto* mpt = mat->GetMaterialPropertiesTable();
  if (!mpt) return;

  G4cout << "\n  ┌─ PPO+bis-MSB summary: " << mat->GetName() << " ─────────────────\n";

  auto printProp = [&](const char* name) {
    if (mpt->ConstPropertyExists(name)) {
      G4cout << "  │  " << std::left << std::setw(30) << name
             << " = " << mpt->GetConstProperty(name) << "\n";
    }
  };

  printProp("SCINTILLATIONYIELD");
  printProp("SCINTILLATIONTIMECONSTANT1");
  printProp("SCINTILLATIONTIMECONSTANT2");
  printProp("SCINTILLATIONYIELD1");
  printProp("WLSTIMECONSTANT");
  printProp("WLS2TIMECONSTANT");
  G4cout << "  └──────────────────────────────────────────────────────\n" << G4endl;
}
/*
#include "PPOIntegratorPatch.hh"
#include "MaterialsWithPPO.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4SystemOfUnits.hh"
#include <iomanip>

void PPOIntegratorPatch::Upgrade(G4Material* mat,
                                  const G4String& tag,
                                  G4double ppoConcGperL,
                                  G4double bisMSBConcGperL)
{
  if (!mat) {
    G4cerr << "[PPOIntegratorPatch] ERROR: null material pointer" << G4endl;
    return;
  }

  G4MaterialPropertiesTable* mpt = mat->GetMaterialPropertiesTable();
  if (!mpt) {
    G4cerr << "[PPOIntegratorPatch] ERROR: no MPT on " << mat->GetName() << G4endl;
    return;
  }

  G4cout << "[PPOIntegratorPatch] Upgrading " << tag
         << " with PPO=" << ppoConcGperL << " g/L"
         << " + bis-MSB=" << bisMSBConcGperL << " g/L" << G4endl;

  if      (tag == "WbLS")     MaterialsWithPPO::AddPPOBisMSB_WbLS    (mpt, ppoConcGperL, bisMSBConcGperL);
  else if (tag == "GdWbLS")   MaterialsWithPPO::AddPPOBisMSB_GdWbLS  (mpt, ppoConcGperL, bisMSBConcGperL);
  else if (tag == "TeWbLS")   MaterialsWithPPO::AddPPOBisMSB_TeWbLS  (mpt, std::max(ppoConcGperL, 1.5), bisMSBConcGperL);
  else if (tag == "D2OLAB")   MaterialsWithPPO::AddPPOBisMSB_D2OLAB  (mpt, ppoConcGperL, bisMSBConcGperL);
  else if (tag == "D2OGdLAB") MaterialsWithPPO::AddPPOBisMSB_D2OGdLAB(mpt, ppoConcGperL, bisMSBConcGperL);
  else if (tag == "D2OTeLAB") MaterialsWithPPO::AddPPOBisMSB_D2OTeLAB(mpt, std::max(ppoConcGperL, 1.5), bisMSBConcGperL);
  else {
    G4cout << "[PPOIntegratorPatch] " << tag
           << " has no scintillation (pure Cherenkov medium) – skipping." << G4endl;
    return;
  }

  PrintSummary(mat);
}

void PPOIntegratorPatch::PrintSummary(G4Material* mat)
{
  auto* mpt = mat->GetMaterialPropertiesTable();
  if (!mpt) return;

  G4cout << "\n  ┌─ PPO+bis-MSB summary: " << mat->GetName() << " ─────────────────\n";

  auto printProp = [&](const char* name) {
    if (mpt->ConstPropertyExists(name)) {
      G4cout << "  │  " << std::left << std::setw(30) << name
             << " = " << mpt->GetConstProperty(name) << "\n";
    }
  };

  printProp("SCINTILLATIONYIELD");
  printProp("FASTTIMECONSTANT");
  printProp("SLOWTIMECONSTANT");
  printProp("YIELDRATIO");
  printProp("FASTSCINTILLATIONRISETIME");
  printProp("SLOWSCINTILLATIONRISETIME");
  printProp("WLSTIMECONSTANT");
  printProp("WLS2TIMECONSTANT");
  G4cout << "  └──────────────────────────────────────────────────────\n" << G4endl;
}
*/
