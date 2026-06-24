#ifndef PPOINTEGRATORPATCH_HH
#define PPOINTEGRATORPATCH_HH

// ============================================================================
//  PPOIntegratorPatch.hh
//
//  Standalone helper that re-opens a material's MPT and upgrades its
//  scintillation/WLS properties to the full PPO + bis-MSB model.
//
//  Call sequence in DetectorConstruction::Construct() after selecting medium:
//
//    fDetMaterial = BuildWbLS();               // sets base RINDEX, ABSLENGTH
//    PPOIntegratorPatch::Upgrade(fDetMaterial, fMediumName);  // upgrades fluors
// ============================================================================

#include "G4Material.hh"
#include "globals.hh"

class PPOIntegratorPatch
{
public:
  // Upgrade detection medium MPT with full PPO+bis-MSB properties
  // ppoConcGperL   default: 1.0 g/L  (WbLS BNL recipe)
  // bisMSBConcGperL default: 0.05 g/L
  static void Upgrade(G4Material* mat,
                      const G4String& mediumTag,
                      G4double ppoConcGperL    = 1.0,
                      G4double bisMSBConcGperL = 0.05);

  // Convenience: print summary of scintillation properties
  static void PrintSummary(G4Material* mat);
};

#endif
