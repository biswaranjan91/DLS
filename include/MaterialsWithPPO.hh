#ifndef MATERIALSWITHPPO_HH
#define MATERIALSWITHPPO_HH

// ============================================================================
//  MaterialsWithPPO.hh
//
//  Provides realistic optical property tables for:
//    – PPO  (2,5-diphenyloxazole): primary scintillation fluor
//    – bis-MSB  (1,4-bis(2-methylstyryl)benzene): secondary wavelength shifter
//
//  PPO + bis-MSB is the standard fluor cocktail used in:
//    • WbLS (BNL recipe, Chen et al.)        ~0.2-2 g/L PPO + 0.01-0.1 g/L bis-MSB
//    • SNO+ LAB scintillator                  2 g/L PPO + 0.05 g/L bis-MSB
//    • JUNO LS                                3 g/L PPO + 0.015 g/L bis-MSB
//    • D2O hybrid media (this study)
//
//  Energy convention: all tables use photon energy in eV, 20-point grid
//    covering 2.0 – 4.7 eV  (264 – 620 nm)
//
//  Physics modelled:
//    1. PPO absorption   (~300-380 nm peak, Stokes-shifted emission ~340-450 nm)
//    2. PPO emission spectrum (FAST component for G4Scintillation)
//    3. bis-MSB WLS absorption (~360-420 nm)  ← absorbs PPO emission
//    4. bis-MSB WLS emission  (~390-480 nm, peak ~420 nm)  → SLOW component
//    5. WLS mean free path (WLSABSLENGTH / WLS2ABSLENGTH)
//    6. WLS time constant (near-prompt for bis-MSB, τ ~ 1.5 ns)
// ============================================================================

#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4SystemOfUnits.hh"
#include "globals.hh"

class MaterialsWithPPO
{
public:
  // ── Photon energy grid (shared by all tables) ──────────────────────────────
  static constexpr int kNE = 25;
  static const G4double kE[kNE];       // photon energy [eV], 2.0 – 4.70 eV

  // ── Per-medium optical property injectors ─────────────────────────────────
  // Each method ADDS PPO+bis-MSB properties to an existing MPT that already
  // has RINDEX, ABSLENGTH, RAYLEIGH defined for the solvent.

  // WbLS variants (H2O-based solvent)
  static void AddPPOBisMSB_WbLS     (G4MaterialPropertiesTable* mpt,
                                     G4double ppoConcGperL  = 1.0,
                                     G4double bisMSBConcGperL = 0.05);

  static void AddPPOBisMSB_GdWbLS   (G4MaterialPropertiesTable* mpt,
                                     G4double ppoConcGperL  = 1.0,
                                     G4double bisMSBConcGperL = 0.05);

  static void AddPPOBisMSB_TeWbLS   (G4MaterialPropertiesTable* mpt,
                                     G4double ppoConcGperL  = 1.5,
                                     G4double bisMSBConcGperL = 0.05);

  // D2O-LAB variants (D2O-based solvent)
  static void AddPPOBisMSB_D2OLAB   (G4MaterialPropertiesTable* mpt,
                                     G4double ppoConcGperL  = 1.0,
                                     G4double bisMSBConcGperL = 0.05);

  static void AddPPOBisMSB_D2OGdLAB (G4MaterialPropertiesTable* mpt,
                                     G4double ppoConcGperL  = 1.0,
                                     G4double bisMSBConcGperL = 0.05);

  static void AddPPOBisMSB_D2OTeLAB (G4MaterialPropertiesTable* mpt,
                                     G4double ppoConcGperL  = 1.5,
                                     G4double bisMSBConcGperL = 0.05);

  // ── Raw spectra (public for diagnostic / plotting) ────────────────────────
  // PPO primary emission spectrum (fast component, normalised)
  static void GetPPOEmissionSpectrum  (G4double* spec);
  // bis-MSB secondary emission spectrum (slow/WLS component, normalised)
  static void GetBisMSBEmissionSpectrum(G4double* spec);
  // PPO absorption cross-section (relative, for WLS mean-free-path scaling)
  static void GetPPOAbsorptionSpectrum (G4double* spec);
  // bis-MSB absorption cross-section (relative)
  static void GetBisMSBAbsorptionSpectrum(G4double* spec);

  // ── Scintillation yield model ─────────────────────────────────────────────
  // Returns effective total photon yield [photons/MeV] given PPO concentration,
  // fraction of light that is Cherenkov vs scintillation, and WLS transfer eff.
  static G4double ComputeEffectiveYield(G4double ppoConcGperL,
                                        G4double bisMSBConcGperL,
                                        G4double labFraction,
                                        G4double wlsTransferEff = 0.95);

private:
  // Internal helper – applies scaled WLS absorption lengths to MPT
  static void ApplyWLSProperties(G4MaterialPropertiesTable* mpt,
                                  G4double ppoConcGperL,
                                  G4double bisMSBConcGperL,
                                  G4double solventScintYield);

  // Scaling: absorption length ∝ 1/concentration (Beer-Lambert)
  // Reference absorption lengths at 1 g/L in LAB
  static constexpr G4double kPPO_AbsRef    = 0.5*CLHEP::cm; // at peak ~310 nm
  static constexpr G4double kBisMSB_AbsRef = 0.8*CLHEP::cm; // at peak ~380 nm
};

#endif // MATERIALSWITHPPO_HH
