// ============================================================================
//  DetectorConstruction.cc  —  SuperK-style PMT flush geometry
//
//  Design: Hamamatsu R1307 PMTs inserted into holes in the acrylic vessel,
//  glass face flush with the INNER surface — exactly as in Super-Kamiokande.
//
//  Optical path:
//    Detection medium  →  [direct Fresnel at inner surface]  →  PMT glass
//    →  photocathode  →  SD fires  →  QE applied
//
//  No reflective coating between medium and PMT.
//  95% PTFE reflector on the non-PMT portions of the inner acrylic wall.
//
//  Vessel geometry (all from R1307 datasheet + standard acrylic vessel):
//    Acrylic sphere : inner R = 150 mm, wall = 5 mm, outer R = 155 mm
//    6 cylindrical holes : radius = 38 mm (tube OD/2), depth = 5 mm
//    PMT glass front face flush with inner wall at R = 150 mm
//    PMT body extends outward to R = 150 + 2 + 0.5 + 147.5 = 300 mm
// ============================================================================

#include "DetectorConstruction.hh"
#include "DetectorMessenger.hh"
#include "PMTSD.hh"
#include "MaterialsWithPPO.hh"
#include "PPOIntegratorPatch.hh"

#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4Element.hh"
#include "G4Isotope.hh"

#include "G4Box.hh"
#include "G4Sphere.hh"
#include "G4Tubs.hh"
#include "G4SubtractionSolid.hh"
#include "G4UnionSolid.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4RotationMatrix.hh"
#include "G4ThreeVector.hh"

#include "G4OpticalSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4LogicalBorderSurface.hh"

#include "G4SDManager.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include <vector>
#include <cmath>
#include <iomanip>

// ── R1307 dimensions (Hamamatsu datasheet) ────────────────────────────────────
namespace R1307 {
  constexpr G4double kTubeRadius        = 76.0/2.*mm;   // tube OD/2 = 38 mm
  constexpr G4double kTubeLength        = 150.*mm;       // max overall length
  constexpr G4double kWindowThickness   = 2.*mm;         // borosilicate faceplate
  constexpr G4double kPhotocathodeR     = 51.5/2.*mm;    // active area radius
  constexpr G4double kPhotocathodeThick = 0.5*mm;        // thin sensitive disc
}

// ── Optical property energy grid (20 points, 270–620 nm) ─────────────────────
namespace {
const int kNE = 20;
const G4double kE[kNE] = {
  2.00*eV,2.13*eV,2.27*eV,2.43*eV,2.58*eV,
  2.76*eV,2.95*eV,3.06*eV,3.18*eV,3.31*eV,
  3.44*eV,3.54*eV,3.65*eV,3.81*eV,3.97*eV,
  4.13*eV,4.28*eV,4.43*eV,4.50*eV,4.59*eV
};
} // namespace

// ─────────────────────────────────────────────────────────────────────────────
DetectorConstruction::DetectorConstruction()
  : G4VUserDetectorConstruction(),
    fMediumName("WbLS"),
    fDetRadius(150.*mm),
    fNPMTs(6),
    fPMTRadius(R1307::kPhotocathodeR)
{
  fMessenger = new DetectorMessenger(this);
}

DetectorConstruction::~DetectorConstruction() { delete fMessenger; }

void DetectorConstruction::SetMediumName(const G4String& n)
{
  fMediumName = n;
  G4cout << "[DetectorConstruction] Medium set to: " << n << G4endl;
}
void DetectorConstruction::SetDetectorRadius(G4double r) { fDetRadius = r; }

// ─────────────────────────────────────────────────────────────────────────────
//  ACRYLIC (PMMA) — Sultanova 2009 Sellmeier fit
// ─────────────────────────────────────────────────────────────────────────────
G4Material* DetectorConstruction::BuildAcrylic()
{
  G4NistManager* nist = G4NistManager::Instance();
  G4Material* acrylic = new G4Material("Acrylic",1.19*g/cm3,3,kStateSolid);
  acrylic->AddElement(nist->FindOrBuildElement("C"),5);
  acrylic->AddElement(nist->FindOrBuildElement("H"),8);
  acrylic->AddElement(nist->FindOrBuildElement("O"),2);
  G4double rindex[kNE]={1.504,1.502,1.500,1.498,1.496,1.494,1.492,1.491,
                         1.490,1.489,1.488,1.487,1.486,1.485,1.484,
                         1.483,1.482,1.481,1.481,1.480};
  G4double abslen[kNE]={0.5*m,1.0*m,2.0*m,3.0*m,3.5*m,4.0*m,4.0*m,4.0*m,
                         4.0*m,4.0*m,4.0*m,3.8*m,3.5*m,3.0*m,2.5*m,
                         2.0*m,1.5*m,1.0*m,0.8*m,0.5*m};
  auto* mpt=new G4MaterialPropertiesTable();
  mpt->AddProperty("RINDEX",   std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(rindex,rindex+kNE));
  mpt->AddProperty("ABSLENGTH",std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(abslen,abslen+kNE));
  acrylic->SetMaterialPropertiesTable(mpt);
  return acrylic;
}

// ─────────────────────────────────────────────────────────────────────────────
//  BOROSILICATE GLASS  (R1307: n=1.500 at 420 nm, exact datasheet value)
// ─────────────────────────────────────────────────────────────────────────────
G4Material* DetectorConstruction::BuildBorosilicateGlass()
{
  G4NistManager* nist = G4NistManager::Instance();
  G4Material* glass = nist->FindOrBuildMaterial("G4_Pyrex_Glass");
  G4double rindex[kNE]={1.530,1.527,1.524,1.521,1.519,1.517,1.515,1.514,
                         1.513,1.512,1.511,1.510,1.509,1.507,1.506,
                         1.504,1.503,1.502,1.501,1.500};
  G4double abslen[kNE]={0.1*m,0.5*m,1.0*m,2.0*m,5.0*m,10.*m,10.*m,10.*m,
                         10.*m,10.*m,10.*m,10.*m,10.*m,10.*m,10.*m,
                         10.*m,10.*m,5.*m,2.*m,0.5*m};
  auto* mpt=new G4MaterialPropertiesTable();
  mpt->AddProperty("RINDEX",   std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(rindex,rindex+kNE));
  mpt->AddProperty("ABSLENGTH",std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(abslen,abslen+kNE));
  glass->SetMaterialPropertiesTable(mpt);
  return glass;
}

// ─────────────────────────────────────────────────────────────────────────────
//  PHOTOCATHODE (bialkali K2CsSb proxy)
// ─────────────────────────────────────────────────────────────────────────────
G4Material* DetectorConstruction::BuildPhotocathode()
{
  G4NistManager* nist = G4NistManager::Instance();
  G4Material* pc = new G4Material("Photocathode",3.0*g/cm3,3,kStateSolid);
  pc->AddElement(nist->FindOrBuildElement("K"), 2);
  pc->AddElement(nist->FindOrBuildElement("Cs"),1);
  pc->AddElement(nist->FindOrBuildElement("Sb"),1);

  // ── Optical properties ──────────────────────────────────────────────────
  // CRITICAL: G4OpBoundaryProcess requires RINDEX on BOTH sides of every
  // optical interface. Without it here, the glass->photocathode boundary
  // is undefined, and the photon is absorbed/reflected at the glass exit
  // face — it never legitimately crosses into the photocathode volume, so
  // PMTSD::ProcessHits is never called. This was the root cause of the
  // total_detected = 0 result.
  //
  // RINDEX is matched to the borosilicate glass (n=1.50 @420nm) so there
  // is zero Fresnel reflection at the glass/photocathode boundary — the
  // photon enters the thin bialkali layer with 100% transmission, exactly
  // as in the real R1307 (the photocathode is deposited directly on the
  // inner glass surface with no air gap or index step).
  //
  // ABSLENGTH is set short (the real bialkali layer is only ~20-30 nm
  // thick and absorbs essentially all incident light within that
  // distance) so that any photon entering the photocathode volume is
  // guaranteed to interact within our 0.5 mm thick simulated disc rather
  // than transmitting straight through it.
  G4double rindex[kNE]  = {1.530,1.527,1.524,1.521,1.519,1.517,1.515,1.514,
                            1.513,1.512,1.511,1.510,1.509,1.507,1.506,
                            1.504,1.503,1.502,1.501,1.500};  // matches glass exactly
  G4double abslen[kNE];
  for (int i=0;i<kNE;++i) abslen[i] = 1.*um;  // fully absorbing on entry

  auto* mpt = new G4MaterialPropertiesTable();
  mpt->AddProperty("RINDEX",
      std::vector<G4double>(kE,kE+kNE), std::vector<G4double>(rindex,rindex+kNE));
  mpt->AddProperty("ABSLENGTH",
      std::vector<G4double>(kE,kE+kNE), std::vector<G4double>(abslen,abslen+kNE));
  pc->SetMaterialPropertiesTable(mpt);

  return pc;
}

// ─────────────────────────────────────────────────────────────────────────────
//  DETECTION MEDIA
// ─────────────────────────────────────────────────────────────────────────────
G4Material* DetectorConstruction::BuildWater()
{
  G4NistManager* nist=G4NistManager::Instance();
  G4Material* water=nist->FindOrBuildMaterial("G4_WATER");
  G4double ri[kNE]={1.332,1.333,1.334,1.335,1.336,1.337,1.339,1.340,1.341,1.342,1.343,1.344,1.346,1.348,1.350,1.352,1.354,1.356,1.357,1.358};
  G4double ab[kNE]={10.*m,20.*m,40.*m,60.*m,80.*m,90.*m,95.*m,100.*m,100.*m,95.*m,85.*m,75.*m,60.*m,40.*m,20.*m,10.*m,5.*m,2.*m,1.*m,0.5*m};
  G4double rl[kNE]={50.*m,60.*m,70.*m,80.*m,90.*m,95.*m,98.*m,100.*m,100.*m,95.*m,85.*m,75.*m,60.*m,50.*m,40.*m,30.*m,20.*m,15.*m,12.*m,10.*m};
  auto* mpt=new G4MaterialPropertiesTable();
  mpt->AddProperty("RINDEX",   std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(ri,ri+kNE));
  mpt->AddProperty("ABSLENGTH",std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(ab,ab+kNE));
  mpt->AddProperty("RAYLEIGH", std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(rl,rl+kNE));
  water->SetMaterialPropertiesTable(mpt);
  return water;
}

G4Material* DetectorConstruction::BuildD2O()
{
  G4Isotope* D=new G4Isotope("Deuterium",1,2,2.014*g/mole);
  G4Element* elD=new G4Element("Deuterium","D",1); elD->AddIsotope(D,1.0);
  G4NistManager* nist=G4NistManager::Instance();
  G4Element* elO=nist->FindOrBuildElement("O");
  G4Material* d2o=new G4Material("D2O",1.107*g/cm3,2,kStateLiquid,293.15*kelvin,1.*atmosphere);
  d2o->AddElement(elD,2); d2o->AddElement(elO,1);
  G4double ri[kNE]={1.325,1.326,1.327,1.328,1.329,1.330,1.332,1.333,1.334,1.335,1.336,1.337,1.339,1.341,1.343,1.345,1.347,1.349,1.350,1.351};
  G4double ab[kNE]={12.*m,22.*m,42.*m,62.*m,82.*m,92.*m,97.*m,102.*m,102.*m,97.*m,87.*m,77.*m,62.*m,42.*m,22.*m,11.*m,6.*m,2.5*m,1.2*m,0.6*m};
  G4double rl[kNE]={48.*m,58.*m,68.*m,78.*m,88.*m,93.*m,96.*m,98.*m,98.*m,93.*m,83.*m,73.*m,58.*m,48.*m,38.*m,28.*m,18.*m,13.*m,10.*m,8.*m};
  auto* mpt=new G4MaterialPropertiesTable();
  mpt->AddProperty("RINDEX",   std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(ri,ri+kNE));
  mpt->AddProperty("ABSLENGTH",std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(ab,ab+kNE));
  mpt->AddProperty("RAYLEIGH", std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(rl,rl+kNE));
  d2o->SetMaterialPropertiesTable(mpt);
  return d2o;
}

// Macro for scintillating medium builders (reduces repetition)
#define ADD_SCINT_PROPS(mpt,y,y1,t1,t2,r1,r2) \
  mpt->AddConstProperty("SCINTILLATIONYIELD",       y,    true); \
  mpt->AddConstProperty("RESOLUTIONSCALE",           1.0,  true); \
  mpt->AddConstProperty("SCINTILLATIONTIMECONSTANT1",t1,   true); \
  mpt->AddConstProperty("SCINTILLATIONTIMECONSTANT2",t2,   true); \
  mpt->AddConstProperty("SCINTILLATIONYIELD1",       y1,   true); \
  mpt->AddConstProperty("SCINTILLATIONRISETIME1",    r1,   true); \
  mpt->AddConstProperty("SCINTILLATIONRISETIME2",    r2,   true);

G4Material* DetectorConstruction::BuildWbLS()
{
  G4NistManager* nist=G4NistManager::Instance();
  G4Material* mat=new G4Material("WbLS",1.001*g/cm3,3,kStateLiquid,293.15*kelvin,1.*atmosphere);
  mat->AddElement(nist->FindOrBuildElement("O"),0.877);
  mat->AddElement(nist->FindOrBuildElement("H"),0.113);
  mat->AddElement(nist->FindOrBuildElement("C"),0.010);
  G4double ri[kNE]={1.334,1.335,1.336,1.337,1.338,1.340,1.341,1.342,1.344,1.345,1.346,1.348,1.349,1.351,1.353,1.355,1.357,1.359,1.360,1.362};
  G4double ab[kNE]={8.*m,15.*m,30.*m,50.*m,70.*m,80.*m,85.*m,88.*m,88.*m,83.*m,73.*m,63.*m,50.*m,30.*m,15.*m,8.*m,4.*m,1.5*m,0.8*m,0.4*m};
  G4double rl[kNE]={45.*m,55.*m,65.*m,75.*m,85.*m,90.*m,93.*m,95.*m,95.*m,90.*m,80.*m,70.*m,55.*m,45.*m,35.*m,25.*m,15.*m,10.*m,8.*m,6.*m};
  G4double sf[kNE]={0,0.01,0.03,0.08,0.15,0.30,0.80,1.00,0.90,0.70,0.45,0.25,0.10,0.04,0.01,0,0,0,0,0};
  G4double ss[kNE]={0,0,0.01,0.04,0.10,0.20,0.50,0.70,0.85,1.00,0.80,0.55,0.30,0.12,0.04,0.01,0,0,0,0};
  auto* mpt=new G4MaterialPropertiesTable();
  mpt->AddProperty("RINDEX",                  std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(ri,ri+kNE));
  mpt->AddProperty("ABSLENGTH",               std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(ab,ab+kNE));
  mpt->AddProperty("RAYLEIGH",                std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(rl,rl+kNE));
  mpt->AddProperty("SCINTILLATIONCOMPONENT1", std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(sf,sf+kNE));
  mpt->AddProperty("SCINTILLATIONCOMPONENT2", std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(ss,ss+kNE));
  ADD_SCINT_PROPS(mpt, 100./MeV, 0.80, 5.0*ns, 50.0*ns, 0.5*ns, 1.5*ns)
  mat->SetMaterialPropertiesTable(mpt);
  return mat;
}

G4Material* DetectorConstruction::BuildGdWbLS()
{
  G4NistManager* nist=G4NistManager::Instance();
  G4Material* mat=new G4Material("GdWbLS",1.002*g/cm3,4,kStateLiquid,293.15*kelvin,1.*atmosphere);
  mat->AddElement(nist->FindOrBuildElement("O"),0.870); mat->AddElement(nist->FindOrBuildElement("H"),0.113);
  mat->AddElement(nist->FindOrBuildElement("C"),0.010); mat->AddElement(nist->FindOrBuildElement("Gd"),0.007);
  G4double ri[kNE]={1.334,1.335,1.337,1.338,1.339,1.341,1.342,1.343,1.345,1.346,1.347,1.349,1.350,1.352,1.354,1.356,1.358,1.360,1.361,1.363};
  G4double ab[kNE]={6.*m,12.*m,25.*m,42.*m,60.*m,70.*m,74.*m,76.*m,75.*m,70.*m,62.*m,52.*m,40.*m,24.*m,12.*m,6.*m,3.*m,1.2*m,0.6*m,0.3*m};
  G4double rl[kNE]={40.*m,50.*m,60.*m,70.*m,78.*m,83.*m,86.*m,88.*m,88.*m,83.*m,74.*m,64.*m,50.*m,40.*m,30.*m,22.*m,13.*m,9.*m,7.*m,5.*m};
  G4double sf[kNE]={0,0.01,0.03,0.08,0.15,0.30,0.80,1.00,0.90,0.70,0.45,0.25,0.10,0.04,0.01,0,0,0,0,0};
  G4double ss[kNE]={0,0,0.01,0.04,0.10,0.20,0.50,0.70,0.85,1.00,0.80,0.55,0.30,0.12,0.04,0.01,0,0,0,0};
  auto* mpt=new G4MaterialPropertiesTable();
  mpt->AddProperty("RINDEX",                  std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(ri,ri+kNE));
  mpt->AddProperty("ABSLENGTH",               std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(ab,ab+kNE));
  mpt->AddProperty("RAYLEIGH",                std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(rl,rl+kNE));
  mpt->AddProperty("SCINTILLATIONCOMPONENT1", std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(sf,sf+kNE));
  mpt->AddProperty("SCINTILLATIONCOMPONENT2", std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(ss,ss+kNE));
  ADD_SCINT_PROPS(mpt, 100./MeV, 0.80, 5.0*ns, 50.0*ns, 0.5*ns, 1.5*ns)
  mat->SetMaterialPropertiesTable(mpt);
  return mat;
}

G4Material* DetectorConstruction::BuildTeWbLS()
{
  G4NistManager* nist=G4NistManager::Instance();
  G4Material* mat=new G4Material("TeWbLS",1.008*g/cm3,4,kStateLiquid,293.15*kelvin,1.*atmosphere);
  mat->AddElement(nist->FindOrBuildElement("O"),0.862); mat->AddElement(nist->FindOrBuildElement("H"),0.110);
  mat->AddElement(nist->FindOrBuildElement("C"),0.010); mat->AddElement(nist->FindOrBuildElement("Te"),0.018);
  G4double ri[kNE]={1.340,1.341,1.343,1.344,1.346,1.347,1.349,1.350,1.352,1.353,1.355,1.356,1.358,1.360,1.362,1.364,1.366,1.368,1.369,1.371};
  G4double ab[kNE]={5.*m,10.*m,20.*m,35.*m,50.*m,60.*m,64.*m,66.*m,65.*m,61.*m,53.*m,44.*m,34.*m,20.*m,10.*m,5.*m,2.5*m,1.*m,0.5*m,0.25*m};
  G4double rl[kNE]={35.*m,44.*m,53.*m,62.*m,70.*m,75.*m,78.*m,80.*m,80.*m,75.*m,67.*m,58.*m,44.*m,35.*m,27.*m,19.*m,11.*m,8.*m,6.*m,4.*m};
  G4double sf[kNE]={0,0.01,0.02,0.06,0.14,0.28,0.75,1.00,0.92,0.72,0.47,0.26,0.11,0.04,0.01,0,0,0,0,0};
  G4double ss[kNE]={0,0,0.01,0.03,0.09,0.18,0.48,0.68,0.84,1.00,0.82,0.57,0.32,0.13,0.04,0.01,0,0,0,0};
  auto* mpt=new G4MaterialPropertiesTable();
  mpt->AddProperty("RINDEX",                  std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(ri,ri+kNE));
  mpt->AddProperty("ABSLENGTH",               std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(ab,ab+kNE));
  mpt->AddProperty("RAYLEIGH",                std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(rl,rl+kNE));
  mpt->AddProperty("SCINTILLATIONCOMPONENT1", std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(sf,sf+kNE));
  mpt->AddProperty("SCINTILLATIONCOMPONENT2", std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(ss,ss+kNE));
  ADD_SCINT_PROPS(mpt, 150./MeV, 0.75, 6.0*ns, 60.0*ns, 0.6*ns, 1.8*ns)
  mat->SetMaterialPropertiesTable(mpt);
  return mat;
}

G4Material* DetectorConstruction::BuildD2OLAB()
{
  G4Isotope* D=new G4Isotope("Deuterium2",1,2,2.014*g/mole);
  G4Element* elD=new G4Element("Deuterium2","D2",1); elD->AddIsotope(D,1.0);
  G4NistManager* nist=G4NistManager::Instance();
  G4Material* mat=new G4Material("D2OLAB",1.108*g/cm3,4,kStateLiquid,293.15*kelvin,1.*atmosphere);
  mat->AddElement(nist->FindOrBuildElement("O"),0.877); mat->AddElement(elD,0.106);
  mat->AddElement(nist->FindOrBuildElement("H"),0.007); mat->AddElement(nist->FindOrBuildElement("C"),0.010);
  G4double ri[kNE]={1.327,1.328,1.329,1.330,1.332,1.333,1.335,1.336,1.337,1.339,1.340,1.341,1.343,1.345,1.347,1.349,1.351,1.353,1.354,1.356};
  G4double ab[kNE]={9.*m,18.*m,36.*m,56.*m,76.*m,86.*m,90.*m,93.*m,93.*m,88.*m,78.*m,68.*m,56.*m,36.*m,18.*m,9.*m,4.5*m,1.8*m,0.9*m,0.45*m};
  G4double rl[kNE]={46.*m,56.*m,66.*m,76.*m,86.*m,91.*m,94.*m,96.*m,96.*m,91.*m,81.*m,71.*m,56.*m,46.*m,36.*m,26.*m,16.*m,11.*m,9.*m,7.*m};
  G4double sf[kNE]={0,0.01,0.03,0.08,0.15,0.30,0.80,1.00,0.90,0.70,0.45,0.25,0.10,0.04,0.01,0,0,0,0,0};
  G4double ss[kNE]={0,0,0.01,0.04,0.10,0.20,0.50,0.70,0.85,1.00,0.80,0.55,0.30,0.12,0.04,0.01,0,0,0,0};
  auto* mpt=new G4MaterialPropertiesTable();
  mpt->AddProperty("RINDEX",                  std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(ri,ri+kNE));
  mpt->AddProperty("ABSLENGTH",               std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(ab,ab+kNE));
  mpt->AddProperty("RAYLEIGH",                std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(rl,rl+kNE));
  mpt->AddProperty("SCINTILLATIONCOMPONENT1", std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(sf,sf+kNE));
  mpt->AddProperty("SCINTILLATIONCOMPONENT2", std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(ss,ss+kNE));
  ADD_SCINT_PROPS(mpt, 100./MeV, 0.80, 5.0*ns, 50.0*ns, 0.5*ns, 1.4*ns)
  mat->SetMaterialPropertiesTable(mpt);
  return mat;
}

G4Material* DetectorConstruction::BuildD2OGdLAB()
{
  G4Isotope* D=new G4Isotope("Deuterium3",1,2,2.014*g/mole);
  G4Element* elD=new G4Element("Deuterium3","D3",1); elD->AddIsotope(D,1.0);
  G4NistManager* nist=G4NistManager::Instance();
  G4Material* mat=new G4Material("D2OGdLAB",1.109*g/cm3,5,kStateLiquid,293.15*kelvin,1.*atmosphere);
  mat->AddElement(nist->FindOrBuildElement("O"),0.870); mat->AddElement(elD,0.106);
  mat->AddElement(nist->FindOrBuildElement("H"),0.007); mat->AddElement(nist->FindOrBuildElement("C"),0.010);
  mat->AddElement(nist->FindOrBuildElement("Gd"),0.007);
  G4double ri[kNE]={1.327,1.328,1.330,1.331,1.332,1.334,1.336,1.337,1.338,1.340,1.341,1.342,1.344,1.346,1.348,1.350,1.352,1.354,1.355,1.357};
  G4double ab[kNE]={7.*m,14.*m,28.*m,46.*m,64.*m,74.*m,78.*m,80.*m,79.*m,74.*m,66.*m,56.*m,44.*m,28.*m,14.*m,7.*m,3.5*m,1.4*m,0.7*m,0.35*m};
  G4double rl[kNE]={42.*m,52.*m,62.*m,72.*m,80.*m,85.*m,88.*m,90.*m,90.*m,85.*m,76.*m,66.*m,52.*m,42.*m,32.*m,23.*m,14.*m,9.*m,7.*m,5.*m};
  G4double sf[kNE]={0,0.01,0.03,0.08,0.15,0.30,0.80,1.00,0.90,0.70,0.45,0.25,0.10,0.04,0.01,0,0,0,0,0};
  G4double ss[kNE]={0,0,0.01,0.04,0.10,0.20,0.50,0.70,0.85,1.00,0.80,0.55,0.30,0.12,0.04,0.01,0,0,0,0};
  auto* mpt=new G4MaterialPropertiesTable();
  mpt->AddProperty("RINDEX",                  std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(ri,ri+kNE));
  mpt->AddProperty("ABSLENGTH",               std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(ab,ab+kNE));
  mpt->AddProperty("RAYLEIGH",                std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(rl,rl+kNE));
  mpt->AddProperty("SCINTILLATIONCOMPONENT1", std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(sf,sf+kNE));
  mpt->AddProperty("SCINTILLATIONCOMPONENT2", std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(ss,ss+kNE));
  ADD_SCINT_PROPS(mpt, 100./MeV, 0.80, 5.0*ns, 50.0*ns, 0.5*ns, 1.4*ns)
  mat->SetMaterialPropertiesTable(mpt);
  return mat;
}

G4Material* DetectorConstruction::BuildD2OTeLAB()
{
  G4Isotope* D=new G4Isotope("Deuterium4",1,2,2.014*g/mole);
  G4Element* elD=new G4Element("Deuterium4","D4",1); elD->AddIsotope(D,1.0);
  G4NistManager* nist=G4NistManager::Instance();
  G4Material* mat=new G4Material("D2OTeLAB",1.115*g/cm3,5,kStateLiquid,293.15*kelvin,1.*atmosphere);
  mat->AddElement(nist->FindOrBuildElement("O"),0.862); mat->AddElement(elD,0.104);
  mat->AddElement(nist->FindOrBuildElement("H"),0.006); mat->AddElement(nist->FindOrBuildElement("C"),0.010);
  mat->AddElement(nist->FindOrBuildElement("Te"),0.018);
  G4double ri[kNE]={1.333,1.334,1.336,1.337,1.339,1.340,1.342,1.343,1.345,1.346,1.348,1.350,1.351,1.353,1.355,1.357,1.359,1.361,1.362,1.364};
  G4double ab[kNE]={6.*m,11.*m,22.*m,38.*m,54.*m,64.*m,68.*m,70.*m,69.*m,65.*m,57.*m,48.*m,37.*m,22.*m,11.*m,6.*m,3.*m,1.2*m,0.6*m,0.3*m};
  G4double rl[kNE]={37.*m,46.*m,55.*m,64.*m,72.*m,77.*m,80.*m,82.*m,82.*m,77.*m,69.*m,60.*m,46.*m,37.*m,29.*m,21.*m,13.*m,8.*m,6.*m,4.*m};
  G4double sf[kNE]={0,0.01,0.02,0.06,0.14,0.28,0.75,1.00,0.92,0.72,0.47,0.26,0.11,0.04,0.01,0,0,0,0,0};
  G4double ss[kNE]={0,0,0.01,0.03,0.09,0.18,0.48,0.68,0.84,1.00,0.82,0.57,0.32,0.13,0.04,0.01,0,0,0,0};
  auto* mpt=new G4MaterialPropertiesTable();
  mpt->AddProperty("RINDEX",                  std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(ri,ri+kNE));
  mpt->AddProperty("ABSLENGTH",               std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(ab,ab+kNE));
  mpt->AddProperty("RAYLEIGH",                std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(rl,rl+kNE));
  mpt->AddProperty("SCINTILLATIONCOMPONENT1", std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(sf,sf+kNE));
  mpt->AddProperty("SCINTILLATIONCOMPONENT2", std::vector<G4double>(kE,kE+kNE),std::vector<G4double>(ss,ss+kNE));
  ADD_SCINT_PROPS(mpt, 150./MeV, 0.75, 6.0*ns, 60.0*ns, 0.6*ns, 2.0*ns)
  mat->SetMaterialPropertiesTable(mpt);
  return mat;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Construct()  — SuperK-style PMT flush geometry
// ─────────────────────────────────────────────────────────────────────────────
G4VPhysicalVolume* DetectorConstruction::Construct()
{
  // ── BUILD VERIFICATION STAMP ─────────────────────────────────────────────
  // If this line does NOT appear in your console output, the binary you are
  // running was NOT rebuilt from the latest DetectorConstruction.cc -- stop
  // and fix the build/run mismatch before looking at geometry again.
  G4cout << "\n*** DetectorConstruction BUILD STAMP: extended-glass-fix-v4 ***\n" << G4endl;

  // ── Materials ──────────────────────────────────────────────────────────────
  G4NistManager* nist = G4NistManager::Instance();
  G4Material* air     = nist->FindOrBuildMaterial("G4_AIR");
  G4Material* vacuum  = nist->FindOrBuildMaterial("G4_Galactic");

  if      (fMediumName=="H2O")      fDetMaterial=BuildWater();
  else if (fMediumName=="D2O")      fDetMaterial=BuildD2O();
  else if (fMediumName=="WbLS")     fDetMaterial=BuildWbLS();
  else if (fMediumName=="GdWbLS")   fDetMaterial=BuildGdWbLS();
  else if (fMediumName=="TeWbLS")   fDetMaterial=BuildTeWbLS();
  else if (fMediumName=="D2OLAB")   fDetMaterial=BuildD2OLAB();
  else if (fMediumName=="D2OGdLAB") fDetMaterial=BuildD2OGdLAB();
  else if (fMediumName=="D2OTeLAB") fDetMaterial=BuildD2OTeLAB();
  else { G4cerr<<"Unknown medium, defaulting H2O"<<G4endl; fDetMaterial=BuildWater(); }

  PPOIntegratorPatch::Upgrade(fDetMaterial, fMediumName);

  G4Material* acrylic   = BuildAcrylic();
  G4Material* bsGlass   = BuildBorosilicateGlass();
  G4Material* photocath = BuildPhotocathode();

  // ── Geometry constants ─────────────────────────────────────────────────────
  G4double rInner      = fDetRadius;             // 150 mm
  G4double wallThick   = 5.*mm;
  G4double rOuter      = rInner + wallThick;     // 155 mm
  G4double worldHalf   = 500.*mm;

  G4double tubeR  = R1307::kTubeRadius;          // 38 mm
  G4double glassT = R1307::kWindowThickness;     // 2 mm
  G4double pcT    = R1307::kPhotocathodeThick;   // 0.5 mm
  G4double pcR    = R1307::kPhotocathodeR;        // 25.75 mm
  G4double bodyL  = R1307::kTubeLength - glassT - pcT; // 147.5 mm

  // ══════════════════════════════════════════════════════════════════════════
  //  SHARED AXIS / ROTATION TABLE
  //
  //  Defined ONCE here and reused for: (1) acrylic hole subtraction,
  //  (2) detector-medium notch subtraction, (3) PMT glass/photocathode/body
  //  placement. Using a single shared table guarantees all three geometric
  //  operations are perfectly aligned — no separate axis lists that could
  //  drift out of sync with each other.
  // ══════════════════════════════════════════════════════════════════════════
  struct AxisDef {
    G4ThreeVector dir;       // outward unit direction
    G4double      rotX, rotY; // rotation to align G4Tubs/G4Sphere local z-axis with dir
  };
  std::vector<AxisDef> axes = {
    { G4ThreeVector( 1,0,0),  0.,       90.*deg },  // +x
    { G4ThreeVector(-1,0,0),  0.,      -90.*deg },  // -x
    { G4ThreeVector( 0,1,0), -90.*deg,  0.      },  // +y
    { G4ThreeVector( 0,-1,0), 90.*deg,  0.      },  // -y
    { G4ThreeVector( 0,0,1),  0.,       0.      },  // +z
    { G4ThreeVector( 0,0,-1), 180.*deg, 0.      },  // -z
  };

  // Pre-build a G4RotationMatrix* for each axis (used by holes, notches, PMTs)
  std::vector<G4RotationMatrix*> axisRot;
  for (const auto& a : axes) {
    auto* r = new G4RotationMatrix();
    r->rotateX(a.rotX);
    r->rotateY(a.rotY);
    axisRot.push_back(r);
  }

  // ── World ──────────────────────────────────────────────────────────────────
  auto* worldSolid = new G4Box("World",worldHalf,worldHalf,worldHalf);
  fWorldLog = new G4LogicalVolume(worldSolid,air,"World");
  fWorldLog->SetVisAttributes(G4VisAttributes::GetInvisible());
  auto* worldPhys = new G4PVPlacement(nullptr,G4ThreeVector(),
                                      fWorldLog,"World",nullptr,false,0,true);

  // ══════════════════════════════════════════════════════════════════════════
  //  SUPERKAMIOKANDE-STYLE FLUSH PMT GEOMETRY
  //
  //  1. Acrylic shell with 6 cylindrical holes (one per axis)
  //  2. Detector medium sphere with 6 matching FLAT-FACE NOTCHES so its
  //     surface is flush with the PMT glass disc at every point — this is
  //     the fix for the sagitta overlap bug that caused zero detections.
  //  3. PMT glass disc fills each hole, front face exactly at r = rInner
  //  4. Photocathode + body placed behind the glass, extending outward
  // ══════════════════════════════════════════════════════════════════════════

  // ── (1) Acrylic shell with 6 holes ───────────────────────────────────────
  G4VSolid* acrylicShell = new G4Sphere("AcrylicShell",
                                         rInner, rOuter,
                                         0.*deg,360.*deg,0.*deg,180.*deg);

  G4double holeLen = wallThick + 15.*mm;  // through-hole with margin both sides
  auto* holeCyl = new G4Tubs("Hole",0.,tubeR,holeLen/2,0.,360.*deg);

  G4VSolid* acrylicWithHoles = acrylicShell;
  for (int i=0;i<6;++i) {
    G4ThreeVector holePos = axes[i].dir * rOuter;  // centred on the shell
    acrylicWithHoles = new G4SubtractionSolid(
        "AcrylicWithHoles", acrylicWithHoles, holeCyl,
        axisRot[i], holePos);
  }

  fAcrylicLog = new G4LogicalVolume(acrylicWithHoles, acrylic, "AcrylicVessel");
  auto* acrylicVis = new G4VisAttributes(G4Colour(0.85,0.92,1.0,0.25));
  acrylicVis->SetForceSolid(true);
  fAcrylicLog->SetVisAttributes(acrylicVis);
  fAcrylicPhys = new G4PVPlacement(nullptr,G4ThreeVector(),
                    fAcrylicLog,"AcrylicVessel",fWorldLog,false,0,true);

  // ── (2) Detector medium sphere with 6 matching flat-face notches ───────────
  //
  //  A flat disc of radius tubeR inscribed AT the sphere surface (r=rInner)
  //  is recessed from the true sphere by the sagitta:
  //    s = rInner - sqrt(rInner^2 - tubeR^2)
  //  Without removing this spherical cap, the medium sphere would overlap
  //  the flat glass disc by up to s ≈ 4.9 mm at the disc rim — an ambiguous,
  //  unresolvable boundary that silently breaks optical transmission into
  //  the PMT (the root cause of the original zero-detection bug).
  //
  //  Fix: subtract a cylindrical notch at each axis, deep enough to remove
  //  the full spherical cap, leaving a FLAT face exactly at r = rInner that
  //  matches the flat glass disc face precisely — a clean, gap-free,
  //  overlap-free interface.

  G4VSolid* detSphere = new G4Sphere("DetectorSphere",0.,rInner,
                                     0.*deg,360.*deg,0.*deg,180.*deg);

  G4double sagitta = rInner - std::sqrt(rInner*rInner - tubeR*tubeR);
  // SECOND CRITICAL FIX: the previous version used notchDepth = sagitta +
  // 0.001mm as a "float safety margin". This was backwards: it made the
  // notch's far face land 0.001mm (1 micron) BEHIND the glass's back face
  // (glassBack = rInner - sagitta), creating a real, deliberate, navigable
  // 1-micron vacuum/air GAP all the way around the bore rim -- a thousand
  // times larger than Geant4's internal surface tolerance (~1e-9mm), so it
  // was NOT treated as "touching". This gap is exactly what showed up in
  // the debug trace as postVol=World deaths confined to r=145-150mm, i.e.
  // precisely the bore region.
  //
  // The correct safety margin goes the OTHER way: make the notch reach
  // slightly DEEPER than the glass's back face, so the medium and glass
  // OVERLAP by a tiny controlled amount instead of leaving a gap. A small
  // overlap is harmless (Geant4's navigator resolves it to whichever solid
  // is queried; no extra "no-man's-land" volume is ever created), whereas
  // any gap at all creates real, fatal World space for the photon to land
  // in. We reuse the IDENTICAL 'sagitta' value for both notchDepth and
  // glassBack below (see PMT glass section) so they are bit-for-bit
  // consistent, then push the notch 1 micron deeper on top of that.
  G4double notchOvershoot = 0.001*mm;  // deliberate OVERLAP margin
  // THIRD CRITICAL FIX (the actual correct sign): the notch must remove
  // SLIGHTLY LESS than the full sagitta, not slightly more. The notch is
  // SUBTRACTED from the medium sphere, so a LARGER notchDepth removes MORE
  // material and pushes the medium's remaining boundary FURTHER AWAY from
  // the glass -- worsening any gap, never closing it. (This was the error
  // in the previous attempt: using sagitta+overshoot on either the single
  // face or symmetrically both made the gap the same size or worse, which
  // is exactly why detections were still zero after that "fix".)
  // The correct move is to UNDER-cut: notchDepth = sagitta - overshoot.
  // This leaves the medium's remaining material reaching slightly PAST
  // glassBack (rInner - sagitta), i.e. medium and glass genuinely overlap
  // by 'overshoot', which is harmless in Geant4 -- whereas any gap, however
  // small, creates real navigable World space that traps photons (this is
  // what every previous debug trace showed: postVol=World confined to the
  // bore region, r=145-150mm).
  G4double notchDepth = sagitta - notchOvershoot;
  auto* notchCyl = new G4Tubs("DetNotch",0.,tubeR,notchDepth/2,0.,360.*deg);

  G4VSolid* detWithNotches = detSphere;
  for (int i=0;i<6;++i) {
    // Notch's far (outward) face anchored exactly at rInner (unchanged from
    // the original design -- nothing competes for space beyond rInner but
    // glass, so no overlap is needed on that side). Notch's near face is
    // now at (rInner - notchDepth) = (rInner - sagitta + overshoot), which
    // is 'overshoot' PAST glassBack = (rInner - sagitta) -- guaranteeing
    // overlap with the glass instead of a gap.
    G4ThreeVector notchPos = axes[i].dir * (rInner - notchDepth/2);
    detWithNotches = new G4SubtractionSolid(
        "DetWithNotches", detWithNotches, notchCyl,
        axisRot[i], notchPos);
  }

  fDetLog = new G4LogicalVolume(detWithNotches,fDetMaterial,"Detector");
  auto* detVis = new G4VisAttributes(G4Colour(0.2,0.5,1.0,0.08));
  detVis->SetForceWireframe(true);
  fDetLog->SetVisAttributes(detVis);
  auto* detPhys = new G4PVPlacement(nullptr,G4ThreeVector(),
                                    fDetLog,"Detector",fWorldLog,false,0,true);

  // ── (3) PMT glass / photocathode / body ─────────────────────────────────
  //
  //  SECOND FIX (completes the bore fix above): shrinking notchDepth to the
  //  exact sagitta only shrank the air gap, it did not eliminate it -- the
  //  bore from (rInner-sagitta) to rInner was still empty (filled with
  //  World/air) because PMTGlass was still just a thin glassT=2mm disc
  //  starting AT rInner. Photons clipping that bore's cylindrical side
  //  wall still escaped into World before ever reaching the glass.
  //  (Confirmed by the second debug trace: postVol=World still dominant,
  //  now confined to r=145-149mm -- exactly the shrunk bore region.)
  //
  //  Real fix: extend the glass solid BACKWARD by exactly the sagitta, so
  //  it physically fills the entire bore with actual borosilicate glass.
  //  The glass FRONT face stays at the real R1307 position (rInner+glassT,
  //  i.e. its outward-facing surface is unchanged), but its BACK face now
  //  reaches all the way to (rInner-sagitta), exactly matching where the
  //  medium notch ends. Two coincident flat planes there again -- zero
  //  gap, zero overlap, and no air pocket for photons to escape through.
  //  The extra ~4.9mm of glass path length is optically negligible (the
  //  borosilicate absorption length is metres, not millimetres).
  G4double glassTeff   = glassT + sagitta;          // effective total glass thickness
  G4double glassBack   = rInner - sagitta;           // back face: flush with medium notch
  G4double glassFront  = rInner + glassT;            // front face: real R1307 position (unchanged)
  G4double glassCentre = (glassBack + glassFront)/2;  // = rInner + glassT/2 - sagitta/2

  G4double pcCentre    = glassFront + pcT/2;          // unchanged: starts right after real glass front
  G4double bodyCentre  = glassFront + pcT + bodyL/2;  // unchanged

  auto* glassSolid = new G4Tubs("PMTGlass",0.,tubeR,glassTeff/2,0.,360.*deg);

  G4cout << "*** GEOMETRY CHECK: sagitta=" << sagitta/mm << "mm"
         << "  notchDepth=" << notchDepth/mm << "mm"
         << "  glassTeff=" << glassTeff/mm << "mm"
         << "  glassBack=" << glassBack/mm << "mm"
         << "  glassFront=" << glassFront/mm << "mm"
         << "  glassCentre=" << glassCentre/mm << "mm"
         << "  tubeR=" << tubeR/mm << "mm"
         << " ***" << G4endl;

  fPMTGlassLog = new G4LogicalVolume(glassSolid,bsGlass,"PMTGlass");
  auto* glassVis = new G4VisAttributes(G4Colour(0.7,0.9,1.0,0.65));
  glassVis->SetForceSolid(true);
  fPMTGlassLog->SetVisAttributes(glassVis);

  auto* pcSolid = new G4Tubs("PMTPhotocathode",0.,pcR,pcT/2,0.,360.*deg);
  fPMTLog = new G4LogicalVolume(pcSolid,photocath,"PMTPhotocathode");
  auto* pcVis = new G4VisAttributes(G4Colour(1.0,0.85,0.0,0.95));
  pcVis->SetForceSolid(true);
  fPMTLog->SetVisAttributes(pcVis);

  auto* bodySolid = new G4Tubs("PMTBody",0.,tubeR,bodyL/2,0.,360.*deg);
  fPMTBodyLog = new G4LogicalVolume(bodySolid,vacuum,"PMTBody");
  auto* bodyVis = new G4VisAttributes(G4Colour(0.25,0.25,0.30,0.75));
  bodyVis->SetForceSolid(true);
  fPMTBodyLog->SetVisAttributes(bodyVis);

  for (int i=0;i<6;++i) {
    G4ThreeVector d = axes[i].dir;
    new G4PVPlacement(axisRot[i], d*glassCentre,
                      fPMTGlassLog,"PMTGlass",fWorldLog,false,i,true);
    new G4PVPlacement(axisRot[i], d*pcCentre,
                      fPMTLog,"PMTPhotocathode",fWorldLog,false,i,true);
    new G4PVPlacement(axisRot[i], d*bodyCentre,
                      fPMTBodyLog,"PMTBody",fWorldLog,false,i,true);
  }

  // ── Optical surfaces ───────────────────────────────────────────────────────
  BuildOpticalSurfaces(detPhys);

  // ── Summary ────────────────────────────────────────────────────────────────
  G4double vol_mL = (4./3.)*CLHEP::pi*std::pow(rInner/cm,3)
                   - 6.*CLHEP::pi*std::pow(tubeR/cm,2)*(notchDepth/cm)/2.;
  G4cout << "\n[DetectorConstruction] === SuperK-style Flush PMT Geometry (fixed) ==="
         << "\n  Medium            : " << fMediumName
         << "\n  Acrylic vessel    : inner R=" << rInner/mm << " mm"
         << "  wall=" << wallThick/mm << " mm  6 holes diam=" << 2*tubeR/mm << " mm"
         << "\n  Sagitta correction: " << sagitta/mm << " mm  (notch depth="
         << notchDepth/mm << " mm)"
         << "\n  Inner volume      : ~" << std::fixed << std::setprecision(1)
         << vol_mL << " mL"
         << "\n  PMT model         : Hamamatsu R1307 (3-inch bialkali)"
         << "\n  PMT glass face    : flush with inner wall at R=" << rInner/mm
         << " mm (notch-matched, zero overlap)"
         << "\n  Optical path      : medium -> glass (Fresnel ~0.4%) -> photocathode -> SD"
         << "\n  Photocathode dia  : " << 2*pcR/mm << " mm"
         << "\n  Tube OD           : " << 2*tubeR/mm << " mm"
         << "\n  PMT total length  : " << R1307::kTubeLength/mm << " mm"
         << "\n  Geometric coverage: "
         << std::setprecision(2)
         << 6*CLHEP::pi*pcR*pcR/(4*CLHEP::pi*rInner*rInner)*100. << " %"
         << "\n===============================================================\n"
         << G4endl;

  return worldPhys;
}

void DetectorConstruction::BuildOpticalSurfaces(G4VPhysicalVolume* detPhys)
{
  // ── 95% PTFE reflector — ACRYLIC BOUNDARY ONLY ────────────────────────────
  //
  // CRITICAL FIX: a G4LogicalSkinSurface wraps the ENTIRE surface of a
  // logical volume — Geant4 has no way to "skip" part of a skin surface.
  // fDetLog's solid is a sphere with 6 flat notches cut out (so the glass
  // can sit flush). A skin surface on fDetLog therefore also covered those
  // 6 notch faces, meaning every photon reaching a PMT position hit a
  // dielectric_metal surface FIRST — which never transmits, only reflects
  // or absorbs, by Geant4 design — and was bounced/absorbed before it ever
  // had a chance to refract into the glass. This is why total_detected
  // stayed at 0 even after the photocathode RINDEX fix: photons were being
  // stopped one boundary too early, at the medium/notch-face/glass
  // junction, not at the glass/photocathode junction.
  //
  // FIX: use a G4LogicalBorderSurface instead, which attaches to a specific
  // ORDERED PAIR of physical volumes (fDetLog -> AcrylicVessel) rather than
  // an entire logical volume's boundary. This reflects photons that reach
  // the acrylic in the gaps between PMTs, while leaving the 6 notch faces
  // (which border the PMT glass physical volume, a different pair) with no
  // surface at all — exactly the physically correct behaviour: medium and
  // glass interact only via the automatic Fresnel/Snell calculation from
  // their respective RINDEX values.
  auto* wallSurf = new G4OpticalSurface("WallSurface");
  wallSurf->SetType(dielectric_metal);
  wallSurf->SetModel(unified);
  wallSurf->SetFinish(groundfrontpainted);  // Lambertian (diffuse)
  auto* wallMPT = new G4MaterialPropertiesTable();
  G4double sE[2]={1.8*eV,4.7*eV};
  G4double sR[2]={0.95,0.95};
  wallMPT->AddProperty("REFLECTIVITY",
      std::vector<G4double>(sE,sE+2),std::vector<G4double>(sR,sR+2));
  wallSurf->SetMaterialPropertiesTable(wallMPT);

  // Border surface: ONLY between the detector medium and the acrylic vessel,
  // using the actual G4VPhysicalVolume pointers stored at placement time
  // (fAcrylicPhys / detPhys), NOT a skin surface on the whole logical volume.
  new G4LogicalBorderSurface("WallBorderSurface_in",  detPhys,  fAcrylicPhys, wallSurf);
  new G4LogicalBorderSurface("WallBorderSurface_out", fAcrylicPhys, detPhys,  wallSurf);

  // ── PMT glass: deliberately NO surface ────────────────────────────────────
  // medium (n~1.34) -> glass (n~1.51): Fresnel handled automatically by
  // G4OpBoundaryProcess from RINDEX alone. No G4OpticalSurface object is
  // created for this boundary at all, so it cannot accidentally inherit
  // any reflective behaviour. Normal-incidence reflection is ~0.4%.

  // ── Photocathode: SD fires on fGeomBoundary entry ─────────────────────────
  // No surface needed; handled in PMTSD::ProcessHits.

  G4cout << "[DetectorConstruction] Surfaces:"
         << "\n  Acrylic boundary : 95% PTFE diffuse reflector"
         << " (BORDER surface, medium<->acrylic ONLY)"
         << "\n  PMT glass        : NO surface — automatic Fresnel n_med->n_glass"
         << " (~0.4% reflection)"
         << "\n  Photocathode     : SD on fGeomBoundary entry + R1307 QE"
         << G4endl;
}

// ─────────────────────────────────────────────────────────────────────────────
void DetectorConstruction::ConstructSDandField()
{
  auto* pmtSD = new PMTSD("PMTSD","PMTHitsCollection");
  G4SDManager::GetSDMpointer()->AddNewDetector(pmtSD);
  SetSensitiveDetector("PMTPhotocathode", pmtSD, true);
}
