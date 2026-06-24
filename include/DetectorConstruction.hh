#ifndef DETECTORCONSTRUCTION_HH
#define DETECTORCONSTRUCTION_HH

#include "G4VUserDetectorConstruction.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Material.hh"
#include "globals.hh"

class DetectorMessenger;

class DetectorConstruction : public G4VUserDetectorConstruction
{
public:
  DetectorConstruction();
  ~DetectorConstruction() override;

  G4VPhysicalVolume* Construct() override;
  void ConstructSDandField() override;

  void SetMediumName   (const G4String& name);
  void SetDetectorRadius(G4double r);

  G4String GetMediumName()     const { return fMediumName; }
  G4double GetDetectorRadius() const { return fDetRadius;  }
  G4int    GetNPMTs()          const { return fNPMTs;      }
  G4LogicalVolume* GetPMTLogical() const { return fPMTLog; }

private:
  // ── Material builders ─────────────────────────────────────────────────────
  G4Material* BuildAcrylic();
  G4Material* BuildBorosilicateGlass();
  G4Material* BuildPhotocathode();

  G4Material* BuildWater();
  G4Material* BuildD2O();
  G4Material* BuildWbLS();
  G4Material* BuildGdWbLS();
  G4Material* BuildTeWbLS();
  G4Material* BuildD2OLAB();
  G4Material* BuildD2OGdLAB();
  G4Material* BuildD2OTeLAB();

  void BuildOpticalSurfaces(G4VPhysicalVolume* detPhys);

  // ── Data members ──────────────────────────────────────────────────────────
  G4String  fMediumName;
  G4double  fDetRadius;
  G4int     fNPMTs;
  G4double  fPMTRadius;

  G4Material*      fDetMaterial  = nullptr;

  G4LogicalVolume* fWorldLog     = nullptr;
  G4LogicalVolume* fAcrylicLog   = nullptr;
  G4LogicalVolume* fDetLog       = nullptr;
  G4LogicalVolume* fPMTGlassLog  = nullptr;
  G4LogicalVolume* fPMTLog       = nullptr;   // photocathode (SD)
  G4LogicalVolume* fPMTBodyLog   = nullptr;
  G4VPhysicalVolume* fAcrylicPhys = nullptr;

  DetectorMessenger* fMessenger  = nullptr;
};

#endif

/*
#ifndef DETECTORCONSTRUCTION_HH
#define DETECTORCONSTRUCTION_HH

#include "G4VUserDetectorConstruction.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Material.hh"
#include "G4OpticalSurface.hh"
#include "globals.hh"
#include <vector>
#include <string>

class G4Box;
class G4Sphere;
class G4Tubs;
class DetectorMessenger;

class DetectorConstruction : public G4VUserDetectorConstruction
{
public:
  DetectorConstruction();
  ~DetectorConstruction() override;

  G4VPhysicalVolume* Construct() override;
  void ConstructSDandField() override;

  // Setters called from macro/messenger
  void SetMediumName(const G4String& name);
  void SetDetectorRadius(G4double r);

  // Getters used by analysis
  G4String       GetMediumName()     const { return fMediumName; }
  G4double       GetDetectorRadius() const { return fDetRadius;  }
  G4int          GetNPMTs()          const { return fNPMTs;      }
  G4LogicalVolume* GetPMTLogical()   const { return fPMTLog;     }

private:
  // ── Material definitions ──────────────────────────────────────────────────
  void DefineMaterials();

  G4Material* BuildWater();
  G4Material* BuildD2O();
  G4Material* BuildLAB();
  G4Material* BuildWbLS();         // H2O + 1% LAB
  G4Material* BuildGdWbLS();       // H2O + Gd-doped LAB
  G4Material* BuildTeWbLS();       // H2O + Te-doped LAB
  G4Material* BuildD2OLAB();       // D2O + 1% LAB
  G4Material* BuildD2OGdLAB();     // D2O + Gd-doped LAB
  G4Material* BuildD2OTeLAB();     // D2O + Te-doped LAB

  void SetOpticalProperties(G4Material* mat, const G4String& tag);

  // Optical property tables
  void AddWaterOpticalProperties    (G4Material* mat);
  void AddD2OOpticalProperties      (G4Material* mat);
  void AddWbLSOpticalProperties     (G4Material* mat);
  void AddGdWbLSOpticalProperties   (G4Material* mat);
  void AddTeWbLSOpticalProperties   (G4Material* mat);
  void AddD2OLABOpticalProperties   (G4Material* mat);
  void AddD2OGdLABOpticalProperties (G4Material* mat);
  void AddD2OTeLABOpticalProperties (G4Material* mat);

  // PMT surface
  void BuildPMTSurface();

  // ── Data members ──────────────────────────────────────────────────────────
  G4String  fMediumName;
  G4double  fDetRadius;
  G4int     fNPMTs;
  G4double  fPMTRadius;   // physical sensitive disk radius

  G4Material*     fDetMaterial  = nullptr;
  G4LogicalVolume* fWorldLog    = nullptr;
  G4LogicalVolume* fDetLog      = nullptr;
  G4LogicalVolume* fPMTLog      = nullptr;

  DetectorMessenger* fMessenger = nullptr;
};

#endif
*/
