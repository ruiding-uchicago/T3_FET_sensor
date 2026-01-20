# FET Descriptor Query System

Smart material property extraction system for Field-Effect Transistor (FET) applications. Queries OPTIMADE databases, extracts 32 physics-validated FET descriptors, and caches results for instant retrieval.

## Quick Start

```bash
# One-command query (checks cache first, queries OPTIMADE if needed)
python get_fet_descriptor.py "GaN"

# Show summary
python get_fet_descriptor.py "Fe2O3" --summary

# Force refresh (ignore cache)
python get_fet_descriptor.py "GaN" --force
```

**Performance:**
- Cached materials (181): < 1 second
- New material: ~25 seconds (query + extract)
- Second query: < 1 second (from cache)

---

## System Overview

### What It Does

1. **Parse** material name → standardized formula (GaN, Fe2O3, etc.)
   - 5-layer parsing strategy (see below)
   - Handles chemical formulas, common names, and mineral names
2. **Check cache** (FET_descriptor_filtered/) for 181 pre-computed materials
3. **If cached** → return JSON instantly ✓
4. **If not cached**:
   - Query OPTIMADE databases (Materials Project, JARVIS, OQMD, Alexandria, etc.)
   - Extract 32 FET descriptors with physics validation
   - Save to cache
   - Return JSON ✓

### Material Name Parsing (5-Layer Strategy)

**Layer 1: Domain Mapping (FASTEST - Priority)**
- 182 formulas with 405 name variations in `domain_frequent_name_mapping.json`
- Instant lookup, no API calls
- Examples:
  - Common names: "sapphire" → Al2O3, "iron oxide" → Fe2O3
  - Mineral names: "pyrite" → FeS2, "galena" → PbS, "wurtzite" → ZnS
  - Technical names: "rutile/anatase" → TiO2, "hematite" → Fe2O3

**Layer 2: Element Names**
- All element names and symbols
- Examples: "gold" → Au, "silicon" → Si

**Layer 3: Formula Parsing**
- Direct chemical formulas via pymatgen
- Examples: "GaN" → GaN, "HfO2" → HfO2

**Layer 4: PubChem API (Fallback for Unknown Materials)**
- Queries online chemical database for materials NOT in domain mapping
- Handles complex/rare materials automatically
- Examples:
  - "lithium niobate" → LiNbO3
  - "calcium fluoride" → CaF2
  - "gallium phosphide" → GaP
  - "boron phosphide" → BP

**Layer 5: Fallback**
- Returns original input with warning if all layers fail

### Cached Materials (181)

**III-V Semiconductors** (12):
- GaN, AlN, InN, GaAs, InP, InAs, InSb, AlAs, AlSb, GaSb, GaSe, GaP

**II-VI Semiconductors** (9):
- CdS, CdSe, CdTe, ZnO, ZnS, ZnSe, ZnTe, HgTe

**Group IV** (5):
- Si, Ge, C (graphene/diamond), SiC, Mg2Si

**Oxides** (30+):
- Al2O3, Fe2O3, Fe3O4, FeO, CuO, Cu2O, ZnO, TiO2, HfO2, SnO2, In2O3, Ga2O3
- BeO, BaO, CaO, SrO, MgO, Sc2O3, Cr2O3, Y2O3, La2O3, CeO2, Nb2O5, Ta2O5
- V2O5, VO2, WO3, MoO3, RuO2, IrO2, Rh2O3, Bi2O3, and more...

**Perovskites & Complex Oxides** (10+):
- BaTiO3, SrTiO3, PbTiO3, LaAlO3, CaTiO3, SrCoO3, and more...

**2D Materials** (8):
- MoS2, WS2, MoSe2, WSe2, MoTe2, WTe2, graphene, h-BN, SnS2, SnSe2

**Chalcogenides** (10+):
- Bi2Se3, Bi2Te3, Sb2Te3, PbS, CuS, FeS2, ReS2, PtSe2, and more...

**Nitrides** (9):
- AlN, GaN, InN, BN, Si3N4, TiN, TaN, HfN, CrN, ScN, VN

**Carbides** (4):
- SiC, TiC, WC, B4C

**Halides** (8):
- AgCl, CaCl2, MgCl2, NaCl, KCl, LiCl, CaF2, MgF2, and more...

**Metals** (20+):
- Fe, Cu, Zn, Au, Pt, Ag, Al, Ti, Ni, Co, W, Ta, Ru, Ir, Pd, Hg, Ca, Mg, and more...

See `FET_descriptor_filtered/` for complete list (181 materials).

---

## 32 FET Descriptors

**Total**: 32 descriptors (26 base + 6 derived)

### BASE DESCRIPTORS (26)

Extracted from OPTIMADE databases with smart multi-source fallback.

#### Composition & Structure (7)

| Descriptor | Unit | Physical Meaning |
|------------|------|------------------|
| `material_formula` | - | Material chemical formula (e.g., GaN, Fe2O3) |
| `chemical_formula_reduced` | - | Reduced chemical formula with smallest integer stoichiometry |
| `elements` | - | List of chemical elements present |
| `nelements` | count | Number of unique chemical elements |
| `crystal_system` | - | Crystal system (cubic, hexagonal, tetragonal, orthorhombic, etc.) |
| `space_group` | - | International space group number (1-230) |
| `nperiodic_dimensions` | count | Number of periodic dimensions (3=bulk, 2=layered, 1=chain, 0=molecule) |

#### Thermodynamics (3)

| Descriptor | Unit | Physical Meaning |
|------------|------|------------------|
| `density` | g/cm³ | Mass density of the material |
| `formation_energy_per_atom` | eV/atom | DFT formation energy per atom (negative = thermodynamically favorable) |
| `energy_above_hull` | eV/atom | Distance to thermodynamic ground state (0 = stable) |

#### Electronic & Transport (5)

| Descriptor | Unit | Physical Meaning |
|------------|------|------------------|
| `band_gap` | eV | Electronic band gap (smart extraction: avoids metallic magnetic artifacts) |
| `band_gap_direct` | eV | Direct band gap (if available) |
| `dos_ef` | states/eV/cell | Density of states at Fermi level (high DOS → metallic) |
| `me_avg` | m₀ | Average electron effective mass |
| `mh_avg` | m₀ | Average hole effective mass |

#### Dielectric (6)

| Descriptor | Unit | Physical Meaning |
|------------|------|------------------|
| `epsilon_x` | - | Static dielectric constant along x-axis |
| `epsilon_y` | - | Static dielectric constant along y-axis |
| `epsilon_z` | - | Static dielectric constant along z-axis |
| `dielectric_total` | - | Total static dielectric constant (electronic + ionic) |
| `dielectric_electronic` | - | Electronic contribution (high-frequency limit) |
| `dielectric_ionic` | - | Ionic contribution (lattice vibrations) |

#### Mechanical (3)

| Descriptor | Unit | Physical Meaning |
|------------|------|------------------|
| `k_vrh` | GPa | Bulk modulus (VRH average); resistance to compression |
| `g_vrh` | GPa | Shear modulus (VRH average); resistance to shear |
| `poisson_ratio` | - | Poisson's ratio (ν); transverse/axial strain ratio |

#### 2D & Magnetism (2)

| Descriptor | Unit | Physical Meaning |
|------------|------|------------------|
| `exfoliation_energy_mev_a2` | meV/Å² | Energy to exfoliate one monolayer from bulk |
| `magnetization` | μ_B/cell | Total magnetization per unit cell |

---

### DERIVED DESCRIPTORS (6)

Automatically computed from base descriptors.

#### 1. `eg_class` - Band Gap Classification

**Unit**: categorical (metal / narrow / moderate / wide)

**Formula**:
```
Eg < 0.05 eV      → metal
0.05 ≤ Eg < 1.0   → narrow
1.0 ≤ Eg < 3.0    → moderate
Eg ≥ 3.0 eV       → wide
```

**Interpretation**:
- **metal**: Metallic conductor
- **narrow**: IR/THz devices
- **moderate**: Visible/UV optoelectronics
- **wide**: Power electronics, deep-UV

#### 2. `eps_mean` - Mean Dielectric Constant

**Unit**: dimensionless

**Formula**: `(εₓ + εᵧ + εᵧ) / 3`

**Interpretation**: Higher → stronger gate coupling

#### 3. `eps_aniso` - Dielectric Anisotropy

**Unit**: dimensionless

**Formula**: `(max(ε) - min(ε)) / eps_mean`

**Interpretation**:
- ≈0: Isotropic
- >0.5: Highly anisotropic (2D materials)

#### 4. `vdw_ready` - 2D/vdW Readiness

**Unit**: categorical (layered_structure / easy / potential / no)

**Formula**:
```
nperiodic_dimensions ≤ 2 → layered_structure
E_exf < 30 meV/Å²        → easy (graphene-like)
30 ≤ E_exf ≤ 130 meV/Å²  → potential
E_exf > 130 meV/Å²       → no
```

Based on Mounet et al., Nature Nanotechnology 2018.

#### 5. `thermo_stable` - Thermodynamic Stability

**Unit**: categorical (on_hull / likely_synthesizable / metastable / risky)

**Formula**:
```
E_hull = 0             → on_hull
0 < E_hull ≤ 0.05      → likely_synthesizable
0.05 < E_hull ≤ 0.1    → metastable
E_hull > 0.1 eV/atom   → risky
```

#### 6. `e_modulus_est` - Young's Modulus Estimate

**Unit**: GPa

**Formula**: `E = 9 × K_VRH × G_VRH / (3 × K_VRH + G_VRH)`

**Interpretation**: Higher → stiffer material

**Typical values**:
- Polymers: 1-10 GPa
- Semiconductors: 100-400 GPa
- Ceramics: >200 GPa

---

## Data Quality

### Physics Validation

All physics violations fixed:
- ✅ Dielectric: ε_total ≥ ε_electronic (21 → 0 violations)
- ✅ Band gap: Elemental metals Eg=0, magnetic oxides non-zero (15 → 0 errors)
- ✅ Mechanical: K>0, G>0, E matches formula (3 → 0 negative values)

### Coverage Statistics

Based on 181 cached materials:

| Descriptor Type | Coverage |
|----------------|----------|
| Composition & Structure | 95-100% |
| Thermodynamics | ~95% |
| Electronic | ~95% |
| Dielectric | ~83% |
| Mechanical | ~76% |
| 2D/Magnetism | ~23-100% |

---

## Usage

### Command Line

```bash
# Basic query (formula)
python get_fet_descriptor.py "GaN"

# Common name resolution (via domain mapping)
python get_fet_descriptor.py "gallium nitride"   # → GaN
python get_fet_descriptor.py "iron oxide"        # → Fe2O3
python get_fet_descriptor.py "sapphire"          # → Al2O3

# Mineral names (via domain mapping)
python get_fet_descriptor.py "pyrite"            # → FeS2
python get_fet_descriptor.py "rutile"            # → TiO2
python get_fet_descriptor.py "wurtzite"          # → ZnS

# Advanced materials (via PubChem API fallback)
python get_fet_descriptor.py "lithium niobate"   # → LiNbO3
python get_fet_descriptor.py "calcium fluoride"  # → CaF2
python get_fet_descriptor.py "gallium phosphide" # → GaP

# Show summary
python get_fet_descriptor.py "Fe2O3" --summary

# Force refresh (re-query OPTIMADE)
python get_fet_descriptor.py "GaN" --force
```

### Python API

```python
from get_fet_descriptor import get_fet_descriptor
import json

# Get descriptor
json_path = get_fet_descriptor("GaN")

# Load data
with open(json_path, 'r') as f:
    data = json.load(f)

# Use descriptors
print(f"Band gap: {data['band_gap']} eV")
print(f"Class: {data['eg_class']}")
print(f"Stability: {data['thermo_stable']}")
print(f"Dielectric: {data['eps_mean']}")
```

### Examples

```bash
# III-V Semiconductors
python get_fet_descriptor.py "GaN"                  # Wide-gap (3.4 eV)
python get_fet_descriptor.py "indium antimonide"    # Narrow-gap (InSb)
python get_fet_descriptor.py "aluminum arsenide"    # AlAs

# II-VI Semiconductors
python get_fet_descriptor.py "cadmium telluride"    # CdTe (solar cells)
python get_fet_descriptor.py "zinc selenide"        # ZnSe (optoelectronics)
python get_fet_descriptor.py "mercury telluride"    # HgTe (infrared)

# Oxides (using common/mineral names)
python get_fet_descriptor.py "hematite"             # Fe2O3 (iron oxide)
python get_fet_descriptor.py "sapphire"             # Al2O3 (corundum)
python get_fet_descriptor.py "rutile"               # TiO2 (titania)
python get_fet_descriptor.py "lime"                 # CaO (calcium oxide)

# Perovskites
python get_fet_descriptor.py "barium titanate"      # BaTiO3 (ferroelectric)
python get_fet_descriptor.py "strontium titanate"   # SrTiO3
python get_fet_descriptor.py "lead titanate"        # PbTiO3

# 2D Materials & Chalcogenides
python get_fet_descriptor.py "MoS2"                 # Molybdenum disulfide
python get_fet_descriptor.py "graphene"             # Carbon monolayer
python get_fet_descriptor.py "bismuth telluride"    # Bi2Te3 (thermoelectric)

# Nitrides & Carbides
python get_fet_descriptor.py "scandium nitride"     # ScN
python get_fet_descriptor.py "vanadium nitride"     # VN (hard coating)
python get_fet_descriptor.py "tungsten carbide"     # WC (tool material)
python get_fet_descriptor.py "boron carbide"        # B4C (armor)

# Minerals & Sulfides
python get_fet_descriptor.py "pyrite"               # FeS2 (fool's gold)
python get_fet_descriptor.py "galena"               # PbS (lead sulfide)
python get_fet_descriptor.py "wurtzite"             # ZnS polymorph

# Advanced materials via PubChem
python get_fet_descriptor.py "lithium niobate"      # LiNbO3 (photonics)
python get_fet_descriptor.py "calcium fluoride"     # CaF2 (optical windows)

# New material workflow (not cached)
python get_fet_descriptor.py "ScN"                  # First query: ~25s (query + extract + cache)
python get_fet_descriptor.py "ScN"                  # Second query: <1s (from cache)
```

---

## How It Works

### Workflow

```
Material Name
    ↓
[1] Parse Name → Standardized Formula
    ↓
[2] Check Cache (FET_descriptor_filtered/)
    ├─ HIT → Return JSON ✓ (< 1s)
    └─ MISS ↓
[3] Query OPTIMADE Databases
    ↓
[4] Save Raw Data (raw_data_pulled/)
    ↓
[5] Extract 32 FET Descriptors
    ↓
[6] Validate Physics (ε, Eg, K/G)
    ↓
[7] Save to Cache (FET_descriptor_filtered/)
    ↓
Return JSON ✓ (~25s for new material)
```

### Multi-Database Fallback

Properties extracted with priority order:

1. **JARVIS** (jarvis.nist.gov) - Priority 10
   - Most comprehensive DFT calculations
   - 60+ specialized properties
   - Best coverage for mechanical/electronic

2. **Materials Project** (materialsproject.org) - Priority 9
   - Highly curated database
   - Excellent thermodynamic data
   - ~150k materials

3. **Alexandria** (alexandria.icams.rub.de) - Priority 8
   - Multiple XC functionals (PBEsol, SCAN)
   - Good electronic structure
   - Formation energies and hull distances

4. **OQMD** (oqmd.org) - Priority 7
   - Well-validated formation energies
   - ~1M structures
   - Structure prototypes

5. **Others** (Materials Cloud, MPDD, NOMAD, etc.)

**Extraction Strategy**:
1. Sort structures by database priority
2. Iterate through field names in priority order
3. Return first valid (non-None, non-placeholder) value
4. Leave empty if no data available

### Smart Band Gap Extraction

For magnetic oxides (Fe₂O₃, CuO, Co₃O₄), preferentially selects **non-zero** band gap values to avoid artifacts from metallic magnetic state DFT calculations.

---

## Files & Structure

```
query_substance_database/
├── get_fet_descriptor.py                   # Smart query script (MAIN ENTRY POINT)
├── query_substance.py                       # OPTIMADE database query
├── extract_comprehensive_properties.py      # 32-descriptor extraction
├── parse_material_name.py                   # Material name parser (5-layer strategy)
├── domain_frequent_name_mapping.json        # 182 formulas, 405 name variations
│
├── raw_data_pulled/                         # OPTIMADE raw JSON (180+ files)
│   ├── gan.json
│   ├── fe2o3.json
│   ├── insb.json
│   └── ...
│
├── FET_descriptor_filtered/                 # Extracted descriptors (181 files)
│   ├── gan.json
│   ├── fe2o3.json
│   ├── insb.json
│   ├── batio3.json
│   └── ...
│
└── README.md                                # This file
```

---

## Advanced: OPTIMADE Query Details

### Manual Query (if needed)

```bash
# Ground state only (default - recommended)
python query_substance.py GaN
python query_substance.py "gallium nitride"

# All structures (polymorphs, supercells, etc.)
python query_substance.py GaN --all-structures

# Custom options
python query_substance.py SiO2 --max-results 50 --output my_data.json
```

**Supported materials:**
- Pure elements: Si, Fe, Au, Cu
- Compounds: GaN, SiO2, Fe2O3
- Common names: "silicon dioxide", "gallium nitride", "alumina"

### Material Name Parser Details

The system uses a 5-layer parsing strategy (detailed at top of README):

1. **Domain mapping** (`domain_frequent_name_mapping.json`) - FASTEST
   - 182 formulas, 405 name variations
   - Examples: sapphire→Al2O3, pyrite→FeS2, wurtzite→ZnS

2. **Element names** - FAST
   - All element names and symbols

3. **Formula parsing** - FAST
   - Direct chemical formulas (pymatgen)

4. **PubChem API** - COMPREHENSIVE (for unknown materials)
   - Online chemical database fallback
   - Examples: lithium niobate→LiNbO3, calcium fluoride→CaF2

5. **Fallback** - Returns input with warning

**Key advantage**: Common materials use instant local mapping (Layer 1), while rare materials automatically fall back to PubChem API (Layer 4).

---

## Advanced: Extraction Details

### Extraction from Raw Data

```bash
# Extract from single file to CSV
python extract_comprehensive_properties.py raw_data_pulled/gan.json output.csv

# Extract from directory to individual JSONs
python extract_comprehensive_properties.py raw_data_pulled/ FET_descriptor_filtered/
```

### Field Name Fallback Examples

**Formation Energy** (tried in order):
```python
[
    '_alexandria_formation_energy_per_atom',      # Alexandria
    '_alexandria_scan_formation_energy_per_atom', # Alexandria (SCAN)
    '_jarvis_form_enp',                           # JARVIS
    '_mp_formation_energy_per_atom',              # Materials Project
    'formation_energy_per_atom',                  # Generic OPTIMADE
]
```

**Band Gap** (tried in order):
```python
[
    '_alexandria_band_gap',           # Alexandria (PBEsol)
    '_alexandria_scan_band_gap',      # Alexandria (SCAN)
    '_jarvis_optb88vdw_bandgap',      # JARVIS (optB88vdW)
    '_jarvis_mbj_bandgap',            # JARVIS (mBJ)
    '_mp_band_gap',                   # Materials Project
    'band_gap',                       # Generic OPTIMADE
]
```

### Physics Validation Rules

1. **Dielectric Constants**:
   - ε_total ≥ ε_electronic (always)
   - ε_ionic = ε_total - ε_electronic
   - If violated: use ε_electronic as fallback

2. **Band Gap**:
   - Elemental metals (Fe, Cu, Au): Force Eg = 0
   - Magnetic oxides (Fe₂O₃, CuO): Prefer non-zero values
   - Remove placeholder values (-99999, -999)

3. **Mechanical Moduli**:
   - K > 0, G > 0 (always)
   - E = 9KG / (3K + G) (validate formula)
   - If violated: mark as invalid

---

## References

- [OPTIMADE API](https://www.optimade.org/) - Materials database query standard
- [Materials Project](https://materialsproject.org/) - DFT database (~150k materials)
- [JARVIS](https://jarvis.nist.gov/) - NIST DFT database
- [OQMD](https://oqmd.org/) - Open Quantum Materials Database (~1M structures)
- [Alexandria](https://alexandria.icams.rub.de/) - Multi-functional DFT database
- Mounet et al., Nature Nanotechnology 2018 - 2D exfoliation criteria

---

## License

MIT License
