# Complete Descriptor Reference

**Total: 164 Descriptors** (32 FET + 132 MAGPIE)

This document provides a comprehensive reference for all material descriptors extracted by the system, including units, physical meanings, and computational methods.

---

## Part 1: FET Descriptors (32)

**Source**: OPTIMADE databases (JARVIS, Materials Project, Alexandria, OQMD, etc.)
**Method**: DFT calculations with multi-database fallback extraction
**Coverage**: 26 base descriptors + 6 derived descriptors

---

### 1.1 Composition & Structure (7 descriptors)

| Descriptor | Type | Unit | Physical Meaning | Typical Values |
|------------|------|------|------------------|----------------|
| `material_formula` | string | - | Material chemical formula (uppercase) | GaN, Fe2O3, Al2O3 |
| `chemical_formula_reduced` | string | - | Reduced chemical formula (smallest integer stoichiometry) | GaN, Fe2O3, TiO2 |
| `elements` | list | - | List of chemical elements present in the material | ['Ga', 'N'], ['Fe', 'O'] |
| `nelements` | integer | count | Number of unique chemical elements | 1-5 (most < 4) |
| `crystal_system` | string | - | Crystal system classification | cubic, hexagonal, tetragonal, orthorhombic, monoclinic, triclinic, trigonal |
| `space_group` | integer | - | International space group number (crystallographic symmetry) | 1-230 |
| `nperiodic_dimensions` | integer | count | Number of periodic dimensions | 3=bulk, 2=layered, 1=chain, 0=molecule |

**Notes**:
- `nperiodic_dimensions=2` indicates layered materials (e.g., MoS2, graphene)
- Space group number determines crystal symmetry operations

---

### 1.2 Thermodynamics (3 descriptors)

| Descriptor | Type | Unit | Physical Meaning | Typical Range | Interpretation |
|------------|------|------|------------------|---------------|----------------|
| `density` | float | g/cm³ | Mass density of crystalline material | 0.5-20 | Light (< 3), Medium (3-8), Heavy (> 8) |
| `formation_energy_per_atom` | float | eV/atom | DFT formation energy per atom from elemental reference states | -5 to +2 | Negative = thermodynamically favorable; < -1 = very stable |
| `energy_above_hull` | float | eV/atom | Energy distance to thermodynamic convex hull (ground state) | 0-1 | 0 = stable; < 0.05 = synthesizable; > 0.1 = metastable/risky |

**Notes**:
- Formation energy: More negative = more stable
- E_hull = 0: Material on convex hull (thermodynamically stable phase)
- E_hull > 0: Metastable; may decompose to lower-energy phases

---

### 1.3 Electronic & Transport (5 descriptors)

| Descriptor | Type | Unit | Physical Meaning | Typical Range | Interpretation |
|------------|------|------|------------------|---------------|----------------|
| `band_gap` | float | eV | Electronic band gap (energy gap between valence/conduction bands) | 0-10 | 0 = metal; 0.1-1 = narrow; 1-3 = moderate; > 3 = wide |
| `band_gap_direct` | float | eV | Direct band gap (k-conserving optical transition) | 0-10 | Direct gap materials have efficient optical emission |
| `dos_ef` | float | states/eV/cell | Electronic density of states at Fermi level | 0-100 | High DOS → metallic; 0 → insulator |
| `me_avg` | float | m₀ | Average electron effective mass (conductivity band) | 0.01-10 | Lower = higher electron mobility |
| `mh_avg` | float | m₀ | Average hole effective mass (valence band) | 0.01-10 | Lower = higher hole mobility |

**Notes**:
- Band gap extraction uses smart algorithm to avoid magnetic material DFT artifacts
- Elemental metals forced to Eg=0 (Fe, Cu, Au, etc.)
- Effective mass in units of electron rest mass (m₀)

---

### 1.4 Dielectric Properties (6 descriptors)

| Descriptor | Type | Unit | Physical Meaning | Typical Range | Application |
|------------|------|------|------------------|---------------|-------------|
| `epsilon_x` | float | - | Static dielectric constant along x-axis (DFPT) | 1-100 | Anisotropic dielectric response |
| `epsilon_y` | float | - | Static dielectric constant along y-axis (DFPT) | 1-100 | Anisotropic dielectric response |
| `epsilon_z` | float | - | Static dielectric constant along z-axis (DFPT) | 1-100 | Anisotropic dielectric response |
| `dielectric_total` | float | - | Total static dielectric constant (ε∞ + ε_ionic) | 1-100 | Low-frequency screening; gate coupling |
| `dielectric_electronic` | float | - | Electronic dielectric contribution (high-frequency limit ε∞) | 1-50 | Optical response; polarizability |
| `dielectric_ionic` | float | - | Ionic dielectric contribution (lattice vibrations) | 0-50 | Phonon contribution; infrared activity |

**Notes**:
- Physics constraint enforced: `ε_total = ε_electronic + ε_ionic`
- Higher ε_total → stronger gate coupling in FETs
- ε_electronic dominates in covalent materials; ε_ionic in ionic crystals

---

### 1.5 Mechanical Properties (3 descriptors)

| Descriptor | Type | Unit | Physical Meaning | Typical Range | Interpretation |
|------------|------|------|------------------|---------------|----------------|
| `k_vrh` | float | GPa | Bulk modulus (Voigt-Reuss-Hill average) | 10-400 | Resistance to uniform compression; stiffness |
| `g_vrh` | float | GPa | Shear modulus (Voigt-Reuss-Hill average) | 5-200 | Resistance to shear deformation; rigidity |
| `poisson_ratio` | float | - | Poisson's ratio ν (transverse/axial strain ratio) | 0-0.5 | Measure of lateral expansion under compression |

**Notes**:
- VRH average: Mean of Voigt (upper bound) and Reuss (lower bound) estimates
- K > 0 and G > 0 enforced (physics validation)
- Higher K, G → stiffer, harder material
- ν ≈ 0.5 → nearly incompressible (rubber-like)
- ν ≈ 0.2 → typical brittle ceramics

---

### 1.6 2D Materials & Magnetism (2 descriptors)

| Descriptor | Type | Unit | Physical Meaning | Typical Range | Interpretation |
|------------|------|------|------------------|---------------|----------------|
| `exfoliation_energy_mev_a2` | float | meV/Å² | Energy required to exfoliate one monolayer from bulk | 10-1000 | < 30 = easy (graphene); 30-130 = potential; > 130 = difficult |
| `magnetization` | float | μ_B/cell | Total magnetic moment per unit cell | 0-10 | 0 = non-magnetic; > 0 = magnetic ordering |

**Notes**:
- Exfoliation energy criteria from Mounet et al., Nat. Nanotech. 2018
- Magnetization in Bohr magnetons (μ_B)

---

### 1.7 Derived Descriptors (6 computed features)

These descriptors are automatically computed from base descriptors.

#### 1.7.1 `eg_class` - Band Gap Classification

**Type**: categorical
**Values**: `metal` | `narrow` | `moderate` | `wide`

**Formula**:
```
Eg < 0.05 eV         → metal
0.05 ≤ Eg < 1.0 eV   → narrow (IR/THz devices)
1.0 ≤ Eg < 3.0 eV    → moderate (visible optoelectronics)
Eg ≥ 3.0 eV          → wide (power electronics, UV)
```

**Special case**: Elemental metals (Fe, Cu, Au) forced to `metal` class.

---

#### 1.7.2 `eps_mean` - Mean Dielectric Constant

**Type**: float
**Unit**: dimensionless

**Formula**: `(εₓ + εᵧ + εᵧ) / 3`

**Interpretation**:
- Higher → stronger electrostatic screening
- Higher → better gate coupling in FETs
- Typical: 2-20 for semiconductors

---

#### 1.7.3 `eps_aniso` - Dielectric Anisotropy

**Type**: float
**Unit**: dimensionless

**Formula**: `(max(εₓ, εᵧ, εᵧ) - min(εₓ, εᵧ, εᵧ)) / eps_mean`

**Interpretation**:
- ≈ 0: Isotropic (cubic crystals)
- 0.2-0.5: Moderately anisotropic
- > 0.5: Highly anisotropic (2D materials, layered structures)

---

#### 1.7.4 `vdw_ready` - 2D/vdW Exfoliation Readiness

**Type**: categorical
**Values**: `layered_structure` | `easy` | `potential` | `no`

**Formula**:
```
nperiodic_dimensions ≤ 2  → layered_structure
E_exf < 30 meV/Å²         → easy (graphene-like)
30 ≤ E_exf ≤ 130 meV/Å²   → potential
E_exf > 130 meV/Å²        → no
```

**Reference**: Mounet et al., Nature Nanotechnology 2018

---

#### 1.7.5 `thermo_stable` - Thermodynamic Stability Classification

**Type**: categorical
**Values**: `on_hull` | `likely_synthesizable` | `metastable` | `risky`

**Formula**:
```
E_hull = 0             → on_hull (thermodynamically stable)
0 < E_hull ≤ 0.05      → likely_synthesizable
0.05 < E_hull ≤ 0.1    → metastable
E_hull > 0.1 eV/atom   → risky (may decompose)
```

**Interpretation**:
- `on_hull`: Experimentally observed, stable phase
- `likely_synthesizable`: Viable with appropriate synthesis conditions
- `metastable`: May exist kinetically, but not ground state
- `risky`: Unlikely to synthesize; may decompose

---

#### 1.7.6 `e_modulus_est` - Young's Modulus Estimate

**Type**: float
**Unit**: GPa

**Formula**: `E = 9 × K_VRH × G_VRH / (3 × K_VRH + G_VRH)`

**Interpretation**:
- Higher → stiffer material (resists tensile/compressive stress)
- Polymers: 1-10 GPa
- Semiconductors: 100-400 GPa
- Ceramics: > 200 GPa
- Diamond: ~1220 GPa (hardest)

**Note**: Only computed if K > 0 and G > 0 (physics validation).

---

---

## Part 2: MAGPIE Descriptors (132)

**Source**: Composition-based (computed from chemical formula only)
**Method**: Statistical aggregation of elemental properties
**Library**: matminer `ElementProperty.from_preset('magpie')`
**Reference**: Ward et al., npj Computational Materials 2016

---

### 2.1 Overview

MAGPIE (Materials-Agnostic Platform for Informatics and Exploration) generates **132 composition-based features** by computing **6 statistical measures** (mean, minimum, maximum, range, avg_dev, mode) over **22 elemental properties** for all elements in the chemical formula.

**Key Advantage**:
- No DFT calculation required
- Instant computation from formula
- Captures chemical intuition (electronegativity, atomic size, electron configuration)

---

### 2.2 Statistical Measures (6)

For each elemental property, the following statistics are computed across all elements in the composition:

| Statistic | Formula | Meaning |
|-----------|---------|---------|
| `mean` | `Σ(wᵢ × pᵢ) / Σwᵢ` | Weighted mean by stoichiometry |
| `minimum` | `min(pᵢ)` | Minimum value among all elements |
| `maximum` | `max(pᵢ)` | Maximum value among all elements |
| `range` | `max(pᵢ) - min(pᵢ)` | Spread of property values |
| `avg_dev` | `Σ(wᵢ × |pᵢ - mean|) / Σwᵢ` | Weighted average deviation from mean |
| `mode` | Most frequent value | Most common property value (discrete properties) |

**Notation**: `wᵢ` = stoichiometry weight, `pᵢ` = elemental property value

---

### 2.3 Elemental Properties (22)

Each of the following 22 properties is combined with 6 statistical measures to generate 132 features.

#### 2.3.1 Basic Atomic Properties (5)

| Property | Symbol | Unit | Physical Meaning | Example (Fe) | Example (O) |
|----------|--------|------|------------------|--------------|-------------|
| `Number` | Z | - | Atomic number (number of protons) | 26 | 8 |
| `MendeleevNumber` | - | - | Mendeleev's periodic table ordering (chemical similarity) | 74 | 103 |
| `AtomicWeight` | M | amu | Atomic mass (weighted isotope average) | 55.845 | 15.999 |
| `Row` | - | - | Periodic table row (period) | 4 | 2 |
| `Column` | - | - | Periodic table column (group) | 8 | 16 |

**Notes**:
- MendeleevNumber: Alternative periodic ordering emphasizing chemical properties
- Row/Column: Periodic trends (electronegativity, size, reactivity)

---

#### 2.3.2 Physical Properties (2)

| Property | Unit | Physical Meaning | Example (Fe) | Example (Si) |
|----------|------|------------------|--------------|--------------|
| `MeltingT` | K | Melting temperature | 1811 K | 1687 K |
| `CovalentRadius` | pm | Covalent atomic radius | 132 pm | 111 pm |

**Notes**:
- Higher melting point → stronger bonding
- Covalent radius: Atomic size in covalent bonds

---

#### 2.3.3 Electronic Structure Properties (3)

| Property | Unit | Physical Meaning | Example (Fe) | Example (O) |
|----------|------|------------------|--------------|-------------|
| `Electronegativity` | - | Pauling electronegativity (tendency to attract electrons) | 1.83 | 3.44 |
| `NValence` | e⁻ | Total valence electrons | 8 | 6 |
| `NUnfilled` | e⁻ | Number of unfilled valence electrons | 2 | 2 |

**Notes**:
- Higher electronegativity → more electron-attracting (O > Fe)
- NUnfilled: Available bonding orbitals

---

#### 2.3.4 Orbital Electron Configuration (10)

Decomposition of valence electrons by orbital type:

| Property | Unit | Physical Meaning | Example (Fe: 3d⁶4s²) |
|----------|------|------------------|---------------------|
| `NsValence` | e⁻ | Valence electrons in s orbitals | 2 |
| `NpValence` | e⁻ | Valence electrons in p orbitals | 0 |
| `NdValence` | e⁻ | Valence electrons in d orbitals | 6 |
| `NfValence` | e⁻ | Valence electrons in f orbitals | 0 |
| `NsUnfilled` | e⁻ | Unfilled s orbital electrons | 0 |
| `NpUnfilled` | e⁻ | Unfilled p orbital electrons | 0 |
| `NdUnfilled` | e⁻ | Unfilled d orbital electrons | 4 |
| `NfUnfilled` | e⁻ | Unfilled f orbital electrons | 0 |

**Total**: 4 valence properties + 4 unfilled properties = 8 orbital features

**Physical Significance**:
- d-electrons → magnetic properties, catalysis, color
- f-electrons → lanthanides/actinides, rare earth magnetism
- Unfilled orbitals → bonding capacity, oxidation states

---

#### 2.3.5 Ground State DFT Properties (3)

Elemental ground state properties from DFT calculations:

| Property | Unit | Physical Meaning | Example (Si) | Example (Fe) |
|----------|------|------------------|--------------|--------------|
| `GSvolume_pa` | Å³/atom | Ground state atomic volume | 20.0 Å³ | 11.8 Å³ |
| `GSbandgap` | eV | Ground state band gap (elemental) | 1.17 eV | 0 eV |
| `GSmagmom` | μB | Ground state magnetic moment | 0 μB | 2.2 μB |

**Notes**:
- GSbandgap: 0 for metals, > 0 for semiconductors/insulators
- GSmagmom: Non-zero for magnetic elements (Fe, Co, Ni, Gd, etc.)

---

#### 2.3.6 Crystallographic Property (1)

| Property | Unit | Physical Meaning | Example (Fe: bcc) | Example (Au: fcc) |
|----------|------|------------------|------------------|------------------|
| `SpaceGroupNumber` | - | Space group of elemental crystal structure | 229 (Im-3m) | 225 (Fm-3m) |

**Notes**:
- Elemental crystal structure varies by element
- Fe: body-centered cubic (bcc)
- Au: face-centered cubic (fcc)
- C: diamond cubic or graphite

---

### 2.4 Complete MAGPIE Feature List (132 features)

**Structure**: `magpie_{statistic}_{property}`

#### Format Pattern:
```
magpie_mean_AtomicWeight
magpie_minimum_AtomicWeight
magpie_maximum_AtomicWeight
magpie_range_AtomicWeight
magpie_avg_dev_AtomicWeight
magpie_mode_AtomicWeight
... (repeat for all 22 properties)
```

#### Full Feature List:

**Statistics**: `mean`, `minimum`, `maximum`, `range`, `avg_dev`, `mode`
**Properties**: 22 elemental properties listed below

1. `Number` (atomic number)
2. `MendeleevNumber`
3. `AtomicWeight`
4. `MeltingT` (melting temperature)
5. `Column` (periodic table column)
6. `Row` (periodic table row)
7. `CovalentRadius`
8. `Electronegativity`
9. `NsValence` (s orbital valence electrons)
10. `NpValence` (p orbital valence electrons)
11. `NdValence` (d orbital valence electrons)
12. `NfValence` (f orbital valence electrons)
13. `NValence` (total valence electrons)
14. `NsUnfilled` (unfilled s electrons)
15. `NpUnfilled` (unfilled p electrons)
16. `NdUnfilled` (unfilled d electrons)
17. `NfUnfilled` (unfilled f electrons)
18. `NUnfilled` (total unfilled electrons)
19. `GSvolume_pa` (ground state atomic volume)
20. `GSbandgap` (ground state band gap)
21. `GSmagmom` (ground state magnetic moment)
22. `SpaceGroupNumber`

**Total**: 6 statistics × 22 properties = **132 MAGPIE features**

---

### 2.5 Example: MAGPIE for GaN

For **GaN** (gallium nitride):
- Elements: Ga (Z=31), N (Z=7)
- Stoichiometry: 1:1

**Sample features**:
```
magpie_mean_AtomicWeight = (69.723 + 14.007) / 2 = 41.865
magpie_minimum_Number = min(31, 7) = 7
magpie_maximum_Electronegativity = max(1.81, 3.04) = 3.04
magpie_range_CovalentRadius = 122 - 71 = 51 pm
magpie_avg_dev_NValence = |13-9| + |5-9| / 2 = 4.0
```

---

### 2.6 Physical Interpretation

#### Why MAGPIE Works:

1. **Chemical Intuition**: Captures periodic trends (electronegativity, size, valence)
2. **Bonding Character**: NValence, Electronegativity → ionic vs. covalent
3. **Electronic Structure**: Orbital occupancy → conductivity, magnetism
4. **Composition Heterogeneity**: `range`, `avg_dev` → complexity, disorder
5. **Elemental Ground States**: GSbandgap, GSmagmom → intrinsic element properties

#### Applications in ML:

- **Property Prediction**: Band gap, formation energy, melting point
- **Phase Stability**: Identify stable/metastable compositions
- **Materials Discovery**: Screen composition space for desired properties
- **Feature Engineering**: Combine with structure-based descriptors (SOAP, Coulomb matrix)

#### Advantages:

✅ **Fast**: No DFT needed, instant from formula
✅ **Universal**: Works for any chemical composition
✅ **Interpretable**: Features map to chemical concepts
✅ **ML-Ready**: Proven performance in materials ML models

#### Limitations:

❌ **No Structure Info**: Cannot distinguish polymorphs (e.g., diamond vs. graphite)
❌ **No Defects**: Assumes perfect stoichiometry
❌ **Approximations**: Elemental properties may not reflect bonding environment

---

---

## Part 3: Summary Statistics

| Category | Count | Source | Computation Time |
|----------|-------|--------|------------------|
| **FET Base Descriptors** | 26 | OPTIMADE (DFT) | ~20s (new material) |
| **FET Derived Descriptors** | 6 | Computed from base | < 0.1s |
| **MAGPIE Features** | 132 | Chemical formula | < 0.1s |
| **Total Descriptors** | **164** | Hybrid | ~20s (new material) |

---

## Part 4: Data Coverage

Based on 181 cached materials:

| Descriptor Group | Coverage | Notes |
|------------------|----------|-------|
| Composition & Structure | 95-100% | Nearly complete |
| Thermodynamics | ~95% | Formation energy highly available |
| Electronic | ~95% | Band gap widely reported |
| Dielectric | ~83% | Some databases lack DFPT data |
| Mechanical | ~76% | Elastic constants computationally expensive |
| 2D/Magnetism | 23-100% | Exfoliation energy only for layered materials |
| **MAGPIE** | **100%** | **Always computable from formula** |

---

## Part 5: Usage Examples

### Python API

```python
import json
from get_fet_descriptor import get_fet_descriptor

# Query material (auto-cache)
json_path = get_fet_descriptor("GaN")

# Load all 164 descriptors
with open(json_path, 'r') as f:
    data = json.load(f)

# Access FET descriptors
print(f"Band gap: {data['band_gap']} eV")
print(f"Formation energy: {data['formation_energy_per_atom']} eV/atom")
print(f"Dielectric constant: {data['eps_mean']}")

# Access MAGPIE features
print(f"Mean electronegativity: {data['magpie_mean_Electronegativity']}")
print(f"Atomic weight range: {data['magpie_range_AtomicWeight']}")

# Count features
magpie_count = sum(1 for k in data.keys() if k.startswith('magpie_'))
print(f"Total descriptors: {len(data)} ({len(data)-magpie_count} FET + {magpie_count} MAGPIE)")
```

### Command Line

```bash
# Query new material with all 164 descriptors
python get_fet_descriptor.py "zinc antimonide"

# Show summary
python get_fet_descriptor.py "BiVO4" --summary

# Batch extraction
python extract_comprehensive_properties.py raw_data_pulled/ FET_descriptor_filtered/
```

---

## Part 6: References

### FET Descriptors
- OPTIMADE: https://www.optimade.org/
- Materials Project: https://materialsproject.org/
- JARVIS: https://jarvis.nist.gov/
- OQMD: https://oqmd.org/
- Alexandria: https://alexandria.icams.rub.de/

### MAGPIE Descriptors
- Ward et al., "A general-purpose machine learning framework for predicting properties of inorganic materials," npj Computational Materials 2, 16028 (2016)
- matminer documentation: https://hackingmaterials.lbl.gov/matminer/
- GitHub: https://github.com/hackingmaterials/matminer

### Exfoliation Criteria
- Mounet et al., "Two-dimensional materials from high-throughput computational exfoliation of experimentally known compounds," Nature Nanotechnology 13, 246-252 (2018)

---

## Part 7: File Locations

- **This file**: `DESCRIPTOR_REFERENCE.md`
- **System README**: `README.md`
- **Extraction script**: `extract_comprehensive_properties.py`
- **Main query script**: `get_fet_descriptor.py`
- **Cached descriptors**: `FET_descriptor_filtered/*.json` (181 materials)
- **Raw OPTIMADE data**: `raw_data_pulled/*.json`

---

**Document Version**: 1.0
**Last Updated**: 2025-10-26
**Total Descriptors**: 164 (32 FET + 132 MAGPIE)
