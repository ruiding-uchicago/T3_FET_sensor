# KEY_DESCRIPTOR_25 - Polymer Numerical Descriptors

**Version**: 1.0 (Nov 2025)
**Purpose**: Optimized set of 25 numerical descriptors for polymer machine learning applications

---

## Overview

This document defines the **recommended 25 numerical descriptors** for polymer feature engineering, balancing:
- ✅ **High data availability** (16-87% fill rate across 292 polymers)
- ✅ **Physical interpretability** (thermal, mechanical, electrical, monomer properties)
- ✅ **ML-readiness** (minimal missing value imputation required)

These 25 descriptors + 256-bit Morgan fingerprint = **281-dimensional feature vector** for polymer property prediction.

---

## Descriptor List

### Polymer Macroscopic Properties (7 descriptors)

| # | Descriptor | Fill Rate | Unit | Physical Meaning | Category |
|---|------------|-----------|------|------------------|----------|
| 1 | `molar_mass_repeat_g_mol` | 87.3% | g/mol | Repeat unit molecular weight | Physical |
| 19 | `density_g_cm3` | 47.6% | g/cm³ | Bulk density | Physical |
| 20 | `Tg_C` | 42.5% | °C | Glass transition temperature | Thermal |
| 21 | `Td_C` | 40.8% | °C | Decomposition temperature | Thermal |
| 23 | `dielectric_constant` | 24.0% | - | Relative permittivity (ε_r) | Electrical |
| 24 | `youngs_modulus_GPa` | 22.6% | GPa | Elastic modulus | Mechanical |
| 25 | `Tm_C` | 16.4% | °C | Melting temperature | Thermal |

### Monomer Descriptors (18 descriptors)

| # | Descriptor | Fill Rate | Category | Physical Meaning |
|---|------------|-----------|----------|------------------|
| 2 | `monomer_Charge` | 84.2% | Electronic | Formal charge |
| 3 | `monomer_TPSA` | 84.2% | Polarity | Topological polar surface area (Ų) |
| 4 | `monomer_HBondDonorCount` | 84.2% | Polarity | H-bond donor count |
| 5 | `monomer_HBondAcceptorCount` | 84.2% | Polarity | H-bond acceptor count |
| 6 | `monomer_MolecularWeight` | 84.2% | Size | Monomer molecular weight (g/mol) |
| 7 | `monomer_HeavyAtomCount` | 84.2% | Size | Non-hydrogen atom count |
| 8 | `monomer_RotatableBondCount` | 84.2% | Flexibility | Rotatable bond count |
| 9 | `monomer_XLogP` | 79.1% | Polarity | Lipophilicity (octanol-water partition) |
| 10 | `monomer_Complexity` | 63.0% | Complexity | Molecular complexity score |
| 11 | `monomer_FeatureRingCount3D` | 56.5% | 3D-Structure | 3D ring count |
| 12 | `monomer_FeatureCationCount3D` | 56.5% | 3D-Electronic | Cationic feature count |
| 13 | `monomer_FeatureAnionCount3D` | 56.5% | 3D-Electronic | Anionic feature count |
| 14 | `monomer_FeatureDonorCount3D` | 56.5% | 3D-Polarity | 3D H-bond donor count |
| 15 | `monomer_FeatureAcceptorCount3D` | 56.5% | 3D-Polarity | 3D H-bond acceptor count |
| 16 | `monomer_Volume3D` | 56.5% | 3D-Size | Molecular volume (ų) |
| 17 | `monomer_EffectiveRotorCount3D` | 56.5% | 3D-Flexibility | Effective rotor count |
| 18 | `monomer_FeatureHydrophobeCount3D` | 56.5% | 3D-Polarity | Hydrophobic feature count |
| 22 | `monomer_aromatic_rings` | 37.3% | Electronic | Aromatic ring count (conjugation) |

---

## Category Distribution

- **Monomer Descriptors**: 18 descriptors (72%)
  - Polarity & Surface: 6 descriptors
  - Size & Shape: 3 descriptors
  - 3D Features: 7 descriptors
  - Electronic: 2 descriptors
  - Flexibility: 1 descriptor
  - Complexity: 1 descriptor

- **Polymer Properties**: 7 descriptors (28%)
  - Thermal: 3 descriptors (Tg, Tm, Td)
  - Physical: 2 descriptors (density, molar_mass)
  - Electrical: 1 descriptor (dielectric_constant)
  - Mechanical: 1 descriptor (youngs_modulus)

---

## Design Rationale

### Why These 25?

1. **Balanced Strategy**: Combines high-fill-rate monomer descriptors (56-84%) with essential polymer properties (16-87%)

2. **Replaced 5 Less-Important 3D Descriptors** from pure Top-25:
   - Removed: `XStericQuadrupole3D`, `YStericQuadrupole3D`, `ZStericQuadrupole3D`, `ConformerModelRMSD3D`, `FeatureCount3D`
   - Added: `Tg_C`, `Tm_C`, `Td_C`, `dielectric_constant`, `youngs_modulus_GPa`, `monomer_aromatic_rings`

3. **FET/Semiconductor Polymer Optimization**:
   - `monomer_aromatic_rings`: Critical for conjugated polymer conductivity
   - `dielectric_constant`: Essential for electronic device performance
   - `Tg/Tm/Td`: Thermal stability for device processing

4. **Missing Value Strategy**:
   - High fill rate (>50%): Direct use with minimal imputation
   - Medium fill rate (20-50%): Median/mean imputation acceptable
   - Low fill rate (16-20%): Use for specialized models (e.g., crystalline polymers need `Tm_C`)

---

## Data Quality (292 Polymers)

| Fill Rate Range | # of Descriptors | Example Descriptors |
|-----------------|------------------|---------------------|
| 80-90% | 8 | molar_mass, monomer_Charge, monomer_TPSA, ... |
| 50-80% | 10 | monomer_Complexity, monomer_3D features |
| 20-50% | 4 | density, Tg_C, Td_C, dielectric_constant |
| <20% | 3 | Tm_C, youngs_modulus, monomer_aromatic_rings |

**Overall Statistics**:
- Median fill rate: **56.5%**
- Mean fill rate: **58.1%**
- Total features: **25 descriptors + 256-bit fingerprint = 281 dimensions**

---

## Usage Guidelines

### For Machine Learning

**Recommended workflow**:
```python
# 1. Extract 25 descriptors from polymer JSON
descriptors_25 = [
    data["data"]["physical_properties"]["molar_mass_repeat_g_mol"],
    data["data"]["monomer_descriptors"]["key_25"]["Charge"],
    data["data"]["monomer_descriptors"]["key_25"]["TPSA"],
    # ... (all 25)
]

# 2. Extract 256-bit fingerprint
fingerprint_256 = data["data"]["fingerprints"]["bit_vector"]

# 3. Concatenate into 281-dim feature vector
feature_vector = descriptors_25 + fingerprint_256  # 281 dimensions
```

**Missing value handling**:
- Use median imputation for descriptors with 50%+ fill rate
- Use mean imputation for descriptors with 20-50% fill rate
- Consider dropping descriptors with <20% fill rate for general models
- For specialized applications (e.g., crystalline polymers), keep low-fill descriptors

### For Polymer Property Prediction

**Target prediction tasks**:
- ✅ Glass transition temperature (Tg)
- ✅ Thermal stability (Td)
- ✅ Density
- ✅ Dielectric properties
- ✅ Mechanical properties
- ✅ Solubility/compatibility
- ✅ Crystallinity (via `Tm_C`)
- ✅ Conductivity (via `monomer_aromatic_rings`)

---

## Alternatives

### Pure Data-Driven (Top 25 by Fill Rate)
- **Pros**: Highest fill rates (42-87%), minimal imputation
- **Cons**: Only 3 polymer macroscopic properties, dominated by monomer 3D features
- **Use case**: General polymer classification/retrieval

### Physics-Driven (Maximizing Physical Coverage)
- **Pros**: Comprehensive coverage of polymer physics
- **Cons**: Many <10% fill rates, requires extensive imputation or domain knowledge
- **Use case**: Expert-guided models with strong priors

### Balanced Strategy (This document) ⭐
- **Pros**: Best trade-off between data availability and physical interpretability
- **Cons**: Some descriptors require imputation (16-50% range)
- **Use case**: FET/semiconductor polymer property prediction, general polymer ML

---

## Version History

- **v1.0** (Nov 2025): Initial release, optimized for FET polymer applications
  - 292 polymers in cache
  - Average monomer descriptor fill rate: 67.7%
  - Molar mass fill rate: 87.3%

---

## References

- **Data Sources**: PubChem, ChEMBL (via query_molecules), RDKit local computation
- **Related Files**:
  - `add_monomer_descriptors.py` - Extract KEY_25 monomer descriptors
  - `fill_computable_properties.py` - RDKit fallback computation
  - `POLYMER_DATA_REFERENCE.md` - Full data schema
  - `README.md` - Project overview
