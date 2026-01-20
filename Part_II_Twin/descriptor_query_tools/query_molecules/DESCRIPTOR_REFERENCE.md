# Molecular Descriptor Reference

Complete reference for all 182 molecular descriptors retrieved by this utility.

## Table of Contents

- [PubChem Descriptors (40)](#pubchem-descriptors-40)
- [ChEMBL Descriptors (14)](#chembl-descriptors-14)
- [Morgan Fingerprint (128)](#morgan-fingerprint-128)
- [References](#references)

---

## PubChem Descriptors (40)

### Identifiers

| Descriptor | Unit | Definition | Type |
|------------|------|------------|------|
| **CID** | - | PubChem Compound Identifier, unique integer ID | Integer |
| **MolecularFormula** | - | Chemical formula showing element symbols and counts (e.g., C9H8O4) | String |
| **InChI** | - | IUPAC International Chemical Identifier, hierarchical text representation | String |
| **InChIKey** | - | Fixed-length hash (27 chars) of InChI for database searching | String |
| **IUPACName** | - | Systematic chemical name according to IUPAC nomenclature rules | String |
| **SMILES** | - | Simplified Molecular Input Line Entry System, linear structure notation | String |
| **ConnectivitySMILES** | - | Canonical SMILES without stereochemistry information | String |

### Basic Properties

| Descriptor | Unit | Definition | Type |
|------------|------|------------|------|
| **MolecularWeight** | Da (g/mol) | Sum of atomic weights of all atoms in the molecule | Float |
| **ExactMass** | Da | Exact mass calculated using isotopic composition (most abundant isotopes) | Float |
| **MonoisotopicMass** | Da | Mass calculated using the most abundant isotope for each element | Float |
| **Charge** | e | Formal charge of the molecule (sum of atomic formal charges) | Integer |

### Physicochemical Properties

| Descriptor | Unit | Definition | Type |
|------------|------|------------|------|
| **XLogP** | log units | Octanol-water partition coefficient (XLogP3 algorithm), measures lipophilicity | Float |
| **TPSA** | Ų | Topological Polar Surface Area, sum of polar atom surface areas (Ertl method) | Float |
| **Complexity** | - | Bertz/Hendrickson/Ihlenfeldt complexity score, rough measure of structural complexity | Float |

### Hydrogen Bonding

| Descriptor | Unit | Definition | Type |
|------------|------|------------|------|
| **HBondDonorCount** | count | Number of hydrogen bond donors (N-H and O-H groups) | Integer |
| **HBondAcceptorCount** | count | Number of hydrogen bond acceptors (N and O atoms) | Integer |

### Structural Counts

| Descriptor | Unit | Definition | Type |
|------------|------|------------|------|
| **HeavyAtomCount** | count | Number of non-hydrogen atoms in the molecule | Integer |
| **AtomStereoCount** | count | Total number of atoms with defined stereochemistry | Integer |
| **DefinedAtomStereoCount** | count | Number of atoms with explicitly defined stereochemistry | Integer |
| **UndefinedAtomStereoCount** | count | Number of atoms with possible but undefined stereochemistry | Integer |
| **IsotopeAtomCount** | count | Number of atoms with non-standard isotopic composition | Integer |
| **CovalentUnitCount** | count | Number of covalently bonded units (usually 1, >1 for salts) | Integer |

### Bond Properties

| Descriptor | Unit | Definition | Type |
|------------|------|------------|------|
| **RotatableBondCount** | count | Number of rotatable single bonds (excludes terminal and ring bonds) | Integer |
| **BondStereoCount** | count | Total number of bonds with stereochemistry (E/Z or cis/trans) | Integer |
| **DefinedBondStereoCount** | count | Number of bonds with explicitly defined stereochemistry | Integer |
| **UndefinedBondStereoCount** | count | Number of bonds with possible but undefined stereochemistry | Integer |

### 3D Conformer Properties

All 3D properties are computed from the lowest energy conformer.

| Descriptor | Unit | Definition | Type |
|------------|------|------------|------|
| **Volume3D** | ų | Van der Waals volume of the 3D conformer | Float |
| **XStericQuadrupole3D** | - | X-component of steric quadrupole moment | Float |
| **YStericQuadrupole3D** | - | Y-component of steric quadrupole moment | Float |
| **ZStericQuadrupole3D** | - | Z-component of steric quadrupole moment | Float |
| **FeatureCount3D** | count | Total number of pharmacophore features | Integer |
| **FeatureAcceptorCount3D** | count | Number of hydrogen bond acceptor features | Integer |
| **FeatureDonorCount3D** | count | Number of hydrogen bond donor features | Integer |
| **FeatureAnionCount3D** | count | Number of anionic (negative charge) features | Integer |
| **FeatureCationCount3D** | count | Number of cationic (positive charge) features | Integer |
| **FeatureRingCount3D** | count | Number of aromatic ring features | Integer |
| **FeatureHydrophobeCount3D** | count | Number of hydrophobic features | Integer |
| **ConformerModelRMSD3D** | Å | Root-mean-square deviation of atomic positions in conformer ensemble | Float |
| **EffectiveRotorCount3D** | count | Number of effectively rotatable bonds in 3D structure | Integer |
| **ConformerCount3D** | count | Number of conformers generated for this compound | Integer |

---

## ChEMBL Descriptors (14)

ChEMBL molecular properties are computed by RDKit within the ChEMBL database.

### Lipophilicity

| Descriptor | Unit | Definition | Type |
|------------|------|------------|------|
| **alogp** | log units | Atomic-contribution based logP (octanol-water partition coefficient) | Float |

### Structural Descriptors

| Descriptor | Unit | Definition | Type |
|------------|------|------------|------|
| **aromatic_rings** | count | Number of aromatic ring systems in the molecule | Integer |
| **heavy_atoms** | count | Number of non-hydrogen atoms (same as PubChem HeavyAtomCount) | Integer |
| **rtb** | count | Number of rotatable bonds (similar to RotatableBondCount) | Integer |

### Molecular Formula and Weight

| Descriptor | Unit | Definition | Type |
|------------|------|------------|------|
| **full_molformula** | - | Complete molecular formula including all atoms | String |
| **full_mwt** | Da (g/mol) | Molecular weight of the full structure | Float |
| **mw_freebase** | Da (g/mol) | Molecular weight of the freebase (neutral) form | Float |

### Hydrogen Bonding (Lipinski Rule of 5)

| Descriptor | Unit | Definition | Type |
|------------|------|------------|------|
| **hba** | count | Number of hydrogen bond acceptors (N and O atoms) | Integer |
| **hbd** | count | Number of hydrogen bond donors (N-H and O-H groups) | Integer |

### Drug-likeness Metrics

| Descriptor | Unit | Definition | Type |
|------------|------|------------|------|
| **psa** | Ų | Polar Surface Area, related to membrane permeability | Float |
| **qed_weighted** | 0-1 | Quantitative Estimate of Drug-likeness, weighted desirability score | Float |
| **np_likeness_score** | -5 to +5 | Natural Product-likeness score (positive = more NP-like) | Float |
| **num_ro5_violations** | count | Number of Lipinski's Rule of Five violations (0-4) | Integer |
| **ro3_pass** | Y/N | Whether compound passes Rule of Three (fragment-like criteria) | String |

---

## Morgan Fingerprint (128)

### Structure

| Descriptor | Unit | Definition | Type |
|------------|------|------------|------|
| **bit_vector** | - | 128-dimensional binary vector encoding molecular structure | List[int] |
| **n_bits** | - | Total number of bits (fixed at 128) | Integer |
| **num_on_bits** | count | Number of bits set to 1 (typically 15-25 for small molecules) | Integer |
| **on_bits** | - | List of indices where bits are set to 1 | List[int] |
| **hex** | - | Hexadecimal representation of the bit vector | String |
| **bit_string** | - | Binary string representation (e.g., "001011...") | String |

### Method

- **Algorithm**: Morgan (Extended-Connectivity Fingerprint, ECFP)
- **Radius**: 2 (equivalent to ECFP4)
- **Purpose**: Encode molecular topology and structure for similarity searching and ML
- **Generator**: RDKit
- **Input**: SMILES or InChI from PubChem/ChEMBL

### Count Fingerprint

| Descriptor | Unit | Definition | Type |
|------------|------|------------|------|
| **count_fingerprint** | - | Dictionary mapping substructure hashes to occurrence counts | Dict[str, int] |

Each key is a hash of a molecular substructure fragment, and the value is how many times that fragment appears.

---

## Descriptor Categories Summary

### Total Dimensions: 182

| Category | Count | Purpose |
|----------|-------|---------|
| **Identifiers** | 7 | Database IDs and structure representations |
| **Basic Properties** | 4 | Mass, charge, formula |
| **Physicochemical** | 3 | Lipophilicity, polarity, complexity |
| **Hydrogen Bonding** | 4 | Donor/acceptor counts (PubChem + ChEMBL) |
| **Structural Counts** | 11 | Atom/bond counts, stereochemistry |
| **3D Properties** | 13 | Conformer-based features |
| **Drug-likeness** | 12 | Lipinski, QED, natural product-likeness |
| **Morgan Fingerprint** | 128 | Structural fingerprint |

### For Machine Learning

**Recommended for LLM/ML (165 numeric features):**
- PubChem numeric: 33 features
- ChEMBL numeric: 12 features
- Morgan fingerprint: 128 bits (binary)
- **Excluded**: String identifiers (CID, SMILES, InChI, etc.) - 9 fields

**Compact option (128 features):**
- Use **Morgan fingerprint only** for small-sample LLM fine-tuning

---

## References

### PubChem

1. **PubChem Documentation**: https://pubchem.ncbi.nlm.nih.gov/docs/
2. **PubChem Glossary**: https://pubchem.ncbi.nlm.nih.gov/docs/glossary
3. **XLogP3 Algorithm**: Wang, R., Fu, Y., Lai, L. (1997). "A New Atom-Additive Method for Calculating Partition Coefficients." *Journal of Chemical Information and Computer Sciences*.
4. **TPSA Method**: Ertl, P., Rohde, B., Selzer, P. (2000). "Fast Calculation of Molecular Polar Surface Area as a Sum of Fragment-Based Contributions." *Journal of Medicinal Chemistry*, 43(20), 3714-3717.
5. **Complexity**: Bertz, S.H. (1981). "The First General Index of Molecular Complexity." *Journal of the American Chemical Society*, 103, 3599-3601.

### ChEMBL

6. **ChEMBL Database**: https://www.ebi.ac.uk/chembl/
7. **ChEMBL Documentation**: https://chembl.gitbook.io/chembl-interface-documentation/
8. **Lipinski's Rule of Five**: Lipinski, C.A. et al. (2001). "Experimental and computational approaches to estimate solubility and permeability." *Advanced Drug Delivery Reviews*, 46(1-3), 3-26.
9. **QED**: Bickerton, G.R. et al. (2012). "Quantifying the chemical beauty of drugs." *Nature Chemistry*, 4(2), 90-98.

### Morgan Fingerprints

10. **Morgan Algorithm**: Rogers, D., Hahn, M. (2010). "Extended-Connectivity Fingerprints." *Journal of Chemical Information and Modeling*, 50(5), 742-754.
11. **RDKit Documentation**: https://www.rdkit.org/docs/

### Additional Resources

12. **Molecular Descriptors Review**: Todeschini, R., Consonni, V. (2009). *Molecular Descriptors for Chemoinformatics*. Wiley-VCH.
13. **IUPAC Nomenclature**: https://iupac.org/what-we-do/nomenclature/

---

## Units Glossary

- **Da (Dalton)**: Atomic mass unit, equivalent to g/mol
- **Å (Ångström)**: 10⁻¹⁰ meters, used for atomic/molecular distances
- **Ų (Square Ångström)**: Å², used for surface areas
- **ų (Cubic Ångström)**: Å³, used for volumes
- **log units**: Logarithmic scale (base 10)
- **e**: Elementary charge unit
- **count**: Dimensionless integer count

---

## Version Information

- **PubChem**: Properties computed using PubChem release 2025.04.14
- **ChEMBL**: Properties from ChEMBL database (RDKit-computed)
- **RDKit**: Morgan fingerprints generated with RDKit 2023.9.1+
- **Utility Version**: 1.0.0

Last updated: 2025-10-27
