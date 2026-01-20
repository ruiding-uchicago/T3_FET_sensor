# Key 25 Descriptors for FET Sensor Applications

Selected based on chemical informatics relevance for FET sensor applications (probes, sensing materials, analytes).

**Overall Availability: 87.3%** across 592 cached molecules

---

## The 25 Descriptors

### **1. Electronic Properties & Charge Transport (5)**
```
1.  Charge                    # PubChem | 100.0% | Molecular charge (affects FET conductivity)
2.  aromatic_rings            # ChEMBL  | 80.3% | Aromatic ring count (π-π stacking, e⁻ transport)
3.  FeatureRingCount3D        # PubChem | 79.9% | 3D aromatic ring features
4.  FeatureCationCount3D      # PubChem | 79.9% | Cationic centers (charge transfer)
5.  FeatureAnionCount3D       # PubChem | 79.9% | Anionic centers (charge transfer)
```

### **2. Polarity & Surface Interactions (6)**
```
6.  TPSA                      # PubChem | 100.0% | Polar surface area (surface adsorption)
7.  XLogP                     # PubChem | 85.4% | Lipophilicity (interface interaction)
8.  HBondDonorCount           # PubChem | 100.0% | H-bond donors (molecular recognition)
9.  HBondAcceptorCount        # PubChem | 100.0% | H-bond acceptors (molecular recognition)
10. FeatureDonorCount3D       # PubChem | 79.9% | 3D H-bond donor features
11. FeatureAcceptorCount3D    # PubChem | 79.9% | 3D H-bond acceptor features
```

### **3. Molecular Size & Shape (6)**
```
12. MolecularWeight           # PubChem | 100.0% | Molecular weight (basic property)
13. HeavyAtomCount            # PubChem | 100.0% | Non-H atom count (structural complexity)
14. Volume3D                  # PubChem | 79.9% | 3D volume (surface coverage)
15. XStericQuadrupole3D       # PubChem | 79.9% | X-direction spatial distribution
16. YStericQuadrupole3D       # PubChem | 79.9% | Y-direction spatial distribution
17. ZStericQuadrupole3D       # PubChem | 79.9% | Z-direction spatial distribution
```

### **4. Molecular Flexibility & Dynamics (3)**
```
18. RotatableBondCount        # PubChem | 100.0% | Rotatable bonds (flexibility, response speed)
19. EffectiveRotorCount3D     # PubChem | 79.9% | Effective rotors (conformational flexibility)
20. ConformerModelRMSD3D      # PubChem | 79.9% | Conformational variation (stability)
```

### **5. Structural Complexity (2)**
```
21. Complexity                # PubChem | 100.0% | Bertz complexity (selectivity)
22. FeatureHydrophobeCount3D  # PubChem | 79.9% | Hydrophobic features (hydrophobic interaction)
```

### **6. Pharmacophore & Recognition (3)**
```
23. FeatureCount3D            # PubChem | 79.9% | Total pharmacophore features (recognition capability)
24. qed_weighted              # ChEMBL  | 80.3% | QED drug-likeness (biosensor applications)
25. np_likeness_score         # ChEMBL  | 80.3% | Natural product likeness (bioactivity)
```

---

## Availability Summary

| Category | Descriptors | Avg Availability |
|----------|-------------|------------------|
| **100% available** | 8 descriptors | 100% |
| **85%+ available** | 1 descriptor | 85.4% |
| **80%+ available** | 3 descriptors | 80.3% |
| **~80% available** | 13 descriptors (all 3D) | 79.9% |

**3D Descriptors Note:**
- All 13 3D descriptors have exactly **79.9% availability** (203/254 molecules)
- Missing in 51 molecules (primarily ions and salts)
- If your FET sensor application focuses on organic molecules (not simple ions), this coverage is sufficient

---

## Rationale for FET Sensor Applications

### **Why these 25?**

1. **Conductivity Modulation (1-5)**: Charge, aromatic rings → directly affect charge carrier concentration
2. **Selective Adsorption (6-11)**: TPSA, XLogP, H-bonding → determine surface binding selectivity
3. **Response Amplitude (12-17)**: Volume3D, spatial distribution → affect signal strength
4. **Response Speed (18-20)**: Flexibility → faster response but may reduce selectivity
5. **Selectivity (21-25)**: Complexity, pharmacophore features → specific target recognition

---

## Python Extraction Code

```python
KEY_DESCRIPTORS_25 = {
    # Electronic properties (5)
    "Charge": "pubchem",
    "aromatic_rings": "chembl",
    "FeatureRingCount3D": "pubchem",
    "FeatureCationCount3D": "pubchem",
    "FeatureAnionCount3D": "pubchem",

    # Polarity & surface (6)
    "TPSA": "pubchem",
    "XLogP": "pubchem",
    "HBondDonorCount": "pubchem",
    "HBondAcceptorCount": "pubchem",
    "FeatureDonorCount3D": "pubchem",
    "FeatureAcceptorCount3D": "pubchem",

    # Size & shape (6)
    "MolecularWeight": "pubchem",
    "HeavyAtomCount": "pubchem",
    "Volume3D": "pubchem",
    "XStericQuadrupole3D": "pubchem",
    "YStericQuadrupole3D": "pubchem",
    "ZStericQuadrupole3D": "pubchem",

    # Flexibility (3)
    "RotatableBondCount": "pubchem",
    "EffectiveRotorCount3D": "pubchem",
    "ConformerModelRMSD3D": "pubchem",

    # Complexity (2)
    "Complexity": "pubchem",
    "FeatureHydrophobeCount3D": "pubchem",

    # Pharmacophore (3)
    "FeatureCount3D": "pubchem",
    "qed_weighted": "chembl",
    "np_likeness_score": "chembl",
}
```

---

## Complete Feature Set

**Total dimensions for ML/LLM:**
- **25 numerical descriptors** (this selection)
- **256-bit Morgan fingerprint** (locally computed via RDKit)
- **Total: 281 features**

---

Last updated: 2025-11-08
