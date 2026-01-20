# Chemical Data Fetcher

A Python utility for retrieving comprehensive molecular descriptors from PubChem and ChEMBL databases. Optimized for LLM fine-tuning and machine learning applications with **182 total descriptors** including 128-bit Morgan fingerprints.

**Total Features**: 182 descriptors (165 numeric + 17 identifiers/strings)
- 40 PubChem properties (physicochemical, 2D/3D descriptors)
- 14 ChEMBL properties (drug-likeness, ADME)
- 128-bit Morgan fingerprints (ECFP4)

## Key Features

- 🔬 **182 Molecular Descriptors**: Complete physicochemical, topological, and 3D properties
- 🧬 **Dual-source Integration**: PubChem (40 properties) + ChEMBL (14 drug properties)
- 🤖 **LLM-Optimized**: 128-bit Morgan fingerprints perfect for small-sample fine-tuning
- 🔍 **Flexible Queries**: Chemical name, SMILES, InChI, PubChem CID, ChEMBL ID, InChI Key
- ⚡ **Smart Caching**: Local JSON storage for instant repeated queries
- 📊 **ML-Ready**: 165 numeric features ready for machine learning pipelines
- 🛡️ **Rate-Limited**: Respects PubChem API limits (5 req/sec, 400 req/min)
- 🔧 **Dual Interface**: CLI tool and Python library

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Usage

```bash
# Query by chemical name (default)
python main.py "aspirin"
python main.py "carbon dioxide"

# Query by SMILES
python main.py --smiles "CCO"

# Query by InChI
python main.py --inchi "InChI=1S/H2O/h1H2"

# Query by PubChem CID
python main.py --cid 962

# Query by ChEMBL ID
python main.py --chembl-id "CHEMBL25"

# Force refresh (ignore cache)
python main.py "aspirin" --refresh

# Get raw JSON output
python main.py "aspirin" --json

# Clear cache
python main.py --clear-cache              # Clear all
python main.py "aspirin" --clear-cache    # Clear specific

# Disable fingerprint generation (faster if not needed)
python main.py "aspirin" --no-fingerprints
```

### Python Library Usage

```python
from src.query_handler import QueryHandler

# Initialize handler
handler = QueryHandler()

# Query by name (default)
result = handler.query("aspirin")

# Query by SMILES
result = handler.query("CCO", query_type="smiles")

# Query by PubChem CID
result = handler.query("962", query_type="cid")

# Force refresh from API
result = handler.query("aspirin", force_refresh=True)

# Access the data
if result:
    pubchem_data = result["data"]["pubchem"]
    chembl_data = result["data"]["chembl"]

    # Get molecular weight
    mw = pubchem_data["properties"]["MolecularWeight"]
    print(f"Molecular Weight: {mw}")
```

Run the example script to see more usage patterns:

```bash
python example_usage.py
```

## Query Types

| Type | Flag | Example | Description |
|------|------|---------|-------------|
| Name | (default) | `"aspirin"` | Chemical name |
| SMILES | `--smiles` | `"CCO"` | SMILES string |
| InChI | `--inchi` | `"InChI=1S/H2O/h1H2"` | InChI string |
| InChI Key | `--inchi-key` | `"BSYNRYMUTXBXSQ..."` | InChI Key |
| PubChem CID | `--cid` | `"962"` | PubChem Compound ID |
| ChEMBL ID | `--chembl-id` | `"CHEMBL25"` | ChEMBL molecule ID |

## Data Structure

Cached data is saved in `cache/` directory as JSON files:

```json
{
  "query": {
    "input": "aspirin",
    "type": "name",
    "timestamp": "2025-10-25T10:30:00Z"
  },
  "data": {
    "pubchem": {
      "source": "PubChem",
      "cid": 2244,
      "properties": {
        "MolecularFormula": "C9H8O4",
        "MolecularWeight": 180.16,
        "CanonicalSMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "InChI": "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)",
        "InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
        "IUPACName": "2-acetyloxybenzoic acid",
        "XLogP": 1.2,
        "TPSA": 63.6,
        "HBondDonorCount": 1,
        "HBondAcceptorCount": 4,
        ...
      }
    },
    "chembl": {
      "source": "ChEMBL",
      "count": 1,
      "molecules": [
        {
          "chembl_id": "CHEMBL25",
          "pref_name": "ASPIRIN",
          "molecule_type": "Small molecule",
          "max_phase": 4,
          "first_approval": 1950,
          "therapeutic_flag": true,
          ...
        }
      ]
    }
  },
  "metadata": {
    "cached": true,
    "last_updated": "2025-10-25T10:30:00Z"
  }
}
```

## Molecular Descriptors (182 Total)

### Complete Descriptor Reference

**📖 See [DESCRIPTOR_REFERENCE.md](DESCRIPTOR_REFERENCE.md) for detailed definitions, units, and references.**

### Quick Overview

**PubChem Properties (40 descriptors)**
- **Identifiers** (7): CID, MolecularFormula, SMILES, InChI, InChIKey, IUPACName
- **Basic Properties** (4): MolecularWeight, ExactMass, MonoisotopicMass, Charge
- **Physicochemical** (3): XLogP (lipophilicity), TPSA (polar surface area), Complexity
- **H-Bonding** (2): HBondDonorCount, HBondAcceptorCount
- **Structural Counts** (11): HeavyAtomCount, RotatableBondCount, AtomStereoCount, etc.
- **3D Properties** (13): Volume3D, conformer features, pharmacophore counts

**ChEMBL Properties (14 descriptors)**
- **Lipophilicity** (1): alogp (atomic contribution logP)
- **Structural** (4): aromatic_rings, heavy_atoms, rtb (rotatable bonds)
- **Molecular Formula/Weight** (3): full_molformula, full_mwt, mw_freebase
- **H-Bonding** (2): hba (acceptors), hbd (donors)
- **Drug-likeness** (4): psa, qed_weighted (QED score), np_likeness_score, num_ro5_violations, ro3_pass

**Morgan Fingerprint (128 bits)**
- **128-bit binary vector**: Circular fingerprint (ECFP4, radius=2)
- **Purpose**: Structural similarity, ML features
- **Optimized**: For LLM fine-tuning with small datasets
- **Additional**: on_bits positions, hex representation, count_fingerprint

## Molecular Fingerprints

The utility automatically generates **128-bit Morgan fingerprints** optimized for LLM and machine learning applications.

### Why 128 bits?

**Optimized for LLM/ML with small datasets:**
- ✅ Compact representation prevents overfitting with limited samples
- ✅ Consistent dimensionality across all molecules
- ✅ Fast computation and training
- ✅ Sufficient information for small molecule structures
- ✅ Perfect for LLM fine-tuning tasks

### What are Morgan Fingerprints?

Morgan fingerprints (ECFP - Extended-Connectivity Fingerprints) encode molecular structure as bit vectors. They're widely used for:

- **LLM fine-tuning**: Compact molecular features for language model training
- **Similarity searching**: Find structurally similar molecules (Tanimoto coefficient)
- **Machine learning**: Use as features for QSAR models, neural networks
- **Clustering**: Group molecules by structural similarity

### Fingerprint Data Format

**128-bit Morgan Fingerprint (ECFP4):**
```python
{
  "source": "RDKit (local)",
  "method": "Morgan",
  "radius": 2,                          # ECFP4
  "n_bits": 128,                        # Compact size for LLM/ML
  "num_on_bits": 18,                    # Number of bits set to 1
  "on_bits": [2, 5, 10, 11, ...],       # Positions of bits set to 1
  "bit_vector": [0, 0, 1, 0, ...],      # Full bit vector (128 values)
  "hex": "4000000...",                  # Hexadecimal representation
  "bit_string": "001001...",            # Bit string representation
  "count_fingerprint": {                # Feature counts for analysis
    "98513984": 2,
    "132611095": 1,
    ...
  }
}
```

### Usage Examples

**Calculate molecular similarity:**

```python
from src.query_handler import QueryHandler
from src.fingerprint_generator import FingerprintGenerator

handler = QueryHandler()
fp_gen = FingerprintGenerator()

# Get two molecules
aspirin = handler.query("aspirin")
ibuprofen = handler.query("ibuprofen")

# Calculate Tanimoto similarity
fp1 = aspirin["data"]["fingerprints"]
fp2 = ibuprofen["data"]["fingerprints"]
similarity = fp_gen.calculate_similarity(fp1, fp2)

print(f"Similarity: {similarity:.3f}")  # 0.167 (128-bit Morgan)
```

**Use fingerprints for LLM/ML:**

```python
import numpy as np

# Get 128-bit fingerprint as numpy array
result = handler.query("aspirin")
fp_vector = np.array(result["data"]["fingerprints"]["bit_vector"])

print(f"Fingerprint shape: {fp_vector.shape}")  # (128,)

# Use as input features:
# 1. For LLM fine-tuning: concatenate with text embeddings
# 2. For ML models: direct input features
# model.predict(fp_vector.reshape(1, -1))
```

**Disable fingerprints if not needed:**

```python
# Skip fingerprint generation for faster queries
handler = QueryHandler(generate_fingerprints=False)
```

See `example_fingerprints.py` for more detailed examples.

## Caching

- First query: Fetches from APIs → Saves to `cache/{sanitized_name}.json`
- Subsequent queries: Instantly loads from cache
- Cache files are named using sanitized query strings
- Use `--refresh` flag to force API fetch
- Use `--clear-cache` to remove cached data

## Project Structure

```
query_molecules/
├── cache/                       # Cached JSON files (auto-created)
├── src/
│   ├── __init__.py
│   ├── utils.py                 # Utilities (sanitization, file ops)
│   ├── cache_manager.py         # Cache operations
│   ├── pubchem_fetcher.py       # PubChem API client
│   ├── chembl_fetcher.py        # ChEMBL API client
│   ├── fingerprint_generator.py # Morgan fingerprint generation
│   └── query_handler.py         # Main query coordinator
├── main.py                      # CLI interface
├── example_usage.py             # Basic usage examples
├── example_fingerprints.py      # Fingerprint examples
├── test_fingerprints.py         # Test script
├── requirements.txt             # Dependencies
├── README.md                    # This file (overview)
├── DESCRIPTOR_REFERENCE.md      # Complete descriptor definitions
└── FINGERPRINT_INFO.md          # Fingerprint configuration guide
```

## API Rate Limits

**PubChem**:
- 5 requests per second
- 400 requests per minute
- Automatically enforced with delays

**ChEMBL**:
- No strict rate limits
- Be respectful with usage

## Error Handling

The utility handles various error cases gracefully:

- Chemical not found in database → Returns None, shows warning
- API connection errors → Shows error, tries other source
- Malformed queries → Falls back to name search
- Invalid cache files → Ignores cache, fetches fresh data

## Use Cases

### 1. LLM Fine-tuning for Drug Discovery

```python
from src.query_handler import QueryHandler
import numpy as np

handler = QueryHandler()

# Prepare training data for LLM
molecules = ["aspirin", "ibuprofen", "morphine", "caffeine"]
fingerprints = []
properties = []

for mol in molecules:
    result = handler.query(mol)
    if result:
        # Get 128-bit fingerprint
        fp = np.array(result["data"]["fingerprints"]["bit_vector"])
        fingerprints.append(fp)

        # Get selected properties
        props = result["data"]["pubchem"]["properties"]
        properties.append({
            "MW": props["MolecularWeight"],
            "LogP": props["XLogP"],
            "TPSA": props["TPSA"]
        })

# Use fingerprints as additional features for LLM
# fingerprints shape: (n_samples, 128)
```

### 2. Similarity Search

```python
from src.query_handler import QueryHandler
from src.fingerprint_generator import FingerprintGenerator

handler = QueryHandler()
fp_gen = FingerprintGenerator()

# Query molecule
target = handler.query("aspirin")
target_fp = target["data"]["fingerprints"]

# Compare with database
database = ["ibuprofen", "naproxen", "acetaminophen", "morphine"]
similarities = []

for mol in database:
    result = handler.query(mol)
    if result:
        fp = result["data"]["fingerprints"]
        sim = fp_gen.calculate_similarity(target_fp, fp)
        similarities.append((mol, sim))

# Sort by similarity
similarities.sort(key=lambda x: x[1], reverse=True)
print("Most similar to aspirin:", similarities[:3])
```

### 3. Batch Property Extraction

```python
from src.query_handler import QueryHandler
import pandas as pd

handler = QueryHandler()
molecules = ["aspirin", "caffeine", "glucose", "ethanol"]

data = []
for mol in molecules:
    result = handler.query(mol)
    if result:
        props = result["data"]["pubchem"]["properties"]
        chembl = result["data"]["chembl"]["molecules"][0] if result["data"]["chembl"] else {}

        data.append({
            "name": mol,
            "MW": props["MolecularWeight"],
            "XLogP": props["XLogP"],
            "TPSA": props["TPSA"],
            "HBD": props["HBondDonorCount"],
            "HBA": props["HBondAcceptorCount"],
            "QED": chembl.get("properties", {}).get("qed_weighted"),
        })

# Create DataFrame for analysis
df = pd.DataFrame(data)
df.to_csv("molecular_properties.csv", index=False)
```

### 4. Drug-likeness Filtering

```python
from src.query_handler import QueryHandler

handler = QueryHandler()

def lipinski_check(mol_name):
    """Check Lipinski's Rule of Five"""
    result = handler.query(mol_name)
    if not result:
        return None

    props = result["data"]["pubchem"]["properties"]

    violations = 0
    if props["MolecularWeight"] > 500:
        violations += 1
    if props["XLogP"] > 5:
        violations += 1
    if props["HBondDonorCount"] > 5:
        violations += 1
    if props["HBondAcceptorCount"] > 10:
        violations += 1

    return {
        "name": mol_name,
        "violations": violations,
        "drug_like": violations <= 1
    }

# Test compounds
compounds = ["aspirin", "paclitaxel", "morphine"]
for comp in compounds:
    result = lipinski_check(comp)
    print(f"{result['name']}: {'Pass' if result['drug_like'] else 'Fail'} "
          f"({result['violations']} violations)")
```

## Troubleshooting

**No data found**: Check chemical name spelling, try alternative names or use SMILES/InChI

**Connection errors**: Check internet connection, ChEMBL may be temporarily down

**Import errors**: Make sure you're in the project root and installed requirements

## License

This is a utility tool for accessing public chemical databases. Please respect the terms of service of PubChem and ChEMBL APIs.

## Documentation

- **[DESCRIPTOR_REFERENCE.md](DESCRIPTOR_REFERENCE.md)** - Complete reference for all 182 descriptors with definitions, units, and citations
- **[FINGERPRINT_INFO.md](FINGERPRINT_INFO.md)** - Detailed guide on Morgan fingerprint configuration and usage

## References

### APIs and Databases

1. **PubChem**: Kim, S. et al. (2023). "PubChem 2023 update." *Nucleic Acids Research*, 51(D1), D1373-D1380. https://pubchem.ncbi.nlm.nih.gov/
2. **ChEMBL**: Zdrazil, B. et al. (2024). "The ChEMBL Database in 2023." *Nucleic Acids Research*, 52(D1), D1180-D1192. https://www.ebi.ac.uk/chembl/

### Molecular Descriptors

3. **XLogP3**: Wang, R., Fu, Y., Lai, L. (1997). "A New Atom-Additive Method for Calculating Partition Coefficients." *J. Chem. Inf. Comput. Sci.*, 37, 615-621.
4. **TPSA**: Ertl, P., Rohde, B., Selzer, P. (2000). "Fast Calculation of Molecular Polar Surface Area." *J. Med. Chem.*, 43(20), 3714-3717.
5. **Complexity**: Bertz, S.H. (1981). "The First General Index of Molecular Complexity." *J. Am. Chem. Soc.*, 103, 3599-3601.

### Drug-likeness

6. **Lipinski's Rule of Five**: Lipinski, C.A. et al. (2001). "Experimental and computational approaches to estimate solubility and permeability." *Adv. Drug Deliv. Rev.*, 46(1-3), 3-26.
7. **QED**: Bickerton, G.R. et al. (2012). "Quantifying the chemical beauty of drugs." *Nat. Chem.*, 4(2), 90-98.

### Fingerprints

8. **Morgan Fingerprints**: Rogers, D., Hahn, M. (2010). "Extended-Connectivity Fingerprints." *J. Chem. Inf. Model.*, 50(5), 742-754.
9. **RDKit**: Landrum, G. (2023). RDKit: Open-source cheminformatics. https://www.rdkit.org/

### Textbooks

10. **Molecular Descriptors**: Todeschini, R., Consonni, V. (2009). *Molecular Descriptors for Chemoinformatics*. Wiley-VCH.
11. **Drug Discovery**: Wermuth, C.G. (2015). *The Practice of Medicinal Chemistry*. Academic Press.
