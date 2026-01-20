# Polymer Data Fetcher

A Python utility for retrieving comprehensive polymer properties from online databases. Optimized for LLM fine-tuning and machine learning applications with **256-bit Morgan fingerprints** on repeat units.

**Version**: 1.1 (Nov 2025 - Added monomer descriptors)

---

## Overview

This tool fetches polymer data from three complementary sources:
1. **PoLyInfo** (NIMS) - Experimental data from literature
2. **PPPDB** (UChicago/NIST) - Predicted solubility parameters and properties
3. **Polymer Genome** - ML-predicted thermal, electrical, optical properties

**Total Features**: ~30 explicit properties + 256-bit structural fingerprint + 25 monomer descriptors

---

## Key Features

- 🔬 **~30 Explicit Properties**: Tg, Tm, Td, density, molecular weight, solubility parameters, dielectric constant, etc.
- 🧬 **Triple-source Integration**: PoLyInfo (experimental) + PPPDB (predicted) + Polymer Genome (ML predicted)
- 🤖 **LLM-Optimized**: 256-bit Morgan fingerprints on repeat units (consistent with query_molecules)
- 🧪 **Monomer Descriptors**: 25 key molecular descriptors (electronic, polarity, size, flexibility) from query_molecules
- 🔍 **Flexible Queries**: Polymer name, repeat unit SMILES, BigSMILES, CAS number
- ⚡ **Smart Caching**: Local JSON storage for instant repeated queries (291 polymers cached, 264 with monomer descriptors)
- 📊 **ML-Ready**: Structured numeric features + high-dimensional fingerprints + monomer properties (~311 total dimensions)
- 🛡️ **Provenance Tracking**: Labels experimental vs predicted data
- 🔧 **Agent-Based Fetching**: Uses LLM agents to query databases without APIs

---

## 🤖 For LLM Agents: How to Query Polymer Data

**If you're an LLM agent looking for polymer property data, follow this workflow:**

### Step 1: Check Cache First (Recommended)

The `cache/` directory contains **293 pre-fetched polymer JSON files**. Always check here first:

```bash
# Convert polymer name to cache filename
# Rule: lowercase, special chars → underscores, append .json
# Example: "poly(3-hexylthiophene)" → "poly_3_hexylthiophene.json"

ls cache/poly_3_hexylthiophene.json
# If exists → Read the JSON file directly
# If not exists → Proceed to Step 2
```

**Sanitization Rules** (polymer name → cache filename):
1. Convert to lowercase
2. Remove quotes (`"` and `'`)
3. Replace special characters with underscores: `[^\w\s\-]` → `_`
4. Replace spaces and hyphens with underscores
5. Replace multiple underscores with single underscore
6. Remove leading/trailing underscores
7. Append `.json`

> **Tip:** Use `src/utils.py::sanitize_filename()` for automatic conversion

**Quick check examples:**
```bash
ls cache/polystyrene.json                    # Polystyrene
ls cache/poly_3_hexylthiophene.json          # poly(3-hexylthiophene)
ls cache/polyethylene_terephthalate.json     # polyethylene terephthalate
ls cache/nafion.json                         # Nafion
```

### Step 2: If Not in Cache, Fetch New Data

If the polymer is not cached, follow **AGENT_PROMPT_TEMPLATE.md** to fetch:

```
Task: Fetch polymer data for "{POLYMER_NAME}"

1. CHECK CACHE: ls cache/{sanitized_name}.json - if exists, STOP
2. Read POLYMER_DATA_REFERENCE.md for format specification
3. Read POLYMER_FETCH_TASK.md for detailed steps
4. Search online (PoLyInfo, PPPDB, Polymer Genome, literature)
5. Generate 256-bit Morgan fingerprint using src/fingerprint_generator.py
6. Save to cache/{sanitized_name}.json
```

**Key Files for Agents:**
- 📋 **AGENT_PROMPT_TEMPLATE.md** - Task instructions for fetching
- 📖 **POLYMER_DATA_REFERENCE.md** - JSON format specification, units, validation rules
- 📝 **POLYMER_FETCH_TASK.md** - Detailed step-by-step guide
- 🔧 **src/fingerprint_generator.py** - Generate 256-bit Morgan fingerprints
- 💾 **cache/** - Pre-fetched polymer data (293 polymers)

**Example workflow:**
```python
# 1. Check if cached
import os
polymer_name = "poly(3-hexylthiophene)"
sanitized = "poly_3_hexylthiophene"  # Apply sanitization rules
cache_path = f"cache/{sanitized}.json"

if os.path.exists(cache_path):
    # Read existing JSON
    with open(cache_path, 'r') as f:
        data = json.load(f)
else:
    # Not cached → fetch following AGENT_PROMPT_TEMPLATE.md
    # Read POLYMER_DATA_REFERENCE.md for format
    # Search online, extract data, generate fingerprint
    # Save to cache_path
```

---

## Architecture

- Fetch explicit properties from 3 databases (PoLyInfo, PPPDB, Polymer Genome)
- Generate 256-bit Morgan fingerprint from `repeat_unit_smiles`
- Maintain consistency with `query_molecules` interface

---

## Data Sources

| Source | Type | Properties | Provenance |
|--------|------|------------|------------|
| **PoLyInfo** | Experimental | repeat_unit, Tg, Tm, Td, density, Mn, Mw, PDI, processing | `experimental` |
| **PPPDB** | Predicted | χ parameter, solubility parameters, Tg, mechanical properties | `predicted` |
| **Polymer Genome** | ML Predicted | Tg, dielectric constant, refractive index, permeability | `predicted` |

---

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Command Line Usage

```bash
# Query by polymer name (default)
python main.py "polystyrene"
python main.py "PMMA"

# Query by repeat unit SMILES
python main.py --smiles "CC(C1=CC=CC=C1)"

# Query by BigSMILES
python main.py --bigsmiles "{[]CC(C1=CC=CC=C1)[]}"

# Query by CAS number
python main.py --cas "9003-53-6"

# Force refresh (ignore cache)
python main.py "polystyrene" --refresh

# Get raw JSON output
python main.py "polystyrene" --json

# Clear cache
python main.py --clear-cache              # Clear all
python main.py "polystyrene" --clear-cache    # Clear specific
```

### Python Library Usage

```python
from src.query_handler import PolymerQueryHandler

# Initialize handler
handler = PolymerQueryHandler()

# Query by name (default)
result = handler.query("polystyrene")

# Query by SMILES
result = handler.query("CC(C1=CC=CC=C1)", query_type="smiles")

# Force refresh from web
result = handler.query("PMMA", force_refresh=True)

# Access the data
if result:
    polyinfo_data = result["data"]["polyinfo"]
    pppdb_data = result["data"]["pppdb"]

    # Get glass transition temperature
    tg = polyinfo_data["thermal_properties"]["Tg_C"]
    print(f"Tg: {tg}°C")

    # Get fingerprint
    fp_vector = result["data"]["fingerprints"]["bit_vector"]
    print(f"Fingerprint: {len(fp_vector)} bits")
```

---

## Data Structure

See **[POLYMER_DATA_REFERENCE.md](POLYMER_DATA_REFERENCE.md)** for complete specification.

Cached data is saved in `cache/` directory as JSON files:

```json
{
  "query": {
    "input": "polystyrene",
    "type": "name",
    "timestamp": "2025-10-27T12:00:00Z"
  },
  "data": {
    "polyinfo": {
      "structure": {
        "polymer_name": "Polystyrene",
        "repeat_unit_smiles": "CC(C1=CC=CC=C1)",
        "cas_number": "9003-53-6"
      },
      "thermal_properties": {
        "Tg_C": 100.0,
        "Tm_C": null
      },
      "physical_properties": {
        "density_g_cm3": 1.05
      },
      "provenance": {
        "source": "experimental",
        "database": "PoLyInfo"
      }
    },
    "pppdb": {
      "solubility_parameters": {
        "chi_parameter": 0.5
      },
      "provenance": {
        "source": "predicted",
        "database": "PPPDB"
      }
    },
    "polymer_genome": {
      "electrical_properties": {
        "dielectric_constant": 3.2
      },
      "provenance": {
        "source": "predicted",
        "database": "PolymerGenome"
      }
    },
    "fingerprints": {
      "method": "morgan_repeat_unit",
      "n_bits": 256,
      "bit_vector": [0, 0, 1, 0, ...],
      "note": "Generated from repeat unit SMILES"
    }
  }
}
```

---

## Polymer Properties (~30 dimensions)

### Complete Property Reference

**📖 See [POLYMER_DATA_REFERENCE.md](POLYMER_DATA_REFERENCE.md) for detailed definitions, units, and validation rules.**

### Quick Overview

**Structure (PoLyInfo)**
- `polymer_name`, `repeat_unit_smiles`, `bigsmiles`, `iupac_name`, `cas_number`

**Thermal Properties (PoLyInfo, PPPDB, Polymer Genome)**
- `Tg_C`, `Tg_K` (Glass transition)
- `Tm_C`, `Tm_K` (Melting point)
- `Td_C`, `Td_K` (Decomposition)

**Physical Properties (PoLyInfo)**
- `density_g_cm3`, `molar_mass_repeat_g_mol`, `refractive_index`

**Molecular Weight (PoLyInfo)**
- `Mn`, `Mw`, `PDI`, `measurement_method`

**Solubility Parameters (PPPDB)**
- `chi_parameter`, `delta_total_MPa`, `delta_d_MPa`, `delta_p_MPa`, `delta_h_MPa`

**Electrical Properties (Polymer Genome)**
- `dielectric_constant`, `dielectric_loss`

**Processing Information (PoLyInfo)**
- `polymerization_method`, `processing_conditions`

**Morgan Fingerprint (256 bits)**
- Generated from `repeat_unit_smiles` using RDKit
- Radius=2 (ECFP4), consistent with `query_molecules`
- Includes: `bit_vector`, `on_bits`, `hex`, `bit_string`

**Monomer Descriptors (25 dimensions) - NEW!**
- Automatically extracted from `query_molecules` repo
- 25 key molecular descriptors organized by:
  - **Electronic** (5): Charge, aromatic_rings, FeatureRingCount3D, etc.
  - **Polarity** (6): TPSA, XLogP, H-bond donors/acceptors, etc.
  - **Size & Shape** (6): MolecularWeight, Volume3D, Steric quadrupoles, etc.
  - **Flexibility** (3): RotatableBondCount, EffectiveRotorCount3D, etc.
  - **Complexity** (2): Complexity, FeatureHydrophobeCount3D
  - **Pharmacophore** (3): FeatureCount3D, qed_weighted, np_likeness_score
- Source: PubChem + ChEMBL APIs via `../query_molecules`
- See `../query_molecules/KEY_25_DESCRIPTORS.md` for details
- Coverage: ~90% of polymers (264/291 have monomer descriptors)

**How to add monomer descriptors to existing/new polymers:**
```bash
# Single polymer
python3 add_monomer_descriptors.py polystyrene.json

# Batch update all polymers
python3 add_monomer_descriptors.py
```

---

## Agent-Based Data Fetching

Since PoLyInfo, PPPDB, and Polymer Genome do not provide APIs, this system uses **LLM agents** to:
1. Navigate to the database websites
2. Search for the polymer
3. Extract data according to **POLYMER_DATA_REFERENCE.md** specification
4. Return structured JSON
5. **Automatically enrich data** with monomer descriptors and RDKit fallback

All agents must strictly follow the format specification to ensure consistency.

### Post-Processing Scripts (Auto-run by Agents)

After saving polymer JSON files, agents automatically run these scripts to maximize data completeness:

**`add_monomer_descriptors.py`** - Add 25 monomer descriptors from query_molecules
```bash
# Single polymer
python3 add_monomer_descriptors.py polystyrene

# Batch mode (all polymers)
python3 add_monomer_descriptors.py
```
- Queries PubChem/ChEMBL via query_molecules repo
- Extracts KEY_25 descriptors: electronic (5), polarity (6), size/shape (6), flexibility (3), complexity (2), pharmacophore (3)
- Success rate: ~90% coverage for polymers with valid SMILES

**`fill_computable_properties.py`** - RDKit fallback for missing data
```bash
# Single polymer
python3 fill_computable_properties.py pbttt_c14

# Batch mode (all polymers)
python3 fill_computable_properties.py
```
- Fills `molar_mass_repeat_g_mol` if missing (now 87.3% coverage, +18.1%)
- Computes 9 RDKit descriptors when PubChem/ChEMBL fails: Charge, aromatic_rings, TPSA, XLogP, H-bond counts, MolecularWeight, HeavyAtomCount, RotatableBondCount
- Reduces "all-null" descriptors from 26.4% → 7.6%

**Current Data Quality** (292 polymers):
- Monomer descriptors: 67.7% filled (avg 16.9/25 per polymer)
- Molar mass: 87.3% filled (255/292 polymers)
- RDKit-computable descriptors: ~90% coverage

---

## Validation & Quality Control

The system validates:
- ✅ Temperature conversions (C ↔ K)
- ✅ Physical constraints (Tm > Tg, PDI ≥ 1.0, etc.)
- ✅ Value ranges (density: 0.5-3.0 g/cm³, etc.)
- ✅ SMILES validity (using RDKit)
- ✅ Data completeness tracking

See validation rules in [POLYMER_DATA_REFERENCE.md](POLYMER_DATA_REFERENCE.md).

---

## Use Cases

### 1. LLM Fine-tuning for Polymer Property Prediction

```python
from src.query_handler import PolymerQueryHandler
import numpy as np

handler = PolymerQueryHandler()

# Prepare training data
polymers = ["polystyrene", "PMMA", "polyethylene", "PVC"]
fingerprints = []
properties = []

for polymer in polymers:
    result = handler.query(polymer)
    if result:
        # Get 256-bit fingerprint
        fp = np.array(result["data"]["fingerprints"]["bit_vector"])
        fingerprints.append(fp)

        # Get Tg
        tg = result["data"]["polyinfo"]["thermal_properties"]["Tg_C"]
        properties.append({"Tg_C": tg})

# Use for LLM fine-tuning
# fingerprints shape: (n_samples, 128)
```

### 2. Property Database Construction

```python
import pandas as pd
from src.query_handler import PolymerQueryHandler

handler = PolymerQueryHandler()
polymers = ["polystyrene", "PMMA", "polyethylene"]

data = []
for polymer in polymers:
    result = handler.query(polymer)
    if result:
        pi = result["data"]["polyinfo"]
        pppdb = result["data"]["pppdb"]
        pg = result["data"]["polymer_genome"]

        data.append({
            "name": pi["structure"]["polymer_name"],
            "SMILES": pi["structure"]["repeat_unit_smiles"],
            "Tg_exp": pi["thermal_properties"]["Tg_C"],
            "Tg_pred_PPPDB": pppdb["thermal_properties"]["Tg_C"],
            "Tg_pred_PG": pg["thermal_properties"]["Tg_C"],
            "density": pi["physical_properties"]["density_g_cm3"],
            "dielectric": pg["electrical_properties"]["dielectric_constant"]
        })

df = pd.DataFrame(data)
df.to_csv("polymer_properties.csv")
```

### 3. Multi-source Comparison

```python
result = handler.query("polystyrene")

# Compare Tg from different sources
tg_exp = result["data"]["polyinfo"]["thermal_properties"]["Tg_C"]
tg_pppdb = result["data"]["pppdb"]["thermal_properties"]["Tg_C"]
tg_pg = result["data"]["polymer_genome"]["thermal_properties"]["Tg_C"]

print(f"Tg (experimental): {tg_exp}°C")
print(f"Tg (PPPDB predicted): {tg_pppdb}°C")
print(f"Tg (Polymer Genome predicted): {tg_pg}°C")
```

---

## Project Structure

```
query_polymers/
├── cache/                          # Cached JSON files (auto-created)
├── src/
│   ├── __init__.py
│   ├── utils.py                    # Utilities (sanitization, validation)
│   ├── cache_manager.py            # Cache operations
│   ├── polyinfo_fetcher.py         # PoLyInfo agent fetcher
│   ├── pppdb_fetcher.py            # PPPDB agent fetcher
│   ├── polymer_genome_fetcher.py   # Polymer Genome agent fetcher
│   ├── fingerprint_generator.py    # Morgan fingerprint (from repeat unit)
│   ├── data_validator.py           # Validation logic
│   └── query_handler.py            # Main query coordinator
├── main.py                         # CLI interface
├── example_usage.py                # Usage examples
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── POLYMER_DATA_REFERENCE.md       # **SPECIFICATION** (for agents)
└── expert_suggestions.md           # Expert recommendations
```

---

## Caching

- First query: Fetches from websites via agents → Saves to `cache/{sanitized_name}.json`
- Subsequent queries: Instantly loads from cache
- Use `--refresh` flag to force re-fetch
- Use `--clear-cache` to remove cached data

---

## Error Handling

- Polymer not found → Returns null for missing sources, continues with available data
- Invalid SMILES → Cannot generate fingerprint, warns user
- Web fetch errors → Tries other sources, reports partial data
- Validation failures → Warns but returns data with quality flags

---

## Comparison with query_molecules

| Feature | query_molecules | query_polymers |
|---------|-----------------|----------------|
| **Target** | Small molecules | Polymers |
| **Data sources** | PubChem + ChEMBL APIs | PoLyInfo + PPPDB + Polymer Genome (agent-fetched) |
| **Fingerprint method** | 256-bit Morgan on full molecule | 256-bit Morgan on repeat unit |
| **# Properties** | 182 descriptors | ~30 explicit + 256-bit FP |
| **Provenance** | Not tracked | Experimental vs Predicted |

---

## Implementation Status

- [x] Define standard format specification (POLYMER_DATA_REFERENCE.md)
- [ ] Implement basic structure (cache, utils, query handler)
- [ ] Create agent-based fetchers for each database
- [ ] Generate Morgan fingerprints from repeat units
- [ ] Data validation and quality control
- [ ] CLI and library interfaces

---

## References

### Databases
1. **PoLyInfo**: National Institute for Materials Science. https://polymer.nims.go.jp/
2. **PPPDB**: Polymer Property Predictor & Database. https://pppdb.uchicago.edu/
3. **Polymer Genome**: https://polymergenome.org/

### Methods
4. **BigSMILES**: Lin, T.-S. et al. (2019). "BigSMILES: A Structurally-Based Line Notation for Describing Macromolecules." *ACS Central Science*, 5(9), 1523-1531.
5. **Morgan Fingerprints**: Rogers, D., Hahn, M. (2010). "Extended-Connectivity Fingerprints." *J. Chem. Inf. Model.*, 50(5), 742-754.

---

## License

This is a utility tool for accessing public polymer databases. Please respect the terms of service of each database.

---

## Documentation

- **[POLYMER_DATA_REFERENCE.md](POLYMER_DATA_REFERENCE.md)** - **REQUIRED READING** for all agents - Complete specification with field definitions, units, validation rules, and examples
- **[expert_suggestions.md](expert_suggestions.md)** - Expert recommendations on polymer fingerprinting approaches
