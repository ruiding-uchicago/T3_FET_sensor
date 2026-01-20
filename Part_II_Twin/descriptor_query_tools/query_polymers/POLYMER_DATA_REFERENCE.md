# Polymer Data Reference Specification

**Version**: 1.1
**Last Updated**: 2025-11-09

This document defines the **standard format** for polymer data retrieval from online databases. All data fetching agents MUST follow this specification to ensure consistent, structured output.

---

## ⚠️ CRITICAL RULES - READ FIRST ⚠️

### ABSOLUTE REQUIREMENTS:

1. **STRICT FORMAT COMPLIANCE**
   - Follow the JSON schema EXACTLY as specified
   - Do NOT add extra fields
   - Do NOT rename fields
   - Do NOT change data types

2. **MISSING DATA HANDLING**
   - If data is NOT available, use `null` (JSON null, not string "null")
   - NEVER use placeholder strings like "N/A", "not available", "unknown", "-", "?"
   - NEVER use empty strings `""` for numeric fields
   - NEVER use 0 or -1 as "missing" indicators

3. **DATA VALIDATION - ZERO TOLERANCE**
   - **NEVER** fill in data that does not conform to the specified format
   - **NEVER** guess or estimate values
   - **NEVER** fill data from unreliable sources
   - If a value is outside the valid range, set it to `null` and add a warning in metadata
   - If a value has wrong units, DO NOT convert unless you are 100% certain - set to `null` instead

4. **NUMERIC PRECISION**
   - Use proper numeric types (float/int), NOT strings
   - Round to 2-4 decimal places maximum
   - Example: `1.05` not `"1.05"` or `1.050000`

5. **REQUIRED FIELDS**
   - `polymer_name` and `repeat_unit_smiles` are REQUIRED
   - If these cannot be obtained, return an error (see Error Handling section)
   - All other fields are OPTIONAL - use `null` if not available

### CONSEQUENCES OF NON-COMPLIANCE:

- Invalid data will corrupt the entire dataset
- Downstream ML models will fail or produce incorrect results
- Data will be REJECTED and agent output will be marked as FAILED

**When in doubt: Use `null` instead of guessing!**

---

## Data Sources

**Use ANY reliable public sources**:
- Chemical databases (ChemicalBook, Sigma-Aldrich, TCI, etc.)
- Wikipedia
- Academic literature / papers
- Polymer databases (if accessible)
- PubChem (for SMILES validation)

**Critical**: Record ALL source URLs in `sources` field.

---

## Standard Output Format

### Complete JSON Schema

```json
{
  "query": {
    "input": "polystyrene",
    "type": "name",
    "timestamp": "2025-10-27T12:00:00Z"
  },
  "data": {
    "structure": {
      "polymer_name": "Polystyrene",
      "repeat_unit_smiles": "CCc1ccccc1",
      "bigsmiles": null,
      "iupac_name": "poly(styrene)",
      "cas_number": "9003-53-6",
      "common_abbreviations": ["PS"],
      "copolymer_components": null
    },
    "thermal_properties": {
      "Tg_C": 100.0,
      "Tg_K": 373.15,
      "Tm_C": null,
      "Tm_K": null,
      "Td_C": 350.0,
      "Td_K": 623.15
    },
    "physical_properties": {
      "density_g_cm3": 1.05,
      "molar_mass_repeat_g_mol": 104.15,
      "refractive_index": 1.59
    },
    "mechanical_properties": {
      "youngs_modulus_GPa": 3.0,
      "tensile_strength_MPa": 40.0
    },
    "electrical_properties": {
      "dielectric_constant": 2.6,
      "dielectric_loss": null
    },
    "solubility_parameters": {
      "delta_total_MPa": 18.5,
      "delta_d_MPa": null,
      "delta_p_MPa": null,
      "delta_h_MPa": null
    },
    "molecular_weight": {
      "Mn": null,
      "Mw": null,
      "PDI": null
    },
    "processing": {
      "polymerization_method": "free radical",
      "notes": null
    },
    "fingerprints": {
      "source": "local_computation",
      "method": "morgan_repeat_unit",
      "radius": 2,
      "n_bits": 256,
      "num_on_bits": 18,
      "on_bits": [2, 5, 10, 11, 15, 23, 28, 34, 45, 56, 67, 78, 89, 90, 101, 112, 123, 127],
      "bit_vector": [0, 0, 1, 0, 0, 1, ...],
      "hex": "4000000...",
      "bit_string": "001001...",
      "note": "Generated from repeat_unit_smiles using RDKit Morgan fingerprint (ECFP4)"
    },
    "monomer_descriptors": {
      "source": "query_molecules",
      "smiles": "CCc1ccccc1",
      "key_25": {
        "Charge": 0,
        "aromatic_rings": 1,
        "FeatureRingCount3D": 1,
        "FeatureCationCount3D": 0,
        "FeatureAnionCount3D": 0,
        "TPSA": 0,
        "XLogP": 3.1,
        "HBondDonorCount": 0,
        "HBondAcceptorCount": 0,
        "FeatureDonorCount3D": 0,
        "FeatureAcceptorCount3D": 0,
        "MolecularWeight": 106.16,
        "HeavyAtomCount": 8,
        "Volume3D": 93,
        "XStericQuadrupole3D": 3.35,
        "YStericQuadrupole3D": 1.35,
        "ZStericQuadrupole3D": 0.75,
        "RotatableBondCount": 1,
        "EffectiveRotorCount3D": 1,
        "ConformerModelRMSD3D": 0.4,
        "Complexity": 51,
        "FeatureHydrophobeCount3D": 1,
        "FeatureCount3D": 2,
        "qed_weighted": 0.51,
        "np_likeness_score": -0.77
      }
    }
  },
  "sources": [
    "https://www.chemicalbook.com/...",
    "https://polymerdatabase.com/...",
    "https://en.wikipedia.org/wiki/Polystyrene"
  ],
  "metadata": {
    "cached": false,
    "last_updated": "2025-10-27T12:00:00Z",
    "data_completeness": 0.75,
    "notes": "Data compiled from multiple public sources"
  }
}
```

---

## Field Definitions and Units

### Structure Fields

| Field | Type | Unit | Required | Description | Example |
|-------|------|------|----------|-------------|---------|
| `polymer_name` | string | - | YES | Common name of the polymer | "Polystyrene" |
| `repeat_unit_smiles` | string | - | YES | Canonical SMILES of repeat unit | "CCc1ccccc1" |
| `bigsmiles` | string | - | NO | BigSMILES notation if available | "{[]CCc1ccccc1[]}" |
| `iupac_name` | string | - | NO | IUPAC systematic name | "poly(styrene)" |
| `cas_number` | string | - | NO | CAS registry number | "9003-53-6" |
| `common_abbreviations` | array | - | NO | Common abbreviations | ["PS"] |
| `copolymer_components` | array | - | NO | For copolymers: links to constituent polymer JSON files | ["cache/component1.json", "cache/component2.json"] |

**Notes for `copolymer_components`:**
- OPTIONAL field, only use for complex copolymers (containing "alt", "co", "block", multiple monomer units)
- If the polymer is a simple homopolymer, set to `null` or omit entirely
- If the polymer is a copolymer and you can decompose it into simpler constituent polymers:
  1. Create separate JSON files for each constituent polymer
  2. List the relative paths to those JSON files in this array
  3. Each constituent file should follow the same format specification
- This enables LLM agents to handle complex copolymers by breaking them into manageable parts

### Thermal Properties

| Field | Type | Unit | Required | Description | Valid Range |
|-------|------|------|----------|-------------|-------------|
| `Tg_C` | float | °C | NO | Glass transition temperature | -200 to 400 |
| `Tg_K` | float | K | NO | Glass transition temperature | 73 to 673 |
| `Tm_C` | float | °C | NO | Melting temperature | -100 to 400 |
| `Tm_K` | float | K | NO | Melting temperature | 173 to 673 |
| `Td_C` | float | °C | NO | Decomposition temperature | 100 to 600 |
| `Td_K` | float | K | NO | Decomposition temperature | 373 to 873 |

**Validation Rules:**
- If both Celsius and Kelvin provided, verify: `Tg_K = Tg_C + 273.15`
- `Tm` must be greater than `Tg` (if both exist)
- `Td` must be greater than `Tm` and `Tg`

### Physical Properties

| Field | Type | Unit | Required | Description | Valid Range |
|-------|------|------|----------|-------------|-------------|
| `density_g_cm3` | float | g/cm³ | NO | Density at room temperature | 0.5 to 3.0 |
| `molar_mass_repeat_g_mol` | float | g/mol | NO | Molar mass of repeat unit | 10 to 10000 |
| `refractive_index` | float | - | NO | Refractive index | 1.3 to 2.0 |

### Molecular Weight

| Field | Type | Unit | Required | Description | Valid Range |
|-------|------|------|----------|-------------|-------------|
| `Mn` | float | g/mol or Da | NO | Number-average molecular weight | 1000 to 10^7 |
| `Mw` | float | g/mol or Da | NO | Weight-average molecular weight | 1000 to 10^7 |
| `PDI` | float | - | NO | Polydispersity index (Mw/Mn) | 1.0 to 10.0 |
| `measurement_method` | string | - | NO | Method used | "GPC", "MALDI-TOF", etc. |

**Validation Rules:**
- `Mw` must be ≥ `Mn`
- `PDI = Mw / Mn` (if all three provided, verify consistency)
- `PDI` must be ≥ 1.0

### Mechanical Properties

| Field | Type | Unit | Required | Description | Valid Range |
|-------|------|------|----------|-------------|-------------|
| `youngs_modulus_GPa` | float | GPa | NO | Young's modulus | 0.001 to 500 |
| `tensile_strength_MPa` | float | MPa | NO | Tensile strength | 1 to 1000 |

### Electrical Properties

| Field | Type | Unit | Required | Description | Valid Range |
|-------|------|------|----------|-------------|-------------|
| `dielectric_constant` | float | - | NO | Relative permittivity (εr) | 1.0 to 100 |
| `dielectric_loss` | float | - | NO | Loss tangent (tan δ) | 0 to 1.0 |

### Solubility Parameters

| Field | Type | Unit | Required | Description |
|-------|------|------|----------|-------------|
| `delta_total_MPa` | float | MPa^0.5 | NO | Total solubility parameter |
| `delta_d_MPa` | float | MPa^0.5 | NO | Dispersion component |
| `delta_p_MPa` | float | MPa^0.5 | NO | Polar component |
| `delta_h_MPa` | float | MPa^0.5 | NO | Hydrogen bonding component |

**Validation Rules:**
- `delta_total^2 ≈ delta_d^2 + delta_p^2 + delta_h^2` (Hansen equation, if all components provided)

### Processing Information

| Field | Type | Unit | Required | Description |
|-------|------|------|----------|-------------|
| `polymerization_method` | string | - | NO | Synthesis method | "free radical", "anionic", "condensation", "ROMP", etc. |
| `notes` | string | - | NO | Any additional processing notes | Free text |

### Monomer Descriptors (OPTIONAL - Auto-generated)

**NOTE**: This field is OPTIONAL and is typically added automatically by `add_monomer_descriptors.py`.
Agents fetching new polymer data should NOT manually add this field.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | string | NO | Source of descriptors (always "query_molecules") |
| `smiles` | string | NO | Monomer SMILES (same as repeat_unit_smiles) |
| `key_25` | object | NO | 25 key molecular descriptors from PubChem/ChEMBL |

**KEY_25 Descriptors** (see `../query_molecules/KEY_25_DESCRIPTORS.md` for details):

| Category | Descriptors | Source |
|----------|-------------|--------|
| Electronic (5) | Charge, aromatic_rings, FeatureRingCount3D, FeatureCationCount3D, FeatureAnionCount3D | PubChem/ChEMBL |
| Polarity (6) | TPSA, XLogP, HBondDonorCount, HBondAcceptorCount, FeatureDonorCount3D, FeatureAcceptorCount3D | PubChem |
| Size & Shape (6) | MolecularWeight, HeavyAtomCount, Volume3D, XStericQuadrupole3D, YStericQuadrupole3D, ZStericQuadrupole3D | PubChem |
| Flexibility (3) | RotatableBondCount, EffectiveRotorCount3D, ConformerModelRMSD3D | PubChem |
| Complexity (2) | Complexity, FeatureHydrophobeCount3D | PubChem |
| Pharmacophore (3) | FeatureCount3D, qed_weighted, np_likeness_score | PubChem/ChEMBL |

**Notes:**
- All descriptor values can be `null` if not available
- Typically 22-25 descriptors are populated (ChEMBL fields may be null)
- Generated from `repeat_unit_smiles` via query_molecules repo
- Adds ~25 dimensions to polymer feature set

### Sources Array (REQUIRED)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `sources` | array of strings | YES | List of ALL URLs where data was obtained | Example: ["https://www.chemicalbook.com/...", "https://polymerdatabase.com/..."] |

**Rules:**
- MUST include at least one source URL
- Include ALL sources consulted
- Prefer direct data sources over aggregators

---

## Query Types

Agents must support the following query types:

| Type | Description | Example |
|------|-------------|---------|
| `name` | Polymer common name | "polystyrene", "PMMA" |
| `smiles` | Repeat unit SMILES | "CC(C1=CC=CC=C1)" |
| `bigsmiles` | BigSMILES notation | "{[]CC(C1=CC=CC=C1)[]}" |
| `cas` | CAS registry number | "9003-53-6" |

---

## Data Retrieval Protocol

### Step 1: Search for Polymer Data
- Use web search, chemical databases, Wikipedia, literature
- Find canonical `repeat_unit_smiles` (use RDKit to validate)
- Collect thermal properties (Tg, Tm, Td), physical properties (density, refractive index)
- Get CAS number, IUPAC name, common abbreviations
- Record ALL source URLs

### Step 2: Generate Fingerprints (Local)
- Extract `repeat_unit_smiles` from collected data
- Use existing `src/fingerprint_generator.py`
- Generate 256-bit Morgan fingerprint (ECFP4, radius=2)
- Store: bit_vector, on_bits, hex, bit_string

**Example code:**
```python
from src.fingerprint_generator import FingerprintGenerator

generator = FingerprintGenerator()
fp_data = generator.generate_from_smiles(repeat_unit_smiles, radius=2, n_bits=256)
```

### Step 3 (Optional): Copolymer Decomposition

**For simple homopolymers (like polystyrene, PMMA):**
- Skip this step entirely
- Set `copolymer_components` to `null`

**For complex copolymers (containing "alt", "co", "block", or multiple distinct monomer units):**
1. **Detect if decomposition is beneficial:**
   - Very long/complex names (>50 characters)
   - Contains alternating or block copolymer indicators: "alt", "co-", "block"
   - Contains multiple distinct monomer units in brackets

2. **Decompose into constituent polymers:**
   - Identify the individual monomer units or simpler polymer segments
   - For each constituent, determine its polymer name
   - Example: `poly[A-alt-B]` → decompose into `poly(A)` and `poly(B)`

3. **Create separate JSON files for each constituent:**
   - Follow the same format specification for each constituent
   - Save as: `cache/{constituent_sanitized_name}.json`
   - Each constituent should have its own properties, fingerprints, etc.

4. **Link in original JSON:**
   - In the original copolymer JSON, set `copolymer_components` to array of paths
   - Example: `["cache/poly_a.json", "cache/poly_b.json"]`

**Example:**
For `poly[[2,5-bis(2-octyldodecyl)pyrrolo[3,4-c]pyrrole]-alt-[thieno[3,2-b]thiophene]]`:

Original JSON (`cache/poly_dpp_alt_tt.json`):
```json
{
  "data": {
    "structure": {
      "polymer_name": "poly[[2,5-bis(2-octyldodecyl)pyrrolo[3,4-c]pyrrole]-alt-[thieno[3,2-b]thiophene]]",
      "repeat_unit_smiles": "...",
      "copolymer_components": [
        "cache/poly_dpp.json",
        "cache/poly_thienothiophene.json"
      ]
    }
  }
}
```

Constituent 1 (`cache/poly_dpp.json`):
```json
{
  "data": {
    "structure": {
      "polymer_name": "poly(diketopyrrolopyrrole)",
      "repeat_unit_smiles": "...",
      "copolymer_components": null
    }
  }
}
```

Constituent 2 (`cache/poly_thienothiophene.json`):
```json
{
  "data": {
    "structure": {
      "polymer_name": "poly(thienothiophene)",
      "repeat_unit_smiles": "...",
      "copolymer_components": null
    }
  }
}
```

**Important Notes:**
- This is OPTIONAL - only use when it helps handle complexity
- Simple polymers should NEVER be decomposed
- All constituent files must follow the same format specification
- This enables gradual building of a polymer component library

---

## Data Validation Rules

### MUST Validate - ENFORCE STRICTLY:

1. **Temperature conversions**:
   - If both `T_C` and `T_K` provided, verify: `T_K = T_C + 273.15` (±0.1 tolerance)
   - If mismatch: **SET BOTH TO `null`** and log error in metadata
   - Do NOT "fix" one based on the other

2. **Physical constraints**:
   - `Tm > Tg` (if both exist and both non-null)
   - `Td > Tm` (if both exist and both non-null)
   - `Td > Tg` (if both exist and both non-null)
   - `PDI ≥ 1.0` (if non-null)
   - `Mw ≥ Mn` (if both non-null)
   - **If violated: SET violating values to `null`**

3. **Range checks**:
   - ALL numeric values MUST be within specified valid ranges
   - **If outside range: SET to `null`**, do NOT clip/truncate
   - Examples:
     - `Tg_C` not in [-200, 400] → `null`
     - `density_g_cm3` not in [0.5, 3.0] → `null`
     - `PDI` < 1.0 → `null`

4. **Required fields**:
   - `polymer_name` (string, non-empty)
   - `repeat_unit_smiles` (string, non-empty, valid SMILES)
   - `sources` (array, at least one URL)
   - **If missing: Return error, do NOT proceed**

5. **Sources array**:
   - MUST include at least one source URL
   - All URLs must be actual sources consulted (not made up)
   - **No exceptions**

6. **Data type enforcement**:
   - Numeric fields: `float` or `int` only, NEVER strings
   - String fields: `string` only
   - Missing: `null` only (JSON null type)
   - **Invalid types will cause rejection**

7. **SMILES validity**:
   - Verify `repeat_unit_smiles` is valid using RDKit `Chem.MolFromSmiles()`
   - **If invalid: Return error, do NOT proceed**
   - If valid but uncertain: Add warning to metadata but keep the SMILES

### VALIDATION FAILURE ACTIONS:

| Validation Type | Action | Example |
|-----------------|--------|---------|
| Temperature mismatch | Set both to `null` | Tg_C=100, Tg_K=300 → both `null` |
| Out of range | Set to `null` | Tg_C=500 → `null` |
| Physical constraint violation | Set violating field(s) to `null` | Tm < Tg → set Tm to `null` |
| Invalid SMILES | **ABORT - return error** | "XYZ123" is not valid |
| Missing required field | **ABORT - return error** | No polymer_name found |
| Wrong data type | **Attempt conversion, else `null`** | "1.05" → 1.05 or `null` |
| Missing sources | **ABORT - return error** | sources array is empty |

### NEVER DO:

- ❌ Do NOT guess missing values based on "similar" polymers
- ❌ Do NOT average values from multiple sources (keep them separate in respective blocks)
- ❌ Do NOT "fix" suspicious but valid data - use `null` and flag
- ❌ Do NOT use default values (like 0, -1, 999, etc.)
- ❌ Do NOT proceed if `repeat_unit_smiles` is invalid
- ❌ Do NOT convert units unless conversion factor is 100% certain
- ❌ Do NOT add extra fields not in specification

---

## Missing Data Handling - STRICT PROTOCOL

### RULES:

1. **Numeric fields**: Use `null` (JSON null type)
   - ✅ Correct: `"Tg_C": null`
   - ❌ Wrong: `"Tg_C": "N/A"`, `"Tg_C": ""`, `"Tg_C": 0`, `"Tg_C": -999`

2. **String fields** (optional): Use `null`
   - ✅ Correct: `"bigsmiles": null`
   - ❌ Wrong: `"bigsmiles": "N/A"`, `"bigsmiles": "unknown"`
   - **Exception**: Required string fields MUST have valid values (see below)

3. **Required fields** (polymer_name, repeat_unit_smiles, sources):
   - `polymer_name`: MUST have valid, non-empty string
   - `repeat_unit_smiles`: MUST have valid, non-empty string
   - `sources`: MUST be array with at least one URL
   - If not available: **Return error, do NOT use null**

4. **Optional data sections missing**:
   - If no data for a section (e.g., mechanical_properties), set all fields to `null`
   - ✅ Correct: `"mechanical_properties": {"youngs_modulus_GPa": null, "tensile_strength_MPa": null}`
   - ❌ Wrong: Omit the section entirely

5. **Data completeness tracking**:
   - Include `data_completeness` in metadata:
     ```
     data_completeness = (non-null fields) / (total possible fields)
     ```
   - Count only data fields, exclude metadata/provenance

### EXAMPLES:

**Correct - missing Tm:**
```json
{
  "thermal_properties": {
    "Tg_C": 100.0,
    "Tg_K": 373.15,
    "Tm_C": null,
    "Tm_K": null
  }
}
```

**WRONG - using placeholder strings:**
```json
{
  "thermal_properties": {
    "Tg_C": 100.0,
    "Tm_C": "not available"  // ❌ WRONG!
  }
}
```

**WRONG - using 0 for missing:**
```json
{
  "physical_properties": {
    "density_g_cm3": 0  // ❌ WRONG! Use null instead
  }
}
```

---

## Error Handling

### If polymer not found:
```json
{
  "query": {...},
  "data": {
    "structure": {
      "polymer_name": null,
      "repeat_unit_smiles": null,
      ...
    },
    ...
  },
  "sources": [],
  "metadata": {
    "error": "Polymer not found in any sources",
    "notes": "No reliable data available"
  }
}
```

### If partial data found:
- Fill available fields
- Set missing fields to `null`
- Continue with fingerprint generation if `repeat_unit_smiles` is available

---

## Example Queries

### Example 1: Polystyrene (Complete Data)

**Input**: `polystyrene`

**Expected Output**: See complete JSON schema above

### Example 2: PMMA (Poly(methyl methacrylate))

**Input**: `PMMA` or `poly(methyl methacrylate)`

**Expected Output**:
```json
{
  "data": {
    "structure": {
      "polymer_name": "Poly(methyl methacrylate)",
      "repeat_unit_smiles": "CC(C)(C(=O)OC)",
      "cas_number": "9011-14-7",
      "common_abbreviations": ["PMMA"]
    },
    "thermal_properties": {
      "Tg_C": 105.0,
      "Tg_K": 378.15,
      "Tm_C": null,
      "Tm_K": null,
      "Td_C": null,
      "Td_K": null
    },
    "physical_properties": {
      "density_g_cm3": 1.18,
      "molar_mass_repeat_g_mol": 100.12,
      "refractive_index": 1.49
    }
  },
  "sources": [
    "https://www.chemicalbook.com/...",
    "https://polymerdatabase.com/..."
  ]
}
```

### Example 3: Query by SMILES

**Input**: `CC(C1=CC=CC=C1)` (type: `smiles`)

**Expected Output**: Same as polystyrene query

---

## Agent Instructions

### ⚠️ MANDATORY COMPLIANCE CHECKLIST ⚠️

**Before returning any data, verify ALL of the following:**

- [ ] JSON is valid and parsable
- [ ] All field names match specification EXACTLY (case-sensitive)
- [ ] All data types are correct (float for numbers, string for text, null for missing)
- [ ] NO placeholder strings ("N/A", "unknown", etc.) - only `null`
- [ ] All numeric values are within valid ranges
- [ ] Temperature conversions are correct (T_K = T_C + 273.15, ±0.1)
- [ ] Physical constraints satisfied (Tm > Tg, PDI ≥ 1.0, etc.)
- [ ] `repeat_unit_smiles` is valid (RDKit can parse it)
- [ ] `polymer_name` and `repeat_unit_smiles` are present and non-empty
- [ ] Each data source has `provenance.source` and `provenance.database`
- [ ] NO extra fields added beyond specification
- [ ] Floats rounded to 2-4 decimal places
- [ ] Timestamps in ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ)

### IMPLEMENTATION RULES:

1. **Format compliance - ZERO TOLERANCE**:
   - Field names: EXACT match, case-sensitive
   - Data types: STRICT enforcement
   - Schema structure: NO deviations
   - **ANY violation = REJECTION**

2. **Data validation - MANDATORY**:
   - Validate BEFORE returning
   - Invalid data → set to `null`
   - Invalid SMILES → return error
   - Out-of-range values → set to `null`

3. **Provenance - REQUIRED**:
   - Every data source block MUST have provenance
   - `source`: "experimental" or "predicted" (exact strings)
   - `database`: "PoLyInfo", "PPPDB", or "PolymerGenome" (exact strings)

4. **Error handling**:
   - Partial data is acceptable IF required fields exist
   - Missing entire data source → set to `null`
   - Missing required fields → return error (see Error Handling)

5. **Caching**:
   - Cache results to avoid repeated web queries
   - Include cache metadata in output

6. **Rate limiting**:
   - Respect website rate limits
   - Add delays between requests if needed

7. **Logging**:
   - Log all URLs visited
   - Log all validation failures
   - Include warnings in metadata

### OUTPUT FORMAT REQUIREMENTS:

1. **JSON validity**:
   - Must be parsable by standard JSON parsers
   - Use proper escaping for special characters
   - Use UTF-8 encoding

2. **Null values**:
   - Use JSON `null` type (lowercase, unquoted)
   - ✅ Correct: `"Tg_C": null`
   - ❌ Wrong: `"Tg_C": "null"`, `"Tg_C": None`, `"Tg_C": undefined`

3. **Numeric precision**:
   - Round to 2-4 decimal places
   - ✅ Correct: `1.05`, `373.15`, `0.5`
   - ❌ Wrong: `1.050000000`, `1`, `"1.05"`

4. **Timestamps**:
   - ISO 8601 format: `YYYY-MM-DDTHH:MM:SSZ`
   - ✅ Correct: `"2025-10-27T12:00:00Z"`
   - ❌ Wrong: `"2025-10-27"`, `"10/27/2025"`, `1698408000`

5. **Strings**:
   - Use double quotes
   - Escape special characters properly
   - No trailing/leading whitespace

### QUALITY ASSURANCE:

**Self-check before returning:**

1. Parse your own JSON output - does it parse without errors?
2. Check for any string "null", "N/A", "unknown" - replace with JSON null
3. Check for any 0, -1, 999 as missing indicators - replace with null
4. Verify all required fields are present
5. Verify SMILES is valid using RDKit
6. Run all validation rules from "Data Validation Rules" section
7. If ANY validation fails, fix or set to `null`

**If uncertain:**
- Use `null` rather than guessing
- Add warning to metadata
- Log the issue for review

---

## Version History

- **v1.0** (2025-10-27): Initial specification
  - Defined schema for PoLyInfo, PPPDB, Polymer Genome
  - 256-bit Morgan fingerprint on repeat unit
  - Complete validation rules
  - Strict format compliance requirements

---

## Quick Reference - Common Mistakes to Avoid

### ❌ WRONG Examples (DO NOT DO):

```json
{
  "thermal_properties": {
    "Tg_C": "100",              // ❌ String instead of number
    "Tm_C": "N/A",              // ❌ Placeholder string
    "Td_C": 0                   // ❌ Zero for missing data
  },
  "physical_properties": {
    "density_g_cm3": null,      // ✅ This is correct
    "refractive_index": -1      // ❌ Invalid indicator for missing
  },
  "molecular_weight": {
    "Mn": "50000 Da",           // ❌ Number with units as string
    "PDI": 0.5                  // ❌ Below valid range (< 1.0)
  },
  "provenance": {
    "source": "experimental"    // ✅ Correct
    // ❌ Missing required "database" field
  }
}
```

### ✅ CORRECT Examples:

```json
{
  "thermal_properties": {
    "Tg_C": 100.0,              // ✅ Numeric type
    "Tm_C": null,               // ✅ Null for missing
    "Td_C": null                // ✅ Null for missing
  },
  "physical_properties": {
    "density_g_cm3": 1.05,      // ✅ Valid value
    "refractive_index": null    // ✅ Null for missing
  },
  "molecular_weight": {
    "Mn": 50000,                // ✅ Numeric value, no units
    "Mw": 100000,               // ✅ Valid
    "PDI": 2.0                  // ✅ Within valid range (≥ 1.0)
  },
  "provenance": {
    "source": "experimental",   // ✅ Exact string
    "database": "PoLyInfo",     // ✅ Exact string
    "reference": null           // ✅ Null for optional missing field
  }
}
```

---

## Final Pre-Submission Checklist

**Run this checklist before submitting ANY output:**

1. **Format**:
   - [ ] JSON parses without errors
   - [ ] All field names match specification (case-sensitive)
   - [ ] No extra fields added
   - [ ] No fields renamed

2. **Data Types**:
   - [ ] All numbers are `float` or `int` (not strings)
   - [ ] All missing values are `null` (not "N/A", "null", 0, -1, etc.)
   - [ ] All strings use double quotes
   - [ ] No boolean values where not specified

3. **Required Fields**:
   - [ ] `polymer_name` is present and non-empty
   - [ ] `repeat_unit_smiles` is present and non-empty
   - [ ] `repeat_unit_smiles` is valid (RDKit can parse)

4. **Validation**:
   - [ ] All temperatures: T_K = T_C + 273.15 (±0.1) or set to null
   - [ ] Tm > Tg (if both exist) or violated value set to null
   - [ ] Td > Tm (if both exist) or violated value set to null
   - [ ] PDI ≥ 1.0 or set to null
   - [ ] Mw ≥ Mn or violated value set to null
   - [ ] All values within specified ranges or set to null

5. **Provenance**:
   - [ ] Each non-null data source has `provenance.source`
   - [ ] Each non-null data source has `provenance.database`
   - [ ] Values are exact strings: "experimental"/"predicted", "PoLyInfo"/"PPPDB"/"PolymerGenome"

6. **Quality**:
   - [ ] Floats rounded to 2-4 decimal places
   - [ ] Timestamps in ISO 8601 format
   - [ ] No placeholder strings anywhere
   - [ ] No trailing/leading whitespace in strings

7. **Metadata**:
   - [ ] Timestamp included
   - [ ] Data completeness calculated
   - [ ] Warnings logged if any validation issues occurred

**If ANY checkbox is unchecked, DO NOT SUBMIT. Fix the issue first.**

---

## Summary

**Key Principles:**
1. **Strict format** - no deviations
2. **Use `null` for missing** - never use placeholders
3. **Validate everything** - reject invalid data
4. **When uncertain** - use `null` and log warning

**Remember:** Invalid data is worse than missing data. When in doubt, use `null`.

---

## References

1. **PoLyInfo**: https://polymer.nims.go.jp/
2. **PPPDB**: https://pppdb.uchicago.edu/
3. **Polymer Genome**: https://polymergenome.org/
4. **BigSMILES**: Lin, T.-S. et al. (2019). "BigSMILES: A Structurally-Based Line Notation for Describing Macromolecules." *ACS Central Science*, 5(9), 1523-1531.
5. **Morgan Fingerprints**: Rogers, D., Hahn, M. (2010). "Extended-Connectivity Fingerprints." *J. Chem. Inf. Model.*, 50(5), 742-754.
6. **RDKit**: Landrum, G. (2023). RDKit: Open-source cheminformatics. https://www.rdkit.org/
