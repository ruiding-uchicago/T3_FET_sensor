# DNA/RNA Feature Extraction Utility

Extract 25 numerical macroscopic properties + 256D DNABERT-S embeddings from DNA/RNA sequences.

**✓ DNABERT-S Integration Complete!** All 198 sequences processed with species-aware deep learning embeddings (256D, nearly lossless compression).

## Quick Start

```python
from dna_feature_utils import extract_dna_features

# Input: sequence string (same format as CSV)
seq = "DNA 5'-GAGTCTGTGGAGGAGGTAGTC-3'"

# Extract features with DNABERT-S embeddings (automatically cached to JSON)
result = extract_dna_features(seq)

# Access results
props = result['macroscopic_properties']  # 25 numerical properties
embedding = result['embedding_256d']      # 256D DNABERT-S embedding
sequence = result['sequence']             # Parsed sequence

print(f"GC content: {props['gc_content']:.3f}")
print(f"Tm: {props['tm_celsius']:.1f}°C")
print(f"ΔG: {props['delta_g_kcal_mol']:.2f} kcal/mol")
print(f"is_rna: {props['is_rna']} (0=DNA, 1=RNA)")
print(f"Embedding (first 5): {embedding[:5]}")
```

## Features Extracted (25 total, all numerical)

### Basic Composition (10)
- `length`, `is_rna`, `gc_content`, `at_content`
- `a_count`, `c_count`, `g_count`, `t_count`, `u_count`
- `purine_pyrimidine_ratio`

**Note**: Sequence type is encoded as `is_rna` (0=DNA, 1=RNA) for full numerical compatibility.

### Thermodynamic (2)
- `tm_celsius` - Melting temperature (Wallace's rule)
- `delta_g_kcal_mol` - Free energy (nearest-neighbor method)

### Structural (5)
- `complexity` - Shannon entropy
- `longest_homopolymer` - Max consecutive identical nucleotides
- `cpg_count` - CpG dinucleotide count
- `gc_skew`, `at_skew` - Strand bias metrics

### Key Dinucleotide Frequencies (8)
- `dinuc_CG`, `dinuc_GC`, `dinuc_AT`, `dinuc_TA`
- `dinuc_GG`, `dinuc_CC`, `dinuc_AA`, `dinuc_TT`

## Files

### Main Files (DNABERT-S)
```
dna_feature_utils.py         # DNABERT-S utility module
process_sequences.py          # Batch process all sequences
dna_features_embeddings.csv   # Output: 25 properties + 256D DNABERT-S embeddings
dna_features_only.csv        # Output: 25 numerical macroscopic properties only
cache/                       # JSON cache (198 files)
```

### Input & Documentation
```
DNA_collection.csv           # Input: 198 DNA/RNA sequences
FEATURE_DEFINITIONS.md       # Feature documentation
```

## Batch Processing

Process all sequences from CSV:

```bash
python3 process_sequences.py
```

Creates:
- `cache/*.json` - Individual cached results (named by original sequence string)
- `dna_features_only.csv` - 25 numerical features for all sequences
- `dna_features_embeddings.csv` - 25 properties + 256D DNABERT-S embeddings

## JSON Cache Structure

Each sequence cached as `cache/{original_sequence_string}.json`:

Example: `cache/DNA 5'-ATCG-3'.json`

```json
{
  "original_input": "DNA 5'-ATCG-3'",
  "macroscopic_properties": {
    "length": 4,
    "gc_content": 0.5,
    "tm_celsius": 12.0,
    "is_rna": 0,
    ...
  },
  "embedding_256d": [-0.055, 0.057, -0.024, ...],
  "sequence": "ATCG"
}
```

**Note**: All embeddings are generated using DNABERT-S (species-aware, validated for 256D compression with nearly lossless performance).

## Statistics (198 sequences)

- DNA: 187, RNA: 11
- Length: 2-100 nt (mean: 28.4)
- GC content: 0.0-1.0 (mean: 0.50)
- Tm: 2-82°C (mean: 56.5°C)
