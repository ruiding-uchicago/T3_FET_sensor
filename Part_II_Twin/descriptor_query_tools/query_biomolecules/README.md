# Protein Information Extraction Tool

A Python-based tool for extracting protein information from UniProt and generating ESM-2 embeddings for protein sequences.

## Overview

This tool queries the UniProt database to retrieve protein sequences and metadata, calculates 25 macroscopic biochemical properties, and generates 320-dimensional ESM-2 embeddings for each protein. The output is stored as JSON files for downstream analysis.

## Features

- **UniProt Integration**: Automatic protein sequence retrieval from UniProt REST API
- **Biochemical Properties**: Calculates 25 macroscopic properties including:
  - Molecular weight, isoelectric point, aromaticity
  - GRAVY (hydropathy index), instability index
  - Amino acid composition (aromatic, aliphatic, polar, charged fractions)
  - Secondary structure propensities (helix, turn, sheet)
  - Special amino acid counts (cysteine, proline, glycine)
  - Extinction coefficients (reduced/oxidized), flexibility, tiny/large residue fractions
- **ESM-2 Embeddings**: Generates 320-dimensional protein sequence embeddings using Facebook's ESM-2 model (t6_8M)
- **Batch Processing**: Process multiple proteins from CSV files
- **Smart Retry**: Automatically retries failed proteins with alternative standardized names
- **Handles Non-standard Amino Acids**: Automatically converts selenocysteine (U), Asx (B), Glx (Z), etc.

## Installation

### Requirements

- Python 3.8+
- pip or conda

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Or using conda:

```bash
conda install requests pandas biopython numpy
conda install pytorch transformers -c pytorch
```

### Required Libraries

- `requests`: UniProt API queries
- `biopython`: Sequence analysis and property calculation
- `numpy`: Numerical operations
- `torch`: PyTorch for ESM-2 model
- `transformers`: Hugging Face transformers for ESM-2

## Usage

### 1. Single Protein Extraction

Extract information for a single protein by name:

```bash
python extract_protein.py "glucose oxidase"
```

**Output**: Creates `cache/glucose oxidase.json` with protein information.

### 2. Batch Processing from CSV

Process all proteins listed in a CSV file:

```bash
python batch_extract.py bio_protein_collection.csv
```

**Features**:
- Automatically skips already-cached proteins
- Handles duplicates (case-insensitive)
- Adds 1-second delay between API requests
- Generates summary report: `cache/_batch_summary.json`

**CSV Format**:
```csv
glucose oxidase
bovine serum albumin
cardiac troponin I
...
```

(One protein name per line, no headers required)

### 3. Retry Failed Proteins

Retry proteins that failed with alternative standardized names:

```bash
python retry_failed.py
```

This script uses a mapping dictionary to retry failed proteins with:
- Modified proteins → Base proteins (e.g., `biotinylated BSA` → `BSA`)
- Protein mutants → Wild-type (e.g., `F88W mutant` → wild-type)
- Long names → Standard abbreviations (e.g., `dCas9...` → `Cas9`)

**Note**: This script does NOT map antibodies to antigens (these are fundamentally different proteins).

## Output Format

Each protein generates a JSON file with the following structure:

```json
{
  "query_name": "glucose oxidase",
  "uniprot_info": {
    "accession": "P18173",
    "protein_name": "Glucose dehydrogenase [FAD, quinone]",
    "organism": "Drosophila melanogaster",
    "sequence_length": 625,
    "sequence": "MSASASACDCL..."
  },
  "macroscopic_properties": {
    "sequence_length": 625,
    "molecular_weight_kDa": 68.39,
    "isoelectric_point": 6.79,
    "aromaticity": 0.0832,
    "instability_index": 34.73,
    "gravy": -0.1651,
    "charge_at_pH7": -1.06,
    "charge_at_pH5": 23.34,
    "aromatic_fraction": 8.32,
    "aliphatic_fraction": 29.44,
    "polar_fraction": 19.84,
    "charged_fraction": 19.84,
    "positive_fraction": 12.48,
    "negative_fraction": 10.08,
    "helix_fraction": 0.2928,
    "turn_fraction": 0.3216,
    "sheet_fraction": 0.3488,
    "cysteine_count": 11,
    "proline_count": 35,
    "glycine_fraction": 9.28
  },
  "embedding": {
    "embedding_dim": 320,
    "embedding_vector": [0.0115, -0.186, 0.130, ...],
    "model_used": "facebook/esm2_t6_8M_UR50D"
  }
}
```

## File Structure

```
query_biomolecules/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── bio_protein_collection.csv         # Input: list of protein names
├── extract_protein.py                 # Core: single protein extraction
├── batch_extract.py                   # Batch processing script
├── retry_failed.py                    # Retry with alternative names
├── cleanup_invalid_mappings.py        # Clean up incorrect mappings
├── analyze_mappings.py                # Analyze mapping validity
└── cache/                             # Output directory
    ├── glucose oxidase.json           # Individual protein data
    ├── cardiac troponin I.json
    ├── ...
    ├── _batch_summary.json            # Batch processing summary
    └── _retry_summary.json            # Retry summary
```

## Important Notes

### What CAN Be Extracted

- ✅ Natural proteins with known sequences in UniProt
- ✅ Modified proteins (if modification doesn't change core sequence)
  - Example: `biotinylated BSA` → extracts BSA sequence
- ✅ Protein mutants (extracts wild-type sequence)
  - Example: `F88W mutant` → extracts wild-type

### What CANNOT Be Extracted

- ❌ **Antibodies**: Specific antibody clones are not in UniProt
  - Example: `anti-cardiac troponin I antibody` ≠ `cardiac troponin I`
  - Antibodies and antigens are completely different proteins
- ❌ **Aptamers**: These are nucleic acids, not proteins
- ❌ **Small molecules**: Hormones, toxins, drugs are not proteins
- ❌ **Buffer solutions**: PBS, Tween-20, etc.

### Non-standard Amino Acids

The tool automatically handles non-standard amino acids:
- U (Selenocysteine) → C (Cysteine)
- B (Asx) → N (Asparagine)
- Z (Glx) → Q (Glutamine)
- X (Unknown) → A (Alanine)
- J (Xle) → L (Leucine)
- O (Pyrrolysine) → K (Lysine)

### ESM-2 Model Details

- **Model**: `facebook/esm2_t6_8M_UR50D` (smallest ESM-2 variant)
- **Embedding dimension**: 320D
- **Context length**: Up to 1024 amino acids (longer sequences are truncated)
- **Pooling**: Mean pooling across sequence length
- **Size**: ~32MB model download on first run

## Example Workflow

```bash
# 1. Extract a single protein
python extract_protein.py "hemoglobin"

# 2. Batch extract all proteins from CSV
python batch_extract.py bio_protein_collection.csv

# 3. Check summary
cat cache/_batch_summary.json

# 4. Retry failed proteins with alternative names
python retry_failed.py

# 5. Use the JSON data for downstream analysis
python your_analysis_script.py
```

## Troubleshooting

### "Protein not found" Error

- Check protein name spelling
- Try alternative names (e.g., `PSA` vs `prostate-specific antigen`)
- Verify it's actually a protein (not antibody, aptamer, or chemical)

### Import Error for `transformers`

```bash
pip install transformers torch
```

### Memory Issues with ESM-2

The t6_8M model is lightweight (~32MB), but if you experience issues:
- Close other applications
- Process proteins one at a time instead of batch
- Consider using a machine with more RAM

### Rate Limiting from UniProt

The batch script includes 1-second delays between requests. If you still get rate-limited:
- Increase the delay in `batch_extract.py` (line with `time.sleep(1)`)
- Process in smaller batches

## Performance

- **Single protein**: ~3-5 seconds (including ESM-2 embedding)
- **Batch (223 proteins)**: ~15-20 minutes
- **Cache hit (already extracted)**: Instant (skips re-extraction)

## Citation

If you use this tool in your research, please cite:

- **UniProt**: The UniProt Consortium (2023) Nucleic Acids Res. 51:D523-D531
- **ESM-2**: Lin et al. (2022) "Language models of protein sequences at the scale of evolution enable accurate structure prediction" bioRxiv
- **BioPython**: Cock et al. (2009) Bioinformatics 25:1422-1423

## License

This tool is provided as-is for research and educational purposes.

## Contact

For questions or issues, please open an issue on the repository.

---

**Last Updated**: November 2025
