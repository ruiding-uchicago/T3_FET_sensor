# Feature Definitions

## Macroscopic Properties (25 numerical features)

All features are numerical values. Sequence type is binary encoded (is_rna: 0=DNA, 1=RNA).

### Basic Composition (10 features)

**length**
- Sequence length in nucleotides
- Unit: nt (nucleotides)

**is_rna**
- Binary encoding for sequence type (0 if DNA, 1 if RNA)
- Unit: binary (0 or 1)

**gc_content**
- Fraction of G and C nucleotides in the sequence
- Unit: dimensionless (0-1)

**at_content**
- Fraction of A and T/U nucleotides in the sequence
- Unit: dimensionless (0-1)

**a_count**
- Total number of adenine (A) bases
- Unit: count

**c_count**
- Total number of cytosine (C) bases
- Unit: count

**g_count**
- Total number of guanine (G) bases
- Unit: count

**t_count**
- Total number of thymine (T) bases in DNA
- Unit: count

**u_count**
- Total number of uracil (U) bases in RNA
- Unit: count

**purine_pyrimidine_ratio**
- Ratio of purines (A, G) to pyrimidines (C, T/U)
- Unit: dimensionless ratio

### Thermodynamic Properties (2 features)

**tm_celsius**
- Melting temperature calculated using Wallace's rule: for sequences <14nt, Tm = 2(A+T) + 4(G+C); for longer sequences, Tm = 64.9 + 41×(GC% - 16.4)
- Unit: °C (degrees Celsius)

**delta_g_kcal_mol**
- Gibbs free energy change estimated by nearest-neighbor method with simplified thermodynamic parameters
- Unit: kcal/mol

### Structural Properties (5 features)

**complexity**
- Shannon entropy normalized to [0,1] measuring sequence diversity: H = -Σ(p_i × log2(p_i)) / log2(4)
- Unit: dimensionless (0-1)

**longest_homopolymer**
- Length of the longest run of consecutive identical nucleotides
- Unit: nt (nucleotides)

**cpg_count**
- Number of CpG dinucleotides, important for gene regulation and methylation
- Unit: count

**gc_skew**
- Strand bias calculated as (G-C)/(G+C), used to predict replication origins
- Unit: dimensionless (-1 to 1)

**at_skew**
- Strand bias calculated as (A-T-U)/(A+T+U), measuring asymmetry between strands
- Unit: dimensionless (-1 to 1)

### Dinucleotide Frequencies (8 features)

**dinuc_CG**
- Frequency of CG dinucleotide in the sequence
- Unit: dimensionless (0-1)

**dinuc_GC**
- Frequency of GC dinucleotide in the sequence
- Unit: dimensionless (0-1)

**dinuc_AT**
- Frequency of AT dinucleotide in the sequence
- Unit: dimensionless (0-1)

**dinuc_TA**
- Frequency of TA dinucleotide in the sequence
- Unit: dimensionless (0-1)

**dinuc_GG**
- Frequency of GG dinucleotide in the sequence
- Unit: dimensionless (0-1)

**dinuc_CC**
- Frequency of CC dinucleotide in the sequence
- Unit: dimensionless (0-1)

**dinuc_AA**
- Frequency of AA dinucleotide in the sequence
- Unit: dimensionless (0-1)

**dinuc_TT**
- Frequency of TT dinucleotide in the sequence
- Unit: dimensionless (0-1)

---

## 256-Dimensional Embedding

### DNABERT-S Embeddings (Primary Method - Species-Aware)

The 256D embedding is generated using **DNABERT-S**, a species-aware pretrained transformer model:

1. **Model**: DNABERT-S (based on DNABERT-2-117M, ISMB 2025)
2. **Process**: Input sequence → Tokenization → BERT transformer → 768D hidden states → Mean pooling → Dimensionality reduction to 256D
3. **Reduction**: The 768D DNABERT-S output is reduced to 256D using average pooling (pooling 3 consecutive values: 768/256 = 3)
4. **Advantages**:
   - Species-aware embeddings that naturally cluster by species
   - Officially validated for 256D compression (nearly lossless performance)
   - Paper states: "maintains nearly the same performance level even when reduced to 256 dimensions"
   - Captures deep contextual patterns and sequence semantics with minimal information loss
5. **Validation**: DNABERT-S paper (ISMB 2025) specifically tested and validated average pooling dimension reduction, confirming that 256D achieves nearly lossless compression

### Alternative: DNABERT-2 Embeddings (General DNA Model)

For general DNA sequence tasks, DNABERT-2 is available (`dna_feature_utils_dnabert2.py`):

- Same 768D → 128D average pooling approach
- General-purpose DNA model without species-specific optimization
- Use when species differentiation is not important

### Alternative: K-mer Based Embeddings (Lightweight)

For scenarios without deep learning requirements, a lightweight k-mer based embedding is also available (`dna_feature_utils.py`):

First, 11 core numerical features are extracted (length, gc_content, at_content, purine_pyrimidine_ratio, tm_celsius, delta_g_kcal_mol, complexity, longest_homopolymer, cpg_count, gc_skew, at_skew). Next, trinucleotide (3-mer) frequencies are calculated across all 64 possible combinations (AAA, AAT, AAC, ..., TTT), capturing local sequence patterns and motifs. These 11 basic features are combined with up to 117 trinucleotide frequency values to form a 128-dimensional vector (11 + 117 = 128).

**Current Implementation**: All 198 sequences have been processed with DNABERT-S 256D embeddings (species-aware, nearly lossless compression). Features are fully numerical with binary encoding for sequence type (is_rna: 0=DNA, 1=RNA).
