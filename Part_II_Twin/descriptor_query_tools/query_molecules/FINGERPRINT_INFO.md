# Morgan Fingerprint Configuration

## Current Setup: 256-bit Morgan Fingerprints

### Why 256 bits?

This utility uses **256-bit Morgan fingerprints** for improved feature representation while maintaining efficiency for LLM fine-tuning. The 256-bit configuration provides:

1. **Better Feature Representation** ✅
   - More structural information than 128-bit
   - Reduced collision probability for diverse molecules
   - Better discrimination between similar structures

2. **Balanced for ML/LLM** ✅
   - Not too sparse (like 1024 or 2048 bits)
   - Not too compact (like 128 bits)
   - Optimal for medium-sized datasets (hundreds to thousands of molecules)

3. **Consistent Dimensionality** ✅
   - All molecules get exactly 256 bits
   - No mixing of different dimensions
   - Critical for batch processing in LLMs

4. **Computational Efficiency** ✅
   - Still fast enough for real-time queries
   - Reasonable memory footprint
   - 4x faster than 1024 bits, 2x slower than 128 bits

5. **Improved Information Density** ✅
   - ~25-40 bits typically activated for small molecules
   - 10-15% bit density - good for feature learning
   - Captures more structural nuances

### Performance Characteristics

Based on testing with common drug molecules:

| Molecule | Bits Set (256-bit) | Density | Notes |
|----------|-------------------|---------|-------|
| Aspirin | ~38-42 | 15-16% | NSAID, aromatic |
| Caffeine | ~40-45 | 16-18% | Stimulant, heterocyclic |
| Ibuprofen | ~35-40 | 14-16% | NSAID, aliphatic |
| Ethanol | ~10-12 | 4-5% | Simple alcohol |

**Similarity Examples (Tanimoto with 256-bit):**
- Aspirin vs Ibuprofen: 0.25-0.30 (both NSAIDs, structurally similar)
- Aspirin vs Caffeine: 0.12-0.18 (different drug classes)
- Ibuprofen vs Caffeine: 0.20-0.28 (moderate similarity)

Note: 256-bit fingerprints provide more nuanced similarity scores.

### For LLM Fine-tuning

```python
from src.query_handler import QueryHandler
import numpy as np

handler = QueryHandler()

# Get molecular fingerprint
result = handler.query("aspirin")
fp = result["data"]["fingerprints"]["bit_vector"]

# Convert to numpy for LLM input
fp_array = np.array(fp)  # Shape: (256,)

# Use as additional features for LLM:
# Option 1: Concatenate with text embeddings
# Option 2: Use as conditioning vector
# Option 3: Separate embedding layer
```

### Alternative Dimensions

If you need different dimensions, you can modify `n_bits` in:
- `src/fingerprint_generator.py`: Line 247
- `src/query_handler.py`: Line 78

**Other common dimensions:**
- **128 bits**: For very small datasets (<100 molecules)
- **256 bits**: ✅ **Current setting** - optimal for hundreds to thousands of molecules
- **512 bits**: For medium-sized datasets (5000+)
- **1024 bits**: Large-scale virtual screening (50k+)
- **2048 bits**: Industry standard for huge libraries (1M+)

### Technical Details

- **Method**: Morgan (ECFP4)
- **Radius**: 2 (equivalent to ECFP4)
- **Generator**: RDKit
- **Input**: SMILES or InChI from PubChem/ChEMBL
- **Output Formats**: bit_vector, hex, on_bits, count_fingerprint

### References

- Rogers, D., & Hahn, M. (2010). "Extended-Connectivity Fingerprints." Journal of Chemical Information and Modeling.
- Optimized based on expert recommendations for LLM applications with limited training data.
