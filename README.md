# T3: Text-Twin-Translation for FET Sensor Design

Reference implementation for the T3 agentic framework.

## Overview

**FET Sensor Digital Twin:** Structured representation of 28 fields across 5 categories extracted from scientific literature.

<p align="center">
  <img src="figures/schematic.png" width="600">
</p>

**LLM-based Information Extraction:** TextGrad optimization pipeline with autonomous prompt refinement.

<p align="center">
  <img src="figures/LLM_pipeline_small.png" width="700">
</p>

## Structure

```
T3_FET_sensor/
├── Part_I_Text/           # TextGrad prompt optimization
│   ├── textgrad_train.py
│   ├── textgrad_evaluate.py
│   └── prompts/
│
├── Part_II_Twin/          # Digital twin construction
│   ├── descriptor_query_tools/   # Material descriptor extraction
│   ├── data_augmentation/        # Physics-aware augmentation
│   ├── gnn_training/             # DTE-GNN model
│   └── sample_data/              # Anonymized samples
│
└── Part_III_Translation/  # Virtual screening & validation
    ├── gnn_inference/            # Inference pipeline
    └── dft_validation/           # DFT config generation
```

## Requirements

- Python 3.9+
- PyTorch, PyTorch Geometric
- RDKit, Transformers, ASE
