# Part II: Twin - Digital Twin Construction for FET Sensors

This module implements the **Twin** phase of the T3 (Text-Twin-Translation) pipeline, constructing digital twin representations of FET sensors through heterogeneous graph neural networks.

## Overview

The Twin phase transforms unstructured sensor descriptions into machine-learning-ready graph representations through three stages:

```
Stage 1: Material Descriptor Injection
    JSON records → Cross-domain material fingerprints (25D macro + 320D fingerprint)

Stage 2: Physics-aware Data Augmentation
    Original samples → Augmented dataset with physical constraints

Stage 3: GNN Training
    Heterogeneous graphs → DTE-GNN predictions (LDL/UDL/Sensitivity)
```

## Directory Structure

```
Part_II_Twin/
├── descriptor_query_tools/     # Material descriptor extraction tools
│   ├── query_molecules/        # PubChem/ChEMBL molecular descriptors
│   ├── query_polymers/         # Polymer repeat unit descriptors
│   ├── query_inorganic/        # Materials Project inorganic properties
│   ├── query_biomolecules/     # ESM-2 protein embeddings
│   └── query_DNA_RNA/          # DNABERT-S nucleic acid embeddings
│
├── sample_data/                # Representative anonymized samples
│   ├── original/               # 5 original sensor records
│   └── augmented_physics/      # 18 physics-augmented variants
│
├── data_augmentation/          # Augmentation and graph construction
│   ├── offline_augmentation_v3_physics.py  # Physics-aware augmentation
│   ├── build_graph_dataset_augmented.py    # Heterogeneous graph builder
│   ├── visualize_graph_topology.py         # Graph visualization
│   └── anonymize_jsons.py                  # Data anonymization utility
│
└── gnn_training/               # DTE-GNN model and training
    ├── train_hetero_gnn_residual_sgnn.py   # Main DTE-GNN training script
    ├── baseline_tabular.py                  # ML baseline models (RF, XGB, SVM)
    ├── baseline_tabular_extended.py         # Extended baselines (GBDT, LDA)
    └── plot_ablation_bar.py                 # Ablation visualization
```

## Stage 1: Material Descriptor Injection

### Descriptor Sources by Material Category

| Category | Macroproperties (25D) | Fingerprint (320D) | Database |
|----------|----------------------|-------------------|----------|
| Molecules | TPSA, XLogP, H-bond counts, etc. | Morgan fingerprint (256-bit) | PubChem, ChEMBL |
| Polymers | Thermal, mechanical, electrical | Morgan fingerprint of monomer | PubChem |
| Inorganic | Band gap, formation energy, etc. | MAGPIE composition embedding | Materials Project |
| Biomolecules | pI, instability index, GRAVY | ESM-2 embedding (650M) | UniProt |
| DNA/RNA | GC content, Tm, thermodynamics | DNABERT-S embedding | DNABERT-S |

### Usage

```bash
# Query molecular descriptors
cd descriptor_query_tools/query_molecules
python main.py --query "dopamine" --output cache/

# Query protein embeddings
cd descriptor_query_tools/query_biomolecules
python extract_protein.py --sequence "MVLSPADKTN..." --output protein.json
```

## Stage 2: Physics-aware Data Augmentation

### Augmentation Strategies

**Discrete Augmentations:**
- Source/Drain swap (FET physical symmetry)
- Dual-gate/Floating-gate flip
- Inert gas substitution (N2 ↔ Ar/He/Ne)

**Continuous Perturbations:**
| Parameter | Perturbation | Physical Basis |
|-----------|-------------|----------------|
| pH | ±0.6-0.8 | Buffer tolerance |
| Dielectric thickness | ×0.85-1.15 | Deposition variation |
| Annealing temperature | ×0.92-1.08 | Furnace calibration |

**Physical Constraints:**
- pH clipped to [0, 14]
- Temperatures remain positive
- Class labels preserved (wide bins span orders of magnitude)

### Heterogeneous Graph Structure

**16 Node Types:**
- Device: channel, gate_top/bottom, dielectric_top/bottom, floating_gate, source, drain, substrate
- Sensing: surface_functionalization, probe_material, detect_target, test_medium, electrolyte
- Process: annealing, condition

**6 Edge Types:**
- Electrical (carrier transport)
- Capacitive (gate-channel coupling)
- Chemical (sensing chain)
- Process (thermal effects)
- Condition (operating parameters)
- Environment (electrolyte-medium)

## Stage 3: GNN Training

### DTE-GNN Architecture

```
DTE-GNN = GNN_branch(macro_graph) + gate × FP_branch(fingerprints)
```

**GNN Branch:**
- HeteroConv with GCNII residuals (α=0.15)
- Jump-knowledge connections
- Attention-based readout

**Fingerprint Branch (3 options):**
- MLP: Simple fully-connected
- Transformer: Self-attention encoder
- SNN: Spiking neural network

**Two-Stage Training:**
1. Pre-train GNN (E_gnn=40 epochs)
2. Freeze GNN, train FP branch (E_fp=20 epochs)

### Training

```bash
cd gnn_training

# Train DTE-GNN
python train_hetero_gnn_residual_sgnn.py \
    --task lower_detection_limit \
    --fp-branch transformer \
    --n-folds 5

# Run baselines
python baseline_tabular.py --task lower_detection_limit
```

### Prediction Tasks

| Task | Class 0 | Class 1 | Class 2 |
|------|---------|---------|---------|
| LDL (Lower Detection Limit) | Good (lowest) | Medium | Poor (highest) |
| UDL (Upper Detection Limit) | Poor (lowest) | Medium | Good (highest) |
| Sensitivity | Poor (lowest) | Medium | Good (highest) |

## Sample Data

The `sample_data/` directory contains 5 anonymized representative samples covering:
- **Bio sensors**: Protein target (influenza hemagglutinin), small molecule (dopamine)
- **Gas sensors**: Ethanol, nitric oxide
- **Liquid sensors**: Hydrogen sulfide

Each sample includes:
- 25D macroproperties for all material fields
- 320D fingerprint vectors
- Complete graph-ready structure

## Requirements

```
torch>=1.12
torch_geometric>=2.0
rdkit>=2022.03
transformers>=4.20  # For ESM-2, DNABERT-S
scikit-learn>=1.0
xgboost>=1.6
matplotlib>=3.5
networkx>=2.8
```

## Citation

If you use this code, please cite:

```bibtex
@article{t3_fet_sensor,
  title={Text-Twin-Translation: An Agentic Framework for FET Sensor Design},
  author={...},
  journal={...},
  year={2025}
}
```

## License

This code is released for research purposes. The original 1600-paper dataset is not included due to copyright restrictions; only anonymized samples are provided.
