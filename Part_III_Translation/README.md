# Part III: Translation - Virtual Screening

GNN-based virtual screening and DFT validation.

## Structure

```
Part_III_Translation/
├── gnn_inference/                    # DTE-GNN inference
│   ├── inference_utils.py            # Core inference engine
│   ├── comprehensive_screening.py    # Multi-task scoring (LDL×UDL×Sens)
│   ├── virtual_screening.py          # Single-task batch screening
│   ├── screening_adapter.py          # Template + candidate adapter
│   ├── original_graph_builder.py     # JSON to graph conversion
│   └── demo_inference.py
│
└── dft_validation/
    ├── generate_inclusion_configs.py # Basic configs (insert + lying)
    └── generate_all_configs.py       # Full configs (+ side-binding)
```

## Scoring Functions

**Single-target score:**
```
S_target = P(LDL=0) × P(UDL=2) × P(Sensitivity=2)
```

**Selectivity score (multi-target):**
```
S_selectivity = (S_PFOS × S_PFOA) / (S_DDS × S_TCAA)
```

## Usage

```bash
# Comprehensive screening (LDL + UDL + Sensitivity combined)
python comprehensive_screening.py \
    --molecules /path/to/molecules \
    --output results.csv

# DFT config generation (32 configs: insert + lying + side-binding)
python generate_all_configs.py \
    --host probe.pdb --guest target.vasp \
    --output configs/ --ring-atoms C1,C2,N1
```
