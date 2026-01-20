# Part III: Translation - Virtual Screening

GNN-based virtual screening and DFT validation.

## Structure

```
Part_III_Translation/
├── gnn_inference/                    # DTE-GNN inference
│   ├── inference_utils.py            # Core inference engine
│   ├── train_hetero_gnn_residual_sgnn.py
│   ├── original_graph_builder.py     # JSON to graph conversion
│   ├── screening_adapter.py          # Template + candidate adapter
│   ├── virtual_screening.py          # Batch screening pipeline
│   └── demo_inference.py
│
└── dft_validation/
    └── generate_inclusion_configs.py # Host-guest config generation
```

## Usage

```bash
# Virtual screening
python virtual_screening.py \
    --checkpoint model.pt \
    --template sensor.json \
    --molecules /path/to/molecules \
    --output results.csv

# DFT config generation
python generate_inclusion_configs.py \
    --host beta_cd.vasp --guest PFOA.vasp \
    --output configs/
```
