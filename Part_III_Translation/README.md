# Part III: Translation - Virtual Screening and Validation

This module implements the **Translation** phase of the T3 pipeline, enabling virtual screening and DFT-based validation.

## Directory Structure

```
Part_III_Translation/
├── gnn_inference/                    # DTE-GNN inference pipeline
│   ├── inference_utils.py            # Core inference engine
│   ├── train_hetero_gnn_residual_sgnn.py  # Model architecture & training
│   ├── original_graph_builder.py     # JSON to HeteroData conversion
│   ├── screening_adapter.py          # Template + candidate to graph
│   ├── virtual_screening.py          # Batch screening pipeline
│   ├── model_saver.py                # Checkpoint utilities
│   └── demo_inference.py             # Usage demonstration
│
└── dft_validation/                   # DFT structure generation
    └── generate_inclusion_configs.py # Host-guest configuration generator
```

## GNN Virtual Screening

Train model first using Part_II_Twin, then run inference:

```python
from inference_utils import InferenceEngine

engine = InferenceEngine.from_checkpoint('path/to/model.pt', device='cuda')
probs = engine.predict(graph)
```

```bash
python virtual_screening.py \
    --checkpoint path/to/model.pt \
    --template sensor_template.json \
    --molecules /path/to/molecules_jsonl \
    --output results.csv --top-k 1000
```

## DFT Validation

Generate host-guest configurations for DFT calculations:

```bash
python generate_inclusion_configs.py \
    --host beta_cd.vasp --guest PFOA.vasp \
    --output beta-CD_PFOA --box-size 35
```

Generates 26 configurations (18 vertical insertions + 8 surface lying) per host-guest pair.

## Requirements

```
torch>=1.12
torch_geometric>=2.0
ase>=3.22
```
