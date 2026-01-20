# Part II: Twin - Digital Twin Construction

Heterogeneous graph construction for FET sensors through three stages.

## Structure

```
Part_II_Twin/
├── descriptor_query_tools/     # Material descriptor extraction
│   ├── query_molecules/        # PubChem/ChEMBL molecular descriptors
│   ├── query_polymers/         # Polymer repeat unit descriptors
│   ├── query_inorganic/        # Materials Project properties
│   ├── query_biomolecules/     # ESM-2 protein embeddings
│   └── query_DNA_RNA/          # DNABERT-S nucleic acid embeddings
│
├── data_augmentation/          # Physics-aware augmentation
│   ├── offline_augmentation_v3_physics.py
│   ├── build_graph_dataset_augmented.py
│   └── visualize_graph_topology.py
│
├── gnn_training/               # DTE-GNN model
│   ├── train_hetero_gnn_residual_sgnn.py
│   ├── baseline_tabular.py
│   └── baseline_tabular_extended.py
│
└── sample_data/                # Anonymized samples
    ├── original/               # 5 original records
    └── augmented_physics/      # 18 augmented variants
```

## Descriptor Format

Each material uses 25D macroproperties + 320D fingerprint vectors.
