#!/usr/bin/env python3
"""
Process DNA/RNA sequences from CSV using DNABERT-S embeddings
"""

import pandas as pd
from dna_feature_utils import DNABERTSFeatureExtractor
from tqdm import tqdm

def main():
    print("Processing DNA/RNA sequences with DNABERT-S...")
    print("=" * 60)
    print("DNABERT-S: Species-aware DNA embeddings with validated 256D compression")
    print("=" * 60)

    # Load CSV
    input_file = 'DNA_collection.csv'
    df = pd.read_csv(input_file, header=None, names=['sequence_raw'])
    print(f"\nLoaded {len(df)} sequences from {input_file}\n")

    # Initialize extractor (this will load DNABERT-S model)
    extractor = DNABERTSFeatureExtractor(cache_dir='cache')

    # Process each sequence
    all_features = []
    all_embeddings = []

    print("\nProcessing sequences with DNABERT-S embeddings...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        seq_str = row['sequence_raw']

        try:
            result = extractor.compute_and_cache(seq_str)
            macroscopic_props = result['macroscopic_properties'].copy()
            macroscopic_props['id'] = idx + 1
            macroscopic_props['original_input'] = seq_str
            macroscopic_props['sequence'] = result['sequence']

            all_features.append(macroscopic_props)
            all_embeddings.append(result['embedding_256d'])

        except Exception as e:
            print(f"\nWarning: Failed to process sequence {idx+1}: {e}")
            continue

    print(f"\nSuccessfully processed {len(all_features)} sequences")

    # Create summary CSV with features
    print("\nCreating summary CSV...")
    features_df = pd.DataFrame(all_features)

    # Add embeddings
    embedding_cols = [f'embedding_{i:03d}' for i in range(256)]
    embedding_df = pd.DataFrame(all_embeddings, columns=embedding_cols)

    # Combine
    result_df = pd.concat([features_df, embedding_df], axis=1)

    # Reorder columns
    cols = ['id', 'original_input', 'sequence', 'is_rna'] + \
           [col for col in result_df.columns if col not in ['id', 'original_input', 'sequence', 'is_rna']]
    result_df = result_df[cols]

    # Save full results
    output_full = 'dna_features_embeddings.csv'
    result_df.to_csv(output_full, index=False)
    print(f"Saved: {output_full}")

    # Save features only
    features_only_df = result_df[[col for col in result_df.columns if not col.startswith('embedding_')]]
    output_features = 'dna_features_only.csv'
    features_only_df.to_csv(output_features, index=False)
    print(f"Saved: {output_features}")

    # Print statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total sequences: {len(result_df)}")
    rna_count = int(result_df['is_rna'].sum())
    dna_count = len(result_df) - rna_count
    print(f"DNA sequences: {dna_count}")
    print(f"RNA sequences: {rna_count}")
    print(f"\nMacroscopic properties: 25 numerical features (is_rna: 0=DNA, 1=RNA)")
    print(f"Embedding: 256D DNABERT-S (species-aware, nearly lossless compression)")
    print(f"\nJSON cache directory: cache/")
    print(f"CSV outputs: {output_features}, {output_full}")
    print("=" * 60)

    # Show sample statistics
    print("\nKey feature statistics:")
    stats_cols = ['length', 'gc_content', 'tm_celsius', 'delta_g_kcal_mol', 'complexity']
    print(result_df[stats_cols].describe().round(3))

if __name__ == '__main__':
    main()
