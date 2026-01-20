#!/usr/bin/env python3
"""
Single Protein Information Extraction Script
Input: Protein name (e.g., "glucose oxidase")
Output: JSON file containing properties + ESM-2 embedding vector
"""

import requests
import json
import os
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import Counter

# ============= 1. UniProt Query =============
def query_uniprot(protein_name):
    """Query UniProt to retrieve protein information"""
    print(f"🔍 Querying UniProt: {protein_name}")

    # Search URL
    search_url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"{protein_name} AND reviewed:true",
        "format": "json",
        "size": 1
    }

    response = requests.get(search_url, params=params)
    response.raise_for_status()
    data = response.json()

    if not data.get('results'):
        print("❌ Protein not found, trying without reviewed constraint...")
        params['query'] = protein_name
        response = requests.get(search_url, params=params)
        data = response.json()

        if not data.get('results'):
            raise ValueError(f"Protein not found: {protein_name}")

    result = data['results'][0]

    # Extract basic information
    uniprot_id = result['primaryAccession']
    sequence = result['sequence']['value']
    organism = result['organism']['scientificName']

    # Extract protein name
    protein_desc = result.get('proteinDescription', {})
    if 'recommendedName' in protein_desc:
        full_name = protein_desc['recommendedName']['fullName']['value']
    elif 'submittedName' in protein_desc:
        full_name = protein_desc['submittedName'][0]['fullName']['value']
    else:
        full_name = protein_name

    print(f"✓ Found: {uniprot_id} - {full_name} ({organism})")
    print(f"  Sequence length: {len(sequence)} amino acids")

    return {
        'uniprot_id': uniprot_id,
        'protein_name': full_name,
        'organism': organism,
        'sequence': sequence
    }


# ============= 2. Macroscopic Properties Calculation =============
def calculate_macroscopic_properties(sequence):
    """Calculate ~20 macroscopic properties from protein sequence"""
    print(f"🧮 Calculating macroscopic properties...")

    # Handle non-standard amino acids (not supported by BioPython)
    # U=Selenocysteine→C, B=Asx→N, Z=Glx→Q, X=Unknown→A, J=Xle→L, O=Pyrrolysine→K
    non_standard_map = {'U': 'C', 'B': 'N', 'Z': 'Q', 'X': 'A', 'J': 'L', 'O': 'K'}
    original_length = len(sequence)
    cleaned_sequence = ''.join([non_standard_map.get(aa, aa) for aa in sequence])

    if cleaned_sequence != sequence:
        replaced = sum(1 for a, b in zip(sequence, cleaned_sequence) if a != b)
        print(f"  ⚠ Replaced {replaced} non-standard amino acids")

    analyzed_seq = ProteinAnalysis(cleaned_sequence)

    # Amino acid composition
    aa_comp = analyzed_seq.amino_acids_percent

    # Calculate amino acid category frequencies
    aromatic = sum([aa_comp.get(aa, 0) for aa in ['F', 'W', 'Y']])  # Aromatic
    aliphatic = sum([aa_comp.get(aa, 0) for aa in ['A', 'V', 'L', 'I']])  # Aliphatic
    polar = sum([aa_comp.get(aa, 0) for aa in ['S', 'T', 'N', 'Q']])  # Polar
    charged = sum([aa_comp.get(aa, 0) for aa in ['K', 'R', 'D', 'E']])  # Charged
    positive = sum([aa_comp.get(aa, 0) for aa in ['K', 'R', 'H']])  # Positive
    negative = sum([aa_comp.get(aa, 0) for aa in ['D', 'E']])  # Negative

    # Secondary structure propensity
    sec_struct = analyzed_seq.secondary_structure_fraction()

    # Molar extinction coefficient (280 nm, M-1 cm-1)
    ext_coeff = analyzed_seq.molar_extinction_coefficient()

    # Average flexibility
    flexibility = analyzed_seq.flexibility()
    avg_flexibility = sum(flexibility) / len(flexibility) if flexibility else 0.0

    # Additional amino acid categories
    tiny = sum([aa_comp.get(aa, 0) for aa in ['A', 'C', 'G', 'S', 'T']])  # Tiny residues
    large = sum([aa_comp.get(aa, 0) for aa in ['F', 'I', 'K', 'L', 'M', 'R', 'W', 'Y']])  # Large residues

    features = {
        # Basic properties
        'sequence_length': len(sequence),
        'molecular_weight_kDa': round(analyzed_seq.molecular_weight() / 1000, 2),
        'isoelectric_point': round(analyzed_seq.isoelectric_point(), 2),
        'aromaticity': round(analyzed_seq.aromaticity(), 4),
        'instability_index': round(analyzed_seq.instability_index(), 2),
        'gravy': round(analyzed_seq.gravy(), 4),  # Hydropathy index

        # pH-related properties
        'charge_at_pH7': round(analyzed_seq.charge_at_pH(7.0), 2),
        'charge_at_pH5': round(analyzed_seq.charge_at_pH(5.0), 2),

        # Amino acid composition statistics
        'aromatic_fraction': round(aromatic, 4),
        'aliphatic_fraction': round(aliphatic, 4),
        'polar_fraction': round(polar, 4),
        'charged_fraction': round(charged, 4),
        'positive_fraction': round(positive, 4),
        'negative_fraction': round(negative, 4),

        # Secondary structure propensity
        'helix_fraction': round(sec_struct[0], 4),
        'turn_fraction': round(sec_struct[1], 4),
        'sheet_fraction': round(sec_struct[2], 4),

        # Special amino acids
        'cysteine_count': sequence.count('C'),
        'proline_count': sequence.count('P'),
        'glycine_fraction': round(aa_comp.get('G', 0), 4),

        # Additional properties (5 new features to reach 25 total)
        'extinction_coefficient_reduced': ext_coeff[0],  # All Cys reduced
        'extinction_coefficient_oxidized': ext_coeff[1],  # All Cys oxidized (disulfide bonds)
        'average_flexibility': round(avg_flexibility, 4),  # Vihinen flexibility scale
        'tiny_fraction': round(tiny, 4),  # Small residues (A,C,G,S,T)
        'large_fraction': round(large, 4),  # Large residues (F,I,K,L,M,R,W,Y)
    }

    print(f"✓ Calculated {len(features)} features")
    return features


# ============= 3. ESM-2 Embedding Generation =============
def generate_esm2_embedding(sequence):
    """Generate sequence embedding vector using ESM-2"""
    print(f"🧬 Generating ESM-2 embedding vector...")

    try:
        from transformers import AutoTokenizer, EsmModel
        import torch

        # Load model (using smallest version for lower dimensionality)
        # Available models: t6_8M(320D) < t12_35M(480D) < t30_150M(640D) < t33_650M(1280D)
        model_name = "facebook/esm2_t6_8M_UR50D"
        print(f"  Loading model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EsmModel.from_pretrained(model_name)
        model.eval()

        # Encode sequence
        inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)

        # Generate embedding
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling to get fixed-dimension vector
            embeddings = outputs.last_hidden_state.mean(dim=1)

        embedding_vector = embeddings[0].cpu().numpy()

        print(f"✓ Generated embedding vector: {embedding_vector.shape[0]} dimensions")

        return {
            'embedding_dim': int(embedding_vector.shape[0]),
            'embedding_vector': embedding_vector.tolist(),
            'model_used': model_name
        }

    except ImportError:
        print("⚠ transformers not installed, skipping ESM-2 embedding")
        return {
            'embedding_dim': 0,
            'embedding_vector': [],
            'model_used': 'none',
            'note': 'transformers library not installed'
        }
    except Exception as e:
        print(f"⚠ ESM-2 generation failed: {e}")
        return {
            'embedding_dim': 0,
            'embedding_vector': [],
            'model_used': 'none',
            'error': str(e)
        }


# ============= 4. Main Function =============
def extract_protein_info(protein_name, cache_dir=None):
    """Extract protein information and save as JSON"""

    # Get the directory of the script itself
    script_dir = os.path.dirname(os.path.realpath(__file__))
    if cache_dir is None:
        cache_dir = os.path.join(script_dir, 'cache')

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    # Output file name
    output_file = os.path.join(cache_dir, f"{protein_name}.json")

    print(f"\n{'='*60}")
    print(f"Extracting protein information: {protein_name}")
    print(f"{'='*60}\n")

    # 1. Query UniProt
    uniprot_data = query_uniprot(protein_name)

    # 2. Calculate macroscopic properties
    macroscopic_properties = calculate_macroscopic_properties(uniprot_data['sequence'])

    # 3. Generate ESM-2 embedding
    embedding_data = generate_esm2_embedding(uniprot_data['sequence'])

    # 4. Merge all data
    result = {
        'query_name': protein_name,
        'uniprot_info': {
            'accession': uniprot_data['uniprot_id'],
            'protein_name': uniprot_data['protein_name'],
            'organism': uniprot_data['organism'],
            'sequence_length': len(uniprot_data['sequence']),
            'sequence': uniprot_data['sequence']
        },
        'macroscopic_properties': macroscopic_properties,
        'embedding': embedding_data
    }

    # 5. Save JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✓ Extraction complete!")
    print(f"📁 Saved to: {output_file}")
    print(f"{'='*60}\n")

    # Print summary
    print("📊 Data Summary:")
    print(f"  UniProt ID: {result['uniprot_info']['accession']}")
    print(f"  Protein Name: {result['uniprot_info']['protein_name']}")
    print(f"  Organism: {result['uniprot_info']['organism']}")
    print(f"  Sequence Length: {result['macroscopic_properties']['sequence_length']} aa")
    print(f"  Molecular Weight: {result['macroscopic_properties']['molecular_weight_kDa']} kDa")
    print(f"  Isoelectric Point: {result['macroscopic_properties']['isoelectric_point']}")
    print(f"  Embedding Dimension: {result['embedding']['embedding_dim']} D")
    print()

    return result


# ============= 5. Command Line Interface =============
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        protein_name = " ".join(sys.argv[1:])
    else:
        protein_name = "glucose oxidase"
        print(f"💡 Using default example: {protein_name}")
        print(f"   Usage: python extract_protein.py <protein_name>\n")

    try:
        extract_protein_info(protein_name)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
