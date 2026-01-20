"""
Monomer descriptor extractor using query_molecules repo
"""
import sys
import os

# Add query_molecules to path
query_molecules_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../query_molecules"))
if os.path.exists(query_molecules_path):
    sys.path.insert(0, query_molecules_path)

from src.query_handler import QueryHandler

# KEY_25 descriptors from query_molecules/KEY_25_DESCRIPTORS.md
KEY_DESCRIPTORS_25 = {
    # Electronic (5)
    "Charge": "pubchem",
    "aromatic_rings": "chembl",
    "FeatureRingCount3D": "pubchem",
    "FeatureCationCount3D": "pubchem",
    "FeatureAnionCount3D": "pubchem",

    # Polarity & surface (6)
    "TPSA": "pubchem",
    "XLogP": "pubchem",
    "HBondDonorCount": "pubchem",
    "HBondAcceptorCount": "pubchem",
    "FeatureDonorCount3D": "pubchem",
    "FeatureAcceptorCount3D": "pubchem",

    # Size & shape (6)
    "MolecularWeight": "pubchem",
    "HeavyAtomCount": "pubchem",
    "Volume3D": "pubchem",
    "XStericQuadrupole3D": "pubchem",
    "YStericQuadrupole3D": "pubchem",
    "ZStericQuadrupole3D": "pubchem",

    # Flexibility (3)
    "RotatableBondCount": "pubchem",
    "EffectiveRotorCount3D": "pubchem",
    "ConformerModelRMSD3D": "pubchem",

    # Complexity (2)
    "Complexity": "pubchem",
    "FeatureHydrophobeCount3D": "pubchem",

    # Pharmacophore (3)
    "FeatureCount3D": "pubchem",
    "qed_weighted": "chembl",
    "np_likeness_score": "chembl",
}


def get_monomer_descriptors(smiles: str, cache_dir=None):
    """
    Get KEY_25 descriptors for a monomer from query_molecules

    Args:
        smiles: Monomer SMILES string (may contain polymer attachment points *)
        cache_dir: query_molecules cache directory (default: auto-detect)

    Returns:
        dict: 25 key descriptors, None for missing values
    """
    if cache_dir is None:
        # Auto-detect query_molecules cache
        cache_dir = os.path.join(query_molecules_path, "cache")

    # Check for polymer attachment point markers (*)
    if '*' in smiles:
        print(f"  ⚠️  SMILES contains polymer markers (*): {smiles}")
        print(f"  ⏭️  Skipping query, returning null values")
        print(f"      (5.7% of polymers have this - can be manually handled later)")
        return {k: None for k in KEY_DESCRIPTORS_25.keys()}

    # Initialize query handler (will use cache and API)
    handler = QueryHandler(cache_dir=cache_dir, use_cache=True, generate_fingerprints=False)

    # Query monomer data by SMILES (checks cache first, then API)
    print(f"  🔍 Querying monomer: {smiles}")
    result = handler.query(smiles, query_type="smiles")

    if not result:
        print(f"  ⚠️  Monomer {smiles} not found, returning null values")
        return {k: None for k in KEY_DESCRIPTORS_25.keys()}

    # Extract KEY_25 descriptors
    descriptors = {}
    for key, source in KEY_DESCRIPTORS_25.items():
        if source == "pubchem":
            value = result["data"]["pubchem"]["properties"].get(key, None)
            # Convert to native Python types
            if value is not None:
                try:
                    if isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                        value = float(value) if '.' in value else int(value)
                except:
                    pass
            descriptors[key] = value
        elif source == "chembl":
            # ChEMBL may return multiple molecules, take first
            chembl_data = result["data"].get("chembl")
            if chembl_data and chembl_data.get("molecules"):
                molecules = chembl_data.get("molecules", [])
                value = molecules[0]["properties"].get(key, None)
                # Convert to native Python types
                if value is not None and isinstance(value, str):
                    try:
                        if value.replace('.', '').replace('-', '').isdigit():
                            value = float(value) if '.' in value else int(value)
                    except:
                        pass
                descriptors[key] = value
            else:
                descriptors[key] = None

    print(f"  ✓ Extracted {sum(1 for v in descriptors.values() if v is not None)}/25 descriptors")
    return descriptors
