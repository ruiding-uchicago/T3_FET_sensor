"""
RDKit fallback for computing polymer descriptors when PubChem/ChEMBL fails
"""
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors


def compute_rdkit_descriptors(smiles: str):
    """
    Compute descriptors using local RDKit when PubChem/ChEMBL fails

    Args:
        smiles: SMILES string (without * markers)

    Returns:
        dict: Computed descriptors, None for unavailable values
    """
    if not smiles or '*' in smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    descriptors = {}

    try:
        # Electronic (partial support)
        descriptors["Charge"] = Chem.GetFormalCharge(mol)
        descriptors["aromatic_rings"] = Descriptors.NumAromaticRings(mol)
        descriptors["FeatureRingCount3D"] = None  # Requires 3D
        descriptors["FeatureCationCount3D"] = None  # Requires 3D
        descriptors["FeatureAnionCount3D"] = None  # Requires 3D

        # Polarity & surface
        descriptors["TPSA"] = Descriptors.TPSA(mol)
        descriptors["XLogP"] = Descriptors.MolLogP(mol)  # Wildman-Crippen LogP
        descriptors["HBondDonorCount"] = Descriptors.NumHDonors(mol)
        descriptors["HBondAcceptorCount"] = Descriptors.NumHAcceptors(mol)
        descriptors["FeatureDonorCount3D"] = None  # Requires 3D
        descriptors["FeatureAcceptorCount3D"] = None  # Requires 3D

        # Size & shape
        descriptors["MolecularWeight"] = Descriptors.MolWt(mol)
        descriptors["HeavyAtomCount"] = Descriptors.HeavyAtomCount(mol)
        descriptors["Volume3D"] = None  # Requires 3D
        descriptors["XStericQuadrupole3D"] = None  # Requires 3D
        descriptors["YStericQuadrupole3D"] = None  # Requires 3D
        descriptors["ZStericQuadrupole3D"] = None  # Requires 3D

        # Flexibility
        descriptors["RotatableBondCount"] = Descriptors.NumRotatableBonds(mol)
        descriptors["EffectiveRotorCount3D"] = None  # Requires 3D
        descriptors["ConformerModelRMSD3D"] = None  # Requires 3D

        # Complexity
        descriptors["Complexity"] = None  # PubChem proprietary algorithm
        descriptors["FeatureHydrophobeCount3D"] = None  # Requires 3D

        # Pharmacophore
        descriptors["FeatureCount3D"] = None  # Requires 3D
        descriptors["qed_weighted"] = None  # ChEMBL-specific
        descriptors["np_likeness_score"] = None  # ChEMBL-specific

    except Exception as e:
        print(f"  ⚠️  RDKit computation failed: {e}")
        return None

    return descriptors


def compute_molar_mass(smiles: str):
    """
    Compute molar mass from SMILES

    Args:
        smiles: SMILES string (without * markers)

    Returns:
        float: Molar mass in g/mol, or None if failed
    """
    if not smiles or '*' in smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        return Descriptors.MolWt(mol)
    except:
        return None
