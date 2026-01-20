"""
Molecular fingerprint generation using RDKit and PubChem fingerprints
"""
from typing import Dict, Any, Optional, List
import base64

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not installed. Local fingerprint generation will be disabled.")


class FingerprintGenerator:
    """
    Generates molecular fingerprints from SMILES, InChI, or mol objects.
    """

    def __init__(self):
        """
        Initialize fingerprint generator.
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for fingerprint generation. Install with: pip install rdkit")

    def generate_from_smiles(self,
                            smiles: str,
                            radius: int = 2,
                            n_bits: int = 2048,
                            include_variants: bool = False) -> Optional[Dict[str, Any]]:
        """
        Generate Morgan fingerprints from SMILES string.

        Args:
            smiles: SMILES string
            radius: Morgan fingerprint radius (default 2, equivalent to ECFP4)
            n_bits: Number of bits in fingerprint (default 2048)
            include_variants: Include different radius variants (default False)

        Returns:
            Dictionary containing fingerprints in various formats
        """
        if not smiles:
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Warning: Could not parse SMILES: {smiles}")
                return None

            return self._generate_fingerprints(mol, radius, n_bits, include_variants)

        except Exception as e:
            print(f"Error generating fingerprint from SMILES: {e}")
            return None

    def generate_from_inchi(self,
                           inchi: str,
                           radius: int = 2,
                           n_bits: int = 2048,
                           include_variants: bool = False) -> Optional[Dict[str, Any]]:
        """
        Generate Morgan fingerprints from InChI string.

        Args:
            inchi: InChI string
            radius: Morgan fingerprint radius (default 2)
            n_bits: Number of bits in fingerprint (default 2048)
            include_variants: Include different radius variants (default False)

        Returns:
            Dictionary containing fingerprints in various formats
        """
        if not inchi:
            return None

        try:
            mol = Chem.MolFromInchi(inchi)
            if mol is None:
                print(f"Warning: Could not parse InChI: {inchi}")
                return None

            return self._generate_fingerprints(mol, radius, n_bits, include_variants)

        except Exception as e:
            print(f"Error generating fingerprint from InChI: {e}")
            return None

    def _generate_fingerprints(self,
                              mol: Any,
                              radius: int = 2,
                              n_bits: int = 2048,
                              include_variants: bool = False) -> Dict[str, Any]:
        """
        Generate Morgan fingerprints from RDKit mol object.

        Args:
            mol: RDKit mol object
            radius: Morgan fingerprint radius
            n_bits: Number of bits in fingerprint
            include_variants: Include different radius variants

        Returns:
            Dictionary containing fingerprints
        """
        result = {
            "method": "Morgan",
            "radius": radius,
            "n_bits": n_bits
        }

        # Generate Morgan fingerprint (bit vector)
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

        # Convert to different formats
        result["bit_vector"] = self._bitvect_to_list(morgan_fp)
        result["bit_string"] = morgan_fp.ToBitString()
        result["hex"] = self._bitvect_to_hex(morgan_fp)
        result["on_bits"] = list(morgan_fp.GetOnBits())
        result["num_on_bits"] = len(result["on_bits"])

        # Count fingerprint (for feature importance)
        morgan_count_fp = AllChem.GetMorganFingerprint(mol, radius)
        result["count_fingerprint"] = dict(morgan_count_fp.GetNonzeroElements())

        # Add variants with different radii if requested
        if include_variants:
            result["variants"] = {}
            for r in [1, 2, 3]:
                if r != radius:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, r, nBits=n_bits)
                    result["variants"][f"radius_{r}"] = {
                        "bit_vector": self._bitvect_to_list(fp),
                        "on_bits": list(fp.GetOnBits()),
                        "num_on_bits": len(list(fp.GetOnBits()))
                    }

        return result

    def _bitvect_to_list(self, bitvect) -> List[int]:
        """
        Convert RDKit bit vector to list of 0s and 1s.

        Args:
            bitvect: RDKit bit vector

        Returns:
            List of integers (0 or 1)
        """
        return [int(b) for b in bitvect.ToBitString()]

    def _bitvect_to_hex(self, bitvect) -> str:
        """
        Convert RDKit bit vector to hexadecimal string.

        Args:
            bitvect: RDKit bit vector

        Returns:
            Hexadecimal string representation
        """
        # Convert bit string to bytes, then to hex
        bit_string = bitvect.ToBitString()
        # Pad to multiple of 8
        padded = bit_string + '0' * (8 - len(bit_string) % 8)
        # Convert to hex
        hex_str = hex(int(padded, 2))[2:]
        return hex_str

    def calculate_similarity(self, fp1_data: Dict[str, Any], fp2_data: Dict[str, Any]) -> float:
        """
        Calculate Tanimoto similarity between two fingerprints.

        Args:
            fp1_data: First fingerprint data (from generate_from_*)
            fp2_data: Second fingerprint data (from generate_from_*)

        Returns:
            Tanimoto similarity coefficient (0-1)
        """
        try:
            # Reconstruct bit vectors from bit_vector lists
            bv1 = DataStructs.ExplicitBitVect(len(fp1_data["bit_vector"]))
            bv2 = DataStructs.ExplicitBitVect(len(fp2_data["bit_vector"]))

            for i, bit in enumerate(fp1_data["bit_vector"]):
                if bit:
                    bv1.SetBit(i)

            for i, bit in enumerate(fp2_data["bit_vector"]):
                if bit:
                    bv2.SetBit(i)

            return DataStructs.TanimotoSimilarity(bv1, bv2)

        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0


def decode_pubchem_fingerprint(fingerprint_b64: str) -> Optional[Dict[str, Any]]:
    """
    Decode PubChem Fingerprint2D from base64 format.

    Args:
        fingerprint_b64: Base64-encoded fingerprint string from PubChem

    Returns:
        Dictionary containing fingerprint data
    """
    try:
        # Decode base64 to bytes
        fp_bytes = base64.b64decode(fingerprint_b64)

        # Convert bytes to bit vector
        bit_vector = []
        for byte in fp_bytes:
            for i in range(8):
                bit_vector.append((byte >> (7 - i)) & 1)

        # Get positions of set bits
        on_bits = [i for i, bit in enumerate(bit_vector) if bit == 1]

        # Convert to hex
        hex_str = fp_bytes.hex()

        return {
            "source": "PubChem",
            "method": "PubChem Fingerprint2D (Substructure Keys)",
            "n_bits": len(bit_vector),
            "num_on_bits": len(on_bits),
            "on_bits": on_bits,
            "bit_vector": bit_vector,
            "hex": hex_str,
            "base64": fingerprint_b64
        }
    except Exception as e:
        print(f"Error decoding PubChem fingerprint: {e}")
        return None


def generate_fingerprint_from_data(data: Dict[str, Any],
                                   radius: int = 2,
                                   n_bits: int = 256,
                                   include_variants: bool = False) -> Optional[Dict[str, Any]]:
    """
    Generate 256-bit Morgan fingerprint from chemical data (optimized for LLM/ML).

    Uses RDKit to compute Morgan fingerprints suitable for:
    - LLM fine-tuning with small samples
    - Machine learning models
    - Consistent dimensionality across all molecules

    Args:
        data: Chemical data from PubChem/ChEMBL
        radius: Morgan fingerprint radius (default 2, ECFP4)
        n_bits: Number of bits (default 256, optimized for LLM)
        include_variants: Include different radius variants

    Returns:
        Fingerprint data or None if generation fails
    """

    # Generate 256-bit Morgan fingerprint using RDKit
    print(f"    Generating {n_bits}-bit Morgan fingerprint...")

    if not RDKIT_AVAILABLE:
        print("    Error: RDKit not available for fallback fingerprint generation")
        return None

    generator = FingerprintGenerator()

    # Try to get SMILES from PubChem data
    smiles = None
    if "pubchem" in data and data["pubchem"]:
        props = data["pubchem"].get("properties", {})
        # Try different SMILES fields
        smiles = (props.get("CanonicalSMILES") or
                 props.get("IsomericSMILES") or
                 props.get("SMILES"))

    # If no SMILES from PubChem, try ChEMBL
    if not smiles and "chembl" in data and data["chembl"]:
        molecules = data["chembl"].get("molecules", [])
        if molecules and "structures" in molecules[0]:
            smiles = molecules[0]["structures"].get("canonical_smiles")

    # If we have SMILES, generate fingerprint
    if smiles:
        fp_data = generator.generate_from_smiles(smiles, radius, n_bits, include_variants)
        if fp_data:
            fp_data["source"] = "RDKit (local)"
            return fp_data

    # Try InChI as fallback
    inchi = None
    if "pubchem" in data and data["pubchem"]:
        inchi = data["pubchem"].get("properties", {}).get("InChI")

    if not inchi and "chembl" in data and data["chembl"]:
        molecules = data["chembl"].get("molecules", [])
        if molecules and "structures" in molecules[0]:
            inchi = molecules[0]["structures"].get("standard_inchi")

    if inchi:
        fp_data = generator.generate_from_inchi(inchi, radius, n_bits, include_variants)
        if fp_data:
            fp_data["source"] = "RDKit (local)"
            return fp_data

    print("    Warning: No SMILES or InChI found to generate fingerprint")
    return None
