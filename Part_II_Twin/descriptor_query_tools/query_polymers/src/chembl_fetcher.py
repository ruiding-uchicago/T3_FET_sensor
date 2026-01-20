"""
ChEMBL API fetcher
"""
import requests
from typing import Dict, Any, Optional, List


class ChEMBLFetcher:
    """
    Fetches chemical data from ChEMBL API.
    """

    BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

    def __init__(self):
        """
        Initialize ChEMBL fetcher.
        """
        pass

    def fetch_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Fetch chemical data by name (searches for molecules with matching preferred name).

        Args:
            name: Chemical name

        Returns:
            Dictionary of chemical properties or None if not found
        """
        url = f"{self.BASE_URL}/molecule.json"
        params = {
            'pref_name__icontains': name,
            'limit': 5  # Get top 5 matches
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "molecules" in data and len(data["molecules"]) > 0:
                # Return all matches with a note about multiple results
                molecules = data["molecules"]
                if len(molecules) > 1:
                    print(f"ChEMBL: Found {len(molecules)} matches for '{name}'")
                    print(f"  Using best match: {molecules[0].get('pref_name', 'N/A')}")

                return self._format_response(molecules)
            else:
                print(f"ChEMBL: Chemical '{name}' not found")
                return None

        except requests.exceptions.RequestException as e:
            print(f"ChEMBL request error: {e}")
            return None
        except Exception as e:
            print(f"ChEMBL error: {e}")
            return None

    def fetch_by_chembl_id(self, chembl_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch chemical data by ChEMBL ID.

        Args:
            chembl_id: ChEMBL ID (e.g., CHEMBL25)

        Returns:
            Dictionary of chemical properties or None if not found
        """
        url = f"{self.BASE_URL}/molecule/{chembl_id}.json"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data:
                return self._format_response([data])
            else:
                print(f"ChEMBL: ID '{chembl_id}' not found")
                return None

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"ChEMBL: ID '{chembl_id}' not found")
            else:
                print(f"ChEMBL HTTP error: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"ChEMBL request error: {e}")
            return None
        except Exception as e:
            print(f"ChEMBL error: {e}")
            return None

    def fetch_by_smiles(self, smiles: str) -> Optional[Dict[str, Any]]:
        """
        Fetch chemical data by SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            Dictionary of chemical properties or None if not found
        """
        # ChEMBL uses molecule_structures__canonical_smiles for exact match
        url = f"{self.BASE_URL}/molecule.json"
        params = {
            'molecule_structures__canonical_smiles__exact': smiles,
            'limit': 5
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "molecules" in data and len(data["molecules"]) > 0:
                molecules = data["molecules"]
                if len(molecules) > 1:
                    print(f"ChEMBL: Found {len(molecules)} matches for SMILES")

                return self._format_response(molecules)
            else:
                print(f"ChEMBL: SMILES '{smiles}' not found")
                return None

        except requests.exceptions.RequestException as e:
            print(f"ChEMBL request error: {e}")
            return None
        except Exception as e:
            print(f"ChEMBL error: {e}")
            return None

    def fetch_by_inchi_key(self, inchi_key: str) -> Optional[Dict[str, Any]]:
        """
        Fetch chemical data by InChI Key.

        Args:
            inchi_key: InChI Key

        Returns:
            Dictionary of chemical properties or None if not found
        """
        url = f"{self.BASE_URL}/molecule.json"
        params = {
            'molecule_structures__standard_inchi_key': inchi_key,
            'limit': 5
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "molecules" in data and len(data["molecules"]) > 0:
                molecules = data["molecules"]
                return self._format_response(molecules)
            else:
                print(f"ChEMBL: InChI Key '{inchi_key}' not found")
                return None

        except requests.exceptions.RequestException as e:
            print(f"ChEMBL request error: {e}")
            return None
        except Exception as e:
            print(f"ChEMBL error: {e}")
            return None

    def _format_response(self, molecules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format the response data.

        Args:
            molecules: List of molecule data from ChEMBL API

        Returns:
            Formatted properties dictionary
        """
        # Extract key information from the molecules
        formatted_molecules = []

        for mol in molecules:
            formatted = {
                "chembl_id": mol.get("molecule_chembl_id"),
                "pref_name": mol.get("pref_name"),
                "molecule_type": mol.get("molecule_type"),
                "max_phase": mol.get("max_phase"),  # Clinical trial phase
                "first_approval": mol.get("first_approval"),
                "oral": mol.get("oral"),
                "parenteral": mol.get("parenteral"),
                "topical": mol.get("topical"),
                "withdrawn_flag": mol.get("withdrawn_flag"),
                "therapeutic_flag": mol.get("therapeutic_flag")
            }

            # Add molecule properties if available
            if "molecule_properties" in mol and mol["molecule_properties"]:
                formatted["properties"] = mol["molecule_properties"]

            # Add structure information if available
            if "molecule_structures" in mol and mol["molecule_structures"]:
                formatted["structures"] = mol["molecule_structures"]

            # Add hierarchy information
            if "molecule_hierarchy" in mol and mol["molecule_hierarchy"]:
                formatted["hierarchy"] = mol["molecule_hierarchy"]

            formatted_molecules.append(formatted)

        return {
            "source": "ChEMBL",
            "count": len(formatted_molecules),
            "molecules": formatted_molecules
        }
