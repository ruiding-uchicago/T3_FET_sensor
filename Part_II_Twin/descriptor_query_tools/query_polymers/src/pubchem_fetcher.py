"""
PubChem PUG REST API fetcher
"""
import requests
import time
from typing import Dict, Any, Optional, List


class PubChemFetcher:
    """
    Fetches chemical data from PubChem PUG REST API.
    """

    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    # Properties to fetch from PubChem
    PROPERTIES = [
        "MolecularFormula",
        "MolecularWeight",
        "CanonicalSMILES",
        "IsomericSMILES",
        "InChI",
        "InChIKey",
        "IUPACName",
        "XLogP",
        "ExactMass",
        "MonoisotopicMass",
        "TPSA",
        "Complexity",
        "Charge",
        "HBondDonorCount",
        "HBondAcceptorCount",
        "RotatableBondCount",
        "HeavyAtomCount",
        "IsotopeAtomCount",
        "AtomStereoCount",
        "DefinedAtomStereoCount",
        "UndefinedAtomStereoCount",
        "BondStereoCount",
        "DefinedBondStereoCount",
        "UndefinedBondStereoCount",
        "CovalentUnitCount",
        "Volume3D",
        "XStericQuadrupole3D",
        "YStericQuadrupole3D",
        "ZStericQuadrupole3D",
        "FeatureCount3D",
        "FeatureAcceptorCount3D",
        "FeatureDonorCount3D",
        "FeatureAnionCount3D",
        "FeatureCationCount3D",
        "FeatureRingCount3D",
        "FeatureHydrophobeCount3D",
        "ConformerModelRMSD3D",
        "EffectiveRotorCount3D",
        "ConformerCount3D"
    ]

    def __init__(self, delay: float = 0.2):
        """
        Initialize PubChem fetcher.

        Args:
            delay: Delay between requests in seconds (default 0.2 for 5 req/sec limit)
        """
        self.delay = delay
        self.last_request_time = 0

    def _rate_limit(self) -> None:
        """
        Enforce rate limiting (5 requests per second).
        """
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.delay:
            time.sleep(self.delay - time_since_last)
        self.last_request_time = time.time()

    def fetch_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Fetch chemical data by name.

        Args:
            name: Chemical name

        Returns:
            Dictionary of chemical properties or None if not found
        """
        self._rate_limit()

        properties_str = ",".join(self.PROPERTIES)
        url = f"{self.BASE_URL}/compound/name/{name}/property/{properties_str}/JSON"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "PropertyTable" in data and "Properties" in data["PropertyTable"]:
                properties = data["PropertyTable"]["Properties"][0]
                return self._format_response(properties)
            else:
                print(f"Warning: Unexpected response structure from PubChem")
                return None

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"PubChem: Chemical '{name}' not found")
            else:
                print(f"PubChem HTTP error: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"PubChem request error: {e}")
            return None
        except Exception as e:
            print(f"PubChem error: {e}")
            return None

    def fetch_by_cid(self, cid: int) -> Optional[Dict[str, Any]]:
        """
        Fetch chemical data by CID (Compound ID).

        Args:
            cid: PubChem Compound ID

        Returns:
            Dictionary of chemical properties or None if not found
        """
        self._rate_limit()

        properties_str = ",".join(self.PROPERTIES)
        url = f"{self.BASE_URL}/compound/cid/{cid}/property/{properties_str}/JSON"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "PropertyTable" in data and "Properties" in data["PropertyTable"]:
                properties = data["PropertyTable"]["Properties"][0]
                return self._format_response(properties)
            else:
                print(f"Warning: Unexpected response structure from PubChem")
                return None

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"PubChem: CID {cid} not found")
            else:
                print(f"PubChem HTTP error: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"PubChem request error: {e}")
            return None
        except Exception as e:
            print(f"PubChem error: {e}")
            return None

    def fetch_by_smiles(self, smiles: str) -> Optional[Dict[str, Any]]:
        """
        Fetch chemical data by SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            Dictionary of chemical properties or None if not found
        """
        self._rate_limit()

        properties_str = ",".join(self.PROPERTIES)
        url = f"{self.BASE_URL}/compound/smiles/{smiles}/property/{properties_str}/JSON"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "PropertyTable" in data and "Properties" in data["PropertyTable"]:
                properties = data["PropertyTable"]["Properties"][0]
                return self._format_response(properties)
            else:
                print(f"Warning: Unexpected response structure from PubChem")
                return None

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"PubChem: SMILES '{smiles}' not found")
            else:
                print(f"PubChem HTTP error: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"PubChem request error: {e}")
            return None
        except Exception as e:
            print(f"PubChem error: {e}")
            return None

    def fetch_by_inchi(self, inchi: str) -> Optional[Dict[str, Any]]:
        """
        Fetch chemical data by InChI string.

        Args:
            inchi: InChI string

        Returns:
            Dictionary of chemical properties or None if not found
        """
        self._rate_limit()

        properties_str = ",".join(self.PROPERTIES)
        url = f"{self.BASE_URL}/compound/inchi/property/{properties_str}/JSON"

        try:
            # InChI needs to be sent as POST data
            response = requests.post(url, data={'inchi': inchi}, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "PropertyTable" in data and "Properties" in data["PropertyTable"]:
                properties = data["PropertyTable"]["Properties"][0]
                return self._format_response(properties)
            else:
                print(f"Warning: Unexpected response structure from PubChem")
                return None

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"PubChem: InChI not found")
            else:
                print(f"PubChem HTTP error: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"PubChem request error: {e}")
            return None
        except Exception as e:
            print(f"PubChem error: {e}")
            return None

    def _format_response(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the response data.

        Args:
            properties: Raw properties from PubChem API

        Returns:
            Formatted properties dictionary
        """
        return {
            "source": "PubChem",
            "cid": properties.get("CID"),
            "properties": properties
        }
