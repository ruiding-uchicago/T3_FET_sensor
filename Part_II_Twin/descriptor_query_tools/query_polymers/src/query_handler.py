"""
Query handler for different input types
"""
from typing import Dict, Any, Optional, Tuple
from .pubchem_fetcher import PubChemFetcher
from .chembl_fetcher import ChEMBLFetcher
from .cache_manager import CacheManager
from .fingerprint_generator import generate_fingerprint_from_data


class QueryHandler:
    """
    Handles queries of different types and fetches data from multiple sources.
    """

    def __init__(self, cache_dir: str = "cache", use_cache: bool = True, generate_fingerprints: bool = True):
        """
        Initialize query handler.

        Args:
            cache_dir: Directory for cache storage
            use_cache: Whether to use caching (default True)
            generate_fingerprints: Whether to generate Morgan fingerprints (default True)
        """
        self.pubchem = PubChemFetcher()
        self.chembl = ChEMBLFetcher()
        self.cache = CacheManager(cache_dir)
        self.use_cache = use_cache
        self.generate_fingerprints = generate_fingerprints

    def query(self,
              query_string: str,
              query_type: str = "name",
              force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        Query chemical data from both PubChem and ChEMBL.

        Args:
            query_string: The query (name, SMILES, InChI, CID, etc.)
            query_type: Type of query - "name", "smiles", "inchi", "cid", "chembl_id", "inchi_key"
            force_refresh: Force refresh from API even if cached (default False)

        Returns:
            Combined data from both sources or None if not found
        """
        # Check cache first (unless force refresh)
        if self.use_cache and not force_refresh:
            cached_data = self.cache.get(query_string)
            if cached_data:
                print(f"✓ Found in cache: {query_string}")
                return cached_data

        print(f"Fetching data for: {query_string} (type: {query_type})")

        # Fetch from both sources
        pubchem_data = self._fetch_from_pubchem(query_string, query_type)
        chembl_data = self._fetch_from_chembl(query_string, query_type)

        # Check if we got any data
        if not pubchem_data and not chembl_data:
            print(f"✗ No data found for: {query_string}")
            return None

        # Combine data
        combined_data = {
            "pubchem": pubchem_data,
            "chembl": chembl_data
        }

        # Generate 128-bit Morgan fingerprints (optimized for LLM/ML)
        fingerprint_data = None
        if self.generate_fingerprints:
            print("  Generating fingerprints...")
            try:
                fingerprint_data = generate_fingerprint_from_data(
                    combined_data,
                    radius=2,      # ECFP4 equivalent
                    n_bits=128,    # Compact size optimized for LLM/ML
                    include_variants=False
                )
                if fingerprint_data:
                    print(f"  ✓ Fingerprint: {fingerprint_data['num_on_bits']} bits set ({fingerprint_data['n_bits']} total)")
            except Exception as e:
                print(f"  Warning: Fingerprint generation failed: {e}")

        # Add fingerprints to combined data
        if fingerprint_data:
            combined_data["fingerprints"] = fingerprint_data

        # Cache the result
        if self.use_cache:
            self.cache.set(query_string, combined_data, query_type)

        return {
            "query": {
                "input": query_string,
                "type": query_type
            },
            "data": combined_data
        }

    def _fetch_from_pubchem(self, query_string: str, query_type: str) -> Optional[Dict[str, Any]]:
        """
        Fetch data from PubChem based on query type.

        Args:
            query_string: The query string
            query_type: Type of query

        Returns:
            PubChem data or None
        """
        print("  Querying PubChem...")

        try:
            if query_type == "name":
                return self.pubchem.fetch_by_name(query_string)
            elif query_type == "cid":
                return self.pubchem.fetch_by_cid(int(query_string))
            elif query_type == "smiles":
                return self.pubchem.fetch_by_smiles(query_string)
            elif query_type == "inchi":
                return self.pubchem.fetch_by_inchi(query_string)
            else:
                print(f"  PubChem: Unsupported query type '{query_type}', trying as name")
                return self.pubchem.fetch_by_name(query_string)
        except Exception as e:
            print(f"  PubChem error: {e}")
            return None

    def _fetch_from_chembl(self, query_string: str, query_type: str) -> Optional[Dict[str, Any]]:
        """
        Fetch data from ChEMBL based on query type.

        Args:
            query_string: The query string
            query_type: Type of query

        Returns:
            ChEMBL data or None
        """
        print("  Querying ChEMBL...")

        try:
            if query_type == "name":
                return self.chembl.fetch_by_name(query_string)
            elif query_type == "chembl_id":
                return self.chembl.fetch_by_chembl_id(query_string)
            elif query_type == "smiles":
                return self.chembl.fetch_by_smiles(query_string)
            elif query_type == "inchi_key":
                return self.chembl.fetch_by_inchi_key(query_string)
            elif query_type == "inchi":
                # ChEMBL doesn't support InChI directly, try to use InChI Key if available
                print(f"  ChEMBL: InChI not directly supported, skipping")
                return None
            elif query_type == "cid":
                # ChEMBL doesn't use PubChem CIDs
                print(f"  ChEMBL: PubChem CID not supported, skipping")
                return None
            else:
                print(f"  ChEMBL: Unsupported query type '{query_type}', trying as name")
                return self.chembl.fetch_by_name(query_string)
        except Exception as e:
            print(f"  ChEMBL error: {e}")
            return None

    def clear_cache(self, query: Optional[str] = None) -> None:
        """
        Clear cache for a specific query or all cache.

        Args:
            query: Query string to clear (if None, clears all)
        """
        self.cache.clear(query)
