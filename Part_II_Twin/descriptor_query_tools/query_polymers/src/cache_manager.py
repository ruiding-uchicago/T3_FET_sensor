"""
Cache management for chemical data
"""
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any
from .utils import get_cache_path, ensure_directory


class CacheManager:
    """
    Manages local cache for chemical data.
    """

    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory to store cache files (default "cache")
        """
        self.cache_dir = cache_dir
        ensure_directory(cache_dir)

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached data for a query.

        Args:
            query: Chemical name or identifier

        Returns:
            Cached data dictionary if found, None otherwise
        """
        cache_path = get_cache_path(query, self.cache_dir)

        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load cache from {cache_path}: {e}")
            return None

    def set(self, query: str, data: Dict[str, Any], query_type: str = "name") -> None:
        """
        Save data to cache.

        Args:
            query: Chemical name or identifier
            data: Data dictionary to cache
            query_type: Type of query (name, smiles, inchi, cid, etc.)
        """
        cache_path = get_cache_path(query, self.cache_dir)

        # Add metadata to cached data
        cache_data = {
            "query": {
                "input": query,
                "type": query_type,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            },
            "data": data,
            "metadata": {
                "cached": True,
                "last_updated": datetime.utcnow().isoformat() + "Z"
            }
        }

        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Data cached to: {cache_path}")
        except IOError as e:
            print(f"Warning: Failed to save cache to {cache_path}: {e}")

    def exists(self, query: str) -> bool:
        """
        Check if cached data exists for a query.

        Args:
            query: Chemical name or identifier

        Returns:
            True if cache exists, False otherwise
        """
        cache_path = get_cache_path(query, self.cache_dir)
        return os.path.exists(cache_path)

    def clear(self, query: Optional[str] = None) -> None:
        """
        Clear cache for a specific query or all cache.

        Args:
            query: Chemical name or identifier (if None, clears all cache)
        """
        if query:
            cache_path = get_cache_path(query, self.cache_dir)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                print(f"✓ Cleared cache for: {query}")
            else:
                print(f"No cache found for: {query}")
        else:
            # Clear all cache files
            if os.path.exists(self.cache_dir):
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.json'):
                        os.remove(os.path.join(self.cache_dir, filename))
                print(f"✓ Cleared all cache from: {self.cache_dir}")
