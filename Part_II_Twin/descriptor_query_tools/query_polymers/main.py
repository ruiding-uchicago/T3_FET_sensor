#!/usr/bin/env python3
"""
Main script for querying polymer properties from multiple databases.

This script follows the same interface as the other query tools:
query_molecules, query_inorganic, query_biomolecules, query_DNA_RNA.
"""

import json
import sys
import argparse
import os
from typing import Dict, Any, Optional

def query_polymer(polymer_name: str) -> Optional[Dict[str, Any]]:
    """
    Query polymer data using agent-based approach.
    
    Args:
        polymer_name: Name of the polymer to query
        
    Returns:
        Dictionary with polymer data or None if not found
    """
    # Sanitize the polymer name to create cache filename
    cache_filename = sanitize_filename(polymer_name)
    cache_path = os.path.join("cache", f"{cache_filename}.json")
    
    # Check if data exists in cache first
    if os.path.exists(cache_path):
        print(f"Found {polymer_name} in cache: {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    print(f"Cache miss for {polymer_name}, would need to implement agent-based fetcher")
    # In a full implementation, this would trigger an LLM agent to:
    # 1. Search PoLyInfo, PPPDB, Polymer Genome
    # 2. Extract data following POLYMER_DATA_REFERENCE.md
    # 3. Generate fingerprints with src/fingerprint_generator.py
    # 4. Save to cache with proper structure
    return None

def sanitize_filename(name: str) -> str:
    """
    Convert polymer name to cache filename following README rules:
    1. Convert to lowercase
    2. Remove quotes
    3. Replace special chars with underscores
    4. Replace spaces/hyphens with underscores
    5. Remove multiple underscores
    6. Strip leading/trailing underscores
    """
    import re
    # Convert to lowercase
    name = name.lower()
    # Remove quotes
    name = name.replace('"', '').replace("'", '')
    # Replace special characters with underscores
    name = re.sub(r'[^\w\s\-]', '_', name)
    # Replace spaces and hyphens with underscores
    name = name.replace(' ', '_').replace('-', '_')
    # Replace multiple underscores with single underscore
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    return name

def main():
    parser = argparse.ArgumentParser(description="Query polymer properties from multiple databases")
    parser.add_argument("polymer", help="Polymer name to query")
    parser.add_argument("--smiles", help="Query by repeat unit SMILES")
    parser.add_argument("--bigsmiles", help="Query by BigSMILES")
    parser.add_argument("--cas", help="Query by CAS number")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--refresh", action="store_true", help="Force refresh (ignore cache)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache for this polymer")
    
    args = parser.parse_args()
    
    # Determine query type and value
    query_type = "name"
    query_value = args.polymer
    
    if args.smiles:
        query_type = "smiles"
        query_value = args.smiles
    elif args.bigsmiles:
        query_type = "bigsmiles"
        query_value = args.bigsmiles
    elif args.cas:
        query_type = "cas"
        query_value = args.cas
    
    # Handle cache clearing
    if args.clear_cache:
        cache_filename = sanitize_filename(args.polymer if not args.smiles else args.smiles)
        cache_path = os.path.join("cache", f"{cache_filename}.json")
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"Cache cleared for {cache_path}")
        else:
            print(f"No cache found for {cache_path}")
        return
    
    # Perform the query
    result = query_polymer(query_value)
    
    if result:
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            # Print summary
            print(f"Successfully retrieved data for: {query_value}")
            # Extract and print key properties if available
            if "data" in result:
                polyinfo = result["data"].get("polyinfo", {})
                if polyinfo:
                    structure = polyinfo.get("structure", {})
                    thermal = polyinfo.get("thermal_properties", {})
                    physical = polyinfo.get("physical_properties", {})
                    
                    print(f"  Polymer name: {structure.get('polymer_name', 'N/A')}")
                    print(f"  Tg (Glass transition): {thermal.get('Tg_C', 'N/A')}°C")
                    print(f"  Density: {physical.get('density_g_cm3', 'N/A')} g/cm³")
    else:
        print(f"Could not find data for: {query_value}")
        if not args.refresh:
            print("Try with --refresh to force fetching from databases")
        sys.exit(1)

if __name__ == "__main__":
    main()