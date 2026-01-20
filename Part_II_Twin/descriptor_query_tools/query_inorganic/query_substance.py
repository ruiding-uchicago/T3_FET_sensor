#!/usr/bin/env python3
"""
Query OPTIMADE databases for materials - get FIRST match from each database.

Usage:
    python query_substance.py GaN
    python query_substance.py HfO2
    python query_substance.py BaTiO3
    python query_substance.py Al2O3
"""

import json
import argparse
from pathlib import Path
from optimade.client import OptimadeClient
from parse_material_name import parse_material_name


def query_substance(substance, output_file=None):
    """
    Query OPTIMADE databases for first match of exact substance formula.

    Args:
        substance: Chemical formula (e.g., 'Si', 'GaN', 'SiO2', 'HfO2')
        output_file: Path to save JSON output (default: {substance}.json)

    Returns:
        Dictionary containing all query results
    """
    # Simple: exact formula match, first result only from each database
    filter_query = f'chemical_formula_reduced="{substance}"'

    print(f"Querying for: {substance}")
    print(f"Getting FIRST match from each database")

    # Create client - limit to 1 result per provider
    print(f"Initializing OPTIMADE client...")
    client = OptimadeClient(max_results_per_provider=1)

    print(f"Filter: {filter_query}")
    print("This may take a moment as we query multiple databases...\n")

    # Execute query
    client.get(filter_query)

    # Get all results
    all_results = client.all_results

    # Print summary
    print("\n=== Query Summary ===")
    if "structures" in all_results:
        total_structures = 0
        for filter_str, providers in all_results["structures"].items():
            print(f"\nFilter: {filter_str}")
            for provider_url, response in providers.items():
                # Handle QueryResults object
                if hasattr(response, 'data'):
                    num_results = len(response.data) if response.data else 0
                elif isinstance(response, dict):
                    num_results = len(response.get("data", []))
                else:
                    num_results = 0
                total_structures += num_results
                print(f"  - {provider_url}: {num_results} structures")

        print(f"\nTotal structures found: {total_structures}")
    else:
        print("No structures found.")

    # Determine output filename
    if output_file is None:
        output_file = f"{substance.lower()}.json"

    # Convert QueryResults objects to dictionaries for JSON serialization
    serializable_results = {}
    if "structures" in all_results:
        serializable_results["structures"] = {}
        for filter_str, providers in all_results["structures"].items():
            serializable_results["structures"][filter_str] = {}
            for provider_url, response in providers.items():
                # Convert QueryResults object to dict
                if hasattr(response, '__dict__'):
                    # Try to get the underlying dict representation
                    if hasattr(response, 'dict'):
                        serializable_results["structures"][filter_str][provider_url] = response.dict()
                    else:
                        # Manually extract data, meta, errors
                        serializable_results["structures"][filter_str][provider_url] = {
                            "data": response.data if hasattr(response, 'data') else [],
                            "meta": response.meta if hasattr(response, 'meta') else {},
                            "errors": response.errors if hasattr(response, 'errors') else []
                        }
                else:
                    serializable_results["structures"][filter_str][provider_url] = response

    # Save to JSON file
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\n✓ Results saved to: {output_path.absolute()}")

    return all_results


def main():
    """Command-line interface for querying substances."""
    parser = argparse.ArgumentParser(
        description='Query OPTIMADE - first match from each database',
        epilog='''
Examples:
  python query_substance.py GaN
  python query_substance.py "gallium nitride"
  python query_substance.py HfO2
  python query_substance.py BaTiO3
        '''
    )
    parser.add_argument(
        'substance',
        help='Chemical name or formula'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output JSON file'
    )

    args = parser.parse_args()

    # Parse material name to chemical formula
    original_input = args.substance
    substance, source = parse_material_name(original_input)

    # Print conversion info
    if substance != original_input:
        print(f"'{original_input}' → {substance} [{source}]")
    else:
        print(f"Using: {substance}")

    if source == 'fallback':
        print(f"Warning: Could not parse '{original_input}', trying as-is")

    print()

    # Query
    query_substance(
        substance=substance,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
