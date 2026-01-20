#!/usr/bin/env python3
"""
Chemical Data Fetcher - Main CLI Interface

Fetches chemical properties from PubChem and ChEMBL APIs and caches them locally.
"""
import argparse
import json
import sys
from src.query_handler import QueryHandler


def print_summary(data: dict) -> None:
    """
    Print a summary of the fetched data.

    Args:
        data: The fetched chemical data
    """
    print("\n" + "="*80)
    print("CHEMICAL DATA SUMMARY")
    print("="*80)

    # Query info
    if "query" in data:
        query_info = data["query"]
        print(f"\nQuery: {query_info.get('input')}")
        print(f"Type:  {query_info.get('type')}")
        if "timestamp" in query_info:
            print(f"Time:  {query_info.get('timestamp')}")

    # PubChem data
    pubchem_data = data.get("data", {}).get("pubchem")
    if pubchem_data:
        print("\n" + "-"*80)
        print("PubChem Data")
        print("-"*80)
        props = pubchem_data.get("properties", {})
        print(f"CID:               {props.get('CID', 'N/A')}")
        print(f"Molecular Formula: {props.get('MolecularFormula', 'N/A')}")
        print(f"Molecular Weight:  {props.get('MolecularWeight', 'N/A')}")
        print(f"IUPAC Name:        {props.get('IUPACName', 'N/A')}")
        print(f"SMILES:            {props.get('CanonicalSMILES', 'N/A')}")
        print(f"InChI Key:         {props.get('InChIKey', 'N/A')}")
        print(f"XLogP:             {props.get('XLogP', 'N/A')}")
        print(f"TPSA:              {props.get('TPSA', 'N/A')}")
        print(f"H-Bond Donors:     {props.get('HBondDonorCount', 'N/A')}")
        print(f"H-Bond Acceptors:  {props.get('HBondAcceptorCount', 'N/A')}")
    else:
        print("\nPubChem: No data found")

    # ChEMBL data
    chembl_data = data.get("data", {}).get("chembl")
    if chembl_data:
        print("\n" + "-"*80)
        print("ChEMBL Data")
        print("-"*80)
        molecules = chembl_data.get("molecules", [])
        if molecules:
            mol = molecules[0]  # Show first match
            print(f"ChEMBL ID:         {mol.get('chembl_id', 'N/A')}")
            print(f"Preferred Name:    {mol.get('pref_name', 'N/A')}")
            print(f"Molecule Type:     {mol.get('molecule_type', 'N/A')}")
            print(f"Max Phase:         {mol.get('max_phase', 'N/A')} (clinical trial)")
            print(f"First Approval:    {mol.get('first_approval', 'N/A')}")
            print(f"Therapeutic:       {mol.get('therapeutic_flag', 'N/A')}")
            print(f"Withdrawn:         {mol.get('withdrawn_flag', 'N/A')}")

            if len(molecules) > 1:
                print(f"\n(+ {len(molecules)-1} more matches in ChEMBL)")
        else:
            print("No molecules found")
    else:
        print("\nChEMBL: No data found")

    print("\n" + "="*80)


def main():
    """
    Main CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description="Fetch chemical properties from PubChem and ChEMBL APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query by chemical name (default)
  python main.py "aspirin"
  python main.py "carbon dioxide"

  # Query by SMILES
  python main.py --smiles "CCO"

  # Query by InChI
  python main.py --inchi "InChI=1S/H2O/h1H2"

  # Query by PubChem CID
  python main.py --cid 962

  # Query by ChEMBL ID
  python main.py --chembl-id "CHEMBL25"

  # Query by InChI Key
  python main.py --inchi-key "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"

  # Force refresh (ignore cache)
  python main.py "aspirin" --refresh

  # Clear all cache
  python main.py --clear-cache

  # Clear cache for specific query
  python main.py "aspirin" --clear-cache

  # Get raw JSON output
  python main.py "aspirin" --json
        """
    )

    # Query argument
    parser.add_argument(
        "query",
        nargs='?',
        help="Chemical name or identifier to query"
    )

    # Query type flags (mutually exclusive)
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument(
        "--smiles",
        action="store_true",
        help="Query is a SMILES string"
    )
    query_group.add_argument(
        "--inchi",
        action="store_true",
        help="Query is an InChI string"
    )
    query_group.add_argument(
        "--inchi-key",
        action="store_true",
        help="Query is an InChI Key"
    )
    query_group.add_argument(
        "--cid",
        action="store_true",
        help="Query is a PubChem CID (Compound ID)"
    )
    query_group.add_argument(
        "--chembl-id",
        action="store_true",
        help="Query is a ChEMBL ID"
    )

    # Options
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh from API (ignore cache)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache (all or for specific query)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of summary"
    )
    parser.add_argument(
        "--cache-dir",
        default="cache",
        help="Cache directory (default: cache)"
    )
    parser.add_argument(
        "--no-fingerprints",
        action="store_true",
        help="Disable Morgan fingerprint generation"
    )

    args = parser.parse_args()

    # Handle clear cache
    if args.clear_cache:
        handler = QueryHandler(cache_dir=args.cache_dir)
        if args.query:
            handler.clear_cache(args.query)
        else:
            handler.clear_cache()
        return

    # Require query for normal operations
    if not args.query:
        parser.print_help()
        sys.exit(1)

    # Determine query type
    query_type = "name"  # default
    if args.smiles:
        query_type = "smiles"
    elif args.inchi:
        query_type = "inchi"
    elif args.inchi_key:
        query_type = "inchi_key"
    elif args.cid:
        query_type = "cid"
    elif args.chembl_id:
        query_type = "chembl_id"

    # Create handler and execute query
    handler = QueryHandler(
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        generate_fingerprints=not args.no_fingerprints
    )

    result = handler.query(
        query_string=args.query,
        query_type=query_type,
        force_refresh=args.refresh
    )

    if result:
        if args.json:
            # Output raw JSON
            print(json.dumps(result, indent=2))
        else:
            # Print formatted summary
            print_summary(result)
    else:
        print(f"\n✗ No data found for '{args.query}'")
        sys.exit(1)


if __name__ == "__main__":
    main()
