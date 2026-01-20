#!/usr/bin/env python3
"""
Smart FET Descriptor Query Script

Automatically:
1. Check local cache (FET_descriptor_filtered/)
2. If cache exists → return JSON directly
3. If cache not exists:
   a. Query OPTIMADE databases
   b. Save raw data to raw_data_pulled/
   c. Extract 32 FET descriptors + 132 MAGPIE features (164 total)
   d. Save to FET_descriptor_filtered/
   e. Return JSON

Usage:
    python get_fet_descriptor.py "GaN"
    python get_fet_descriptor.py "gallium nitride"
    python get_fet_descriptor.py "iron oxide"
"""

import sys
import json
import argparse
from pathlib import Path
from parse_material_name import parse_material_name
from query_substance import query_substance
import subprocess


def get_fet_descriptor(material_name, force_refresh=False):
    """
    Smart retrieval of material FET descriptors.

    Args:
        material_name: Material name or chemical formula
        force_refresh: Force refresh (ignore cache)

    Returns:
        Path to FET descriptor JSON file, or None if failed
    """
    print("=" * 70)
    print("FET Descriptor Smart Query")
    print("=" * 70)
    print()

    # Step 1: Parse material name
    print(f"[Step 1] Parsing material name: '{material_name}'")
    formula, source = parse_material_name(material_name)

    if source == 'fallback':
        print(f"⚠️  Warning: Could not parse '{material_name}', trying as-is")
    else:
        print(f"✓ Parsed as: {formula} [source: {source}]")

    filename = f"{formula.lower()}.json"
    print()

    # Step 2: Check local cache
    cache_path = Path("FET_descriptor_filtered") / filename

    if cache_path.exists() and not force_refresh:
        print(f"[Step 2] Checking local cache...")
        print(f"✓ Found in cache: {cache_path}")
        print()
        print("=" * 70)
        print("✓ SUCCESS - Using cached descriptor")
        print("=" * 70)
        print(f"FET Descriptor JSON: {cache_path.absolute()}")
        return cache_path

    if force_refresh:
        print(f"[Step 2] Forcing refresh (ignoring cache)")
    else:
        print(f"[Step 2] Checking local cache...")
        print(f"⚠️  Not in cache: {cache_path}")
    print()

    # Step 3: Query OPTIMADE databases
    print(f"[Step 3] Querying OPTIMADE databases for {formula}...")
    try:
        query_substance(formula, output_file=filename)
    except Exception as e:
        print(f"✗ OPTIMADE query failed: {e}")
        return None

    # Check if query result file exists
    if not Path(filename).exists():
        print(f"✗ Query did not produce {filename}")
        return None

    print()

    # Step 4: Move to raw_data_pulled/
    print(f"[Step 4] Saving raw data to raw_data_pulled/...")
    raw_data_dir = Path("raw_data_pulled")
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_data_dir / filename

    # Delete if already exists
    if raw_path.exists():
        raw_path.unlink()

    Path(filename).rename(raw_path)
    print(f"✓ Saved: {raw_path}")
    print()

    # Step 5: Extract FET descriptors
    print(f"[Step 5] Extracting FET descriptors...")

    # Ensure output directory exists
    Path("FET_descriptor_filtered").mkdir(parents=True, exist_ok=True)

    try:
        # Call extraction script
        extraction_script_path = Path(__file__).parent / "extract_comprehensive_properties.py"
        result = subprocess.run(
            [
                sys.executable,
                str(extraction_script_path),
                str(raw_path),
                "FET_descriptor_filtered/"
            ],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"✗ Extraction failed:")
            print(result.stderr)
            return None

        # Display extraction output
        print(result.stdout)

    except subprocess.TimeoutExpired:
        print(f"✗ Extraction timeout (>60s)")
        return None
    except Exception as e:
        print(f"✗ Extraction error: {e}")
        return None

    # Step 6: Verify output
    if not cache_path.exists():
        print(f"✗ Extraction did not produce {cache_path}")
        return None

    print()
    print("=" * 70)
    print("✓ SUCCESS - FET descriptor extracted and cached")
    print("=" * 70)
    print(f"FET Descriptor JSON: {cache_path.absolute()}")

    return cache_path


def display_descriptor_summary(json_path):
    """Display a summary of the FET descriptor."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        print()
        print("=" * 70)
        print("FET Descriptor Summary")
        print("=" * 70)

        # Key descriptors
        key_descriptors = [
            ('material_formula', 'Formula'),
            ('band_gap', 'Band Gap (eV)'),
            ('eg_class', 'Band Gap Class'),
            ('formation_energy_per_atom', 'Formation Energy (eV/atom)'),
            ('energy_above_hull', 'E_hull (eV/atom)'),
            ('thermo_stable', 'Stability'),
            ('crystal_system', 'Crystal System'),
            ('space_group', 'Space Group'),
            ('density', 'Density (g/cm³)'),
        ]

        for key, label in key_descriptors:
            value = data.get(key)
            if value is not None and value != '':
                print(f"  {label:30s}: {value}")

        # Mechanical properties
        if data.get('k_vrh') is not None:
            print(f"\n  Mechanical Properties:")
            print(f"    K_VRH (GPa)                  : {data.get('k_vrh')}")
            print(f"    G_VRH (GPa)                  : {data.get('g_vrh')}")
            print(f"    E_modulus (GPa)              : {data.get('e_modulus_est')}")

        # Dielectric properties
        if data.get('eps_mean'):
            print(f"\n  Dielectric Properties:")
            print(f"    ε_mean                       : {data.get('eps_mean')}")
            print(f"    ε_anisotropy                 : {data.get('eps_aniso')}")

        # Count MAGPIE features
        magpie_count = sum(1 for k in data.keys() if k.startswith('magpie_'))
        fet_count = len(data) - magpie_count

        print()
        print(f"Total descriptors: {len(data)} ({fet_count} FET + {magpie_count} MAGPIE)")
        print(f"For full details, see: {json_path}")

    except Exception as e:
        print(f"Could not display summary: {e}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Smart FET Descriptor Query - Auto-cache and extract',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Query material (cache first)
  python get_fet_descriptor.py "GaN"
  python get_fet_descriptor.py "gallium nitride"
  python get_fet_descriptor.py "iron oxide"

  # Force refresh (ignore cache)
  python get_fet_descriptor.py "GaN" --force

  # Show detailed summary
  python get_fet_descriptor.py "GaN" --summary

Workflow:
  1. Parse material name → standardized formula
  2. Check cache (FET_descriptor_filtered/)
     - Exists → return JSON ✓
     - Not exists → continue
  3. Query OPTIMADE databases
  4. Extract 164 descriptors (32 FET + 132 MAGPIE)
  5. Save to cache
  6. Return JSON ✓
        '''
    )

    parser.add_argument(
        'material',
        help='Material name or chemical formula'
    )

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force refresh (ignore cache)'
    )

    parser.add_argument(
        '--summary', '-s',
        action='store_true',
        help='Display descriptor summary'
    )

    args = parser.parse_args()

    # Get FET descriptor
    json_path = get_fet_descriptor(args.material, force_refresh=args.force)

    if json_path is None:
        print()
        print("✗ Failed to get FET descriptor")
        sys.exit(1)

    # Display summary if requested
    if args.summary:
        display_descriptor_summary(json_path)

    print()
    print("✓ Done")
    sys.exit(0)


if __name__ == '__main__':
    main()
