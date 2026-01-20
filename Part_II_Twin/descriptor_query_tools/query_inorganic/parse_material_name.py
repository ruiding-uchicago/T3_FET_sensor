#!/usr/bin/env python3
"""
Flexible material name to chemical formula converter.
Uses multi-layer strategy for maximum coverage.
"""

from pymatgen.core import Composition, Element
import pubchempy as pcp
import json
from pathlib import Path


def parse_material_name(input_name):
    """
    Convert material name to chemical formula using multi-layer strategy.

    Strategy (in order):
    1. Check local name_map for common materials (avoids misparsing)
    2. Check element name/symbol mapping (gold→Au, avoids GOLD→GLHO)
    3. Try parsing as chemical formula directly (GaN, HfO2, etc.)
    4. Query PubChem API for chemical name resolution
    5. Return original input as fallback

    Args:
        input_name: Material name or formula (str)

    Returns:
        tuple: (formula, source) where source indicates how it was resolved
    """
    original = input_name

    # Layer 1: Load name_map from domain_frequent_name_mapping.json
    # This provides 50+ formulas with multiple name variations
    name_map = {}
    json_path = Path(__file__).parent / "domain_frequent_name_mapping.json"
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                formula_to_names = json.load(f)

            # Reverse mapping: name -> formula
            for formula, names in formula_to_names.items():
                for name in names:
                    name_map[name.lower()] = formula
        except Exception as e:
            print(f"Warning: Could not load domain_frequent_name_mapping.json: {e}")

    if input_name.lower() in name_map:
        return (name_map[input_name.lower()], "domain_mapping")

    # Layer 2: Element name/symbol mapping (before formula parse to avoid "GOLD"→"GLHO")
    element_names = {
        **{elem.symbol.lower(): elem.symbol for elem in Element},
        'hydrogen': 'H', 'helium': 'He', 'lithium': 'Li', 'beryllium': 'Be', 'boron': 'B',
        'carbon': 'C', 'nitrogen': 'N', 'oxygen': 'O', 'fluorine': 'F', 'neon': 'Ne',
        'sodium': 'Na', 'magnesium': 'Mg', 'aluminum': 'Al', 'aluminium': 'Al', 'silicon': 'Si',
        'phosphorus': 'P', 'sulfur': 'S', 'sulphur': 'S', 'chlorine': 'Cl', 'argon': 'Ar',
        'potassium': 'K', 'calcium': 'Ca', 'scandium': 'Sc', 'titanium': 'Ti', 'vanadium': 'V',
        'chromium': 'Cr', 'manganese': 'Mn', 'iron': 'Fe', 'cobalt': 'Co', 'nickel': 'Ni',
        'copper': 'Cu', 'zinc': 'Zn', 'gallium': 'Ga', 'germanium': 'Ge', 'arsenic': 'As',
        'selenium': 'Se', 'bromine': 'Br', 'krypton': 'Kr', 'rubidium': 'Rb', 'strontium': 'Sr',
        'yttrium': 'Y', 'zirconium': 'Zr', 'niobium': 'Nb', 'molybdenum': 'Mo', 'technetium': 'Tc',
        'ruthenium': 'Ru', 'rhodium': 'Rh', 'palladium': 'Pd', 'silver': 'Ag', 'cadmium': 'Cd',
        'indium': 'In', 'tin': 'Sn', 'antimony': 'Sb', 'tellurium': 'Te', 'iodine': 'I',
        'xenon': 'Xe', 'cesium': 'Cs', 'barium': 'Ba', 'lanthanum': 'La', 'cerium': 'Ce',
        'hafnium': 'Hf', 'tantalum': 'Ta', 'tungsten': 'W', 'rhenium': 'Re', 'osmium': 'Os',
        'iridium': 'Ir', 'platinum': 'Pt', 'gold': 'Au', 'mercury': 'Hg', 'thallium': 'Tl',
        'lead': 'Pb', 'bismuth': 'Bi', 'polonium': 'Po', 'uranium': 'U',
    }

    if input_name.lower() in element_names:
        return (element_names[input_name.lower()], "element")

    # Layer 3: Try parsing as chemical formula (GaN, HfO2, etc.)
    try:
        comp = Composition(input_name)
        formula = comp.reduced_formula
        # Check if it's actually a valid formula (has elements)
        if len(comp.elements) > 0:
            return (formula, "formula_parse")
    except:
        pass

    # Layer 4: PubChem API (slowest, but most comprehensive for chemical names)
    try:
        compounds = pcp.get_compounds(input_name, 'name', listkey_count=1)
        if compounds and compounds[0].molecular_formula:
            # Convert PubChem formula to reduced formula via pymatgen
            try:
                comp = Composition(compounds[0].molecular_formula)
                formula = comp.reduced_formula
                return (formula, "pubchem")
            except:
                # Use PubChem formula as-is if pymatgen fails
                return (compounds[0].molecular_formula, "pubchem_raw")
    except Exception as e:
        pass  # API failed, continue to fallback

    # Layer 5: Fallback - return original input with warning
    return (input_name, "fallback")


if __name__ == "__main__":
    # Test cases
    tests = [
        'gold', 'Au', 'GOLD',
        'silicon dioxide', 'SiO2',
        'iron oxide',
        'GaN', 'gallium nitride',
        'BaTiO3', 'barium titanate',
        'sapphire', 'graphene',
        'HfO2', 'hafnium oxide',
        'polystyrene',  # Should fail gracefully
        'unknown_material_xyz',  # Should fallback
    ]

    print("Testing material name parser:\n")
    for test in tests:
        formula, source = parse_material_name(test)
        status = "✓" if source != "fallback" else "⚠"
        print(f"{status} {test:25s} → {formula:15s} [{source}]")
