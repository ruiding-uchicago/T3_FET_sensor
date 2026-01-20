#!/usr/bin/env python3
"""
Offline Data Augmentation for FET Sensor Dataset (V3 Random Field Selection)

Generates augmented JSON files in descriptor_injected_jsons_augmented_v3/
Original files remain in descriptor_injected_jsons/

Augmentation strategies:
1. Discrete (topology/atmosphere):
   - S/D Swap - swap source and drain materials
   - Dual-gate Flip - flip top/bottom gate for dual_gate devices
   - Floating-gate Flip - flip gate/dielectric layers for floating_gate devices
   - Carrier Gas Replacement - replace N2 with Ar, He, or Ne
   - Annealing Atmosphere Replacement - replace inert atmospheres

2. Numerical perturbations (RANDOM FIELD SELECTION + 2^k combinations):
   - Each field has 50% probability of being selected for perturbation
   - Selected fields: Two levels (LOW and HIGH), full combinations
   - Non-selected fields: Keep original value

   Strategy: Global coverage with reduced per-DOI burden
   - Total files < 16k
   - All field types covered across dataset
   - Fixed random seed for reproducibility

   Fields:
   - pH: bio ±0.8, liquid ±1.0
   - Dielectric thickness: ×0.75 or ×1.25
   - Substrate thickness: ×0.70 or ×1.30
   - Annealing temperature: ×0.92 or ×1.08
   - Annealing time: ×0.75 or ×1.25
"""

import json
import copy
import re
import shutil
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Iterator
from datetime import datetime
from collections import defaultdict
from itertools import product

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR / "descriptor_injected_jsons"
OUTPUT_DIR = SCRIPT_DIR / "descriptor_injected_jsons_augmented_v3"
MOLECULES_CACHE = SCRIPT_DIR / "query_molecules" / "cache"

# V3 specific: Random field selection probability
FIELD_SELECTION_PROB = 0.15  # Each field has 15% chance to be selected for perturbation

# V3.1 NEW: Material descriptor perturbation (mild, stacked on augmented samples)
ENABLE_MATERIAL_PERTURBATION = True  # Enable material descriptor perturbation
MACRO_NOISE_STD = 0.08  # ±8% Gaussian noise for macro_vec (mild)
FINGERPRINT_FLIP_PROB = 0.08  # 8% bit flip probability (mild)

# Inert gas descriptors will be loaded from cache
INERT_GASES = {
    'argon': 'argon.json',
    'helium': 'helium.json',
    'neon': 'neon.json',
    'nitrogen': 'nitrogen.json'
}

# All inert gases that can be interchanged
INERT_GAS_NAMES = {
    'nitrogen': 'nitrogen',
    'n2': 'nitrogen',
    'argon': 'argon',
    'ar': 'argon',
    'helium': 'helium',
    'he': 'helium',
    'neon': 'neon',
    'ne': 'neon',
    'vacuum': 'vacuum',  # special case
    'air': 'air',  # treat as nitrogen equivalent
    'dry air': 'air',
    'dry_air': 'air',
}

# Inert atmospheres that can be swapped
INERT_ATMOSPHERES = set(INERT_GAS_NAMES.keys()) | {'inert atmosphere', 'nitrogen/argon'}

# Numerical perturbation levels (MILD - conservative physics-aware augmentation)
# V3.1: Gaussian sampling within these bounds (not fixed values)
# Further reduced ranges for gentler perturbations
PERTURBATION_LEVELS = {
    'temperature': {
        'gas': [0.92, 1.08],      # ×0.92, ×1.08 (±8%)
        'bio': [-4, +4],          # additive °C (±4°C)
        'liquid': [0.92, 1.08],   # ×0.92, ×1.08 (±8%)
    },
    'pH': {
        'gas': None,  # Not applicable
        'bio': [-0.6, +0.6],      # ±0.6
        'liquid': [-0.8, +0.8],   # ±0.8
    },
    'dielectric_thickness': [0.85, 1.15],  # ×0.85, ×1.15 (±15%)
    'substrate_thickness': [0.85, 1.15],   # ×0.85, ×1.15 (±15%)
    'annealing_temperature': [0.92, 1.08],  # ×0.92, ×1.08 (±8%)
    'annealing_time': [0.85, 1.15],  # ×0.85, ×1.15 (±15%)
}

# ============================================================================
# Load Descriptor Caches
# ============================================================================

def load_gas_descriptor(gas_name: str) -> Optional[Dict]:
    """Load complete gas descriptor from query_molecules cache."""
    cache_file = MOLECULES_CACHE / INERT_GASES.get(gas_name, f"{gas_name}.json")
    if not cache_file.exists():
        print(f"  Warning: Cache not found for {gas_name}")
        return None

    with open(cache_file) as f:
        cache_data = json.load(f)

    # Extract descriptor in the format used by enriched JSONs
    pubchem = cache_data.get('data', {}).get('pubchem', {}).get('properties', {})
    fingerprints = cache_data.get('data', {}).get('fingerprints', {})

    # Build macroproperties (25D)
    macroproperties = {
        "Charge": pubchem.get("Charge"),
        "aromatic_rings": None,
        "FeatureRingCount3D": pubchem.get("FeatureRingCount3D"),
        "FeatureCationCount3D": pubchem.get("FeatureCationCount3D"),
        "FeatureAnionCount3D": pubchem.get("FeatureAnionCount3D"),
        "TPSA": pubchem.get("TPSA"),
        "XLogP": pubchem.get("XLogP"),
        "HBondDonorCount": pubchem.get("HBondDonorCount"),
        "HBondAcceptorCount": pubchem.get("HBondAcceptorCount"),
        "FeatureDonorCount3D": pubchem.get("FeatureDonorCount3D"),
        "FeatureAcceptorCount3D": pubchem.get("FeatureAcceptorCount3D"),
        "MolecularWeight": _to_number(pubchem.get("MolecularWeight")),
        "HeavyAtomCount": pubchem.get("HeavyAtomCount"),
        "Volume3D": pubchem.get("Volume3D"),
        "XStericQuadrupole3D": pubchem.get("XStericQuadrupole3D"),
        "YStericQuadrupole3D": pubchem.get("YStericQuadrupole3D"),
        "ZStericQuadrupole3D": pubchem.get("ZStericQuadrupole3D"),
        "RotatableBondCount": pubchem.get("RotatableBondCount"),
        "EffectiveRotorCount3D": pubchem.get("EffectiveRotorCount3D"),
        "ConformerModelRMSD3D": pubchem.get("ConformerModelRMSD3D"),
        "Complexity": pubchem.get("Complexity"),
        "FeatureHydrophobeCount3D": pubchem.get("FeatureHydrophobeCount3D"),
        "FeatureCount3D": pubchem.get("FeatureCount3D"),
        "qed_weighted": None,
        "np_likeness_score": None
    }

    # Build fingerprint (256D + 64 nulls = 320D)
    fp_256 = fingerprints.get('bit_vector', [0] * 256)
    fp_320 = fp_256 + [None] * 64

    return {
        "name": gas_name,
        "canonical_name": gas_name,
        "substance_type": "molecules",
        "macroproperties": macroproperties,
        "fingerprint": fp_320
    }

def _to_number(value):
    """Convert string numbers to actual numbers."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return int(value) if '.' not in value else float(value)
        except:
            return None
    return value

# Pre-load gas descriptors
print("Loading inert gas descriptors from cache...")
GAS_DESCRIPTORS = {}
for gas_name in INERT_GASES:
    desc = load_gas_descriptor(gas_name)
    if desc:
        GAS_DESCRIPTORS[gas_name] = desc
        print(f"  Loaded {gas_name}")

# ============================================================================
# Utility: Parse and Modify Value Strings
# ============================================================================

def parse_value_with_unit(value_str: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Parse a string like "25 °C" or "300 nm" into (number, unit).
    Returns (None, None) if parsing fails.
    """
    if not isinstance(value_str, str):
        return None, None

    # Match number (with optional decimal, negative, scientific notation) followed by optional unit
    match = re.match(r'^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(.*)$', value_str.strip())
    if match:
        try:
            num = float(match.group(1))
            unit = match.group(2).strip() if match.group(2) else None
            return num, unit
        except ValueError:
            return None, None
    return None, None


def format_value_with_unit(num: float, unit: Optional[str]) -> str:
    """Format number with unit back to string."""
    if unit:
        return f"{num:.4g} {unit}"
    return f"{num:.4g}"


def perturb_value(value_str: str, factor_or_delta: float, mode: str = 'multiply') -> Optional[str]:
    """
    Perturb a value string.

    Args:
        value_str: Original value like "25 °C"
        factor_or_delta: Perturbation amount
        mode: 'multiply' (×factor) or 'add' (+delta)

    Returns:
        Perturbed value string, or None if parsing fails
    """
    num, unit = parse_value_with_unit(value_str)
    if num is None:
        return None

    if mode == 'multiply':
        new_num = num * factor_or_delta
    else:  # add
        new_num = num + factor_or_delta

    return format_value_with_unit(new_num, unit)


# ============================================================================
# Discrete Augmentation Functions
# ============================================================================

def has_different_sd(data: Dict) -> bool:
    """Check if any record has different source and drain."""
    for record in data.get('records', []):
        if record.get('source') != record.get('drain'):
            return True
    return False


def swap_source_drain(data: Dict) -> Optional[Dict]:
    """
    Swap source and drain materials in all records.

    Physics basis: FET S/D are physically symmetric (99.1% identical materials).
    Only measurement convention defines S/D.

    Returns None if S/D are identical (no point in swapping).
    """
    # Skip if all records have identical S/D
    if not has_different_sd(data):
        return None

    augmented = copy.deepcopy(data)

    for record in augmented.get('records', []):
        source = record.get('source')
        drain = record.get('drain')
        record['source'] = drain
        record['drain'] = source

    return augmented


def flip_dual_gate(data: Dict) -> Optional[Dict]:
    """
    Flip top/bottom gate for dual_gate devices.

    Physics basis: Top/bottom gates are capacitively equivalent.

    Returns None if no flippable dual_gate records found (need list with >=2 elements).
    """
    has_flipped = False
    augmented = copy.deepcopy(data)

    for record in augmented.get('records', []):
        design = record.get('structure_design_type', '')
        if design == 'dual_gate' or 'dual' in str(design).lower():
            # Swap gate layers if it's a list with >=2 elements
            gate = record.get('gate')
            if isinstance(gate, list) and len(gate) >= 2:
                record['gate'] = gate[::-1]  # Reverse the list
                has_flipped = True

            # Also swap dielectric if it's a list with >=2 elements
            dielectric = record.get('dielectric_layer')
            if isinstance(dielectric, list) and len(dielectric) >= 2:
                record['dielectric_layer'] = dielectric[::-1]
                has_flipped = True

    return augmented if has_flipped else None


def flip_floating_gate(data: Dict) -> Optional[Dict]:
    """
    Flip gate/dielectric layers for floating_gate devices with multi-layer stacks.

    Physics basis: Control gate and floating gate layers have symmetric capacitive coupling.

    Returns None if no flippable floating_gate records found.
    """
    has_flippable = False
    augmented = copy.deepcopy(data)

    for record in augmented.get('records', []):
        design = record.get('structure_design_type', '')
        if design == 'floating_gate' or 'floating' in str(design).lower():
            gate = record.get('gate')
            dielectric = record.get('dielectric_layer')

            # Only flip if both have multiple layers
            gate_flippable = isinstance(gate, list) and len(gate) >= 2
            dielectric_flippable = isinstance(dielectric, list) and len(dielectric) >= 2

            if gate_flippable or dielectric_flippable:
                has_flippable = True
                if gate_flippable:
                    record['gate'] = gate[::-1]
                if dielectric_flippable:
                    record['dielectric_layer'] = dielectric[::-1]

    return augmented if has_flippable else None


def replace_carrier_gas(data: Dict, new_gas: str) -> Optional[Dict]:
    """
    Replace carrier gas (in test_medium) with an equivalent inert gas.

    Physics basis: N2, Ar, He are all inert and don't participate in chemisorption.
    Only affect diffusion coefficient slightly.

    Returns None if not applicable (not a gas sensor or no nitrogen carrier).
    """
    if new_gas not in GAS_DESCRIPTORS:
        return None

    new_descriptor = GAS_DESCRIPTORS[new_gas]
    has_replacement = False
    augmented = copy.deepcopy(data)

    for record in augmented.get('records', []):
        # Only for gas sensors
        if record.get('sensor_type') != 'gas':
            continue

        test_medium = record.get('test_medium')
        if test_medium is None:
            continue

        # Check if test_medium contains nitrogen (carrier gas)
        if isinstance(test_medium, dict):
            name = test_medium.get('name', '').lower()
            if name in ['nitrogen', 'n2', 'air', 'dry air', 'dry_air']:
                record['test_medium'] = copy.deepcopy(new_descriptor)
                has_replacement = True
        elif isinstance(test_medium, list):
            for i, medium in enumerate(test_medium):
                name = medium.get('name', '').lower()
                if name in ['nitrogen', 'n2', 'air', 'dry air', 'dry_air']:
                    test_medium[i] = copy.deepcopy(new_descriptor)
                    has_replacement = True

    return augmented if has_replacement else None


def replace_annealing_atmosphere(data: Dict, new_atm: str) -> Optional[Dict]:
    """
    Replace annealing atmosphere with an equivalent inert atmosphere.

    Physics basis: N2, Ar, vacuum are all inert/non-reactive environments.
    They don't participate in chemical reactions during annealing.

    Returns None if no applicable records found.
    """
    if new_atm not in GAS_DESCRIPTORS:
        return None

    new_descriptor = GAS_DESCRIPTORS[new_atm]
    has_replacement = False
    augmented = copy.deepcopy(data)

    for record in augmented.get('records', []):
        atm = record.get('annealing_atmosphere')
        if atm is None:
            continue

        # Check if it's an inert atmosphere
        if isinstance(atm, dict):
            name = atm.get('name', '').lower() if atm.get('name') else ''
            if name in INERT_ATMOSPHERES and name != new_atm.lower():
                record['annealing_atmosphere'] = copy.deepcopy(new_descriptor)
                has_replacement = True
        elif isinstance(atm, list):
            for i, a in enumerate(atm):
                name = a.get('name', '').lower() if a.get('name') else ''
                if name in INERT_ATMOSPHERES and name != new_atm.lower():
                    atm[i] = copy.deepcopy(new_descriptor)
                    has_replacement = True

    return augmented if has_replacement else None


# ============================================================================
# Numerical Perturbation Functions
# ============================================================================

def get_sensor_type(data: Dict) -> str:
    """Get the primary sensor type from records."""
    for record in data.get('records', []):
        st = record.get('sensor_type', '')
        if st in ['gas', 'bio', 'liquid']:
            return st
    return 'gas'  # default


def parse_range_value(value_str: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Parse a range string like "7.4-7.4" or single value like "25 °C".
    Returns the first/mean value and unit.
    """
    if not isinstance(value_str, str):
        return None, None

    # Check if it's a range (e.g., "7.4-7.4" or "20-30 °C")
    range_match = re.match(r'^([-+]?\d*\.?\d+)\s*[-–]\s*([-+]?\d*\.?\d+)\s*(.*)$', value_str.strip())
    if range_match:
        try:
            num1 = float(range_match.group(1))
            num2 = float(range_match.group(2))
            unit = range_match.group(3).strip() if range_match.group(3) else None
            # Return mean of range
            return (num1 + num2) / 2, unit
        except ValueError:
            pass

    # Fall back to single value parsing
    return parse_value_with_unit(value_str)


def format_range_value(num: float, unit: Optional[str], original_format: str) -> str:
    """Format value back to string, preserving range format if original was a range."""
    # Check if original was a range
    if '-' in original_format or '–' in original_format:
        range_match = re.match(r'^([-+]?\d*\.?\d+)\s*[-–]\s*([-+]?\d*\.?\d+)', original_format.strip())
        if range_match:
            # Keep as range with same span
            orig_low = float(range_match.group(1))
            orig_high = float(range_match.group(2))
            span = orig_high - orig_low
            new_low = num - span / 2
            new_high = num + span / 2
            if unit:
                return f"{new_low:.4g}-{new_high:.4g} {unit}"
            return f"{new_low:.4g}-{new_high:.4g}"

    # Single value
    return format_value_with_unit(num, unit)


def has_perturbable_field(data: Dict, field: str) -> bool:
    """Check if data has a perturbable field with valid value."""
    for record in data.get('records', []):
        value = record.get(field)
        if isinstance(value, str):
            num, _ = parse_range_value(value)
            if num is not None and num > 0:
                return True
    return False


def is_ph_sensor(data: Dict) -> bool:
    """Check if this is a pH/H+ sensor (should not perturb pH)."""
    for record in data.get('records', []):
        target = str(record.get('detect_target', '')).lower()
        if 'ph' in target or 'h+' in target or 'hydrogen ion' in target:
            return True
    return False


def perturb_range_value(value_str: str, factor_or_delta: float, mode: str = 'multiply') -> Optional[str]:
    """
    Perturb a value string that might be a range (e.g., "7.4-7.4" or "2-12").
    """
    if not isinstance(value_str, str) or not value_str.strip():
        return None

    # Check if it's a range
    range_match = re.match(r'^([-+]?\d*\.?\d+)\s*[-–]\s*([-+]?\d*\.?\d+)\s*(.*)$', value_str.strip())
    if range_match:
        try:
            num1 = float(range_match.group(1))
            num2 = float(range_match.group(2))
            unit = range_match.group(3).strip() if range_match.group(3) else ''

            if mode == 'multiply':
                new_num1 = num1 * factor_or_delta
                new_num2 = num2 * factor_or_delta
            else:  # add
                new_num1 = num1 + factor_or_delta
                new_num2 = num2 + factor_or_delta

            if unit:
                return f"{new_num1:.4g}-{new_num2:.4g} {unit}"
            return f"{new_num1:.4g}-{new_num2:.4g}"
        except ValueError:
            pass

    # Fall back to single value
    return perturb_value(value_str, factor_or_delta, mode)


def apply_mild_noise_to_materials(materials: List[Dict], macro_std: float, fp_prob: float) -> List[Dict]:
    """
    Apply mild random noise to material descriptors (V3.1).

    This is stacked on top of existing augmentations to add material perturbation
    without increasing file count.

    Args:
        materials: List of material dictionaries
        macro_std: Noise std for macroproperties (±5%, mild)
        fp_prob: Bit flip probability for fingerprints (5%, mild)

    Returns:
        Noisy material list
    """
    noisy_materials = []
    for mat in materials:
        noisy_mat = copy.deepcopy(mat)

        # 1. Macroproperties - handle both formats
        # Format 1: macro_vec (25D vector) - for molecules/materials
        macro_vec = noisy_mat.get('macro_vec', [])
        if macro_vec and len(macro_vec) == 25:
            noisy_macro = []
            for val in macro_vec:
                if val is not None and isinstance(val, (int, float)):
                    noise = np.random.normal(0, macro_std * abs(val)) if val != 0 else np.random.normal(0, macro_std)
                    noisy_macro.append(val + noise)
                else:
                    noisy_macro.append(val)
            noisy_mat['macro_vec'] = noisy_macro

        # Format 2: macroproperties dict - for DNA/RNA
        macro_dict = noisy_mat.get('macroproperties', {})
        if isinstance(macro_dict, dict) and macro_dict:
            noisy_macro_dict = {}
            for key, val in macro_dict.items():
                if val is not None and isinstance(val, (int, float)):
                    noise = np.random.normal(0, macro_std * abs(val)) if val != 0 else np.random.normal(0, macro_std)
                    noisy_macro_dict[key] = val + noise
                else:
                    noisy_macro_dict[key] = val
            noisy_mat['macroproperties'] = noisy_macro_dict

        # 2. Fingerprints - handle both formats
        fingerprint = noisy_mat.get('fingerprint', [])
        if fingerprint:
            # Format 1: 320D binary vector (0/1) - for molecules/materials
            if len(fingerprint) == 320 and all(isinstance(x, (int, float)) and x in [0, 1, None] for x in fingerprint[:10]):
                noisy_fp = []
                for bit in fingerprint:
                    if bit is not None and np.random.rand() < fp_prob:
                        # Flip bit: 0→1, 1→0
                        noisy_fp.append(1 - bit if bit in [0, 1] else bit)
                    else:
                        noisy_fp.append(bit)
                noisy_mat['fingerprint'] = noisy_fp

            # Format 2: Embedding vector (floats) - for DNA/RNA
            elif len(fingerprint) > 0 and isinstance(fingerprint[0], float):
                noisy_fp = []
                for val in fingerprint:
                    if val is not None and isinstance(val, (int, float)):
                        # Add Gaussian noise to embedding
                        noise = np.random.normal(0, macro_std * abs(val)) if val != 0 else np.random.normal(0, macro_std * 0.1)
                        noisy_fp.append(val + noise)
                    else:
                        noisy_fp.append(val)
                noisy_mat['fingerprint'] = noisy_fp

        noisy_materials.append(noisy_mat)

    return noisy_materials


def apply_numerical_perturbation(data: Dict, perturbations: Dict[str, float]) -> Dict:
    """
    Apply numerical perturbations to data.

    Args:
        data: Original JSON data
        perturbations: Dict mapping field names to (factor, mode) tuples
            e.g. {'temperature': (1.08, 'multiply'), 'pH_value': (0.8, 'add')}

    Returns:
        Perturbed data copy
    """
    augmented = copy.deepcopy(data)

    for record in augmented.get('records', []):
        # Temperature perturbation (operating_temperature)
        if 'temperature' in perturbations:
            factor, mode = perturbations['temperature']
            temp_val = record.get('operating_temperature')
            if isinstance(temp_val, str) and temp_val.strip():
                new_val = perturb_value(temp_val, factor, mode)
                if new_val:
                    record['operating_temperature'] = new_val

        # pH perturbation (pH_value field, skip for gas sensors)
        if 'pH_value' in perturbations and record.get('sensor_type') != 'gas':
            delta, mode = perturbations['pH_value']
            ph_val = record.get('pH_value')
            # Skip if pH is -1 (gas sensor code) or empty
            if isinstance(ph_val, str) and ph_val.strip() and ph_val.strip() != '-1':
                new_val = perturb_range_value(ph_val, delta, mode)
                if new_val:
                    # Enforce 0-14 range for each number in range
                    range_match = re.match(r'^([-+]?\d*\.?\d+)\s*[-–]\s*([-+]?\d*\.?\d+)\s*(.*)$', new_val.strip())
                    if range_match:
                        n1 = max(0, min(14, float(range_match.group(1))))
                        n2 = max(0, min(14, float(range_match.group(2))))
                        unit = range_match.group(3).strip()
                        new_val = f"{n1:.4g}-{n2:.4g} {unit}".strip()
                    else:
                        num, unit = parse_value_with_unit(new_val)
                        if num is not None:
                            num = max(0, min(14, num))
                            new_val = format_value_with_unit(num, unit)
                    record['pH_value'] = new_val

        # Dielectric thickness perturbation (dielectric_layer_thickness field)
        if 'dielectric_thickness' in perturbations:
            factor, mode = perturbations['dielectric_thickness']
            diel_thick = record.get('dielectric_layer_thickness')
            if isinstance(diel_thick, str) and diel_thick.strip():
                new_val = perturb_value(diel_thick, factor, mode)
                if new_val:
                    record['dielectric_layer_thickness'] = new_val

        # Substrate thickness perturbation
        if 'substrate_thickness' in perturbations:
            factor, mode = perturbations['substrate_thickness']
            sub_thick = record.get('substrate_thickness')
            if isinstance(sub_thick, str) and sub_thick.strip():
                new_val = perturb_value(sub_thick, factor, mode)
                if new_val:
                    record['substrate_thickness'] = new_val

        # Annealing temperature perturbation
        if 'annealing_temperature' in perturbations:
            factor, mode = perturbations['annealing_temperature']
            anneal_temp = record.get('annealing_temperature')
            if isinstance(anneal_temp, str) and anneal_temp.strip():
                new_val = perturb_value(anneal_temp, factor, mode)
                if new_val:
                    record['annealing_temperature'] = new_val

        # Annealing time perturbation
        if 'annealing_time' in perturbations:
            factor, mode = perturbations['annealing_time']
            anneal_time = record.get('annealing_time')
            if isinstance(anneal_time, str) and anneal_time.strip():
                new_val = perturb_value(anneal_time, factor, mode)
                if new_val:
                    record['annealing_time'] = new_val

    # Material perturbation will be applied in save_augmented() to avoid duplication
    return augmented


def has_valid_ph(data: Dict) -> bool:
    """Check if data has valid pH_value (not -1, not empty, bio/liquid only)."""
    for record in data.get('records', []):
        if record.get('sensor_type') == 'gas':
            continue
        ph_val = record.get('pH_value')
        if isinstance(ph_val, str) and ph_val.strip() and ph_val.strip() != '-1':
            # Try to parse it
            num, _ = parse_range_value(ph_val)
            if num is not None and num >= 0:
                return True
    return False


def has_valid_thickness(data: Dict, field: str) -> bool:
    """Check if data has valid thickness field."""
    for record in data.get('records', []):
        val = record.get(field)
        if isinstance(val, str) and val.strip():
            num, _ = parse_value_with_unit(val)
            if num is not None and num > 0:
                return True
    return False


def generate_perturbation_combinations(data: Dict) -> Iterator[Tuple[str, Dict]]:
    """
    Generate perturbation combinations with GAUSSIAN NOISE within physical bounds (V3.1).

    Strategy:
    - Each field has FIELD_SELECTION_PROB (15%) chance to be selected
    - Selected fields: Gaussian noise within physical bounds (not fixed extremes)
    - Non-selected fields: Not perturbed (keep original value)
    - Each DOI generates 2 variants (if any field is selected)

    This provides smooth perturbations while respecting physical constraints.

    Yields: (suffix, perturbation_dict) tuples
    """
    sensor_type = get_sensor_type(data)
    is_ph = is_ph_sensor(data)

    # Check which fields are perturbable (using correct field names!)
    has_temp = has_perturbable_field(data, 'operating_temperature')
    has_ph = not is_ph and has_valid_ph(data)
    has_diel = has_valid_thickness(data, 'dielectric_layer_thickness')
    has_sub = has_valid_thickness(data, 'substrate_thickness')
    has_anneal_temp = has_perturbable_field(data, 'annealing_temperature')
    has_anneal_time = has_perturbable_field(data, 'annealing_time')

    # V3.1: Random field selection - store selected fields with their bounds
    selected_fields = []

    if has_temp and random.random() < FIELD_SELECTION_PROB:
        temp_bounds = PERTURBATION_LEVELS['temperature'].get(sensor_type, [])
        if temp_bounds:
            mode = 'add' if sensor_type == 'bio' else 'multiply'
            selected_fields.append(('t', 'temperature', temp_bounds, mode))

    if has_ph and random.random() < FIELD_SELECTION_PROB:
        ph_bounds = PERTURBATION_LEVELS['pH'].get(sensor_type, [])
        if ph_bounds:
            selected_fields.append(('p', 'pH_value', ph_bounds, 'add'))

    if has_diel and random.random() < FIELD_SELECTION_PROB:
        diel_bounds = PERTURBATION_LEVELS['dielectric_thickness']
        selected_fields.append(('d', 'dielectric_thickness', diel_bounds, 'multiply'))

    if has_sub and random.random() < FIELD_SELECTION_PROB:
        sub_bounds = PERTURBATION_LEVELS.get('substrate_thickness', [0.75, 1.25])
        selected_fields.append(('s', 'substrate_thickness', sub_bounds, 'multiply'))

    if has_anneal_temp and random.random() < FIELD_SELECTION_PROB:
        anneal_temp_bounds = PERTURBATION_LEVELS['annealing_temperature']
        selected_fields.append(('at', 'annealing_temperature', anneal_temp_bounds, 'multiply'))

    if has_anneal_time and random.random() < FIELD_SELECTION_PROB:
        anneal_time_bounds = PERTURBATION_LEVELS['annealing_time']
        selected_fields.append(('ai', 'annealing_time', anneal_time_bounds, 'multiply'))

    # No fields selected for perturbation - skip this DOI
    if not selected_fields:
        return

    # V3.1: Generate 2 variants per DOI (Gaussian sampling within bounds)
    # This maintains similar data volume as before while using continuous perturbations
    num_variants = 2

    for variant_id in range(num_variants):
        perturbations = {}
        suffix_parts = []

        for key, field_name, bounds, mode in selected_fields:
            # Sample from Gaussian within physical bounds
            lower, upper = bounds

            # For multiply mode: mean=1.0, std chosen so ±2σ covers [lower, upper]
            # For add mode: mean=0, std chosen so ±2σ covers [lower, upper]
            if mode == 'multiply':
                mean = 1.0
                std = (upper - lower) / 4.0  # ±2σ ≈ full range
                value = np.random.normal(mean, std)
                value = np.clip(value, lower, upper)
            else:  # add mode
                mean = 0.0
                std = (upper - lower) / 4.0  # ±2σ ≈ full range
                value = np.random.normal(mean, std)
                value = np.clip(value, lower, upper)

            perturbations[field_name] = (value, mode)
            suffix_parts.append(f"{key}{variant_id}")

        suffix = "_".join(suffix_parts)
        yield suffix, perturbations


# ============================================================================
# Main Pipeline
# ============================================================================

def get_doi_stem(filepath: Path) -> str:
    """Extract DOI stem from filepath (without .json extension)."""
    return filepath.stem


def save_augmented(data: Dict, original_path: Path, suffix: str, output_dir: Path):
    """Save augmented JSON with suffix to output directory."""
    stem = get_doi_stem(original_path)
    new_path = output_dir / f"{stem}_{suffix}.json"

    # V3.1 NEW: Apply material perturbation before saving (stacked on augmented data)
    if ENABLE_MATERIAL_PERTURBATION:
        # Apply material noise to all records
        for record in data.get('records', []):
            material_fields = ['channel', 'dielectric_layer', 'gate', 'source', 'drain',
                             'substrate', 'probe_material', 'surface_functionalization',
                             'annealing_atmosphere', 'detect_target', 'test_medium']

            for field in material_fields:
                material = record.get(field)
                if material and isinstance(material, dict):
                    # Apply mild noise to this material
                    noisy_materials = apply_mild_noise_to_materials([material], MACRO_NOISE_STD, FINGERPRINT_FLIP_PROB)
                    record[field] = noisy_materials[0]

    # Add augmentation metadata
    data['_augmentation'] = suffix
    data['_original_doi'] = data.get('DOI', stem)

    with open(new_path, 'w') as f:
        json.dump(data, f, indent=2)

    return new_path


def process_discrete_augmentations(data: Dict, filepath: Path, output_dir: Path) -> Dict[str, int]:
    """Process discrete augmentations (topology + atmosphere)."""
    counts = defaultdict(int)
    doi = data.get('DOI', filepath.stem)

    # Check properties of records
    has_gas_sensor = any(r.get('sensor_type') == 'gas' for r in data.get('records', []))
    has_dual_gate = any('dual' in str(r.get('structure_design_type', '')).lower() for r in data.get('records', []))
    has_floating_gate = any('floating' in str(r.get('structure_design_type', '')).lower() for r in data.get('records', []))

    def check_inert_atm(r):
        atm = r.get('annealing_atmosphere')
        if isinstance(atm, dict):
            name = atm.get('name')
            if name:
                return name.lower() in INERT_ATMOSPHERES
        return False

    has_inert_atm = any(check_inert_atm(r) for r in data.get('records', []))

    # 1. S/D Swap (only if S/D are different)
    sd_swapped = swap_source_drain(data)
    if sd_swapped:
        save_augmented(sd_swapped, filepath, 'sd_swap', output_dir)
        counts['sd_swap'] += 1

    # 2. Dual-gate flip
    if has_dual_gate:
        dg_flipped = flip_dual_gate(data)
        if dg_flipped:
            save_augmented(dg_flipped, filepath, 'dual_gate_flip', output_dir)
            counts['dual_gate_flip'] += 1

    # 3. Floating-gate flip
    if has_floating_gate:
        fg_flipped = flip_floating_gate(data)
        if fg_flipped:
            save_augmented(fg_flipped, filepath, 'floating_gate_flip', output_dir)
            counts['floating_gate_flip'] += 1

    # 4. Carrier gas replacement (gas sensors only)
    if has_gas_sensor:
        for gas in ['argon', 'helium', 'neon']:
            gas_replaced = replace_carrier_gas(data, gas)
            if gas_replaced:
                save_augmented(gas_replaced, filepath, f'carrier_{gas}', output_dir)
                counts[f'carrier_{gas}'] += 1

    # 5. Annealing atmosphere replacement
    if has_inert_atm:
        # Determine current atmosphere
        for rec in data.get('records', []):
            atm = rec.get('annealing_atmosphere')
            if isinstance(atm, dict) and atm.get('name'):
                current_atm = atm.get('name').lower()
                break
        else:
            current_atm = ''

        # Normalize current atmosphere name
        current_normalized = INERT_GAS_NAMES.get(current_atm, current_atm)

        # All possible replacements
        all_inert = ['nitrogen', 'argon', 'helium', 'neon']

        # For vacuum/air, treat as nitrogen-equivalent, replace with Ar/He/Ne
        if current_normalized in ['vacuum', 'air']:
            alternatives = ['argon', 'helium', 'neon']
        else:
            # Replace with all others except self
            alternatives = [g for g in all_inert if g != current_normalized]

        for alt_atm in alternatives:
            atm_replaced = replace_annealing_atmosphere(data, alt_atm)
            if atm_replaced:
                save_augmented(atm_replaced, filepath, f'anneal_{alt_atm}', output_dir)
                counts[f'anneal_{alt_atm}'] += 1

    return counts


def process_numerical_augmentations(data: Dict, filepath: Path, output_dir: Path) -> Dict[str, int]:
    """Process numerical perturbation augmentations (full combination)."""
    counts = defaultdict(int)

    for suffix, perturbations in generate_perturbation_combinations(data):
        augmented = apply_numerical_perturbation(data, perturbations)
        save_augmented(augmented, filepath, f'num_{suffix}', output_dir)
        counts[f'numerical_{suffix}'] += 1

    return counts


def main():
    print("=" * 80)
    print("Offline Data Augmentation for FET Sensor Dataset (V3 Physics-Aware)")
    print("=" * 80)
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Field Selection Probability: {FIELD_SELECTION_PROB*100:.0f}%")
    print(f"Perturbation Strategy: Gaussian within physical bounds + material perturbation (stacked)")
    print(f"Numerical: Gaussian noise within bounds (not fixed extremes)")
    print(f"Material Perturbation: ±{MACRO_NOISE_STD*100:.0f}% macro_vec, {FINGERPRINT_FLIP_PROB*100:.0f}% fingerprint flip")
    print(f"Target: ~6-7k total files")
    print(f"\nSource Directory: {SOURCE_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Find all original JSON files (exclude already augmented ones)
    json_files = sorted(SOURCE_DIR.glob("*.json"))
    original_files = [f for f in json_files if not any(
        suffix in f.stem for suffix in ['_sd_swap', '_dual_gate_flip', '_floating_gate_flip',
                                         '_carrier_', '_anneal_', '_num_']
    ) and f.name != 'TUTORIAL_FOR_AGENTS.md']

    # Step 0: Copy original files to output directory
    print(f"\nCopying {len(original_files)} original files to output directory...")
    for filepath in original_files:
        dest = OUTPUT_DIR / filepath.name
        if not dest.exists():
            shutil.copy2(filepath, dest)
    print(f"  Done.")

    # Re-read for processing
    json_files = sorted(SOURCE_DIR.glob("*.json"))
    original_files = [f for f in json_files if not any(
        suffix in f.stem for suffix in ['_sd_swap', '_dual_gate_flip', '_floating_gate_flip',
                                         '_carrier_', '_anneal_', '_num_']
    ) and f.name != 'TUTORIAL_FOR_AGENTS.md']

    print(f"Found {len(original_files)} original JSON files")

    # Process each file
    discrete_counts = defaultdict(int)
    numerical_counts = defaultdict(int)
    processed = 0

    for filepath in original_files:
        with open(filepath) as f:
            original_data = json.load(f)

        # Skip if this is already an augmented file
        if '_augmentation' in original_data:
            continue

        # Process discrete augmentations
        d_counts = process_discrete_augmentations(original_data, filepath, OUTPUT_DIR)
        for k, v in d_counts.items():
            discrete_counts[k] += v

        # Process numerical augmentations
        n_counts = process_numerical_augmentations(original_data, filepath, OUTPUT_DIR)
        for k, v in n_counts.items():
            numerical_counts[k] += v

        processed += 1

        if processed % 100 == 0:
            total_so_far = sum(discrete_counts.values()) + sum(numerical_counts.values())
            print(f"  Processed {processed}/{len(original_files)} files... ({total_so_far} augmented files generated)")

    # Summary
    print("\n" + "=" * 80)
    print("Augmentation Summary")
    print("=" * 80)
    print(f"\nOriginal files (unchanged in source): {len(original_files)}")

    print(f"\n--- Discrete Augmentations ---")
    discrete_total = 0
    for aug_type, count in sorted(discrete_counts.items()):
        print(f"  {aug_type}: {count}")
        discrete_total += count
    print(f"  Subtotal: {discrete_total}")

    print(f"\n--- Numerical Perturbation Augmentations ---")
    # Group numerical counts
    numerical_total = sum(numerical_counts.values())
    print(f"  Total numerical combinations: {numerical_total}")

    # Show breakdown by perturbation type
    field_counts = defaultdict(int)
    for key, count in numerical_counts.items():
        # Parse which fields were perturbed
        if '_t0' in key or '_t2' in key or key.startswith('numerical_t0') or key.startswith('numerical_t2'):
            field_counts['temperature_perturbed'] += count
        if '_p0' in key or '_p2' in key:
            field_counts['pH_value_perturbed'] += count
        if '_d0' in key or '_d2' in key:
            field_counts['dielectric_thickness_perturbed'] += count
        if '_s0' in key or '_s2' in key:
            field_counts['substrate_thickness_perturbed'] += count
        if '_at0' in key or '_at2' in key:
            field_counts['anneal_temp_perturbed'] += count
        if '_ai0' in key or '_ai2' in key:
            field_counts['anneal_time_perturbed'] += count

    for field, count in sorted(field_counts.items()):
        print(f"    (includes {field}: {count})")

    total_augmented = discrete_total + numerical_total
    print(f"\n--- Total ---")
    print(f"Total augmented files: {total_augmented}")
    print(f"Original files: {len(original_files)}")
    print(f"Expansion factor (augmented only): {total_augmented / len(original_files):.2f}x")
    print(f"Total effective dataset (orig + aug): {len(original_files) + total_augmented}")

    # Save log
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'source_dir': str(SOURCE_DIR),
        'output_dir': str(OUTPUT_DIR),
        'original_files': len(original_files),
        'discrete_augmentation_counts': dict(discrete_counts),
        'numerical_augmentation_total': numerical_total,
        'numerical_field_involvement': dict(field_counts),
        'total_augmented_files': total_augmented,
        'expansion_factor': total_augmented / len(original_files),
        'effective_dataset_size': len(original_files) + total_augmented,
        'strategies': {
            'discrete': {
                'sd_swap': {
                    'description': 'Swap source and drain materials',
                    'physics_basis': 'FET S/D are physically symmetric (99.1% identical materials)',
                    'risk': 'zero'
                },
                'dual_gate_flip': {
                    'description': 'Flip top/bottom gate for dual_gate devices',
                    'physics_basis': 'Top/bottom gates are capacitively equivalent',
                    'risk': 'low'
                },
                'floating_gate_flip': {
                    'description': 'Flip gate/dielectric layers for floating_gate devices',
                    'physics_basis': 'Control gate and floating gate layers have symmetric capacitive coupling',
                    'risk': 'low'
                },
                'carrier_gas': {
                    'description': 'Replace carrier gas (N2) with Ar, He, or Ne using real descriptors',
                    'physics_basis': 'N2/Ar/He/Ne are all inert, only affect diffusion coefficient',
                    'risk': 'low'
                },
                'annealing_atmosphere': {
                    'description': 'Replace annealing atmosphere with equivalent inert gas',
                    'physics_basis': 'N2/Ar/He/Ne/vacuum are all inert, prevent oxidation equally well',
                    'risk': 'low'
                }
            },
            'numerical': {
                'temperature': {
                    'description': 'Perturb operating temperature',
                    'levels': PERTURBATION_LEVELS['temperature'],
                    'physics_basis': 'Arrhenius kinetics: ±8% change in T causes <10× response change',
                    'risk': 'low'
                },
                'pH': {
                    'description': 'Perturb test pH for bio/liquid sensors',
                    'levels': PERTURBATION_LEVELS['pH'],
                    'physics_basis': 'Surface potential: ψ ≈ 59mV × ΔpH, small changes negligible',
                    'risk': 'low',
                    'skip_for': 'gas sensors and pH sensors'
                },
                'dielectric_thickness': {
                    'description': 'Perturb dielectric layer thickness',
                    'levels': PERTURBATION_LEVELS['dielectric_thickness'],
                    'physics_basis': 'gm ∝ 1/d, 25% thickness change → ~20% LDL change',
                    'risk': 'low-medium'
                },
                'annealing_temperature': {
                    'description': 'Perturb annealing temperature',
                    'levels': PERTURBATION_LEVELS['annealing_temperature'],
                    'physics_basis': 'Grain size ∝ exp(-Ea/kT), ±8% change is within process tolerance',
                    'risk': 'low'
                },
                'annealing_time': {
                    'description': 'Perturb annealing time',
                    'levels': PERTURBATION_LEVELS['annealing_time'],
                    'physics_basis': 'Grain growth ~t^0.5, ±25% change is within process tolerance',
                    'risk': 'low'
                }
            }
        }
    }

    log_path = Path("offline_augmentation_v2_log.json")
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"\nLog saved to {log_path}")


if __name__ == "__main__":
    main()
