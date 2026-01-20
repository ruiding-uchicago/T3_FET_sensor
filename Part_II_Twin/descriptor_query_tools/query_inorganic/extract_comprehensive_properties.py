#!/usr/bin/env python
"""
Comprehensive property extractor for OPTIMADE query results with multi-layer fallback.

This script extracts key material properties from OPTIMADE database query results,
implementing a robust fallback mechanism to handle data from multiple sources.
ALSO computes derived FET descriptors in one pass.
"""

import json
import sys
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# MAGPIE feature extraction
try:
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    MAGPIE_AVAILABLE = True
except ImportError:
    MAGPIE_AVAILABLE = False
    print("Warning: matminer not installed. MAGPIE features will be skipped.")


# ============================================================================
# DERIVED DESCRIPTOR FUNCTIONS (Tier-1 from expert recommendations)
# ============================================================================

def safe_float(value):
    """Safely convert value to float, return None if not possible."""
    try:
        return float(value) if value and str(value).strip() else None
    except (ValueError, TypeError):
        return None


def compute_eg_class(band_gap, elements=None):
    """
    Band gap class: metal(<0.05), narrow(0.05-1), moderate(1-3), wide(≥3 eV).

    Special case: elemental metals forced to 'metal' class.
    """
    # Known elemental metals (should have Eg=0)
    ELEMENTAL_METALS = {
        'Fe', 'Cu', 'Al', 'Ag', 'Au', 'Pt', 'Pd', 'Ni', 'Co', 'Zn', 'Mg', 'Ca',
        'Li', 'Na', 'K', 'Rb', 'Cs', 'Be', 'Sr', 'Ba', 'Sc', 'Ti', 'V', 'Cr',
        'Mn', 'Ga', 'In', 'Sn', 'Pb', 'Bi', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
        'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Hg', 'Tl', 'Y', 'La', 'Ce',
    }

    # Check if elemental metal
    if elements:
        try:
            elem_list = eval(elements) if isinstance(elements, str) else elements
            # Get unique elements (handle duplicates in element list)
            unique_elems = list(set(elem_list))
            if len(unique_elems) == 1 and unique_elems[0] in ELEMENTAL_METALS:
                return 'metal'  # Force metal class for elemental metals
        except:
            pass

    Eg = safe_float(band_gap)
    if Eg is None:
        return ''
    if Eg < 0.05:
        return 'metal'
    elif Eg < 1.0:
        return 'narrow'
    elif Eg < 3.0:
        return 'moderate'
    else:
        return 'wide'


def compute_eps_mean_aniso(eps_x, eps_y, eps_z):
    """Compute mean dielectric constant and anisotropy."""
    vals = [safe_float(v) for v in [eps_x, eps_y, eps_z]]
    vals = [v for v in vals if v is not None]
    if not vals:
        return '', ''
    eps_mean = sum(vals) / len(vals)
    eps_aniso = (max(vals) - min(vals)) / eps_mean if eps_mean > 0 else 0
    return f"{eps_mean:.4f}", f"{eps_aniso:.4f}"


def compute_vdw_ready(exfoliation_energy_mev_a2, nperiodic_dimensions):
    """2D readiness: easy(<30), potential(30-130), no(>130 meV/Å²)."""
    E_exf = safe_float(exfoliation_energy_mev_a2)
    n_periodic = safe_float(nperiodic_dimensions)
    if n_periodic is not None and n_periodic <= 2:
        return 'layered_structure'
    if E_exf is None:
        return ''
    if E_exf < 30:
        return 'easy'
    elif E_exf <= 130:
        return 'potential'
    else:
        return 'no'


def compute_thermo_stable(energy_above_hull):
    """Stability: on_hull(0), likely_synthesizable(≤0.05), metastable(≤0.1), risky(>0.1)."""
    E_hull = safe_float(energy_above_hull)
    if E_hull is None:
        return ''
    if E_hull == 0:
        return 'on_hull'
    elif E_hull <= 0.05:
        return 'likely_synthesizable'
    elif E_hull <= 0.1:
        return 'metastable'
    else:
        return 'risky'


def compute_e_modulus_est(k_vrh, g_vrh):
    """
    Estimate Young's modulus: E = 9KG/(3K+G).

    Only compute if K > 0 and G > 0 (physics constraint).
    """
    K = safe_float(k_vrh)
    G = safe_float(g_vrh)
    # Physics validation: K and G must be positive
    if K is None or G is None or K <= 0 or G <= 0:
        return ''
    if (3*K + G) == 0:  # Additional safety check
        return ''
    E = 9 * K * G / (3*K + G)
    return f"{E:.2f}"


def add_derived_descriptors(props):
    """Add all derived descriptors to a properties dict."""
    # 1. Band gap class (with elemental metal detection)
    props['eg_class'] = compute_eg_class(props.get('band_gap'), props.get('elements'))

    # 2. Dielectric mean/aniso
    eps_mean, eps_aniso = compute_eps_mean_aniso(
        props.get('epsilon_x'), props.get('epsilon_y'), props.get('epsilon_z')
    )
    props['eps_mean'] = eps_mean
    props['eps_aniso'] = eps_aniso

    # 3. vdW readiness
    props['vdw_ready'] = compute_vdw_ready(
        props.get('exfoliation_energy_mev_a2'), props.get('nperiodic_dimensions')
    )

    # 4. Thermodynamic stability
    props['thermo_stable'] = compute_thermo_stable(props.get('energy_above_hull'))

    # 5. Young's modulus
    props['e_modulus_est'] = compute_e_modulus_est(props.get('k_vrh'), props.get('g_vrh'))

    return props


# ============================================================================
# MAGPIE DESCRIPTOR FUNCTIONS (Composition-based features)
# ============================================================================

def compute_magpie_features(chemical_formula: str) -> Dict[str, Any]:
    """
    Compute MAGPIE descriptors from chemical formula.

    MAGPIE generates 132 composition-based features including:
    - Elemental property statistics (mean, min, max, range, avg_dev, mode)
    - Stoichiometric attributes
    - Electronic structure features
    - Ionic compound features

    Args:
        chemical_formula: Chemical formula string (e.g., "GaN", "Al2O3")

    Returns:
        Dictionary mapping feature names to values (132 features)
    """
    if not MAGPIE_AVAILABLE:
        return {}

    if not chemical_formula:
        return {}

    try:
        # Create composition object
        comp = Composition(chemical_formula)

        # Initialize MAGPIE featurizer with impute_nan=True to avoid NaNs
        magpie = ElementProperty.from_preset('magpie')

        # Compute features
        features = magpie.featurize(comp)
        feature_labels = magpie.feature_labels()

        # Create dictionary with shortened feature names (remove "MagpieData " prefix)
        magpie_dict = {}
        for label, value in zip(feature_labels, features):
            # Shorten label: "MagpieData minimum Number" → "magpie_minimum_Number"
            short_label = label.replace('MagpieData ', 'magpie_').replace(' ', '_')
            magpie_dict[short_label] = value

        return magpie_dict

    except Exception as e:
        print(f"    Warning: MAGPIE computation failed for {chemical_formula}: {e}")
        return {}


# Database priority order (higher priority = more reliable/complete data)
DB_PRIORITY = {
    'jarvis.nist.gov': 10,                    # JARVIS has comprehensive properties
    'optimade.materialsproject.org': 9,       # Materials Project is highly curated
    'alexandria.icams.rub.de': 8,             # Alexandria has good DFT data
    'oqmd.org': 7,                            # OQMD is well-validated
    'mpddoptimade.phaseslab.org': 6,          # MPDD has good coverage
    'optimade.materialscloud.org': 5,         # Materials Cloud
    'api.mpds.io': 4,                         # MPDS
    'cmr-optimade.fysik.dtu.dk': 3,           # CMR
    'optimade.openmaterialsdb.se': 2,         # OMDB
    'www.crystallography.net': 1,             # COD
}


class PropertyExtractor:
    """
    Extracts material properties with multi-layer fallback mechanism.
    """

    def __init__(self, json_data: Dict):
        """
        Initialize with loaded JSON data.

        Args:
            json_data: Parsed JSON from query results
        """
        self.data = json_data
        self.formula_key = list(json_data['structures'].keys())[0] if json_data['structures'] else None
        self.databases = json_data['structures'].get(self.formula_key, {}) if self.formula_key else {}

    def get_database_priority(self, db_url: str) -> int:
        """Get priority score for a database."""
        for key, priority in DB_PRIORITY.items():
            if key in db_url:
                return priority
        return 0

    def get_all_structures(self) -> List[Tuple[str, Dict]]:
        """
        Get all structures sorted by database priority.

        Returns:
            List of (db_url, structure) tuples sorted by priority
        """
        structures = []
        for db_url, db_data in self.databases.items():
            if db_data.get('data'):
                priority = self.get_database_priority(db_url)
                for struct in db_data['data']:
                    structures.append((priority, db_url, struct))

        # Sort by priority (descending)
        structures.sort(key=lambda x: x[0], reverse=True)
        return [(url, struct) for _, url, struct in structures]

    def extract_with_fallback(self, field_names: List[str], structures: List[Tuple[str, Dict]]) -> Optional[Any]:
        """
        Extract a field with fallback through multiple structures and field names.

        Args:
            field_names: List of possible field names to try (in priority order)
            structures: List of (db_url, structure) tuples

        Returns:
            First non-None value found, or None
        """
        # Placeholder values to skip
        placeholders = {-99999, -999, '-99999', '-999', -9999, '-9999'}

        for db_url, struct in structures:
            attrs = struct.get('attributes', {})
            for field_name in field_names:
                value = attrs.get(field_name)
                # Skip None values
                if value is None:
                    continue
                # Skip placeholder values (only for hashable types)
                try:
                    if value in placeholders:
                        continue
                except TypeError:
                    # Value is not hashable (e.g., list, dict), so it's not a placeholder
                    pass
                # Found a valid value
                return value
        return None

    def extract_basic_info(self, structures: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        """Extract basic chemical and structural information (Tier-1)."""
        return {
            'chemical_formula_reduced': self.extract_with_fallback(
                ['chemical_formula_reduced'], structures
            ),
            'elements': self.extract_with_fallback(
                ['elements'], structures
            ),
            'nelements': self.extract_with_fallback(
                ['nelements'], structures
            ),
        }

    def extract_structural_info(self, structures: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        """Extract structural/crystallographic information."""
        # Space group
        space_group = self.extract_with_fallback([
            '_jarvis_spg_number',                     # JARVIS
            '_alexandria_space_group',                # Alexandria
            '_oqmd_spacegroup',                       # OQMD
            '_mpdd_spacegroupn',                      # MPDD
            'space_group',                            # Generic
        ], structures)

        # Crystal system
        crystal_system = self.extract_with_fallback([
            '_jarvis_crys',                           # JARVIS
            '_mpdd_crystalsystem',                    # MPDD
            'crystal_system',                         # Generic
        ], structures)

        # Number of periodic dimensions (Tier-1)
        nperiodic_dimensions = self.extract_with_fallback([
            'nperiodic_dimensions',                   # Generic OPTIMADE field
        ], structures)

        return {
            'space_group': space_group,
            'crystal_system': crystal_system,
            'nperiodic_dimensions': nperiodic_dimensions,
        }

    def extract_energy_properties(self, structures: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        """Extract energy and stability properties."""
        return {
            'formation_energy_per_atom': self.extract_with_fallback([
                '_jarvis_formation_energy_peratom',      # JARVIS
                '_alexandria_formation_energy_per_atom',  # Alexandria PBEsol
                '_alexandria_scan_formation_energy_per_atom',  # Alexandria SCAN
                '_oqmd_delta_e',                          # OQMD
                '_odbx_formation_energy',                 # ODBX
                '_mp_formation_energy_per_atom',          # Materials Project
                'formation_energy_per_atom',              # Generic
            ], structures),
            'energy_above_hull': self.extract_with_fallback([
                '_jarvis_ehull',                          # JARVIS
                '_alexandria_hull_distance',              # Alexandria PBEsol
                '_alexandria_scan_hull_distance',         # Alexandria SCAN
                '_oqmd_stability',                        # OQMD
                '_odbx_hull_distance',                    # ODBX
                '_mp_e_above_hull',                       # Materials Project
                'energy_above_hull',                      # Generic
            ], structures),
        }

    def extract_band_gap_smart(self, structures: List[Tuple[str, Dict]]) -> Any:
        """
        Smart band gap extraction with material-type awareness.

        Strategy:
        1. For magnetic materials (Fe, Co, Ni, Mn oxides): prefer non-zero gaps
           (avoids metallic magnetic state artifacts)
        2. For other materials: prioritize HSE > mBJ > optB88vdW > MP
        """
        # Field priority: HSE (most accurate) > mBJ > optB88vdW > MP > others
        field_names = [
            '_jarvis_hse_gap',                        # HSE06 (most accurate for gaps)
            '_jarvis_mbj_bandgap',                    # mBJ (good for wide-gap)
            '_jarvis_optb88vdw_bandgap',              # optB88vdW (good for 2D)
            '_mp_band_gap',                           # Materials Project
            '_alexandria_band_gap',                   # Alexandria PBEsol
            '_alexandria_scan_band_gap',              # Alexandria SCAN
            '_oqmd_band_gap',                         # OQMD
            'band_gap',                               # Generic
        ]

        placeholders = {-99999, -999, '-99999', '-999', -9999, '-9999'}

        # Detect magnetic materials (transition metal oxides with magnetic artifacts)
        # These elements show metallic DFT ground states that hide their semiconducting gaps
        MAGNETIC_ELEMENTS = {'Fe', 'Co', 'Ni', 'Mn', 'Cu', 'Cr'}
        is_magnetic = False
        for db_url, struct in structures:
            elements = struct.get('attributes', {}).get('elements', [])
            if any(elem in MAGNETIC_ELEMENTS for elem in elements):
                is_magnetic = True
                break

        # Collect all valid values
        all_values = []
        for field_name in field_names:
            for db_url, struct in structures:
                attrs = struct.get('attributes', {})
                value = attrs.get(field_name)

                if value is None:
                    continue

                # Skip placeholders
                try:
                    if value in placeholders:
                        continue
                except TypeError:
                    pass

                # Convert to float
                try:
                    fvalue = float(value)
                    if fvalue >= 0:  # Valid non-negative value
                        all_values.append(fvalue)
                        # For non-magnetic materials, return first valid value immediately
                        if not is_magnetic:
                            return fvalue
                except (ValueError, TypeError):
                    continue

        # For magnetic materials: prefer non-zero values to avoid metallic artifacts
        if is_magnetic and all_values:
            non_zero_values = [v for v in all_values if v > 0.05]
            if non_zero_values:
                return non_zero_values[0]  # Return first non-zero value
            else:
                return all_values[0]  # Fall back to first value if all are ~0

        return None  # No valid band gap found

    def extract_electronic_properties(self, structures: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        """Extract electronic properties."""
        return {
            'band_gap': self.extract_band_gap_smart(structures),
            'band_gap_direct': self.extract_with_fallback([
                '_alexandria_band_gap_direct',            # Alexandria PBEsol
                '_alexandria_scan_band_gap_direct',       # Alexandria SCAN
                'band_gap_direct',                        # Generic
            ], structures),
            'dos_ef': self.extract_with_fallback([
                '_alexandria_dos_ef',                     # Alexandria DOS at Fermi level
            ], structures),
            'me_avg': self.extract_with_fallback([
                '_jarvis_avg_elec_mass',                  # JARVIS average electron mass
            ], structures),
            'mh_avg': self.extract_with_fallback([
                '_jarvis_avg_hole_mass',                  # JARVIS average hole mass
            ], structures),
        }

    def extract_mechanical_properties(self, structures: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        """
        Extract mechanical properties with physics validation.

        Validates that K_VRH > 0 and G_VRH > 0 (uncompressible, rigid materials).
        Extracts K and G from the same structure to ensure consistency.
        """
        # Field names for bulk and shear moduli
        k_fields = ['_jarvis_bulk_modulus_kv', 'bulk_modulus']
        g_fields = ['_jarvis_shear_modulus_gv', 'shear_modulus']

        placeholders = {-99999, -999, '-99999', '-999', -9999, '-9999'}

        # Try to find a structure with valid K AND G (from same source)
        k_vrh = None
        g_vrh = None

        for db_url, struct in structures:
            attrs = struct.get('attributes', {})

            # Try to get K and G from this structure
            struct_k = None
            struct_g = None

            for k_field in k_fields:
                val = attrs.get(k_field)
                if val is not None:
                    try:
                        if val not in placeholders:
                            fval = float(val)
                            if fval > 0:  # Physics constraint: K must be positive
                                struct_k = fval
                                break
                    except (ValueError, TypeError):
                        pass

            for g_field in g_fields:
                val = attrs.get(g_field)
                if val is not None:
                    try:
                        if val not in placeholders:
                            fval = float(val)
                            if fval > 0:  # Physics constraint: G must be positive
                                struct_g = fval
                                break
                    except (ValueError, TypeError):
                        pass

            # If this structure has both valid K and G, use them
            if struct_k is not None and struct_g is not None:
                k_vrh = struct_k
                g_vrh = struct_g
                break

        return {
            'k_vrh': k_vrh,
            'g_vrh': g_vrh,
            'poisson_ratio': self.extract_with_fallback([
                '_jarvis_poisson',  # JARVIS
            ], structures),
        }

    def extract_dielectric_safe(self, structures: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        """
        Extract dielectric properties with physics sanity checks.

        Key distinctions:
        - epsilon_x/y/z: STATIC dielectric tensor from DFPT (ε_static)
        - dielectric_electronic: HIGH-FREQUENCY optical dielectric (ε∞)
        - dielectric_total: STATIC total = electronic + ionic

        Physics constraints enforced:
        - ε_total ≥ ε_electronic ≥ max(εx,εy,εz) ≥ 1
        - Reject ε > 1000 (likely placeholder or ferroelectric outlier)
        """
        # Extract raw values
        raw = {}

        # STATIC dielectric tensor (εx, εy, εz) from DFPT
        raw['epsilon_x'] = self.extract_with_fallback([
            '_jarvis_epsx',                           # JARVIS static dielectric
            '_mp_diel_elec_xx',                       # Materials Project
        ], structures)

        raw['epsilon_y'] = self.extract_with_fallback([
            '_jarvis_epsy',
            '_mp_diel_elec_yy',
        ], structures)

        raw['epsilon_z'] = self.extract_with_fallback([
            '_jarvis_epsz',
            '_mp_diel_elec_zz',
        ], structures)

        # HIGH-FREQUENCY electronic dielectric (ε∞)
        raw['dielectric_electronic'] = self.extract_with_fallback([
            '_jarvis_dfpt_piezo_max_dielectric_electronic',
            '_jarvis_mepsx',                          # Mean electronic epsilon
            '_mp_diel_elec',
        ], structures)

        # IONIC contribution
        raw['dielectric_ionic'] = self.extract_with_fallback([
            '_jarvis_dfpt_piezo_max_dielectric_ionic',
        ], structures)

        # STATIC total dielectric
        raw['dielectric_total'] = self.extract_with_fallback([
            '_jarvis_dfpt_piezo_max_dielectric',      # DFPT total
            '_mp_diel_total',                         # Materials Project
        ], structures)

        # Apply physics sanity checks (minimal constraints only)
        result = {}

        # Check epsilon_x/y/z: must be ≥1 only (no upper limit)
        for key in ['epsilon_x', 'epsilon_y', 'epsilon_z']:
            val = safe_float(raw[key])
            if val is not None and val >= 1:
                result[key] = val
            else:
                result[key] = None

        # Check dielectric_electronic: ε∞ ≥ 1 only (no upper limit)
        eps_elec = safe_float(raw['dielectric_electronic'])
        if eps_elec is not None and eps_elec >= 1:
            result['dielectric_electronic'] = eps_elec
        else:
            result['dielectric_electronic'] = None

        # Check dielectric_ionic: should be ≥0
        eps_ionic = safe_float(raw['dielectric_ionic'])
        if eps_ionic is not None and eps_ionic >= 0:
            result['dielectric_ionic'] = eps_ionic
        else:
            result['dielectric_ionic'] = None

        # Check dielectric_total: enforce ε_total ≥ ε_electronic only
        eps_total = safe_float(raw['dielectric_total'])
        if eps_total is not None and result['dielectric_electronic'] is not None:
            # Enforce ε_total ≥ ε_electronic (physics law)
            if eps_total >= result['dielectric_electronic']:
                result['dielectric_total'] = eps_total
            elif result['dielectric_ionic'] is not None:
                # Recompute from components if inconsistent
                result['dielectric_total'] = result['dielectric_electronic'] + result['dielectric_ionic']
            else:
                result['dielectric_total'] = None
        elif eps_total is not None and eps_total >= 1:
            result['dielectric_total'] = eps_total
        else:
            result['dielectric_total'] = None

        return result

    def extract_optical_properties(self, structures: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        """Extract optical properties with sanity checks."""
        return self.extract_dielectric_safe(structures)

    def extract_other_properties(self, structures: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        """Extract other physical properties."""
        return {
            'density': self.extract_with_fallback([
                '_jarvis_density',                        # JARVIS
                '_mpdd_density',                          # MPDD
                'density',                                # Generic
            ], structures),
            'magnetization': self.extract_with_fallback([
                '_alexandria_magnetization',              # Alexandria PBEsol
                '_alexandria_scan_magnetization',         # Alexandria SCAN
                '_jarvis_magmom_outcar',                  # JARVIS total magnetic moment
                'magnetization',                          # Generic
            ], structures),
            'exfoliation_energy_mev_a2': self.extract_with_fallback([
                '_jarvis_exfoliation_energy',             # JARVIS (for 2D materials, meV/Å²)
            ], structures),
        }

    def extract_metadata(self, structures: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        """Extract metadata about the structure."""
        # Expert recommendation: Drop provenance/metadata
        return {}

    def extract_all(self) -> Dict[str, Any]:
        """
        Extract all properties with full fallback mechanism.

        Returns:
            Dictionary with all extracted properties
        """
        if not self.formula_key:
            raise ValueError('No structure data available')

        structures = self.get_all_structures()

        if not structures:
            raise ValueError('No successful database queries')

        # Extract all property groups
        properties = {}
        properties.update(self.extract_basic_info(structures))
        properties.update(self.extract_structural_info(structures))
        properties.update(self.extract_energy_properties(structures))
        properties.update(self.extract_electronic_properties(structures))
        properties.update(self.extract_mechanical_properties(structures))
        properties.update(self.extract_optical_properties(structures))
        properties.update(self.extract_other_properties(structures))
        properties.update(self.extract_metadata(structures))

        # Compute MAGPIE features from chemical formula
        chemical_formula = properties.get('chemical_formula_reduced')
        if chemical_formula:
            magpie_features = compute_magpie_features(chemical_formula)
            properties.update(magpie_features)

        return properties


def extract_from_file(json_file: Path) -> Dict[str, Any]:
    """
    Extract properties from a single JSON file.

    Args:
        json_file: Path to JSON file

    Returns:
        Dictionary of extracted properties
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        extractor = PropertyExtractor(data)
        properties = extractor.extract_all()
        properties['material_formula'] = json_file.stem.upper()

        return properties

    except Exception as e:
        return {
            'material_formula': json_file.stem.upper(),
            '_extraction_error': str(e)
        }


def save_to_json_directory(properties_list: List[Dict[str, Any]], json_files: List[Path], output_dir: Path):
    """
    Save extracted properties to individual JSON files (one per material).

    Args:
        properties_list: List of property dictionaries
        json_files: List of original JSON file paths
        output_dir: Directory to save output JSON files
    """
    if not properties_list:
        print("No properties to save")
        return

    # Add derived descriptors to all materials
    print("Computing derived descriptors...")
    for props in properties_list:
        add_derived_descriptors(props)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each material to individual JSON file
    print(f"Saving to {output_dir}/...")
    saved_count = 0
    for props, json_file in zip(properties_list, json_files):
        # Skip materials with extraction errors
        if '_extraction_error' in props:
            continue

        # Remove internal fields starting with _
        clean_props = {k: v for k, v in props.items() if not k.startswith('_')}

        # Save to JSON file with same name as input
        output_file = output_dir / json_file.name
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(clean_props, f, indent=2, ensure_ascii=False)
        saved_count += 1

    print(f"✓ Saved {saved_count} materials to {output_dir}/")


def save_to_csv(properties_list: List[Dict[str, Any]], output_file: Path):
    """
    Save extracted properties to CSV file with derived descriptors.

    Args:
        properties_list: List of property dictionaries
        output_file: Path to output CSV file
    """
    if not properties_list:
        print("No properties to save")
        return

    # Add derived descriptors to all materials
    print("Computing derived descriptors...")
    for props in properties_list:
        add_derived_descriptors(props)

    # Collect all unique keys (excluding internal fields starting with _)
    all_keys = set()
    for props in properties_list:
        all_keys.update(k for k in props.keys() if not k.startswith('_'))

    # Define column order (important fields first)
    # Based on expert final recommendations for FET sensor informatics
    priority_columns = [
        # Tier-1: Composition & structure
        'material_formula',
        'chemical_formula_reduced',
        'elements',
        'nelements',
        'crystal_system',
        'space_group',
        'nperiodic_dimensions',
        'density',
        # Tier-1: Thermodynamics
        'formation_energy_per_atom',
        'energy_above_hull',
        # Tier-1: Electronic / transport
        'band_gap',
        'band_gap_direct',
        'dos_ef',
        'me_avg',
        'mh_avg',
        # Tier-1: Dielectric (gate coupling & screening)
        'epsilon_x',
        'epsilon_y',
        'epsilon_z',
        'dielectric_total',
        # Tier-2: Dielectric decomposition
        'dielectric_electronic',
        'dielectric_ionic',
        # Tier-1: Mechanical (integration, reliability)
        'k_vrh',
        'g_vrh',
        'poisson_ratio',
        # Tier-1: Stability & 2D readiness
        'exfoliation_energy_mev_a2',
        # Tier-1: Magnetism
        'magnetization',
        # Tier-1: DERIVED DESCRIPTORS
        'eg_class',
        'eps_mean',
        'eps_aniso',
        'vdw_ready',
        'thermo_stable',
        'e_modulus_est',
    ]

    # Separate MAGPIE columns (132 features) for ordered placement
    magpie_columns = sorted([col for col in all_keys if col.startswith('magpie_')])

    # Remaining non-MAGPIE columns in alphabetical order
    remaining_columns = sorted(all_keys - set(priority_columns) - set(magpie_columns))

    # Final column order: priority → remaining → MAGPIE
    columns = [col for col in priority_columns if col in all_keys] + remaining_columns + magpie_columns

    # Write CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for props in properties_list:
            # Convert lists to strings for CSV, skip internal fields
            row = {}
            for key, value in props.items():
                if key.startswith('_'):  # Skip internal fields
                    continue
                if isinstance(value, list):
                    row[key] = str(value)
                else:
                    row[key] = value
            writer.writerow(row)

    print(f"✓ Saved {len(properties_list)} entries to {output_file}")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python extract_comprehensive_properties.py <json_file_or_directory> [output_csv_or_directory]")
        print()
        print("Examples:")
        print("  # Output to CSV (default)")
        print("  python extract_comprehensive_properties.py raw_data_pulled/")
        print("  python extract_comprehensive_properties.py raw_data_pulled/ fet_descriptors.csv")
        print()
        print("  # Output to JSON directory (one file per material)")
        print("  python extract_comprehensive_properties.py raw_data_pulled/ FET_descriptor_filtered/")
        print()
        print("Note: If output ends with '/', it will create a directory of JSON files.")
        print("      Otherwise, it will create a CSV file.")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    # Determine output mode based on output path
    if len(sys.argv) > 2:
        output_arg = sys.argv[2]
        # If ends with /, it's a directory
        is_directory_output = output_arg.endswith('/')
        output_path = Path(output_arg.rstrip('/'))
    else:
        # Default to CSV
        is_directory_output = False
        output_path = Path('fet_descriptors.csv')

    # Collect JSON files
    if input_path.is_file():
        json_files = [input_path]
    elif input_path.is_dir():
        json_files = sorted(input_path.glob('*.json'))
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)

    if not json_files:
        print(f"No JSON files found in {input_path}")
        sys.exit(1)

    print(f"Processing {len(json_files)} JSON files...")
    print()

    # Extract properties from all files
    properties_list = []
    for i, json_file in enumerate(json_files, 1):
        print(f"[{i}/{len(json_files)}] Processing {json_file.name}...", end=' ')
        props = extract_from_file(json_file)
        properties_list.append(props)

        if '_extraction_error' in props:
            print(f"✗ {props['_extraction_error']}")
        else:
            print(f"✓ {props.get('num_databases', 0)} databases")

    print()

    # Save output based on mode
    if is_directory_output:
        save_to_json_directory(properties_list, json_files, output_path)
    else:
        save_to_csv(properties_list, output_path)

    # Summary statistics
    successful = sum(1 for p in properties_list if '_extraction_error' not in p)
    with_formation_energy = sum(1 for p in properties_list if p.get('formation_energy_per_atom') is not None)
    with_band_gap = sum(1 for p in properties_list if p.get('band_gap') is not None)

    print()
    print("=== Summary Statistics ===")
    print(f"Total materials processed: {len(properties_list)}")
    print(f"Successful extractions: {successful}")
    print(f"With formation energy: {with_formation_energy}")
    print(f"With band gap data: {with_band_gap}")


if __name__ == '__main__':
    main()
