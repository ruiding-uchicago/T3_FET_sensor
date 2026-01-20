#!/usr/bin/env python3
"""
Generate host-guest inclusion complex configurations for DFT calculations.

This script creates systematic sampling of:
1. Vertical insertion along cavity axis (9 depths × 2 modes = 18 configs)
2. Horizontal lying on surfaces (2 surfaces × 4 rotations = 8 configs)

Total: 26 configurations per host-guest pair

Usage:
    python generate_inclusion_configs.py --host beta_cd.vasp --guest PFOA.vasp --output beta-CD_PFOA
    python generate_inclusion_configs.py --host beta_cd.vasp --guest PFOS.vasp --output beta-CD_PFOS
    python generate_inclusion_configs.py --host beta_cd.vasp --guest SDS.vasp --output beta-CD_SDS
"""

import argparse
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from ase.io import read, write
from ase import Atoms


def get_principal_axes(positions):
    """Get principal axes via SVD. Returns axes sorted by variance (largest first)."""
    centered = positions - positions.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered)
    return Vt, S


def align_axis_to_target(positions, source_axis, target_axis):
    """Rotate positions so source_axis aligns with target_axis."""
    source_axis = np.array(source_axis) / np.linalg.norm(source_axis)
    target_axis = np.array(target_axis) / np.linalg.norm(target_axis)

    dot = np.dot(source_axis, target_axis)
    if np.abs(dot) > 0.9999:
        if dot < 0:
            # Flip
            positions = positions.copy()
            positions[:, 2] = -positions[:, 2]
        return positions

    rot_axis = np.cross(source_axis, target_axis)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    angle = np.arccos(np.clip(dot, -1, 1))
    rot = R.from_rotvec(angle * rot_axis)

    center = positions.mean(axis=0)
    return rot.apply(positions - center) + center


def process_host(atoms, element_for_axis='C'):
    """
    Process host molecule:
    - Find cavity axis using specified element atoms
    - Align cavity axis to Z-axis
    - Center at origin
    """
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    # Find cavity axis using specified element
    mask = [s == element_for_axis for s in symbols]
    elem_pos = positions[mask]
    Vt, S = get_principal_axes(elem_pos)

    # Cavity axis is the smallest variance direction (perpendicular to ring)
    cavity_axis = Vt[2]
    if cavity_axis[2] < 0:
        cavity_axis = -cavity_axis

    # Center at origin
    positions = positions - positions.mean(axis=0)

    # Align cavity axis to Z
    positions = align_axis_to_target(positions, cavity_axis, [0, 0, 1])

    return symbols, positions


def process_guest(atoms):
    """
    Process guest molecule:
    - Find principal axis (longest direction)
    - Center at origin
    Returns both Z-aligned (vertical) and X-aligned (horizontal) versions
    """
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    # Center at origin
    positions = positions - positions.mean(axis=0)

    # Find main axis
    Vt, S = get_principal_axes(positions)
    main_axis = Vt[0]
    if main_axis[2] < 0:
        main_axis = -main_axis

    # Vertical: align to Z
    pos_vertical = align_axis_to_target(positions, main_axis, [0, 0, 1])

    # Horizontal: align to X
    pos_horizontal = align_axis_to_target(positions, main_axis, [1, 0, 0])

    return symbols, pos_vertical, pos_horizontal


def create_complex(host_symbols, host_pos, guest_symbols, guest_pos,
                   box_size=50.0, center_at_box=True):
    """Create combined host-guest structure."""
    combined_symbols = list(host_symbols) + list(guest_symbols)
    combined_pos = np.vstack([host_pos, guest_pos])

    if center_at_box:
        box_center = np.array([box_size/2, box_size/2, box_size/2])
        combined_pos = combined_pos - combined_pos.mean(axis=0) + box_center

    atoms = Atoms(
        symbols=combined_symbols,
        positions=combined_pos,
        cell=[box_size, box_size, box_size],
        pbc=True
    )
    return atoms


def generate_vertical_insertions(host_symbols, host_pos, guest_symbols, guest_pos_vertical,
                                  depths=None, box_size=50.0):
    """Generate vertical insertion configurations at different depths."""
    if depths is None:
        depths = [-8, -6, -4, -2, 0, 2, 4, 6, 8]

    configs = []
    box_center = np.array([box_size/2, box_size/2, box_size/2])

    for depth in depths:
        for mode in ['A', 'B']:
            h_pos = host_pos.copy() + box_center
            g_pos = guest_pos_vertical.copy()

            if mode == 'B':
                # Flip around X-axis (180° rotation)
                g_pos[:, 1] = -g_pos[:, 1]
                g_pos[:, 2] = -g_pos[:, 2]

            # Apply depth offset
            g_pos = g_pos + box_center + np.array([0, 0, depth])

            atoms = create_complex(host_symbols, h_pos, guest_symbols, g_pos,
                                   box_size=box_size, center_at_box=False)

            min_dist = cdist(h_pos, g_pos).min()
            name = f"d{depth:+d}_m{mode}"

            configs.append({
                'name': name,
                'atoms': atoms,
                'min_dist': min_dist,
                'type': 'vertical'
            })

    return configs


def generate_surface_lying(host_symbols, host_pos, guest_symbols, guest_pos_horizontal,
                           surface_distance=3.0, rotations=None, box_size=50.0):
    """Generate horizontal lying configurations on upper/lower surfaces."""
    if rotations is None:
        rotations = [0, 90, 180, 270]

    configs = []
    box_center = np.array([box_size/2, box_size/2, box_size/2])

    host_z_max = host_pos[:, 2].max()
    host_z_min = host_pos[:, 2].min()

    for surface in ['upper', 'lower']:
        for rot_angle in rotations:
            g_pos = guest_pos_horizontal.copy()

            # Rotate around Z-axis
            rot = R.from_euler('z', rot_angle, degrees=True)
            g_pos = rot.apply(g_pos)

            # Position on surface
            if surface == 'upper':
                g_z_min = g_pos[:, 2].min()
                z_shift = host_z_max - g_z_min + surface_distance
            else:
                g_z_max = g_pos[:, 2].max()
                z_shift = host_z_min - g_z_max - surface_distance

            g_pos[:, 2] += z_shift

            h_pos = host_pos.copy() + box_center
            g_pos = g_pos + box_center

            atoms = create_complex(host_symbols, h_pos, guest_symbols, g_pos,
                                   box_size=box_size, center_at_box=False)

            min_dist = cdist(h_pos, g_pos).min()
            name = f"lying_{surface}_r{rot_angle}"

            configs.append({
                'name': name,
                'atoms': atoms,
                'min_dist': min_dist,
                'type': 'lying'
            })

    return configs


def main():
    parser = argparse.ArgumentParser(
        description='Generate host-guest inclusion complex configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--host', required=True, help='Host molecule file (e.g., beta_cd.vasp)')
    parser.add_argument('--guest', required=True, help='Guest molecule file (e.g., PFOA.vasp)')
    parser.add_argument('--output', required=True, help='Output directory name')
    parser.add_argument('--host-element', default='C', help='Element to use for finding host cavity axis (default: C)')
    parser.add_argument('--prefix', default=None, help='Filename prefix (default: derived from output)')
    parser.add_argument('--box-size', type=float, default=50.0, help='Box size in Angstroms (default: 50)')
    parser.add_argument('--surface-distance', type=float, default=3.0, help='Distance from surface for lying configs (default: 3.0)')
    parser.add_argument('--depths', type=str, default='-8,-6,-4,-2,0,2,4,6,8',
                        help='Comma-separated insertion depths (default: -8,-6,-4,-2,0,2,4,6,8)')

    args = parser.parse_args()

    # Parse depths
    depths = [int(d) for d in args.depths.split(',')]

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Determine prefix
    if args.prefix:
        prefix = args.prefix
    else:
        prefix = args.output.replace('-', '_').replace(' ', '_')

    print("=" * 60)
    print("Host-Guest Inclusion Complex Generator")
    print("=" * 60)

    # Load and process host
    print(f"\nLoading host: {args.host}")
    host_atoms = read(args.host)
    host_symbols, host_pos = process_host(host_atoms, args.host_element)
    print(f"  Formula: {host_atoms.get_chemical_formula()}")
    print(f"  Atoms: {len(host_atoms)}")
    print(f"  Cavity Z range: {host_pos[:,2].min():.1f} to {host_pos[:,2].max():.1f} Å")

    # Load and process guest
    print(f"\nLoading guest: {args.guest}")
    guest_atoms = read(args.guest)
    guest_symbols, guest_pos_v, guest_pos_h = process_guest(guest_atoms)
    print(f"  Formula: {guest_atoms.get_chemical_formula()}")
    print(f"  Atoms: {len(guest_atoms)}")
    print(f"  Length (vertical): {guest_pos_v[:,2].max() - guest_pos_v[:,2].min():.1f} Å")

    # Generate vertical insertions
    print(f"\n--- Generating Vertical Insertions ---")
    print(f"Depths: {depths}")
    vertical_configs = generate_vertical_insertions(
        host_symbols, host_pos, guest_symbols, guest_pos_v,
        depths=depths, box_size=args.box_size
    )

    for cfg in vertical_configs:
        fname = f"{prefix}_{cfg['name']}.vasp"
        fpath = os.path.join(args.output, fname)
        write(fpath, cfg['atoms'], format='vasp', vasp5=True, sort=True)
        status = "✓" if cfg['min_dist'] >= 1.5 else "⚠️"
        print(f"  {fname}: min_dist={cfg['min_dist']:.2f}Å {status}")

    # Generate surface lying
    print(f"\n--- Generating Surface Lying ---")
    lying_configs = generate_surface_lying(
        host_symbols, host_pos, guest_symbols, guest_pos_h,
        surface_distance=args.surface_distance, box_size=args.box_size
    )

    for cfg in lying_configs:
        fname = f"{prefix}_{cfg['name']}.vasp"
        fpath = os.path.join(args.output, fname)
        write(fpath, cfg['atoms'], format='vasp', vasp5=True, sort=True)
        status = "✓" if cfg['min_dist'] >= 1.5 else "⚠️"
        print(f"  {fname}: min_dist={cfg['min_dist']:.2f}Å {status}")

    # Summary
    total = len(vertical_configs) + len(lying_configs)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Output directory: {args.output}/")
    print(f"Vertical insertions: {len(vertical_configs)} configs")
    print(f"Surface lying: {len(lying_configs)} configs")
    print(f">>> TOTAL: {total} configurations <<<")
    print("\nNote: Configs with ⚠️ have atomic clashes - this is expected.")
    print("      DFT geometry optimization will resolve overlaps.")


if __name__ == '__main__':
    main()
