#!/usr/bin/env python3
"""
Generate host-guest configurations for DFT validation.
Hosts: beta-CD, CID_422, CID_545, CID_702, CID_670_iso1/iso2, CID_736_iso1/iso2
Guests: PFOA, PFOS, SDS, TCAA

Configuration types:
- insert_d{depth}_m{A/B}.vasp: vertical insertion at various depths (-8 to +8 Å)
- lying_{upper/lower}_r{angle}.vasp: surface lying configurations
- side_a{angle}.vasp: side binding configurations
"""

import numpy as np
import os
import shutil

def standardize_element(elem):
    """Standardize element symbol: first letter uppercase, rest lowercase (e.g., CL -> Cl)"""
    elem = elem.strip()
    if len(elem) == 1:
        return elem.upper()
    return elem[0].upper() + elem[1:].lower()

def parse_pdb(filename):
    atoms = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('HETATM') or line.startswith('ATOM'):
                parts = line.split()
                x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
                element = standardize_element(parts[-1])
                atom_name = parts[2]
                atoms.append({'element': element, 'coords': np.array([x, y, z]), 'name': atom_name})
    return atoms

def parse_vasp(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    elements_line = lines[5].split()
    counts = list(map(int, lines[6].split()))
    coords, elements = [], []
    idx = 8
    for elem, count in zip(elements_line, counts):
        for _ in range(count):
            x, y, z = map(float, lines[idx].split()[:3])
            coords.append([x, y, z])
            elements.append(elem)
            idx += 1
    return elements, np.array(coords)

def write_vasp(filename, elements, coords, box_size=50.0):
    unique_elements = sorted(set(elements))
    sorted_coords, element_counts = [], {}
    for elem in unique_elements:
        elem_coords = [c for e, c in zip(elements, coords) if e == elem]
        sorted_coords.extend(elem_coords)
        element_counts[elem] = len(elem_coords)
    with open(filename, 'w') as f:
        f.write(' '.join(unique_elements) + '\n')
        f.write(' 1.0000000000000000\n')
        f.write(f'    {box_size:.16f}    0.0000000000000000    0.0000000000000000\n')
        f.write(f'     0.0000000000000000   {box_size:.16f}    0.0000000000000000\n')
        f.write(f'     0.0000000000000000    0.0000000000000000   {box_size:.16f}\n')
        f.write(' ' + '  '.join(unique_elements) + ' \n')
        f.write(' ' + '  '.join(str(element_counts[e]) for e in unique_elements) + '\n')
        f.write('Cartesian\n')
        for coord in sorted_coords:
            f.write(f' {coord[0]:.16f} {coord[1]:.16f} {coord[2]:.16f}\n')

def rotation_matrix_axis_angle(axis, angle):
    axis = np.array(axis) / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)

def generate_configs(host_name, pdb_file, ring_atom_names, output_base, guests, depths=None):
    """Generate configurations for a host molecule based on its ring atoms."""
    if depths is None:
        depths = [-8, -6, -4, -2, 0, 2, 4, 6, 8]

    box_center = np.array([25.0, 25.0, 25.0])

    host_atoms = parse_pdb(pdb_file)
    host_coords = np.array([a['coords'] for a in host_atoms])
    host_elements = [a['element'] for a in host_atoms]

    # Find ring atoms and compute ring center/normal
    ring_coords = []
    for atom in host_atoms:
        if atom['name'] in ring_atom_names:
            ring_coords.append(atom['coords'])
    ring_coords = np.array(ring_coords)
    ring_center = ring_coords.mean(axis=0)
    host_coords_centered = host_coords - ring_center

    # PCA for ring normal
    ring_centered = ring_coords - ring_center
    cov = np.cov(ring_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    ring_normal = eigenvectors[:, 0]
    if ring_normal[2] < 0:
        ring_normal = -ring_normal

    # Perpendicular directions
    perp1 = np.cross(ring_normal, [0, 0, 1])
    if np.linalg.norm(perp1) < 0.1:
        perp1 = np.cross(ring_normal, [0, 1, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(ring_normal, perp1)

    host_along_normal = np.dot(host_coords_centered, ring_normal)
    host_top = host_along_normal.max()
    host_bottom = host_along_normal.min()

    print(f"{host_name}: ring normal=[{ring_normal[0]:.3f}, {ring_normal[1]:.3f}, {ring_normal[2]:.3f}]")

    for guest_name, guest_file in guests.items():
        guest_elements, guest_coords = parse_vasp(guest_file)
        guest_center = guest_coords.mean(axis=0)
        guest_coords_centered = guest_coords - guest_center

        guest_cov = np.cov(guest_coords_centered.T)
        g_eigenvalues, g_eigenvectors = np.linalg.eigh(guest_cov)
        guest_axis = g_eigenvectors[:, 2]

        output_dir = os.path.join(output_base, f"{host_name}_{guest_name}")
        os.makedirs(output_dir, exist_ok=True)

        # Insert configs
        for mode in ['mA', 'mB']:
            cross = np.cross(guest_axis, ring_normal)
            if np.linalg.norm(cross) > 0.001:
                R = rotation_matrix_axis_angle(cross, np.arccos(np.clip(np.dot(guest_axis, ring_normal), -1, 1)))
            else:
                R = np.eye(3)
            guest_rotated = (R @ guest_coords_centered.T).T

            if mode == 'mB':
                R_flip = rotation_matrix_axis_angle(perp1, np.pi)
                guest_rotated = (R_flip @ guest_rotated.T).T

            for depth in depths:
                offset = ring_normal * depth
                guest_final = guest_rotated + box_center + offset
                host_final = host_coords_centered + box_center

                all_coords = np.vstack([host_final, guest_final])
                all_elements = host_elements + guest_elements

                depth_str = f"d{depth:+d}" if depth != 0 else "d0"
                write_vasp(f"{output_dir}/insert_{depth_str}_{mode}.vasp", all_elements, all_coords)

        # Lying configs
        for surface in ['upper', 'lower']:
            for rot_angle in [0, 90, 180, 270]:
                cross = np.cross(guest_axis, perp1)
                if np.linalg.norm(cross) > 0.001:
                    R_align = rotation_matrix_axis_angle(cross, np.arccos(np.clip(np.dot(guest_axis, perp1), -1, 1)))
                else:
                    R_align = np.eye(3)
                guest_rotated = (R_align @ guest_coords_centered.T).T
                R_z = rotation_matrix_axis_angle(ring_normal, np.radians(rot_angle))
                guest_rotated = (R_z @ guest_rotated.T).T

                guest_along_normal_rot = np.dot(guest_rotated, ring_normal)
                if surface == 'upper':
                    offset = ring_normal * (host_top - guest_along_normal_rot.min() + 2.0)
                else:
                    offset = ring_normal * (host_bottom - guest_along_normal_rot.max() - 2.0)

                guest_final = guest_rotated + box_center + offset
                host_final = host_coords_centered + box_center
                all_coords = np.vstack([host_final, guest_final])
                all_elements = host_elements + guest_elements
                write_vasp(f"{output_dir}/lying_{surface}_r{rot_angle}.vasp", all_elements, all_coords)

        # Side configs
        for angle in [0, 60, 120, 180, 240, 300]:
            rad = np.radians(angle)
            direction = np.cos(rad) * perp1 + np.sin(rad) * perp2

            cross = np.cross(guest_axis, -direction)
            if np.linalg.norm(cross) > 0.001:
                R_align = rotation_matrix_axis_angle(cross, np.arccos(np.clip(np.dot(guest_axis, -direction), -1, 1)))
            else:
                R_align = np.eye(3)
            guest_rotated = (R_align @ guest_coords_centered.T).T

            guest_along_dir = np.dot(guest_rotated, direction)
            host_along_dir = np.dot(host_coords_centered, direction)
            offset_dist = host_along_dir.max() - guest_along_dir.min() + 2.0
            offset = direction * offset_dist

            guest_final = guest_rotated + box_center + offset
            host_final = host_coords_centered + box_center
            all_coords = np.vstack([host_final, guest_final])
            all_elements = host_elements + guest_elements
            write_vasp(f"{output_dir}/side_a{angle}.vasp", all_elements, all_coords)


def generate_betaCD_configs(host_file, output_base, guests, depths=None):
    """Generate configurations for beta-CD (VASP format host)."""
    if depths is None:
        depths = [-6, -4, -2, 0, 2, 4, 6]

    box_center = np.array([25.0, 25.0, 25.0])

    host_elements, host_coords = parse_vasp(host_file)
    host_center = host_coords.mean(axis=0)
    host_coords_centered = host_coords - host_center

    # For beta-CD, use PCA to find the ring plane
    cov = np.cov(host_coords_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    ring_normal = eigenvectors[:, 0]
    if ring_normal[2] < 0:
        ring_normal = -ring_normal

    perp1 = np.cross(ring_normal, [0, 0, 1])
    if np.linalg.norm(perp1) < 0.1:
        perp1 = np.cross(ring_normal, [0, 1, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(ring_normal, perp1)

    host_along_normal = np.dot(host_coords_centered, ring_normal)
    host_top = host_along_normal.max()
    host_bottom = host_along_normal.min()

    print(f"beta-CD: ring normal=[{ring_normal[0]:.3f}, {ring_normal[1]:.3f}, {ring_normal[2]:.3f}]")

    for guest_name, guest_file in guests.items():
        guest_elements, guest_coords = parse_vasp(guest_file)
        guest_center = guest_coords.mean(axis=0)
        guest_coords_centered = guest_coords - guest_center

        guest_cov = np.cov(guest_coords_centered.T)
        g_eigenvalues, g_eigenvectors = np.linalg.eigh(guest_cov)
        guest_axis = g_eigenvectors[:, 2]

        output_dir = os.path.join(output_base, f"beta-CD_{guest_name}")
        os.makedirs(output_dir, exist_ok=True)

        # Insert configs
        for mode in ['mA', 'mB']:
            cross = np.cross(guest_axis, ring_normal)
            if np.linalg.norm(cross) > 0.001:
                R = rotation_matrix_axis_angle(cross, np.arccos(np.clip(np.dot(guest_axis, ring_normal), -1, 1)))
            else:
                R = np.eye(3)
            guest_rotated = (R @ guest_coords_centered.T).T

            if mode == 'mB':
                R_flip = rotation_matrix_axis_angle(perp1, np.pi)
                guest_rotated = (R_flip @ guest_rotated.T).T

            for depth in depths:
                offset = ring_normal * depth
                guest_final = guest_rotated + box_center + offset
                host_final = host_coords_centered + box_center

                all_coords = np.vstack([host_final, guest_final])
                all_elements = host_elements + guest_elements

                depth_str = f"d{depth:+d}" if depth != 0 else "d0"
                write_vasp(f"{output_dir}/insert_{depth_str}_{mode}.vasp", all_elements, all_coords)

        # Lying configs
        for surface in ['upper', 'lower']:
            for rot_angle in [0, 90, 180]:
                cross = np.cross(guest_axis, perp1)
                if np.linalg.norm(cross) > 0.001:
                    R_align = rotation_matrix_axis_angle(cross, np.arccos(np.clip(np.dot(guest_axis, perp1), -1, 1)))
                else:
                    R_align = np.eye(3)
                guest_rotated = (R_align @ guest_coords_centered.T).T
                R_z = rotation_matrix_axis_angle(ring_normal, np.radians(rot_angle))
                guest_rotated = (R_z @ guest_rotated.T).T

                guest_along_normal_rot = np.dot(guest_rotated, ring_normal)
                if surface == 'upper':
                    offset = ring_normal * (host_top - guest_along_normal_rot.min() + 2.0)
                else:
                    offset = ring_normal * (host_bottom - guest_along_normal_rot.max() - 2.0)

                guest_final = guest_rotated + box_center + offset
                host_final = host_coords_centered + box_center
                all_coords = np.vstack([host_final, guest_final])
                all_elements = host_elements + guest_elements
                write_vasp(f"{output_dir}/lying_{surface}_r{rot_angle}.vasp", all_elements, all_coords)


def main():
    """
    Command-line interface for generating host-guest configurations.

    Example usage:
        # Generate for a PDB host with specific ring atoms
        python generate_all_configs.py --host probe.pdb --guest target.vasp \\
            --output configs/ --ring-atoms C1,C2,C3,N1,N2

        # Generate for a VASP host (beta-CD style)
        python generate_all_configs.py --host beta_cd.vasp --guest PFOA.vasp \\
            --output configs/ --host-format vasp
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate host-guest configurations for DFT (insert, lying, side-binding)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=main.__doc__
    )
    parser.add_argument('--host', required=True, help='Host molecule file (PDB or VASP)')
    parser.add_argument('--guest', required=True, help='Guest molecule file (VASP format)')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--host-format', choices=['pdb', 'vasp'], default='pdb',
                        help='Host file format (default: pdb)')
    parser.add_argument('--ring-atoms', type=str, default=None,
                        help='Comma-separated ring atom names for PDB hosts (e.g., C1,C2,N1)')
    parser.add_argument('--depths', type=str, default='-8,-6,-4,-2,0,2,4,6,8',
                        help='Comma-separated insertion depths (default: -8 to +8)')

    args = parser.parse_args()

    depths = [int(d) for d in args.depths.split(',')]
    guests = {'guest': args.guest}

    if args.host_format == 'vasp':
        generate_betaCD_configs(args.host, args.output, guests, depths=depths)
    else:
        if args.ring_atoms is None:
            print("Warning: --ring-atoms not specified, using all C atoms for ring detection")
            ring_atoms = None
        else:
            ring_atoms = [a.strip() for a in args.ring_atoms.split(',')]

        host_name = os.path.splitext(os.path.basename(args.host))[0]
        generate_configs(host_name, args.host, ring_atoms or [], args.output, guests, depths=depths)

    print(f"\nDone! Configurations saved to {args.output}/")


if __name__ == "__main__":
    main()
