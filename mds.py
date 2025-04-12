#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end pipeline:
   - Read a PDB file and extract backbone (N, CA, C).
   - Compute Cβ coordinates using the standard empirical formula.
   - Build an image tensor (a 3-channel distogram) from the backbone.
   - Save the image.
   - Extract the distance matrix from the tensor (first channel) and map back to distances.
   - Use classical MDS to reconstruct the Cβ positions from the distance matrix.
   - Align the reconstructed coordinates using Procrustes analysis and compute the RMSD.
   - Write out a new PDB with the reconstructed Cβ positions (while keeping the original backbone)
   - Visualize the original and reconstructed Cβ positions.
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser, PDBIO, Model, Chain, Residue, Atom

# ---------------------------
# Geometry helper functions
# ---------------------------

def get_dihedrals(a, b, c, d):
    """
    Compute the dihedral angle for four points a, b, c, d.
    """
    np.seterr(divide='ignore', invalid='ignore')
    b0 = -1.0 * (b - a)
    b1 = c - b
    b2 = d - c
    b1_norm = np.linalg.norm(b1, axis=-1, keepdims=True)
    b1_unit = b1 / b1_norm
    v = b0 - np.sum(b0 * b1_unit, axis=-1, keepdims=True) * b1_unit
    w = b2 - np.sum(b2 * b1_unit, axis=-1, keepdims=True) * b1_unit
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1_unit, v) * w, axis=-1)
    return np.arctan2(y, x)

def get_angles(a, b, c):
    """
    Compute the angle at b formed by a--b--c.
    """
    v = a - b
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    v_unit = v / v_norm
    w = c - b
    w_norm = np.linalg.norm(w, axis=-1, keepdims=True)
    w_unit = w / w_norm
    x = np.sum(v_unit * w_unit, axis=-1)
    return np.arccos(np.clip(x, -1.0, 1.0))

# ---------------------------
# Build the distogram tensor
# ---------------------------

def get_coords6d(xyz, dmax=80.0, normalize=True):
    """
    Given an array xyz of shape (L, 3, 3) containing the backbone atoms for each residue,
    compute a 3-channel tensor encoding distance and angle information.
    The tensor channels represent:
      - Channel 0: Combined channel with upper triangle = dihedral omega and lower triangle = distance.
      - Channel 1: Dihedral angle theta.
      - Channel 2: Bond angle phi.
    """
    nres = xyz.shape[0]
    N = xyz[:, 0]   # N coordinates of shape (L,3)
    Ca = xyz[:, 1]  # Cα
    C = xyz[:, 2]   # C

    # Compute approximate Cβ positions.
    b = Ca - N
    c = C - Ca
    a = np.cross(b, c)
    Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

    # Use cKDTree to find all residue pairs within dmax.
    kd = cKDTree(Cb)
    indices = kd.query_ball_tree(kd, dmax)
    idx_list = []
    for i, neigh in enumerate(indices):
        for j in neigh:
            if i != j:
                idx_list.append([i, j])
    if len(idx_list) == 0:
        dist6d = np.full((nres, nres), dmax, dtype=float)
        omega6d = np.zeros((nres, nres), dtype=float)
        theta6d = np.zeros((nres, nres), dtype=float)
        phi6d = np.zeros((nres, nres), dtype=float)
    else:
        idx = np.array(idx_list).T
        idx0, idx1 = idx
        dist6d = np.full((nres, nres), dmax, dtype=float)
        diff = Cb[idx1] - Cb[idx0]
        dist6d[idx0, idx1] = np.linalg.norm(diff, axis=-1)
        omega6d = np.zeros((nres, nres), dtype=float)
        omega6d[idx0, idx1] = get_dihedrals(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])
        theta6d = np.zeros((nres, nres), dtype=float)
        theta6d[idx0, idx1] = get_dihedrals(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])
        phi6d = np.zeros((nres, nres), dtype=float)
        phi6d[idx0, idx1] = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])

    # Normalize channels to [-1, 1]
    if normalize:
        dist6d = (dist6d / dmax) * 2.0 - 1.0
        omega6d = omega6d / math.pi
        theta6d = theta6d / math.pi
        phi6d = (phi6d / math.pi) * 2.0 - 1.0

    # Combine the upper triangle of omega and lower triangle of distance.
    combined_channel = np.triu(omega6d) + np.tril(dist6d)
    coords_3 = np.stack([combined_channel, theta6d, phi6d], axis=-1)
    # Rearrange to shape (3, L, L)
    coords_3 = coords_3.transpose(2, 0, 1).astype(np.float32)
    coords_3 = np.nan_to_num(coords_3)
    return coords_3

# ---------------------------
# Functions to extract distance matrix and reconstruct via MDS
# ---------------------------

def extract_distance_matrix(tensor, dmax=80.0):
    """
    Extract the distance matrix from the first channel of the tensor.
    The combined channel was built as:
      combined = np.triu(omega6d) + np.tril(dist6d)
    so the lower triangle (including the diagonal) holds the normalized distances.
    Recover distances from:
         normalized = (d / dmax)*2.0 - 1.0   =>   d = ((normalized + 1)/2)*dmax
    """
    combined = tensor[0]  # first channel; shape: (L, L)
    L = combined.shape[0]
    # Extract the lower triangle (including the diagonal)
    d_lower = np.tril(combined)
    # Symmetrize to get the full distance matrix
    D_norm = d_lower + d_lower.T - np.diag(np.diag(d_lower))
    D = ((D_norm + 1) / 2.0) * dmax
    np.fill_diagonal(D, 0)
    return D

def classical_mds(D, n_components=3):
    """
    Perform classical multidimensional scaling (MDS) to reconstruct points in R^(n_components).
    """
    n = D.shape[0]
    # Centering matrix
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    L_eig = np.maximum(eigenvalues[:n_components], 0)
    coords = eigenvectors[:, :n_components] * np.sqrt(L_eig)
    return coords

def procrustes_align(X, Y):
    """
    Align Y (reconstructed) to X (true coordinates) using Procrustes analysis.
    X and Y are expected to be (L, 3).
    """
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)
    U, s, Vt = np.linalg.svd(Y_centered.T @ X_centered)
    R = U @ Vt
    Y_aligned = (Y_centered @ R) + X.mean(axis=0)
    return Y_aligned

def compute_rmsd(X, Y):
    """
    Compute the root-mean-square deviation (RMSD) between two sets of points.
    X and Y are (L, 3) arrays.
    """
    diff = X - Y
    rmsd = np.sqrt(np.sum(diff ** 2) / X.shape[0])
    return rmsd

# ---------------------------
# PDB parsing and backbone extraction
# ---------------------------

def read_backbone(pdb_path):
    """
    Read the PDB file and extract backbone coordinates (N, CA, C) for the first model and first chain.
    Returns:
      - bb_coords: numpy array of shape (L, 3, 3), where for each residue the rows are (N, CA, C)
      - residues: list of corresponding residue objects (to preserve residue id info)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = next(structure.get_models())
    chain = next(model.get_chains())
    bb_coords = []
    residues = []
    for res in chain:
        # Skip hetero atoms/residues if needed.
        try:
            N = res['N'].get_coord()
            CA = res['CA'].get_coord()
            C = res['C'].get_coord()
        except KeyError:
            continue
        bb_coords.append(np.stack([N, CA, C], axis=0))
        residues.append(res)
    bb_coords = np.array(bb_coords, dtype=np.float32)
    return bb_coords, residues

def compute_cb_from_backbone(bb):
    """
    Given backbone coordinates (N, CA, C) for one residue (shape (3,)) compute approximate Cβ.
    """
    N, CA, C = bb
    b = CA - N
    c = C - CA
    a = np.cross(b, c)
    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
    return CB

# ---------------------------
# PDB writer for the reconstructed structure
# ---------------------------

def write_reconstructed_pdb(output_path, residues, bb_coords, cb_recon):
    """
    Write a new PDB file containing the original backbone (N, CA, C) and the reconstructed Cβ positions.
    For each residue, output ATOM records for N, CA, C (from bb_coords) and CB (from cb_recon).
    """
    pdb_lines = []
    atom_serial = 1
    chain_id = "A"
    for i, res in enumerate(residues):
        res_name = res.get_resname()
        res_id = res.get_id()[1]
        # Get backbone atoms from bb_coords; order: N, CA, C
        N_coord, CA_coord, C_coord = bb_coords[i]
        CB_coord = cb_recon[i]  # aligned reconstructed Cβ
        # Format each ATOM record (columns are fixed-width, here using a simple format)
        pdb_lines.append("ATOM  {:5d}  N   {:3s} {}   {:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00 20.00           N".format(
            atom_serial, res_name, chain_id, res_id, *N_coord))
        atom_serial += 1
        pdb_lines.append("ATOM  {:5d}  CA  {:3s} {}   {:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00 20.00           C".format(
            atom_serial, res_name, chain_id, res_id, *CA_coord))
        atom_serial += 1
        pdb_lines.append("ATOM  {:5d}  C   {:3s} {}   {:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00 20.00           C".format(
            atom_serial, res_name, chain_id, res_id, *C_coord))
        atom_serial += 1
        pdb_lines.append("ATOM  {:5d}  CB  {:3s} {}   {:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00 20.00           C".format(
            atom_serial, res_name, chain_id, res_id, *CB_coord))
        atom_serial += 1
    pdb_lines.append("END")
    with open(output_path, "w") as f:
        for line in pdb_lines:
            f.write(line + "\n")

# ---------------------------
# Main pipeline
# ---------------------------

def main():
    # Set paths and constants
    pdb_path = "input.pdb"  # Replace with your pdb filename.
    output_image_path = "distogram.png"
    output_pdb_path = "reconstructed.pdb"
    DMAX = 80.0

    # 1. Read the PDB backbone.
    bb_coords, residues = read_backbone(pdb_path)
    if bb_coords.shape[0] < 3:
        print("Not enough backbone residues found. Exiting.")
        return
    L = bb_coords.shape[0]
    print(f"Read {L} residues from {pdb_path}")

    # 2. Compute original Cβ positions for comparison.
    cb_orig = np.array([compute_cb_from_backbone(bb_coords[i]) for i in range(L)])

    # 3. Build the image tensor from backbone data.
    tensor = get_coords6d(bb_coords, dmax=DMAX, normalize=True)
    
    # 4. Save the tensor as an image.
    # Scale from [-1, 1] to [0, 255] and transpose to (L, L, 3)
    tensor_scaled = ((tensor + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    # Note: tensor has shape (3, L, L) --> transpose to (L, L, 3)
    img_array = tensor_scaled.transpose(1, 2, 0)
    img = Image.fromarray(img_array)
    img.save(output_image_path)
    print(f"Saved distogram image to {output_image_path}")

    # 5. Extract the distance matrix from the first channel.
    D = extract_distance_matrix(tensor, dmax=DMAX)

    # 6. Reconstruct Cβ positions by classical MDS.
    cb_recon = classical_mds(D, n_components=3)

    # 7. Align the reconstructed coordinates to the original Cβ positions.
    cb_recon_aligned = procrustes_align(cb_orig, cb_recon)

    # 8. Compute the RMSD between original and reconstructed Cβ coordinates.
    rmsd_val = compute_rmsd(cb_orig, cb_recon_aligned)
    print(f"RMSD between original and reconstructed Cβ positions: {rmsd_val:.3f} Å")

    # 9. Write out a PDB with the original backbone and reconstructed Cβ coordinates.
    write_reconstructed_pdb(output_pdb_path, residues, bb_coords, cb_recon_aligned)
    print(f"Reconstructed PDB written to {output_pdb_path}")

    # 10. Visualize the original vs. reconstructed Cβ positions.
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cb_orig[:,0], cb_orig[:,1], cb_orig[:,2], c='blue', label='Original Cβ', s=40)
    ax.scatter(cb_recon_aligned[:,0], cb_recon_aligned[:,1], cb_recon_aligned[:,2], c='red', label='Reconstructed Cβ', s=40, marker='^')
    ax.set_title("Cβ Positions: Original vs. Reconstructed")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
