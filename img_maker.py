#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make PNG images of distograms from .pdb or .cif files, binned by residue length.

- Streams files in CHUNKS to avoid listing/shuffling millions at once.
- If a file has <40 or >1024 residues, skip.
- Save .png in e.g. bin_40_64/<stem>.png
- Also write a FASTA file with the protein sequence (e.g. "MTFPY...") 
 (the FASTA file will have the same base name as the PNG).
- Shuffle each chunk to distribute load.
"""

import os
import sys
import math
import logging
import warnings
import random
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
import scipy.spatial
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure
from PIL import Image

###############################################################################
# 1) Configuration & Mappings
###############################################################################

non_standard_to_standard = {
    '2AS':'ASP', '3AH':'HIS', '5HP':'GLU', 'ACL':'ARG', 'AGM':'ARG', 'AIB':'ALA', 'ALM':'ALA',
    'ALO':'THR', 'ALY':'LYS', 'ARM':'ARG', 'ASA':'ASP', 'ASB':'ASP', 'ASK':'ASP', 'ASL':'ASP',
    'ASQ':'ASP', 'ASX':'ASP', 'AYA':'ALA', 'BCS':'CYS', 'BHD':'ASP', 'BMT':'THR', 'BNN':'ALA',
    'BUC':'CYS', 'BUG':'LEU', 'C5C':'CYS', 'C6C':'CYS', 'CAS':'CYS', 'CCS':'CYS', 'CEA':'CYS',
    'CGU':'GLU', 'CHG':'ALA', 'CLE':'LEU', 'CME':'CYS', 'CSD':'ALA', 'CSO':'CYS', 'CSP':'CYS',
    'CSS':'CYS', 'CSW':'CYS', 'CSX':'CYS', 'CXM':'MET', 'CY1':'CYS', 'CY3':'CYS', 'CYG':'CYS',
    'CYM':'CYS', 'CYQ':'CYS', 'DAH':'PHE', 'DAL':'ALA', 'DAR':'ARG', 'DAS':'ASP', 'DCY':'CYS',
    'DGL':'GLU', 'DGN':'GLN', 'DHA':'ALA', 'DHI':'HIS', 'DIL':'ILE', 'DIV':'VAL', 'DLE':'LEU',
    'DLY':'LYS', 'DNP':'ALA', 'DPN':'PHE', 'DPR':'PRO', 'DSN':'SER', 'DSP':'ASP', 'DTH':'THR',
    'DTR':'TRP', 'DTY':'TYR', 'DVA':'VAL', 'EFC':'CYS', 'FLA':'ALA', 'FME':'MET', 'GGL':'GLU',
    'GL3':'GLY', 'GLZ':'GLY', 'GMA':'GLU', 'GSC':'GLY', 'HAC':'ALA', 'HAR':'ARG', 'HIC':'HIS',
    'HIP':'HIS', 'HMR':'ARG', 'HPQ':'PHE', 'HTR':'TRP', 'HYP':'PRO', 'IAS':'ASP', 'IIL':'ILE',
    'IYR':'TYR', 'KCX':'LYS', 'LLP':'LYS', 'LLY':'LYS', 'LTR':'TRP', 'LYM':'LYS', 'LYZ':'LYS',
    'MAA':'ALA', 'MEN':'ASN', 'MHS':'HIS', 'MIS':'SER', 'MLE':'LEU', 'MPQ':'GLY', 'MSA':'GLY',
    'MSE':'MET', 'MVA':'VAL', 'NEM':'HIS', 'NEP':'HIS', 'NLE':'LEU', 'NLN':'LEU', 'NLP':'LEU',
    'NMC':'GLY', 'OAS':'SER', 'OCS':'CYS', 'OMT':'MET', 'PAQ':'TYR', 'PCA':'GLU', 'PEC':'CYS',
    'PHI':'PHE', 'PHL':'PHE', 'PR3':'CYS', 'PRR':'ALA', 'PTR':'TYR', 'PYL':'LYS', 'PYX':'CYS',
    'SAC':'SER', 'SAR':'GLY', 'SCH':'CYS', 'SCS':'CYS', 'SCY':'CYS', 'SEC':'CYS', 'SEL':'SER',
    'SEP':'SER', 'SET':'SER', 'SHC':'CYS', 'SHR':'LYS', 'SMC':'CYS', 'SOC':'CYS', 'STY':'TYR',
    'SVA':'SER', 'TIH':'ALA', 'TPL':'TRP', 'TPO':'THR', 'TPQ':'ALA', 'TRG':'LYS', 'TRO':'TRP',
    'TYB':'TYR', 'TYI':'TYR', 'TYQ':'TYR', 'TYS':'TYR', 'TYY':'TYR'
}

three_to_one_letter = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M', 'UNK': 'X'
}

BIN_RANGES = [
    (40, 64), (65, 128), (129, 192), (193, 256),
    (257, 320), (321, 384), (385, 448), (449, 512),
    (513, 576), (577, 640), (641, 704), (705, 768),
    (769, 832), (833, 896), (897, 960), (961, 1024)
]

def bin_name_for_length(L):
    for low, high in BIN_RANGES:
        if low <= L <= high:
            return f"bin_{low}_{high}"
    return None

MIN_RES = 40
MAX_RES = 1024

# Paths
STRUCT_DIR = Path("/path/to/pdb_or_cif_files")  # directory with many .pdb/.cif
OUTPUT_BASE = Path("/path/for/saving_img")

STRUCT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
# FASTA output directory
FASTA_DIR = Path("/path/for/saving_fasta")
FASTA_DIR.mkdir(parents=True, exist_ok=True)

###############################################################################
# 2) Distogram / Utility
###############################################################################

def get_dihedrals(a, b, c, d):
    np.seterr(divide='ignore', invalid='ignore')
    b0 = -1.0 * (b - a)
    b1 = c - b
    b2 = d - c
    b1_norm = np.linalg.norm(b1, axis=-1, keepdims=True)
    b1 = b1 / b1_norm
    v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)
    return np.arctan2(y, x)

def get_angles(a, b, c):
    v = a - b
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    v = v / v_norm
    w = c - b
    w_norm = np.linalg.norm(w, axis=-1, keepdims=True)
    w = w / w_norm
    x = np.sum(v * w, axis=-1)
    return np.arccos(x)

def get_coords6d(xyz, dmax=80.0, normalize=True):
    nres = xyz.shape[0]
    N = xyz[:, 0]
    Ca = xyz[:, 1]
    C = xyz[:, 2]

    # Approximate Cb
    b = Ca - N
    c = C - Ca
    a = np.cross(b, c)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca

    kd = scipy.spatial.cKDTree(Cb)
    indices = kd.query_ball_tree(kd, dmax)
    idx = np.array([
        [i, j] for i in range(len(indices)) for j in indices[i] if i != j
    ]).T

    if idx.size == 0:
        dist6d = np.full((nres, nres), dmax, dtype=float)
        omega6d = np.zeros((nres, nres), dtype=float)
        theta6d = np.zeros((nres, nres), dtype=float)
        phi6d = np.zeros((nres, nres), dtype=float)
    else:
        idx0, idx1 = idx
        dist6d = np.full((nres, nres), dmax, dtype=float)
        dist6d[idx0, idx1] = np.linalg.norm(Cb[idx1] - Cb[idx0], axis=-1)
        omega6d = np.zeros((nres, nres), dtype=float)
        omega6d[idx0, idx1] = get_dihedrals(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])
        theta6d = np.zeros((nres, nres), dtype=float)
        theta6d[idx0, idx1] = get_dihedrals(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])
        phi6d = np.zeros((nres, nres), dtype=float)
        phi6d[idx0, idx1] = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])

    if normalize:
        dist6d = (dist6d / dmax) * 2.0 - 1.0
        omega6d = omega6d / math.pi
        theta6d = theta6d / math.pi
        phi6d = (phi6d / math.pi) * 2.0 - 1.0

    combined_channel = np.triu(omega6d) + np.tril(dist6d)
    coords_3 = np.stack([combined_channel, theta6d, phi6d], axis=-1)
    coords_3 = coords_3.transpose(2, 0, 1).astype(np.float32)
    coords_3 = np.nan_to_num(coords_3)
    return coords_3

###############################################################################
# 3) File Parsing -> Distogram + PNG and FASTA
###############################################################################

def extract_features_from_structure(path: Path) -> dict:
    ext = path.suffix.lower()
    if ext not in [".pdb", ".cif", ""]:
        return None

    try:
        if ext == ".pdb" or ext == "":
            with open(path, 'r') as f:
                pdbf = PDBFile.read(f)
            structure = pdbf.get_structure()
            if pdbf.get_model_count() != 1:
                return None  # skip multi-model or zero-model
        else:  # .cif
            cif_obj = CIFFile.read(path)
            structure = get_structure(cif_obj)
            # skip multi-model check if needed

        if struc.get_chain_count(structure) > 1:
            return None  # skip multi-chain

        # Get residues and build the sequence string
        _, aa = struc.get_residues(structure)
        for i, a3 in enumerate(aa):
            if a3 not in three_to_one_letter:
                aa[i] = non_standard_to_standard.get(a3, "X")
        seq_str = "".join(three_to_one_letter.get(a3, "X") for a3 in aa)

        nres = len(aa)
        if nres < MIN_RES or nres > MAX_RES:
            return None

        # backbone coords
        bb_coords = []
        for residue in struc.residue_iter(structure):
            atom_names = residue.get_annotation("atom_name")
            coords = residue.coord[0]
            triple = []
            for aname in ["N", "CA", "C"]:
                idx_atom = np.where(atom_names == aname)[0]
                if idx_atom.size == 0:
                    triple.append([0, 0, 0])
                else:
                    triple.append(coords[idx_atom[0]])
            bb_coords.append(triple)
        bb_coords = np.array(bb_coords, dtype=np.float32)

        arr_3_l_l = get_coords6d(bb_coords, dmax=80.0, normalize=True)
        arr_3_l_l = np.nan_to_num(arr_3_l_l)

        return {
            "id": path.stem,
            "coords_6d": arr_3_l_l,
            "nres": nres,
            "seq": seq_str
        }
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None


def tensor_to_image(sample: dict, bin_folder: Path):
    coords_6d = sample["coords_6d"]
    protein_id = sample["id"]
    seq_str = sample.get("seq", "")
    coords_normalized = ((coords_6d + 1)/2 * 255).clip(0, 255).astype(np.uint8)
    img_array = coords_normalized.transpose(1,2,0)
    img = Image.fromarray(img_array)
    outpath = bin_folder / f"{protein_id}.png"
    tmp_outpath = bin_folder / f"{protein_id}.png.tmp"
    try:
        img.save(tmp_outpath, format='PNG')
        tmp_outpath.rename(outpath)
    except Exception as e:
        print(f"Error saving image for {protein_id}: {e}")
        return False
    fasta_path = Path("/scratch/xwang213/fasta") / f"{protein_id}.fasta"
    try:
        with open(fasta_path, "w") as f:
            f.write(f">{protein_id}\n")
            f.write(seq_str + "\n")
    except Exception as e:
        print(f"Error saving FASTA for {protein_id}: {e}")
        return False
    return True


def process_wrapper(path: Path):
    """
    Worker function: parse -> bin -> save PNG (and FASTA) if not exists.
    After successful saving of both PNG and FASTA, delete the source .pdb/.cif file.
    """
    sample = extract_features_from_structure(path)
    if sample is None:
        return
    nres = sample["nres"]
    bin_name = bin_name_for_length(nres)
    if bin_name is None:
        return
    bin_folder = OUTPUT_BASE / bin_name
    bin_folder.mkdir(parents=True, exist_ok=True)
    outpath = bin_folder / f"{sample['id']}.png"
    if outpath.exists():
        return
    # Save PNG and FASTA.
    saved = tensor_to_image(sample, bin_folder)
    # If saving succeeded, then delete the original .pdb/.cif file.
    if saved and outpath.exists():
        try:
            os.remove(path)
        except Exception as e:
         
            print(f"Error deleting {path}: {e}")

###############################################################################
# 4) Chunked Streaming + Multiprocessing
###############################################################################

def initializer():
    warnings.filterwarnings("ignore", ".*elements were guessed from atom_.*")

def chunked_path_iterator(base_dir: Path, chunk_size=10000):
    """
    Streams .pdb/.cif files from 'base_dir' in chunks to avoid listing them all at once.
    Yields blocks of up to 'chunk_size' paths. Each chunk is partially shuffled.
    """
    exts = {".pdb", ".cif", ""}
    buffer = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            p = Path(root) / file
            if p.suffix.lower() in exts:
                buffer.append(p)
                if len(buffer) >= chunk_size:
                    random.shuffle(buffer)
                    yield buffer
                    buffer = []
    if buffer:
        random.shuffle(buffer)
        yield buffer

def count_files(base_dir: Path):
    """
    Optional 1st pass: count how many .pdb/.cif are in 'base_dir' so we can do a global TQDM.
    """
    exts = {".pdb", ".cif", ""}
    total = 0
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if Path(file).suffix.lower() in exts:
                total += 1
    return total

###############################################################################
# 5) Main
###############################################################################

def main():
    logging.basicConfig(
        filename='pdb_to_img.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    print("Counting .pdb/.cif files (first pass)...")
    total_files = count_files(STRUCT_DIR)
    if total_files == 0:
        print("No structure files found. Exiting.")
        return

    print(f"Found {total_files} files in {STRUCT_DIR}")
    logging.info(f"Found {total_files} structure files in {STRUCT_DIR}")

    chunk_size = 20000  # adjust as needed
    print(f"Using chunk_size = {chunk_size}")

    num_procs = max(1, cpu_count() - 4)
    print(f"Using {num_procs} worker processes...")

    processed_count = 0

    with Pool(processes=num_procs, initializer=initializer) as pool:
        with tqdm(total=total_files, desc="Processing") as pbar:
            for chunk_paths in chunked_path_iterator(STRUCT_DIR, chunk_size=chunk_size):
                for _ in pool.imap_unordered(process_wrapper, chunk_paths, chunksize=50):
                    processed_count += 1
                    pbar.update()

    print(f"All structures processed: {processed_count} total.")
    logging.info(f"All structures processed: {processed_count} total.")

if __name__ == "__main__":
    main()
