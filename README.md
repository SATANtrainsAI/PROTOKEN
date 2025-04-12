# Repository Overview

This repository contains two major parts:

1. **Protein Structure to Image**  
   A Python script (`pdb_to_png_distogram.py`) for parsing protein structures (*.pdb or *.cif) and converting them into PNG “distogram” images (plus associated FASTA files). These images capture backbone distances and angular features.

2. **Hierarchical VQ-VAE Training**  
   A hierarchical VQ-VAE model implemented with PyTorch and Transformer blocks (`model.py`, `train.py`, and `dataloader.py`). This model is designed to learn quantized latent representations of the generated protein images.

Below is an overview of the file structure and detailed explanations of both the implementation and the underlying mathematical concepts.

---

## 1. Protein Structure to Distogram Image

**File:** `pdb_to_png_distogram.py`

### Overview

- **Reads protein structures** (*.pdb or *.cif) using Biotite (or a related library).
- **Extracts backbone atoms** `N`, `CA`, `C` (and estimates `CB`).
- **Computes distances, dihedral angles, and planar angles** between residues, forming a 6D representation.
- **Normalizes and merges these channels** into a 3-channel image (RGB). Each pixel \((i, j)\) corresponds to backbone geometry between residue \(i\) and residue \(j\).
- **Saves the result as a PNG image.** Also exports the protein sequence in a FASTA file.
- **Bins the resulting images** according to the protein length (number of residues) to keep image sizes manageable.
- **Uses multiprocessing** to parallelize processing and automatically deletes original *.pdb/*.cif files after successful conversion.

### Key Sections

#### Configuration & Mappings

- A dictionary `non_standard_to_standard` is used to map nonstandard residues to standard residue names.
- Another dictionary `three_to_one_letter` maps three-letter codes (e.g., `VAL`) to one-letter codes (e.g., `V`).
- The code ensures multi-chain or multi-model structures are skipped to reduce complexity.

#### Backbone Extraction & 6D Representation

##### 2.1 Coordinate and Geometry Extraction

For each residue, we collect coordinates of the backbone atoms: `N`, `CA`, `C`, and we also estimate `CB`.

- **Approximating \( C_\beta \):**  
  The script uses a geometric approach to approximate the position of \( C_\beta \) if it is not explicitly present. The approximate formula is:

  $$
  C_\beta \approx -0.5827\,\vec{a} \;+\; 0.5680\,\vec{b} \;-\; 0.5407\,\vec{c} \;+\; C_\alpha,
  $$

  where

  $$
  \vec{b} = C_\alpha - N,\quad \vec{c} = C - C_\alpha,\quad \vec{a} = \vec{b} \times \vec{c}.
  $$

  This is a known approximation in structural biology for generating a placeholder side-chain direction.

##### 2.2 Distance and Angles

Each pair of residues \((i, j)\) is described by:

- **Distance between \( C_{\beta_i} \) and \( C_{\beta_j} \):**

  $$
  d_{ij} = \|\,C_{\beta_j} - C_{\beta_i}\|.
  $$

- **Dihedral angles \(\omega\) and**
