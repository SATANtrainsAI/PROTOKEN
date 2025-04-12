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

- **Dihedral angles \(\omega\) and \(\theta\):**  
  Using the standard 4-point dihedral formula (with vectors \(N\), \(CA\), \(C\), \(C_\beta\)). If we define `get_dihedrals(a, b, c, d)`, then we compute:

  $$
  \begin{aligned}
  b_0 &= -\,(b - a), \\
  b_1 &= c - b, \quad \text{(normalized such that } b_1 \leftarrow \frac{b_1}{\|b_1\|}\text{)}, \\
  b_2 &= d - c, \\
  v &= b_0 - (b_0 \cdot b_1)\,b_1, \\
  w &= b_2 - (b_2 \cdot b_1)\,b_1, \\
  x &= v \cdot w, \\
  y &= (b_1 \times v) \cdot w, \\
  \text{dihedral} &= \arctan2\,(y,\, x).
  \end{aligned}
  $$

- **Planar angle \(\phi\):**  
  This is computed using three points \(\bigl(C_{\alpha_i},\; C_{\beta_i},\; C_{\beta_j}\bigr)\) to measure the angle \(\angle_{\,i_{Cb},\,i_{Ca},\,j_{Cb}}\).

These angles and distances are stored in separate 2D arrays of shape \((L, L)\) if the protein has \(L\) residues. The script normalizes them (e.g., dividing distances by a maximum range, scaling angles by \(\pi\)) and merges some channels to limit the final image shape.

#### Normalization & Image Generation

- **Distances** are scaled to \([-1, 1]\) based on a maximum \( d_{\max} \) (e.g., 80.0 Å).
- **Angles** are scaled or offset so that, for example, \(\omega \in [-\pi, \pi]\) becomes \(\omega/\pi\), etc.
- The channels are merged to form a \(3 \times L \times L\) array, which is then saved as a PNG using `PIL.Image`.

#### FASTA Output

- For each structure, a corresponding FASTA file is generated to store the linear sequence of the protein for reference.

#### Binning by Residue Count

- Proteins are grouped into directories such as `bin_40_64/` or `bin_65_128/` to limit the size of the 2D images.
- For example, a 65-residue protein leads to a \(3 \times 65 \times 65\) image, stored in `bin_65_128/`.

#### Parallel Processing & Deletion

- The script uses `multiprocessing.Pool` for parallelism.
- After successfully generating the PNG and FASTA, the original *.pdb/*.cif file is deleted to free space.

---

## 2. Dataloader & Infinite Streaming

**File:** `dataloader.py`

### Purpose

- Provides **Iterable Datasets** that stream large volumes of images without loading them all at once.
- Uses chunked or streaming shuffling to feed data in random order.
- Handles train/validation splits in a **deterministic** way based on the MD5 hash of the file path.
- Supports multi-processing and Distributed Data Parallel (DDP) to shard data among workers and ranks.

### Key Mechanics

#### IterableImageDataset

- **Walks through a single “bin” folder** of PNGs.
- Maintains a shuffle buffer to randomize the order.
- Applies standard PyTorch transforms (e.g., `T.ToTensor()`, normalization).
- Partitions data into “train” or “val” subsets by hashing each filename and comparing against a given `valid_frac`.

#### IterableImageDatasetUni

- Similar to the above, but loops over multiple bin folders (e.g., `bin_40_64/`, `bin_65_128/`, etc.).
- Optionally skips certain bins if desired.

#### Batch Collation & Padding

- The function `collate_with_padding` ensures that each batch is zero-padded to a dimension that is a multiple of a specified `patch_size`, so that the subsequent model can operate on uniformly sized inputs.
- This is crucial when training Transformers that assume consistent input dimensions.

#### Infinite Loader

- The function `cycle(dl)` yields data from a DataLoader in an endless loop.
- This is useful for large-scale or indefinite training cycles.

---

## 3. Hierarchical VQ-VAE & Transformer Blocks

**Files:**

- **`model.py`** – Contains the core hierarchical VQ-VAE, Transformer blocks, quantization modules, etc.
- **`train.py`** – The training script that orchestrates gradient updates, logging, checkpointing, and other training logistics.

### 3.1. Transformer Blocks & Rotary Embedding

#### Multi-Head Attention with Rotary Embedding

- The code uses standard multi-head self-attention, partitioning the embedding dimension `n_embd` into `n_head` heads.
- **Queries \( Q \), Keys \( K \), and Values \( V \)** each have a dimension given by

  $$
  \text{latent\_dim} = n_{\text{head}} \times \left(\frac{\text{latent\_dim}}{n_{\text{head}}}\right).
  $$

- The attention scores are computed as:

  $$
  \text{Attention}(Q, K, V) = \text{softmax}\!\Bigl(\frac{Q\,K^\top}{\sqrt{d_{\text{head}}}}\Bigr)V,
  $$

  where

  $$
  d_{\text{head}} = \frac{\text{latent\_dim}}{n_{\text{head}}}.
  $$

- **Rotary Embedding:**  
  The module `rotary_embedding_torch` is used to incorporate relative positional information by rotating the \( Q/K \) vectors depending on position indices. This is a more continuous version of positional encoding that can elegantly handle 2D sequences.

#### MLP Feedforward

- A two-layer feedforward block with a hidden dimension `mlp_hidden_dim` (typically \(2 \times n_{\text{embd}}\)), using a GELU activation.

#### Residual + LayerNorm

- Standard Transformer architecture: each block has two residual connections (one around the multi-head self-attention, one around the MLP), each preceded by Layer Normalization.

### 3.2. Vision Transformer Encoders / Decoders

Rather than splitting images into fixed-size patches (as in a traditional Vision Transformer), the code uses convolution-based downsampling (or upsampling) to reduce (or restore) resolution by factors of 2 multiple times.

#### VitEncoder

- A **DownsampleStack** repeatedly halves the resolution using `Conv2d(stride=2)`. For example, if the `downscale_factor` is 2, it will be applied once; if 4, it may be applied twice, etc.
- The resulting feature map is flattened to a shape of \((B,\, H \times W,\, C)\).
- The flattened features are passed through several Transformer Blocks (self-attention).
- The output is then reshaped back to \((B,\, C,\, H,\, W)\).

#### VitDecoder

- This module takes features from the codebook (or a concatenation of multiple code levels).
- A 1×1 convolution is applied to project the features to the base dimension `n_embd`.
- The projected features are then processed through several Transformer Blocks.
- An **UpsampleStack** with `ConvTranspose2d(stride=2)` is used to repeatedly upscale the features until the original resolution is reached.

### 3.3. Vector Quantization (VQ) and Codebook

The `Quantize` module uses a codebook

$$
\mathbf{E} \in \mathbb{R}^{\text{codebook\_dim} \times \text{codebook\_size}}
$$

to quantize the feature maps. The main steps are:

1. **Projection:**  
   Project the input of shape \((B,\, \text{in\_channels},\, H,\, W)\) to a new tensor of shape \((B,\, \text{codebook\_dim},\, H,\, W)\).

2. **Flattening:**  
   Flatten each spatial location so that the tensor becomes a set of vectors.

3. **Nearest Code Selection:**  
   Compute the squared Euclidean distance from each vector to every codebook entry and use either argmin or (equivalently) argmax (of the negative distances) to select the nearest code.

4. **Quantization:**  
   Replace each feature vector with its corresponding codebook embedding.

5. **EMA Updates:**  
   Update the codebook entries using Exponential Moving Average (EMA) to keep them stable (as described in Oord et al., "Neural Discrete Representation Learning").

6. **Commitment Loss:**  
   Enforce that the encoder output stays close to the discrete codes by applying the loss:

   $$
   L_{\text{commit}} = \Bigl\|\text{sg}\bigl[z_q\bigr] - z_e\Bigr\|^2,
   $$

   where \(\text{sg}(\cdot)\) is the stop-gradient operator, \(z_q\) is the quantized (nearest) code, and \(z_e\) is the continuous encoder output.

### 3.4. Hierarchical Approach

- The model is organized into multiple levels \(\ell = 0, 1, 2, \dots\), where each level compresses the image at a successively coarser scale.
- The top-level code has the smallest spatial resolution (e.g., \(\frac{1}{8}\) or \(\frac{1}{16}\) of the original).
- During decoding, the top-level code is first decoded into a coarse representation. The next level then uses a combination of the corresponding encoder output from that level **plus** the upsampled coarse-level decoded output, followed by codebook quantization, and this process is repeated down to the finest scale.

### 3.5. Loss Function

- **Reconstruction Loss:**

  $$
  L_{\text{recon}} = \|\hat{x} - x\|^2,
  $$

  where \( x \) is the original image and \(\hat{x}\) is the final decoded (reconstructed) image.

- **VQ Commitment Loss:**  
  The commitment losses across all levels are summed as follows:

  $$
  L_{\text{vq}} = \sum_{\ell}\,\Bigl\|\text{sg}\bigl[q_\ell\bigr] - e_\ell\Bigr\|^2.
  $$

- **Total Loss:**

  $$
  L = L_{\text{recon}} + \beta\, L_{\text{vq}},
  $$

  where \(\beta\) is a hyperparameter (default value 0.25).

### 3.6. Training Script (train.py)

- **DistributedDataParallel (DDP):** Utilized for multi-GPU scaling.
- **Mixed Precision:** Implemented with `autocast` (using float16 or bfloat16) to increase training efficiency.
- **WandB Logging:** Used for logging training curves and reconstruction samples.
- **Checkpointing:** Saves the model's `state_dict`, optimizer state, and iteration count to allow resuming training.

---

## Usage & Utility

### Converting PDB/CIF to Distograms

1. Run `pdb_to_png_distogram.py`.
2. Adjust input directories, output directories, and bin ranges as needed.
3. This will produce a large collection of *.png images in `bin_*_*` folders along with corresponding *.fasta sequences.

### Training the Hierarchical VQ-VAE

1. Adjust the arguments in `train.py` (e.g., batch size, learning rate).
2. Specify the input path to the generated images (the parent folder containing multiple `bin_*_*` subdirectories).
3. Run the training script, for example:
