# Protein Structure Tokenization and Representation via Autoencoding Transformers:
This repo intorduces a novel approach for protein structre backbone tokenization via VQVAE:

This repository contains two major parts:

1. **Protein Structure to Image**  
   A Python script (`img_maker.py`) for parsing protein structures (files with extensions .pdb or .cif) and converting them into PNG “distogram” images (with accompanying FASTA files). These images capture backbone distances and angular features.

2. **Hierarchical VQ-VAE Training**  
   A hierarchical VQ-VAE model implemented in PyTorch using Transformer blocks (files include `model.py`, `train.py`, and `dataloader.py`). This model learns quantized latent representations from the generated protein images.

Below is an overview of the file structure and detailed explanations of the implementation and its underlying concepts. for more mathematical details, check out the `info.pdf` file.

---

## 1. Protein Structure to Distogram Image

**File:** `pdb_to_png_distogram.py`

### Overview

- **Input:** Reads protein structures from .pdb or .cif files using Biotite (or a similar library).
- **Atom Extraction:** Extracts backbone atoms (N, CA, C) and estimates the CB (beta carbon) if it is not directly provided.
- **Feature Computation:** Computes distances between beta carbons, as well as dihedral and planar angles between residues, creating a six-dimensional representation.
- **Normalization and Merging:** Normalizes the computed distances and angles, and merges them into a three-channel (RGB) image where each pixel corresponds to the geometric relationship between two residues.
- **Output:** Saves the image as a PNG file and writes out a FASTA file containing the protein sequence.
- **Binning:** Organizes images into directories based on the number of residues (for example, files with 40 to 64 residues, 65 to 128 residues, etc.) to maintain manageable image sizes.
- **Parallel Processing:** Uses Python’s multiprocessing to process files in parallel and deletes the original structure files after successful conversion.

### Key Sections

#### Configuration & Mappings

- Maps nonstandard residue names to standard ones via a dictionary.
- Converts three-letter residue codes to one-letter codes using a second dictionary.
- Skips structures with multiple chains or models to reduce complexity.

#### Backbone Extraction & 6D Representation

- **Coordinate Extraction:** For each residue, retrieves coordinates for the backbone atoms (N, CA, C) and estimates the CB position using a known geometric approximation.
- **Feature Calculation:**  
  - Computes the distance between estimated beta carbons for each pair of residues.  
  - Calculates dihedral angles using a sequence of four atoms (e.g., N, CA, C, and CB) by determining the orientation between two planes.  
  - Computes a planar angle using three points (CA, CB, and another CB) to capture the angular relationship.
- **Data Organization:** The distances and angles are stored in two-dimensional arrays (with dimensions corresponding to the number of residues) and then merged to form the final three-channel image.

#### Normalization & Image Generation

- **Normalization:** Scales the distance values to a fixed range and adjusts angle values to standard scales.
- **Image Creation:** Merges the normalized channels into a three-dimensional array with dimensions corresponding to the image channels and the number of residues. The image is then saved in PNG format.
  
#### FASTA Output

- Generates a FASTA file for each structure, which contains the protein’s sequence derived from the one-letter residue codes.

#### Binning by Residue Count

- Groups proteins into different directories based on the number of residues (e.g., images for proteins with 40–64 residues are stored in a directory labeled "bin_40_64").
- This binning helps manage and standardize image sizes.

#### Parallel Processing & Deletion

- Uses Python’s `multiprocessing.Pool` to distribute the processing of many files.
- Once the PNG image and FASTA file are successfully generated, the original protein structure file is deleted to free up disk space.

---

## 2. Dataloader & Infinite Streaming

**File:** `dataloader.py`

### Purpose

- Provides iterable datasets that efficiently stream large numbers of images without requiring all file paths to be loaded into memory.
- Uses techniques such as chunked or streaming shuffling to randomize data order.
- Supports train/validation splits in a deterministic manner using the MD5 hash of file paths.
- Facilitates multi-processing and Distributed Data Parallel (DDP) by sharding the dataset across multiple workers and processing nodes.

### Key Mechanics

#### IterableImageDataset

- Walks through a single bin directory containing PNG images.
- Maintains a buffer to shuffle file paths randomly.
- Applies standard image transformations using PyTorch (such as converting to tensors and normalizing).
- Determines whether a file belongs to the training or validation subset based on a hash of the filename.

#### IterableImageDatasetUni

- Similar to the above, but iterates over multiple bin directories (such as those corresponding to different residue count bins).
- May exclude specific bins if required.

#### Batch Collation & Padding

- Uses a custom collate function (`collate_with_padding`) that zero-pads images so that all inputs in a batch are of the same size, a necessity for training Transformer models with consistent input dimensions.

#### Infinite Loader

- Implements a function (`cycle(dl)`) that wraps a DataLoader in an endless loop, allowing for continuous training over an indefinitely large dataset.

---

## 3. Hierarchical VQ-VAE & Transformer Blocks

**Files:**

- **`model.py`** – Contains the core implementation of the hierarchical VQ-VAE, including Transformer blocks and quantization modules.
- **`train.py`** – The training script that handles gradient updates, logging, checkpointing, and other training processes.

### 3.1. Transformer Blocks & Rotary Embedding

#### Multi-Head Attention with Rotary Embedding

- The model employs multi-head self-attention by splitting the embedding dimensions among several attention heads.
- Queries, keys, and values are generated from the input features, and the attention mechanism computes the relationships between different parts of the input.
- Rotary embedding is applied to the query and key vectors to encode positional information in a smooth and continuous manner, which is especially effective for two-dimensional image data.

#### MLP Feedforward

- Each Transformer block includes a two-layer feedforward network with an expanded hidden dimension and uses GELU activation for non-linearity.

#### Residual Connections & Layer Normalization

- Transformer blocks are equipped with residual connections around both the self-attention and the feedforward sub-layers.
- Layer normalization is applied before these sub-layers to stabilize training.

### 3.2. Vision Transformer Encoders / Decoders

- Instead of dividing the image into fixed-size patches, the model uses convolution-based downsampling to reduce the resolution progressively.
- **VitEncoder:**  
  - Uses a sequence of convolutional layers (DownsampleStack) with a stride of 2 to reduce the image resolution.
  - The output is flattened and processed by Transformer blocks before being reshaped back into image form.
- **VitDecoder:**  
  - Begins with features from the quantized representation and uses a 1×1 convolution to project these features.
  - Transformer blocks then process the projected features, followed by a series of transposed convolution layers (UpsampleStack) that progressively restore the original image resolution.

### 3.3. Vector Quantization (VQ) and Codebook

- The quantization module creates a discrete codebook that represents the learned features.
- The steps include:
  - Projecting input feature maps to a lower-dimensional space.
  - Flattening spatial locations to create a set of feature vectors.
  - Finding the nearest codebook vector for each feature (using distance metrics such as the squared Euclidean distance).
  - Replacing feature vectors with their nearest codebook embeddings.
  - Using an Exponential Moving Average (EMA) to update the codebook vectors during training.
  - Applying a commitment loss to encourage the encoder outputs to remain close to the chosen codebook vectors.

### 3.4. Hierarchical Model Architecture

- The model is designed with multiple hierarchical levels where each level compresses the image further.
- The top level has the coarsest representation, while lower levels refine the output.
- During decoding, the process begins with the coarsest representation and gradually adds finer details by incorporating outputs from lower levels, including upsampled features from previous decoding stages.

### 3.5. Loss Function

- **Reconstruction Loss:** Measures the difference between the original image and the final reconstructed image.
- **VQ Commitment Loss:** Sums the losses from all levels that measure the distance between the encoder outputs and their corresponding quantized representations.
- The total loss is a weighted sum of the reconstruction and the commitment losses, with a hyperparameter controlling the relative weight.

### 3.6. Training Script (train.py)

- The training script leverages DistributedDataParallel (DDP) to scale training across multiple GPUs.
- It uses mixed precision training (with support for float16 or bfloat16) to enhance training speed.
- Training progress, including loss curves and sample reconstructions, are logged using WandB.
- Checkpointing is implemented to save the model’s state, optimizer state, and the iteration count for resuming training.

