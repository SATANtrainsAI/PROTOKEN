# Protein Structure Tokenization and Representation via Autoencoding Transformers:
This repo intorduces a novel approach for protein structre backbone tokenization via VQVAE:

# Repository Overview

This repository is an end-to-end framework for processing protein structure data and learning compact representations using hierarchical vector quantized autoencoders.

## Protein Structure to Image

- **Conversion Script:**  
  The `pdb_to_png_distogram.py` script reads protein structures (PDB/CIF files) and extracts backbone atom coordinates (N, CA, C) while estimating the CB position. It computes inter-residue distances and angular features that are encoded into a three-channel (RGB) distogram image, with each channel capturing distinct geometric information.

- **Outputs:**  
  Generates PNG images and corresponding FASTA files (protein sequences), organizing them into bins by residue count for size consistency.

- **Efficiency:**  
  Parallel processing via multiprocessing is used to handle large datasets and automatically cleans up original files.

## Hierarchical VQ-VAE Training

- **Model Architecture:**  
  The model employs a Vision Transformer encoder with latent attention and rotary positional embeddings, using a UNet-style downsampling to encode the images. A hierarchical, multilevel vector quantization captures both coarse global features and fine details.

- **Decoding & Refinement:**  
  The decoder upsamples features using attention-based processing, complemented by a refinement head to enhance reconstruction quality.

- **Training Pipeline:**  
  Managed by `train.py` and `dataloader.py`, the training framework supports efficient data streaming, Distributed Data Parallel (DDP) training, and mixed precision, with integrated logging and checkpointing.

## Applications

This framework is designed for analyzing protein geometry via deep generative modeling. The learned discrete representations can be applied in generative design, protein classification, or as latent embeddings in downstream structure-based machine learning tasks.

