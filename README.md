# Protein Structure Tokenization and Representation via Autoencoding Transformers:
This repository is an end-to-end framework and a novel approach for protein structre backbone tokenization via VQVAE:


## Protein Structure to Image

- **Conversion Script:**  
  The `img_maker.py` script reads protein structures (PDB/CIF files) and extracts backbone atom coordinates (N, CA, C) while estimating the CB position. It computes inter-residue distances and angular features that are encoded into a three-channel (RGB) distogram image, with each channel capturing distinct geometric information.(for more information check out `info.pdf` file)

- **Outputs:**  
  Generates PNG images and corresponding FASTA files (protein sequences), organizing them into bins by residue count for size consistency.

- **Efficiency:**  
  Parallel processing via multiprocessing is used to handle large datasets and automatically cleans up original files.

## Hierarchical VQ-VAE Training

- **Model Architecture:**  
  The model employs a Vision Transformer encoder with latent attention and rotary positional embeddings, using a UNet-style downsampling to encode the images. A hierarchical, multilevel vector quantization captures both coarse global features and fine local details.

- **Decoding & Refinement:**  
  The decoder upsamples features using attention-based processing, complemented by a refinement head to enhance reconstruction quality.

## Applications

This framework is designed for tokenizarion and representation learning of protein geometry via deep generative modeling. The learned discrete representations can be applied in generative design, protein classification, or as latent embeddings in downstream structure-based machine learning tasks.

