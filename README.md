\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath, amssymb}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{enumitem}
\usepackage{hyperref}
\usepackage{parskip}

\geometry{margin=1in}

\title{Repository Overview: Protein Structure to Image and Hierarchical VQ-VAE Training}
\author{}
\date{}

\begin{document}

\maketitle

\section*{Repository Overview}

This repository contains two major parts:
\begin{enumerate}[label=\arabic*.]
  \item \textbf{Protein Structure to Image} \\
    A Python script (\texttt{pdb\_to\_png\_distogram.py}) for parsing protein structures (\texttt{*.pdb} or \texttt{*.cif}) and converting them into PNG “distogram” images (plus associated FASTA files). These images capture backbone distances and angular features.
    
  \item \textbf{Hierarchical VQ-VAE Training} \\
    A hierarchical VQ-VAE model implemented with PyTorch and Transformer blocks (\texttt{model.py}, \texttt{train.py}, and \texttt{dataloader.py}). This model is designed to learn quantized latent representations of the generated protein images.
\end{enumerate}

Below is an overview of the file structure and detailed explanations of both the implementation and the underlying mathematical concepts.

\section{Protein Structure to Distogram Image}

\subsection*{File: \texttt{pdb\_to\_png\_distogram.py}}

\subsubsection*{Overview}

\begin{itemize}
  \item Reads protein structures (\texttt{*.pdb} or \texttt{*.cif}) using Biotite (or a related library).
  \item Extracts backbone atoms \textbf{N}, \textbf{CA}, and \textbf{C} (and estimates \textbf{CB}).
  \item Computes distances, dihedral angles, and planar angles between residues, forming a 6D representation.
  \item Normalizes and merges these channels into a 3-channel image (RGB). Each pixel $(i,j)$ corresponds to the backbone geometry between residue $i$ and residue $j$.
  \item Saves the result as a PNG image and also exports the protein sequence in a FASTA file.
  \item Bins the resulting images according to the protein length (number of residues) to keep image sizes manageable.
  \item Uses multiprocessing to parallelize processing and automatically deletes original \texttt{*.pdb} or \texttt{*.cif} files after successful conversion.
\end{itemize}

\subsubsection*{Key Sections}

\paragraph{Configuration \& Mappings}
\begin{itemize}
  \item A dictionary \texttt{non\_standard\_to\_standard} is used to map nonstandard residues to standard residue names.
  \item Another dictionary \texttt{three\_to\_one\_letter} maps three-letter codes (e.g., \texttt{VAL}) to one-letter codes (e.g., \texttt{V}).
  \item The code ensures multi-chain or multi-model structures are skipped to reduce complexity.
\end{itemize}

\paragraph{Backbone Extraction \& 6D Representation}

\subparagraph{2.1 Coordinate and Geometry Extraction}

For each residue, we collect coordinates of the backbone atoms: \texttt{N}, \texttt{CA}, \texttt{C}, and we also estimate \texttt{CB}.

\textbf{Approximating \( C_\beta \):}\\[1ex]
The script uses a geometric approach to approximate the position of \( C_\beta \) if it is not explicitly present. The approximate formula is
\[
C_\beta \approx -0.5827\,\vec{a} \;+\; 0.5680\,\vec{b} \;-\; 0.5407\,\vec{c} \;+\; C_\alpha,
\]
where
\[
\vec{b} = C_\alpha - N,\quad \vec{c} = C - C_\alpha,\quad \vec{a} = \vec{b} \times \vec{c}.
\]
This is a known approximation in structural biology for generating a placeholder side-chain direction.

\subparagraph{2.2 Distance and Angles}

Each pair of residues $(i, j)$ is described by:

\begin{itemize}
  \item \textbf{Distance between \( C_{\beta_i} \) and \( C_{\beta_j} \):}
    \[
    d_{ij} = \|\,C_{\beta_j} - C_{\beta_i}\|.
    \]
  \item \textbf{Dihedral angles \(\omega\) and \(\theta\):}\\[1ex]
    Using the standard 4-point dihedral formula (with vectors \(N\), \(CA\), \(C\), \(C_\beta\)). If we define \texttt{get\_dihedrals(a, b, c, d)}, then we compute:
    \[
    \begin{aligned}
    b_0 &= -\,(b - a), \\
    b_1 &= c - b,\quad \text{(normalized so that } b_1 \leftarrow \frac{b_1}{\|b_1\|}\text{)}, \\
    b_2 &= d - c, \\
    v &= b_0 - (b_0 \cdot b_1)\,b_1, \\
    w &= b_2 - (b_2 \cdot b_1)\,b_1, \\
    x &= v \cdot w, \\
    y &= (b_1 \times v) \cdot w, \\
    \text{dihedral} &= \arctan2\,(y,\,x).
    \end{aligned}
    \]
  \item \textbf{Planar angle \(\phi\):}\\[1ex]
    This angle is computed using three points \(\bigl(C_{\alpha_i},\; C_{\beta_i},\; C_{\beta_j}\bigr)\) to measure the angle \(\angle_{\,i_{Cb},\,i_{Ca},\,j_{Cb}}\).
\end{itemize}

These angles and distances are stored in separate 2D arrays of shape $(L, L)$ if the protein has $L$ residues. The script normalizes these values (e.g., dividing distances by a maximum range and scaling angles by $\pi$) and merges some channels to limit the final image shape.

\subsubsection*{Normalization \& Image Generation}

\begin{itemize}
  \item \textbf{Distances} are scaled to the range \([-1, 1]\) based on a maximum value \( d_{\max} \) (e.g., 80.0 \AA).
  \item \textbf{Angles} are scaled or offset so that, for example, \(\omega \in [-\pi, \pi]\) becomes \(\omega/\pi\).
  \item The channels are merged to form a \(3 \times L \times L\) array, which is saved as a PNG using the \texttt{PIL.Image} module.
\end{itemize}

\subsubsection*{FASTA Output}

For each structure, a corresponding FASTA file is generated to store the linear sequence of the protein for reference.

\subsubsection*{Binning by Residue Count}

\begin{itemize}
  \item Proteins are grouped into directories such as \texttt{bin\_40\_64/} or \texttt{bin\_65\_128/} to limit the size of the 2D images.
  \item For example, a 65-residue protein leads to a \(3 \times 65 \times 65\) image that is stored in \texttt{bin\_65\_128/}.
\end{itemize}

\subsubsection*{Parallel Processing \& Deletion}

\begin{itemize}
  \item The script uses \texttt{multiprocessing.Pool} for parallelism.
  \item After successfully generating the PNG and FASTA files, the original \texttt{*.pdb} or \texttt{*.cif} file is deleted to free up space.
\end{itemize}

\section{Dataloader \& Infinite Streaming}

\subsection*{File: \texttt{dataloader.py}}

\subsubsection*{Purpose}

\begin{itemize}
  \item Provides \textbf{Iterable Datasets} that stream large volumes of images without loading them all at once.
  \item Uses chunked or streaming shuffling to feed data in random order.
  \item Handles train/validation splits in a \textbf{deterministic} way based on the MD5 hash of the file path.
  \item Supports multi-processing and Distributed Data Parallel (DDP) to shard data among workers and ranks.
\end{itemize}

\subsubsection*{Key Mechanics}

\paragraph{IterableImageDataset}
\begin{itemize}
  \item Walks through a single ``bin'' folder of PNG images.
  \item Maintains a shuffle buffer to randomize the order.
  \item Applies standard PyTorch transforms (e.g., \texttt{T.ToTensor()}, normalization).
  \item Partitions data into ``train'' or ``val'' subsets by hashing each filename and comparing the hash value against a specified \texttt{valid\_frac}.
\end{itemize}

\paragraph{IterableImageDatasetUni}
\begin{itemize}
  \item Similar to \texttt{IterableImageDataset} but loops over multiple bin folders (e.g., \texttt{bin\_40\_64/}, \texttt{bin\_65\_128/}, etc.).
  \item Optionally skips certain bins if desired.
\end{itemize}

\paragraph{Batch Collation \& Padding}
\begin{itemize}
  \item The function \texttt{collate\_with\_padding} ensures each batch is zero-padded to a dimension that is a multiple of a defined \texttt{patch\_size} so that the subsequent model can operate on uniformly sized inputs.
  \item This is especially important when training Transformers that assume consistent input dimensions.
\end{itemize}

\paragraph{Infinite Loader}
\begin{itemize}
  \item The function \texttt{cycle(dl)} yields data from a DataLoader in an endless loop.
  \item This mechanism is useful for large-scale or indefinite training cycles.
\end{itemize}

\section{Hierarchical VQ-VAE \& Transformer Blocks}

\subsection*{Files:}
\begin{itemize}
  \item \texttt{model.py} --- Contains the core hierarchical VQ-VAE, Transformer blocks, quantization modules, etc.
  \item \texttt{train.py} --- The training script orchestrating gradient updates, logging, checkpointing, and other training logistics.
\end{itemize}

\subsection{Transformer Blocks \& Rotary Embedding}

\subsubsection*{Multi-Head Attention with Rotary Embedding}

\begin{itemize}
  \item The code uses standard multi-head self-attention, partitioning the embedding dimension \texttt{n\_embd} into \texttt{n\_head} heads.
  \item \textbf{Queries} \( Q \), \textbf{Keys} \( K \), and \textbf{Values} \( V \) each have a dimension given by
    \[
    \text{latent\_dim} = n_{\text{head}} \times \left(\frac{\text{latent\_dim}}{n_{\text{head}}}\right).
    \]
  \item The attention scores are computed as:
    \[
    \text{Attention}(Q, K, V) = \text{softmax}\!\Biggl(\frac{Q\,K^\top}{\sqrt{d_{\text{head}}}}\Biggr)V,
    \]
    where
    \[
    d_{\text{head}} = \frac{\text{latent\_dim}}{n_{\text{head}}}.
    \]
  \item \textbf{Rotary Embedding:} \\
    The module \texttt{rotary\_embedding\_torch} is used to incorporate relative positional information by rotating the \( Q \) and \( K \) vectors in a manner that depends on position indices. This provides a more continuous version of positional encoding capable of handling 2D sequences elegantly.
\end{itemize}

\subsubsection*{MLP Feedforward}

\begin{itemize}
  \item A two-layer feedforward block with hidden dimension \texttt{mlp\_hidden\_dim} (typically \(2 \times n_{\text{embd}}\)), utilizing a GELU activation function.
\end{itemize}

\subsubsection*{Residual \& LayerNorm}

\begin{itemize}
  \item Standard Transformer architecture with two residual connections per block:
    \begin{itemize}
      \item One residual branch around the multi-head self-attention sub-layer.
      \item One residual branch around the MLP sub-layer.
    \end{itemize}
  \item Each residual branch is preceded by Layer Normalization.
\end{itemize}

\subsection{Vision Transformer Encoders / Decoders}

Rather than splitting images into fixed-size patches (as in a traditional Vision Transformer), the code uses convolution-based downsampling (or upsampling) to reduce (or restore) resolution by factors of 2 multiple times.

\subsubsection*{VitEncoder}

\begin{itemize}
  \item Implements a \textbf{DownsampleStack} that repeatedly halves the resolution using \texttt{Conv2d} with \texttt{stride=2}. For example, if \texttt{downscale\_factor} is 2, the convolution is applied once; if it is 4, it may be applied twice, etc.
  \item The resulting feature map is flattened into a shape \((B,\, H \times W,\, C)\), where \(B\) is the batch size.
  \item The flattened features are processed through several Transformer Blocks (self-attention).
  \item Finally, the sequence is reshaped back to \((B,\, C,\, H,\, W)\).
\end{itemize}

\subsubsection*{VitDecoder}

\begin{itemize}
  \item Takes features from the codebook (or a concatenation of multiple code levels).
  \item Applies a 1×1 convolution to project the features to the base dimension \texttt{n\_embd}.
  \item Processes the projected features through several Transformer Blocks.
  \item Uses an \textbf{UpsampleStack} with \texttt{ConvTranspose2d} (with \texttt{stride=2}) repeatedly to upscale the feature maps until the original resolution is reached.
\end{itemize}

\subsection{Vector Quantization (VQ) and Codebook}

The \texttt{Quantize} module uses a codebook 
\[
\mathbf{E} \in \mathbb{R}^{\text{codebook\_dim} \times \text{codebook\_size}}
\]
to quantize the feature maps. The main idea is as follows:

\begin{enumerate}
  \item \textbf{Projection:} \\
    Project the input tensor of shape \((B,\, \text{in\_channels},\, H,\, W)\) to a tensor of shape \((B,\, \text{codebook\_dim},\, H,\, W)\).
  \item \textbf{Flattening:} \\
    Flatten each spatial location so that the tensor becomes a collection of vectors.
  \item \textbf{Nearest Code Selection:} \\
    Compute the squared Euclidean distance from each vector to every codebook entry and use \texttt{argmin} (or equivalently, \texttt{argmax} of the negative distance) to select the nearest code.
  \item \textbf{Quantization:} \\
    Replace each feature vector with its corresponding nearest codebook embedding.
  \item \textbf{EMA Updates:} \\
    Update the codebook entries using an Exponential Moving Average (EMA) to keep them stable (refer to Oord et al., \emph{Neural Discrete Representation Learning}).
  \item \textbf{Commitment Loss:} \\
    Enforce that the encoder output remains close to the discrete codes by computing the loss:
    \[
    L_{\text{commit}} = \Bigl\|\text{sg}\bigl[z_q\bigr] - z_e\Bigr\|^2,
    \]
    where \(\text{sg}(\cdot)\) is the stop-gradient operator, \(z_q\) is the quantized (nearest) code, and \(z_e\) is the continuous encoder output.
\end{enumerate}

\subsection{Hierarchical Approach}

\begin{itemize}
  \item The model is structured into multiple levels \(\ell = 0, 1, 2, \dots\), with each level compressing the image at a successively coarser scale.
  \item The top-level code has the smallest spatial resolution (e.g., \(\frac{1}{8}\) or \(\frac{1}{16}\) of the original image).
  \item During decoding, the top-level code is first decoded into a coarse representation. The next level then uses a combination of the corresponding encoder output at that level \textbf{plus} the upsampled coarse-level decoded output, followed by codebook quantization. This process is repeated until the finest scale is reached.
\end{itemize}

\subsection{Loss Function}

\subsubsection*{Reconstruction Loss}

\[
L_{\text{recon}} = \|\hat{x} - x\|^2,
\]
where \(x\) is the original image and \(\hat{x}\) is the final decoded (reconstructed) image.

\subsubsection*{VQ Commitment Loss}

The commitment losses from all levels are summed:
\[
L_{\text{vq}} = \sum_{\ell} \Bigl\|\text{sg}\bigl[q_\ell\bigr] - e_\ell\Bigr\|^2.
\]

\subsubsection*{Total Loss}

\[
L = L_{\text{recon}} + \beta\, L_{\text{vq}},
\]
where \(\beta\) is a hyperparameter (default value 0.25).

\subsection{Training Script (train.py)}

\begin{itemize}
  \item \textbf{DistributedDataParallel (DDP):} \\
    The training script uses DDP for multi-GPU scaling.
  \item \textbf{Mixed Precision:} \\
    Implemented with \texttt{autocast} (using \texttt{float16} or \texttt{bfloat16}) to improve training efficiency.
  \item \textbf{WandB Logging:} \\
    Used for logging training curves and saving reconstruction sample images.
  \item \textbf{Checkpointing:} \\
    The script saves the model’s \texttt{state\_dict}, optimizer state, and iteration count to facilitate resuming training.
\end{itemize}

\section{Usage \& Utility}

\subsection{Converting PDB/CIF to Distograms}

\begin{enumerate}
  \item Run \texttt{pdb\_to\_png\_distogram.py}.
  \item Adjust input directories, output directories, and bin ranges as needed.
  \item This will generate a large collection of \texttt{*.png} images in the \texttt{bin\_*_*} folders, along with corresponding \texttt{*.fasta} sequences.
\end{enumerate}

\subsection{Training the Hierarchical VQ-VAE}

\begin{enumerate}
  \item Adjust arguments in \texttt{train.py} (e.g., batch size, learning rate).
  \item Specify the input path to the generated images (i.e., the parent folder containing multiple \texttt{bin\_*_*} subdirectories).
  \item Run the training script:
    \begin{center}
      \texttt{python train.py --folder /path/to/images \ldots}
    \end{center}
  \item The script will stream data in a memory-efficient manner and train a multi-level VQ-VAE model to compress the 2D distograms.
\end{enumerate}

\subsection{Applications}

\begin{itemize}
  \item This pipeline is useful for analyzing and learning compressed representations of protein geometry.
  \item The resulting discrete codes can be used in generative modeling or as latent embeddings for tasks such as protein classification or structure-based machine learning.
\end{itemize}

\end{document}
