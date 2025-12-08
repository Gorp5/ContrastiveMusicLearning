# **Myna-RPE: Relative Positional Embeddings for Track-Level Music Representations**

This repository contains an implementation of **Myna-RPE**, an extension of the Myna masked contrastive learning framework for audio representation learning.  
Myna-RPE introduces **Relative Positional Embeddings (RPE)** — including **2D ALiBi**, **1D ALiBi**, and **RoPE** variants — enabling **end-to-end track-level embeddings** from full-length mel-spectrograms **without chunking or aggregation**.

This code reproduces the methods and experiments described in:

> **Relative Positional Embeddings for Track-Level Representations in Masked Contrastive Learning**

---

##  Overview

Modern self-supervised music models typically train on short fixed-length segments and later average multiple embeddings at inference time — losing long-range structure.

**Myna-RPE removes this limitation** by adapting Relative Positional Embedding algorithms to ViT-based audio encoders under patchout masking.  
This enables models that can:

-  Process **entire tracks** in a single forward pass  
-  **Extrapolate** to spectrograms longer than those seen during training  
-  Preserve **global structure** across time and frequency  
-  Improve **downstream MIR performance** on multiple benchmarks  

---

##  Features

Myna-RPE is fully modular — each positional embedding scheme can be toggled independently.

- Full PyTorch implementation of:
  - **2D ALiBi**
  - **1D ALiBi**
  - **Learned frequency embeddings**
  - **RoPE** positional embeddings
  - **Sinusoidal** positional embeddings
- Patchout-aware RPE (coordinates preserved after masking)
- Myna-style ViT encoder with CLS token
- Contrastive pretraining with **InfoNCE**
- Mel-spectrogram preprocessing pipeline
- Training scripts for **MTG-Jamendo Top-50 Tags**
- Evaluation scripts for:
  - GTZAN (genre)
  - GiantSteps Key (key detection)
  - EmoMusic (emotion regression)

---

##  Results (Linear Probes)

| Model               | GTZAN Acc | GiantSteps Acc | EmoMusic A | EmoMusic V | Avg      |
| ------------------- | --------- | -------------- | ---------- | ---------- | -------- |
| 1D ALiBi + F-Embed  | 74.87     | **82.57**      | 59.37      | 43.45      | –        |
| **2D ALiBi (Ours)** | **78.39** | 76.50          | **68.06**  | **44.05**  | **≈66%** |

**2D ALiBi** shows consistent improvements over 1D, particularly for **genre classification** and **emotion prediction**.

---

##  Method

Myna-RPE builds on the Myna contrastive learning framework (itself extending CLMR).  
The goal is to learn discriminative audio representations via **self-supervised contrastive learning**.

### Training Pipeline

For each batch:

1. Convert a track to a mel-spectrogram.  
2. Sample two random fixed-length chunks → **positive pair**.  
3. Treat chunks from different tracks as **negative pairs**.  
4. Patchify the spectrogram and apply **patchout masking**.  
5. Feed tokens into a ViT encoder.  
6. Apply **InfoNCE** to make positives similar & negatives dissimilar.
7. Basic scripts to interface with the Spotify API

This trains the encoder to model high-level musical structure while remaining robust to masking and augmentation.

### Encoder Architecture

A modified Audio Spectrogram Transformer (from Myna) featuring:

- A prepended **CLS token** for global pooling
- **16×16 non-overlapping patches**
- **Patchout** (random token dropout)
- Standard Transformer encoder layers
- **No mean pooling** — only CLS is used

The key innovation:
> **Absolute positional embeddings are replaced with Relative Positional Embeddings**, allowing the model to process arbitrary-length spectrograms — including **full songs**.

---

##  Further Exploration

Several promising research directions extend naturally from Myna-RPE:

### **1. Latent Space Structuring via K-Means and Convex Hull Losses**
Building on the techniques described in  
**“Convex Hull and K-Means Loss for Self-Supervised Representations” (Eng. Appl. AI, 2024 — https://doi.org/10.1016/j.engappai.2024.108612)**,  
Myna-RPE can incorporate additional geometric constraints on the embedding space.  
These losses encourage:

- **K-Means Loss:** tighter, more coherent clusters of songs or track segments  
- **Convex Hull Loss:** embeddings that expand to better capture diversity within musical categories

This could improve downstream retrieval, tagging, and clustering tasks beyond contrastive-only training.

---

### **2. Training on Variable-Length Song Segments**
Instead of sampling fixed-length chunks, an extended Myna-RPE could:

- Randomly sample **variable-length excerpts**  
- Mix short phrases, mid-length clips, and near-full sections  
- Train encoders to adapt seamlessly to the natural duration variability in music

Because RPE allows arbitrary sequence length, the model can generalize across varying temporal scales without modification.

---

### **3. Playlist Generation via Convex Hull Geometry**
A geometry-aware embedding space opens the door to **playlist and collection modeling**:

- Treat a playlist as a **convex hull** enclosing the embeddings of its songs  
- Measure whether a new track lies **inside**, **near**, or **outside** the playlist hull  
- Generate playlists by:
  - selecting songs whose embeddings fill out a target hull shape  
  - expanding a playlist’s convex hull and sampling points near its boundary  
  - using hull interpolation to create *thematic transitions* between playlists

This yields playlists driven by **global musical geometry**, not just nearest neighbors.

---

All models were trained using a single A100 for 512 epochs in parallel. Compute was gratefully sourced from Yonsei University's ai compute cluster (Thank you!)

---

##  License

MIT License  

---
