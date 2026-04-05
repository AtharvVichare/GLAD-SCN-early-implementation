# GlAD-SCN — Graph-Level Anomaly Detection via Stable ChebNet

> ** 2026 · ML4Sci · Phase 1 Early Implementation**  
> Unsupervised graph-level anomaly detection for new physics searches at the LHC.  
> Trained exclusively on Standard Model jets — no BSM signal seen during training.

⚠ Note on Current Evaluation Metrics
Quantitative metrics such as AUC-ROC, signal efficiency, and anomaly score separation are directly dependent on the convergence of the SimCLR contrastive loss (L₂). Full convergence of InfoNCE at this scale is computationally intensive — I do not have access to the high computation power required to drive L₂ to its optimal minimum.
Despite this, the model demonstrates meaningful progress even in the under-converged regime:

L₃ successfully reduces the distance between Z_G and Z'_G for SM jets across training, confirmed by PCA visualization
The latent space develops emerging geometric structure, with paired embeddings converging and different jets beginning to separate — indicating that the contrastive objective is functioning directionally even without full convergence

With increased computational resources allowing full SimCLR convergence, the anomaly scoring pipeline is expected to reach its full discriminative potential. The architecture and training protocol are validated — compute is the remaining bottleneck.
---

## Overview

GlAD-SCN detects anomalous jet events at the LHC by learning what a normal Standard Model (SM) jet looks like in graph space, then flagging jets that deviate from that learned distribution at inference time. The model combines a dual-graph encoder (clean + weight-perturbed), a graph autoencoder, SimCLR contrastive learning, and a representation error anomaly score — all without ever using BSM labels during training.



**Dataset:** LHC Olympics 2020 R&D Dataset (`events_anomalydetection_v2.h5`, Zenodo DOI: 10.5281/zenodo.4536377)  
**Framework:** PyTorch · PyTorch Geometric  
**Phase:** 1 (ChebNet backbone) — Phase 2 (Stable ChebNet + improvements) in progress
---
## Using Stable Chebnet
---
Stable ChebNet: ODE Formulation with Antisymmetric Weights
To enable large K without instability, Stable-ChebNet reformulates the layer dynamics as a continuoustime ODE[5]:
<img width="1784" height="400" alt="image" src="https://github.com/user-attachments/assets/5da4072e-8a7a-4943-9aba-451c7aff5736" />

By enforcing antisymmetric weight matrices W⊤k = −Wk and using the symmetric normalised Laplacian, Theorem 3 guarantees purely imaginary Jacobian eigenvalues Re(λi(J)) = 0, ensuring
non-dissipative information propagation: energy is preserved and distant nodes remain sensitive to
far-away inputs.
Discretising with forward Euler and adding a damping term γI yields the Stable-ChebNet update:
<img width="985" height="155" alt="image" src="https://github.com/user-attachments/assets/339d5451-d058-4c6f-9592-6be3fb70406e" />
where ϵ > 0 is the step size. Theorem 4 proves second-order stability:
<img width="445" height="135" alt="image" src="https://github.com/user-attachments/assets/f225e9c7-f106-4dc2-b947-7844da18e746" />


---

## Architecture

```
Input jet graph G  (nodes: pT, η, φ  ·  k-NN edges, k=8)
        │
        ├─────────────────────────┐
        ▼                         ▼
Clean encoder f(θ)        Perturbed encoder f(θ′)
3× ChebNet(K=5)           θ′ = θ + η·Δθ, Δθ~N(0,σ²)
  + BN + ReLU             3× ChebNet(K=5) + BN + ReLU
        │                         │
   z_node, Z_G               Ẑ_G (graph embed)
        │                         │
        ├─────────────────────────┘
        │              │
        ▼              ▼
  Graph autoencoder    SimCLR contrastive loss (L₂)
  structure + attr     Z_G · Ẑ_G → MLP heads → InfoNCE
  decoders → Ĝ · L₁         
        │
        ▼
  Re-encoder f(θ) on Ĝ  (shared weights)
  3× ChebNet(K=5) + BN + ReLU
  → z′_node · Z′_G · L₃
        │
        ▼
  L_total = L₁ + λ₁·L₂ + λ₂·L₃
        │  (training only)
        ▼
  Anomaly score S(G) = Σ_v‖z_node,v − z′_node,v‖²/|V| + ‖Z_G − Z′_G‖²
  (inference only — high S(G) → BSM candidate)
```

---

## Training Protocol

Training follows a two-stage protocol to prevent the contrastive loss from dominating an uninitialised latent space.

**Stage 1 — Warm-up:** L₁ (reconstruction) and L₃ (representation error) only. The autoencoder establishes a stable SM reconstruction baseline before contrastive gradients are introduced.

**Stage 2 — Full training:** All three losses active jointly. L₂ (SimCLR / InfoNCE) is introduced to shape the latent space, pulling same-jet pairs together and pushing different jets apart.

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-4 with cosine annealing |
| Batch size | 256 |
| ChebNet order K | 5 |
| Contrastive temperature τ | 0.5 |
| Perturbation coefficient η | 0.1 |
| Early stopping | Patience 20 (on SM val loss) |

---

## Phase 1 Findings

### 1. Latent space convergence
<img width="1950" height="750" alt="image" src="https://github.com/user-attachments/assets/45f2544b-8717-4ad0-88f8-14252b0d7e46" />

<img width="1950" height="750" alt="image" src="https://github.com/user-attachments/assets/470b8553-363b-43d8-b490-8f25352a0395" />


The PCA visualization of Z_G (clean encoder) and Z'_G (re-encoder on reconstructed graph) shows clear evidence that all three losses work together as intended.

**Pre-training:** Z_G and Z'_G are entirely disjoint — large arbitrary gap across the latent space with no correspondence between original and reconstructed embeddings. Gray lines connecting paired points cross the entire PCA plane.

**After full training (CT mature phase):** Z_G and Z'_G converge substantially. Paired points are close together across the distribution, confirming that L₃ successfully reduces the representation error for SM jets. The model also develops partial geometric structure in the embedding space — jets are not uniformly scattered but begin forming loose directional groupings, indicating that the SimCLR contrastive objective is shaping jet identity in the latent space. The combined action of L₁, L₂, and L₃ produces both faithful reconstruction (small Z_G − Z'_G gap) and emerging discriminative structure.

### 2. Dirichlet energy — encoder stability comparison
<img width="1650" height="750" alt="image" src="https://github.com/user-attachments/assets/d7b1c80a-166d-4169-9100-d46a1ee37cca" />

Dirichlet energy (DE) measures how well node embeddings retain distinct information throughout training. A high, stable DE means nodes remain distinguishable — critical for a fine-grained anomaly score. Collapse toward zero means over-smoothing: all nodes become identical and the encoder loses discriminative power.

The benchmark across all GLAD variants shows a clear hierarchy:

| Model | DE behaviour | Assessment |
|---|---|---|
| **StableChebNet-GLAD** | Starts ~21, remains stable at ~21 throughout 80 epochs | Best — high and consistent |
| ChebNet-GLAD | Starts ~7, stabilises at ~5–6 | Moderate — acceptable but lower |
| **EdgeConv-GLAD** | Starts very high (~66) but collapses sharply to ~0 by epoch 5 — early stopped (ES@ep5) | Unstable — DE collapse triggered early stopping |

StableChebNet-GLAD is the only model that combines high Dirichlet energy with training stability across 80 epochs. EdgeConv achieves the highest initial DE by far but is completely unstable — its message-passing mechanism destroys node distinguishability within a handful of epochs, making it unsuitable for long-range anomaly detection on jet graphs. ChebNet is stable but sits at a significantly lower DE floor (~5–6) compared to Stable ChebNet (~21), directly justifying the Phase 2 upgrade.

### 3. Known limitation — latent space cluster formation
<img width="1950" height="750" alt="image" src="https://github.com/user-attachments/assets/52b7a56c-672e-4333-9706-a2938db90c21" />

PCA analysis of the trained latent space reveals a tight cluster of SM jets forming in one region of the embedding space. This is a manifold collapse artifact: the autoencoder learns a degenerate reconstruction shortcut, mapping a large fraction of diverse SM jets onto a single dominant "average SM" representation.

The consequence for anomaly detection is a failure mode: a BSM jet whose graph structure partially resembles the SM average can land near this cluster in Z_G space, and its autoencoder reconstruction — being SM-ified toward the same cluster — produces a Z'_G that is also nearby. The L₃ representation error is then small despite the input being genuinely anomalous. This reduces sensitivity precisely in the region of BSM signal space that is hardest to detect.

This limitation is identified as the primary open problem from Phase 1 and is the central motivation for Phase 2 architectural improvements.





---

## Evaluation Metrics

- **AUC-ROC** — primary benchmark metric
- **Signal efficiency at fixed background rejection** — 10%, 1%, 0.1% (standard LHCO format)
- **Anomaly score distributions** — SM vs BSM histogram separation
- **Dirichlet energy per epoch** — encoder stability diagnostic
- **PCA of Z_G vs Z'_G** — latent space convergence diagnostic

---

## References

[1] Hariri et al. — Graph VAE for Detector Reconstruction, NeurIPS ML4PS 2020  
[2] Luo et al. — Deep Graph Level Anomaly Detection with Contrastive Learning, *Scientific Reports* 2022  
[3] Hariri et al. — Return of ChebNet, arXiv:2104.01725  
[5] Collins, Howe, Nachman — Anomaly Detection for Resonant New Physics, *PRL* 2018  
[7] Larkoski, Moult, Nachman — Jet Substructure at the LHC, *Physics Reports* 2020  
[10] Tang, Li, Yu — ChebNet with Rectified Power Units, 2020  
[*] Kasieczka et al. — LHC Olympics 2020, *Reports on Progress in Physics* 2021  
    R&D Dataset: zenodo.org/record/4536377

# RnD Olympics 2020: Graph Preprocessing Pipeline

This repository contains the preprocessing workflow for the **RnD Olympics Dataset 2020**. The pipeline transforms raw, tabular particle physics data into graph-structured objects suitable for Geometric Deep Learning (GDL).

<img width="690" height="895" alt="image" src="https://github.com/user-attachments/assets/f4b28fc0-ccf2-48d6-a340-657d1057ef23" />


## 1. Dataset Overview
The dataset consists of simulated high-energy physics events used to train models for anomaly detection and event classification.

* **Total Events**: 1.1 million total events.
* **Standard Model (SM)**: 1,000,000 events used for baseline training.
* **Beyond Standard Model (BSM)**: 100,000 events.

---

## 2. Data Structure
The raw dataset is stored in a structured tabular format where each row represents an entire event.

* **Event Capacity**: Each row (event) contains data for up to **700 particles**.
* **Particle Features**: Each particle is defined by three kinematic variables: transverse momentum ($p_T$), pseudorapidity ($\eta$), and azimuthal angle ($\phi$).
* **Padding**: Events with fewer than 700 particles are filled with **zero padding** to maintain consistent row length.

---

## 3. Preprocessing Workflow
The transformation from raw rows to graphs follows these sequential steps:

### Step 1: Extraction & Filtering
* **Extraction**: Particles are extracted from the raw dataset on an event-by-event basis.
* **Zero-Padding Removal**: All $(0, 0, 0)$ entries are stripped during extraction to isolate actual particle hits.

### Step 2: Jet Clustering
* **Algorithm**: The **anti-$k_T$** clustering algorithm is used to group particles into jets.
* **Parameters**: Radius parameter $R = 1$.
* **Tool**: Implementation via the `pyjet` library.

### Step 3: Graph Construction
* **Algorithm**: A graph is built for each jet using the **k-Nearest Neighbors (k-NN)** algorithm.
* **Connectivity**: Each node is connected to its $k=8$ nearest neighbors.

---

## 4. Output Representation
The final processed data is wrapped as a **PyTorch Geometric (PyG)** `Data` object:

* **Node Features (`x`)**: A matrix containing $[p_T, \eta, \phi]$ for each particle.
* **Edge Index (`edge_index`)**: A graph connectivity matrix in COO format.
* **Labels (`y`)**: Truth labels for event classification.

---

> **Note**: This pipeline is optimized for training on Standard Model (SM) backgrounds while remaining compatible with BSM signal testing.
