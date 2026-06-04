# Explainable Social Recommendation through Diffusion-based Denoising

A graph-based social recommendation framework that combines **Graph Autoencoder (GAE)** denoising, **Social GCN** propagation, and **Graph Attention Networks (GAT)** to produce accurate and interpretable recommendations. Evaluated on two real-world benchmarks: **CiaoDVD** and **Epinions**.

> **Paper:** Varsha Balaji, Rishita Kakarlapudi, Shanjana Pulagala, Pranathi Kamisetty, Simran Mishra —
> *Explainable Social Recommendation through Diffusion-based Denoising*, UIC CS, 2024.

---

## Repository Structure

```
SocialDiff-XAI/
├── Data/
│   ├── ciao_with_rating_timestamp_txt.zip     # CiaoDVD ratings + trust (txt format)
│   └── epinions_with_timestamps_11.rar        # Epinions ratings + trust (mat format)
├── Scripts/
│   ├── ExplainableSRS_CIAO.ipynb              # Full pipeline on CiaoDVD dataset
│   └── ExplainableSRS_Epinions.ipynb          # Full pipeline on Epinions dataset
└── README.md
```

---

## Overview

Modern recommender systems often function as black boxes — generating ranked suggestions without explaining why. This is especially problematic in social recommendation, where trust networks influence outputs but the source of that influence remains hidden.

This project addresses two core challenges:
1. **Noisy trust graphs** — real-world social links can be outdated, irrelevant, or misleading. We use a Graph Autoencoder to denoise the trust graph before learning.
2. **Lack of interpretability** — we replace GCN propagation with a Graph Attention layer so each neighbor's influence is explicitly quantified and explainable.

---

## Datasets

| Dataset  | Users  | Items   | Ratings | Trust Edges | Format |
|----------|--------|---------|---------|-------------|--------|
| CiaoDVD  | 2,215  | 16,790  | 35,550  | 52,721      | `.txt` |
| Epinions | 22,164 | 296,277 | 922,267 | —           | `.mat` |

> **Note:** Dataset archives are stored in `Data/` and must be extracted before running the notebooks. Epinions extraction requires `unrar` (`apt-get install -y unrar`).

---

## Pipeline

### Phase 1 — Data Preparation
- Load ratings and trust edges; remove self-loops, duplicates, and invalid ratings
- Normalize rating values to [0, 1]
- Align users across rating and trust sources; keep only users present in both
- Build sparse matrices:
  - `R` — user–item interaction matrix
  - `A` — symmetrized user–user trust adjacency

### Phase 2 — Graph Autoencoder Denoising (GAE)
- A two-layer GCN encoder learns 64-dimensional user embeddings from the trust graph
- Decoder reconstructs edge scores; trained with Binary Cross-Entropy against positive and randomly sampled negative edges
- After training, top-10 most similar neighbors per user are retained to form the **denoised adjacency A'**
- Result: original graph A had **80,244 edges**; denoised A' retains **43,400 edges**

### Phase 3 — Social GCN Recommendation
- A two-layer **SocialGCN** propagates user embeddings over A'
- Trained with hybrid loss: BPR (ranking) + MSE (rating prediction)
- Top-K recommendations generated via dot-product scoring between user and item embeddings
- Data split: 75% train / 10% validation / 15% test

### Phase 4 — Explainability via Graph Attention
- A **sparse GAT layer** replaces GCN propagation, computing attention scores `e_ij = a(Wh_i ∥ Wh_j)` per trust edge
- Produces per-neighbor influence weights for every user
- Explainability outputs:
  - **Ego-graph visualization** — edge thickness proportional to attention weight
  - **Ranked influencer table** — top-K neighbors with attention scores
  - **Embedding dimension importance** — bar chart of `|u_k × i_k|` per dimension for any (user, item) pair
  - **Fidelity metric** — change in predicted score after removing the top-attention neighbor

---

## Models Compared

| Model                | Description                                               |
|----------------------|-----------------------------------------------------------|
| `MF_baseline`        | BPR-trained matrix factorization, ratings only           |
| `SocialGCN_raw`      | SocialGCN over original (noisy) trust graph A            |
| `SocialGCN_denoised` | SocialGCN over GAE-denoised graph A'                     |

---

## Results

### CiaoDVD

| Model                | Split | P@10   | R@10   | nDCG@10 |
|----------------------|-------|--------|--------|---------|
| MF_baseline          | val   | 0.0065 | 0.0260 | 0.0280  |
|                      | test  | 0.0049 | 0.0225 | 0.0251  |
| SocialGCN_raw        | val   | 0.0078 | 0.0295 | 0.0305  |
|                      | test  | 0.0062 | 0.0258 | 0.0284  |
| SocialGCN_denoised   | val   | 0.0089 | 0.0312 | 0.0331  |
|                      | **test**  | **0.0071** | **0.0281** | **0.0309**  |

SocialGCN_denoised achieves the best results on CiaoDVD, improving test nDCG@10 from 0.0251 (MF baseline) to 0.0309.

### Epinions

| Model                | Split | P@10   | R@10     | nDCG@10  |
|----------------------|-------|--------|----------|----------|
| MF_baseline          | val   | 0.0026 | 0.009219 | 0.012584 |
|                      | test  | 0.0036 | 0.014530 | 0.015113 |
| SocialGCN_raw        | val   | 0.0026 | 0.009111 | 0.009984 |
|                      | test  | 0.0032 | 0.008585 | 0.014175 |
| SocialGCN_denoised   | val   | 0.0024 | 0.006737 | 0.011595 |
|                      | **test**  | **0.0044** | **0.008618** | **0.018315**  |

On Epinions, SocialGCN_denoised achieves the best test nDCG@10 (0.0183), clearly outperforming SocialGCN_raw and confirming that graph denoising is critical for noisy, large-scale trust networks.

---

## Setup & Requirements

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib networkx tqdm tabulate seaborn ipywidgets
```

Both notebooks are designed for **Google Colab** and mount Google Drive for data access. To run locally, replace `/content/...` paths with your local equivalents.

### Running the Notebooks

1. Place the dataset archives in `Data/` and extract them (extraction cells are included in the notebooks)
2. Open the relevant notebook from `Scripts/`:
   - `ExplainableSRS_CIAO.ipynb` — CiaoDVD dataset
   - `ExplainableSRS_Epinions.ipynb` — Epinions dataset
3. Run all cells top to bottom — each notebook is fully self-contained across all phases
4. To inspect explainability for a specific user, set `ego_user` and `item_id` in the visualization cells

---

## References

- Li et al., *RecDiff: Diffusion Model for Social Recommendation*, CIKM 2024. https://doi.org/10.1145/3627673.3679630
- Kipf & Welling, *Variational Graph Auto-Encoders*, arXiv 2016. https://arxiv.org/abs/1611.07308
- Wang et al., *Neural Graph Collaborative Filtering*, SIGIR 2019. https://doi.org/10.1145/3331184.3331267
- Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions (SHAP)*, NeurIPS 2017. https://arxiv.org/abs/1705.07874
- Tintarev & Masthoff, *Explaining Recommendations: Design and Evaluation*, RecSys 2007. https://doi.org/10.1145/1297231.1297255
