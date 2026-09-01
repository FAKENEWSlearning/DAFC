# Multi-View Fake News Detection

> [中文](README.md)

A multi-view fake news detection model based on dual-branch text-image encoders and Mixture-of-Experts (MoE), evaluated on three public datasets.

## Overview

This repository contains the official code of the paper "Multi-View Fake News Detection". The model adopts a dual-branch text-image architecture, leveraging CLIP / Chinese-CLIP / BERT-family pretrained encoders, and performs cross-view fusion and alignment through Mixture-of-Heads gating (MoH), local/global Mixture-of-Experts (UMoE / IMOE + MoEGlobal), cross-modal attention (BiMGRIA / multi-scale attention), and contrastive losses.

| Dataset | Language | Command |
| --- | --- | --- |
| Weibo | Chinese | `python train.py --dataset weibo` |
| Twitter | English | `python train.py --dataset twitter` |
| GossipCop | English | `python train.py --dataset gossipcop` |

> PolitiFact parsing is also retained and can be run with `python train.py --dataset politifact`, but it is not part of the official experiments in the paper.

## Model Architecture

- Dual-branch encoders: text + image, with CLIP / Chinese-CLIP / BERT-family pretrained encoders (`src/models/model.py`)
- Mixture-of-Heads attention gating (MoH): `src/models/dmoh.py`
- Local / global Mixture-of-Experts: `src/models/umoe.py` (UMoE), `src/models/imoe.py` (IMOE + MoEGlobal)
- Cross-modal attention and alignment: `src/models/attention.py` (BiMGRIA / multi-scale attention MSAA / MoHAttention)
- Contrastive and distribution losses: `src/losses/losses.py`
- Ablation study (three variants): `ablationstudy/`

## Repository Structure

```
.
├── train.py                  # Unified training entry (the only entry; select dataset via --dataset)
├── requirements.txt
├── README.md                 # 中文说明
├── README_EN.md              # English documentation
├── .gitignore
├── src/                      # Source package
│   ├── data/
│   │   ├── loader.py         # Unified data loading load_data (PreDataset / pretrained features)
│   │   └── datasets.py       # Parsing and filtering for all four datasets
│   ├── models/
│   │   ├── model.py          # Main model load_model
│   │   ├── dmoh.py           # MoH multi-head attention gating
│   │   ├── umoe.py           # UMoE layers
│   │   ├── imoe.py           # IMOE + MoEGlobal (local/global experts)
│   │   ├── attention.py      # BiMGRIA / multi-scale attention / MoHAttention
│   │   └── aca.py            # Cross-modal cross-attention (used by the ablation study)
│   ├── losses/losses.py      # Contrastive / JS / KL / Triplet losses
│   ├── engine/trainer.py     # train_test training and evaluation pipeline
│   └── utils.py              # Text filtering, image transforms, etc.
├── ablationstudy/            # Ablation study code (I / L / R variants)
├── scripts/                  # One-off data preprocessing scripts
│   ├── preprocess_gossipcop.py
│   └── preprocess_politifact.py
├── data/                     # Datasets (git-ignored, not committed)
└── bert-*/ roberta-*/ xlm-*/ chinese-*/   # Local pretrained weights (git-ignored)
```

## Getting Started

### Installation

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git   # OpenAI CLIP (install from source)
```

### Pretrained Weights

The code loads HuggingFace-format weights via relative paths (e.g. `./bert-base-chinese`, `./chinese-clip-vit-large-patch14`). Place them in the same directory as this README (too large to be committed).

### Data Preparation

- `data/weibo_dataset/{tweets, rumor_images, nonrumor_images}`
- `data/twitter_dataset/{devset, testset}`
- `data/gossipcop_dataset/{images, gossipcop_train_tweets.txt, gossipcop_test_tweets.txt, news_data_cleaned.txt}`
- Parsed caches are stored at `data/<dataset>/processed/*.pkl`

### Training

Always run from the repository root (relative paths and package imports depend on it):

```bash
python train.py --dataset weibo
python train.py --dataset twitter
python train.py --dataset gossipcop
```

### Command-Line Arguments

| Argument | Description |
| --- | --- |
| `--dataset` | `weibo` / `twitter` / `gossipcop` / `politifact` |
| `--batch-size` / `--epochs` / `--lr` | Default to the per-dataset paper settings below when omitted |
| `--criterion` | `ce` or `focal` (per-dataset default when omitted) |
| `--focal-gamma` / `--focal-alpha` | Focal Loss parameters |
| `--weight-decay` / `--seed` | Default 1e-4 / 42 |

### Default Hyperparameters per Dataset

Consistent with the paper experiments:

| Dataset | batch_size | epochs | lr | criterion |
| --- | --- | --- | --- | --- |
| weibo | 64 | 300 | 1e-4 | CE |
| twitter | 32 | 300 | 1e-4 | Focal (γ=2, α=[1.0, 2.5]) |
| gossipcop | 4 | 200 | 1e-3 | CE |
| politifact | 32 | 200 | 1e-4 | CE (not officially evaluated) |

## Ablation Study

```bash
python -m ablationstudy.I   # or: python ablationstudy/I.py (has its own sys.path bootstrap)
python -m ablationstudy.L
python -m ablationstudy.R
```

The three scripts correspond to the I / L / R variants in the paper's ablation table, with the matching `*-engine.py` training engines. Note: the `DDPM`-related code in these scripts is commented out because the original `DDPMmodel.py` module is missing; restore it from your history/backup if the DDPM ablation is needed.

## Notes

- This repository was refactored from flat scripts: the original `main.py / main-twitter.py / main-poli.py` entries were merged into `train.py`; the original `main-poli.py` actually ran the GossipCop dataset.
- The original `FocalLoss` hard-coded `alpha` on the `cuda` device; it now uses `register_buffer`, making it work on both CPU and GPU with identical behavior.
- The current code assumes CUDA (e.g. `torch.autograd.set_detect_anomaly(True)`); adapt it yourself for CPU-only environments.
- The preprocessing scripts under `scripts/` contain hard-coded local paths that must be updated when migrating environments.
