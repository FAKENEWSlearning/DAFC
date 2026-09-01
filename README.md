# 多视图虚假新闻检测 / Multi-View Fake News Detection

基于文本-图像双分支与混合专家(MoE)的多视图虚假新闻检测模型,在三个公开数据集上验证。

A multi-view fake news detection model based on dual-branch text-image encoders and Mixture-of-Experts (MoE), evaluated on three public datasets.

---

# 中文

## 简介

本仓库是论文「多视图虚假新闻检测」的官方代码。模型采用文本-图像双分支结构,融合 CLIP / Chinese-CLIP / BERT 系预训练编码器,并通过多头注意力门控(MoH)、局部/全局专家混合(UMoE / IMOE + MoEGlobal)、跨模态注意力(BiMGRIA / 多尺度注意力)与对比损失进行跨视图融合与对齐。

| 数据集 | 语言 | 运行命令 |
| --- | --- | --- |
| Weibo | 中文 | `python train.py --dataset weibo` |
| Twitter | 英文 | `python train.py --dataset twitter` |
| GossipCop | 英文 | `python train.py --dataset gossipcop` |

> 代码同时保留 PolitiFact 数据解析,可用 `python train.py --dataset politifact` 运行,但该数据集不在论文正式实验中。

## 模型结构概览

- 双分支编码:文本 + 图像,融合 CLIP / Chinese-CLIP / BERT 系预训练编码器(`src/models/model.py`)
- 多头注意力门控 MoH:`src/models/dmoh.py`
- 局部 / 全局专家混合:`src/models/umoe.py`(UMoE)、`src/models/imoe.py`(IMOE + MoEGlobal)
- 跨模态注意力与对齐:`src/models/attention.py`(BiMGRIA / 多尺度注意力 MSAA / MoHAttention)
- 对比与分布损失:`src/losses/losses.py`
- 论文消融实验(三个变体):`ablationstudy/`

## 目录结构

```
.
├── train.py                  # 统一训练入口(唯一入口,--dataset 选择数据集)
├── requirements.txt
├── README.md
├── .gitignore
├── src/                      # 源码包
│   ├── data/
│   │   ├── loader.py         # 统一数据加载 load_data(PreDataset / 预训练特征)
│   │   └── datasets.py       # 四个数据集的解析与过滤
│   ├── models/
│   │   ├── model.py          # 主模型 load_model
│   │   ├── dmoh.py           # MoH 多头注意力门控
│   │   ├── umoe.py           # UMoE 层
│   │   ├── imoe.py           # IMOE + MoEGlobal(局部/全局专家)
│   │   ├── attention.py      # BiMGRIA / 多尺度注意力 / MoHAttention
│   │   └── aca.py            # 跨模态交叉注意力(消融实验使用)
│   ├── losses/losses.py      # 对比损失 / JS / KL / Triplet 等
│   ├── engine/trainer.py     # train_test 训练评估流程
│   └── utils.py              # 文本过滤、图像变换等工具
├── ablationstudy/            # 消融实验代码(I / L / R 三个变体)
├── scripts/                  # 一次性数据预处理脚本
│   ├── preprocess_gossipcop.py
│   └── preprocess_politifact.py
├── data/                     # 数据集(不入库,gitignore)
└── bert-*/ roberta-*/ xlm-*/ chinese-*/   # 本地预训练权重(不入库)
```

## 快速开始

### 环境安装

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git   # OpenAI CLIP(源码安装)
```

### 预训练权重

代码按相对路径(`./bert-base-chinese`、`./chinese-clip-vit-large-patch14` 等)加载 HuggingFace 格式权重,需与本 README 同目录放置(体积大,不入库)。

### 数据准备

- `data/weibo_dataset/{tweets, rumor_images, nonrumor_images}`
- `data/twitter_dataset/{devset, testset}`
- `data/gossipcop_dataset/{images, gossipcop_train_tweets.txt, gossipcop_test_tweets.txt, news_data_cleaned.txt}`
- 解析结果缓存于 `data/<dataset>/processed/*.pkl`

### 运行

必须在仓库根目录执行(以保证相对路径与包导入正常):

```bash
python train.py --dataset weibo
python train.py --dataset twitter
python train.py --dataset gossipcop
```

### train.py 参数

| 参数 | 说明 |
| --- | --- |
| `--dataset` | `weibo` / `twitter` / `gossipcop` / `politifact` |
| `--batch-size` / `--epochs` / `--lr` | 不指定时使用下表论文默认配置 |
| `--criterion` | `ce` 或 `focal`(不指定时按数据集默认) |
| `--focal-gamma` / `--focal-alpha` | Focal Loss 参数 |
| `--weight-decay` / `--seed` | 默认 1e-4 / 42 |

### 各数据集默认超参数

与论文实验保持一致:

| 数据集 | batch_size | epochs | lr | criterion |
| --- | --- | --- | --- | --- |
| weibo | 64 | 300 | 1e-4 | CE |
| twitter | 32 | 300 | 1e-4 | Focal(γ=2, α=[1.0, 2.5]) |
| gossipcop | 4 | 200 | 1e-3 | CE |
| politifact | 32 | 200 | 1e-4 | CE(未正式实验) |

## 消融实验

```bash
python -m ablationstudy.I   # 或 python ablationstudy/I.py(自带路径引导)
python -m ablationstudy.L
python -m ablationstudy.R
```

三个脚本分别对应论文消融表中的 I / L / R 变体,配套的 `*-engine.py` 为对应训练引擎。
注意:脚本中 `DDPM` 相关代码已被注释(原 `DDPMmodel.py` 模块缺失),如需恢复 DDPM 消融,请从历史版本/备份中找回该模块。

## 注意事项

- 仓库由扁平脚本重构而来:原 `main.py / main-twitter.py / main-poli.py` 三个入口已合并为 `train.py`;原 `main-poli.py` 实际运行的是 GossipCop 数据集。
- `FocalLoss` 原实现将 `alpha` 硬编码在 `cuda` 设备上,现改为 `register_buffer`,兼容 CPU/GPU,行为与原版一致。
- 当前代码默认使用 CUDA(`torch.autograd.set_detect_anomaly(True)` 等),无 GPU 环境需自行适配。
- `scripts/` 预处理脚本含硬编码本机路径,迁移环境时需修改。

---

# English

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
├── README.md
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
