# -*- coding: utf-8 -*-
"""
统一训练入口:多视图虚假新闻检测模型。

用法(在仓库根目录执行):
    python train.py --dataset weibo
    python train.py --dataset twitter
    python train.py --dataset gossipcop
    python train.py --dataset politifact    # 可选,未正式实验

超参数:未显式指定时使用各数据集在论文实验中的默认配置(见 DATASET_CONFIGS)。
"""
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.loader import load_data
from src.models.model import load_model
from src.engine.trainer import train_test


class FocalLoss(nn.Module):
    """Focal Loss,alpha / gamma 用于类别不平衡调节。"""

    def __init__(self, gamma: float = 2.0, alpha=(2.5, 1.0), reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", torch.tensor(alpha))
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(input, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha[target] * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()


# 各数据集默认超参数(与原 main*.py 中的实验设置保持一致)
DATASET_CONFIGS = {
    "weibo":      dict(batch_size=64, epochs=300, lr=1e-4, criterion="ce"),
    "twitter":    dict(batch_size=32, epochs=300, lr=1e-4, criterion="focal",
                      focal_gamma=2.0, focal_alpha=(1.0, 2.5)),
    "gossipcop":  dict(batch_size=4,  epochs=200, lr=1e-3, criterion="ce"),
    "politifact": dict(batch_size=32, epochs=200, lr=1e-4, criterion="ce"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多视图虚假新闻检测:统一训练入口")
    parser.add_argument("--dataset", type=str, default="weibo",
                        choices=list(DATASET_CONFIGS), help="数据集名称")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="训练批次大小(默认取各数据集配置)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数(默认取各数据集配置)")
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率(默认取各数据集配置)")
    parser.add_argument("--criterion", type=str, default=None,
                        choices=["ce", "focal"], help="损失函数(默认取各数据集配置)")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, nargs=2, default=(1.0, 2.5),
                        metavar=("ALPHA0", "ALPHA1"))
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.autograd.set_detect_anomaly(True)

    cfg = DATASET_CONFIGS[args.dataset]
    batch_size = args.batch_size if args.batch_size is not None else cfg["batch_size"]
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    lr = args.lr if args.lr is not None else cfg["lr"]
    criterion_name = args.criterion if args.criterion is not None else cfg["criterion"]

    dataset = args.dataset + "_dataset"
    train_loader, test_loader = load_data(dataset, batch_size=batch_size)
    model = load_model(dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    if criterion_name == "focal":
        criterion = FocalLoss(gamma=cfg.get("focal_gamma", args.focal_gamma),
                              alpha=tuple(cfg.get("focal_alpha", tuple(args.focal_alpha))))
    else:
        criterion = nn.CrossEntropyLoss()

    train_test(model, train_loader, test_loader, optimizer=optimizer,
               criterion=criterion, epochs=epochs, dataset=dataset)


if __name__ == "__main__":
    main(parse_args())
