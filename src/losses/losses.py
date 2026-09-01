import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional



def info_nce_loss(features, temperature=0.1):
    """
    计算InfoNCE损失（对比交叉熵损失）
    参数:
        features: 特征向量，形状为 [batch_size, feature_dim]
        temperature: 温度参数，缩放点积相似度
    """
    # 特征归一化，确保使用余弦相似度
    features = F.normalize(features, dim=1)
    
    batch_size = features.shape[0]
    labels = torch.arange(batch_size).to(features.device)
    
    # 计算相似度矩阵（使用余弦相似度）
    sim_matrix = torch.matmul(features, features.T) / temperature
    
    # 应用掩码以排除自身匹配
    mask = torch.eye(batch_size, dtype=torch.bool).to(features.device)
    sim_matrix = sim_matrix.masked_fill(mask, -1e9)
    
    # 计算交叉熵损失
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

def semantic_triplet_learning(t_emb, labels, margin, epoch):
    """
    Computes triplet loss for binary classification (real/fake), with hard/semi-hard/hardest negative mining.
    
    :param t_emb: Embeddings of shape (batch_size, embedding_dim)
    :param labels: Binary labels of shape (batch_size,)
    :param margin: Margin for triplet loss
    :param epoch: Current training epoch to control sampling strategy
    :return: Scalar loss or None if no valid triplets
    """
    batch_size = t_emb.size(0)

    # Compute pairwise distances
    dist = torch.pow(t_emb, 2).sum(dim=1, keepdim=True).expand(batch_size, batch_size)
    dist = dist + dist.t()
    dist.addmm_(t_emb, t_emb.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()

    # Masks for positive and negative samples
    pos_mask = labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).t())
    neg_mask = ~pos_mask

    diss_diff = []

    for i in range(batch_size):
        # Anchor-positive distances (same class)
        dist_pos = dist[i][pos_mask[i]]
        if len(dist_pos) == 0:
            continue

        # Anchor-negative distances (different class)
        dist_neg = dist[i][neg_mask[i]]
        if len(dist_neg) == 0:
            continue

        # Compute a_n - a_p distances
        dist_an_sub_ap = dist_neg.unsqueeze(1) - dist_pos

        # Sample based on epoch
        if epoch < 5:
            # Only semi-hard negatives
            semi_hard = dist_an_sub_ap[(dist_an_sub_ap > 0) & (dist_an_sub_ap < margin)]
        elif 5 <= epoch < 10:
            # Semi-hard + hard negatives
            semi_hard = dist_an_sub_ap[dist_an_sub_ap < margin]
        else:
            # Hardest: include all where dist(a,n) <= dist(a,p)
            semi_hard = dist_an_sub_ap[dist_an_sub_ap <= margin]

        if len(semi_hard) == 0:
            continue

        # Clamp values and compute loss contribution
        loss_part = torch.clamp(-semi_hard + margin, min=0.0)
        diss_diff.append(loss_part.mean().unsqueeze(0))

    if not diss_diff:
        return None

    return torch.cat(diss_diff).mean()
# 1. 对比损失 (Contrastive Loss) - 适用于成对样本
class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x1, x2: 特征向量，形状为 [batch_size, feature_dim]
            y: 标签，1表示正例对，0表示负例对，形状为 [batch_size]
        """
        # 计算欧氏距离
        d = F.pairwise_distance(x1, x2, p=2)
        # 对比损失公式
        loss = y * 0.5 * d.pow(2) + (1 - y) * 0.5 * torch.clamp(self.margin - d, min=0).pow(2)
        return loss.mean()

# 2. 三元组损失 (Triplet Loss)
class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0, distance_metric: str = 'cosine'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        输入:
            anchor, positive, negative: 特征向量，形状为 [batch_size, feature_dim]
        """
        if self.distance_metric == 'euclidean':
            # 欧氏距离
            pos_dist = F.pairwise_distance(anchor, positive, p=2)
            neg_dist = F.pairwise_distance(anchor, negative, p=2)
        elif self.distance_metric == 'cosine':
            # 余弦距离 (1-余弦相似度)
            pos_dist = 1 - F.cosine_similarity(anchor, positive, dim=1)
            neg_dist = 1 - F.cosine_similarity(anchor, negative, dim=1)
        else:
            raise ValueError(f"不支持的距离度量: {self.distance_metric}")
            
        # 三元组损失公式
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0)
        return loss.mean()

# 3. NT-Xent损失 (用于SimCLR等批量对比学习)
class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, device: str = 'cuda'):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        输入:
            features: 增强样本的特征向量，形状为 [2*batch_size, feature_dim]
                     前batch_size个是原始样本的第一种增强，后batch_size个是第二种增强
        """
        batch_size = features.shape[0] // 2
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        sim_matrix = torch.exp(sim_matrix)
        
        # 创建标签: 每个样本的正例是其另一种增强版本
        labels = torch.cat([
            torch.arange(batch_size, 2*batch_size),
            torch.arange(0, batch_size)
        ]).to(self.device)
        
        # 构建对比学习的"伪标签"矩阵
        mask = torch.eye(2*batch_size, dtype=torch.bool).to(self.device)
        sim_matrix = sim_matrix.masked_fill(mask, 0)
        
        # 计算交叉熵损失
        loss = self.criterion(sim_matrix, labels)
        return loss

# 4. 监督对比损失 (Supervised Contrastive Learning)
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, device: str = 'cuda'):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        输入:
            features: 样本特征向量，形状为 [batch_size, feature_dim]
            labels: 样本标签，形状为 [batch_size]
        """
        batch_size = features.shape[0]
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        sim_matrix = torch.exp(sim_matrix)
        
        # 创建标签掩码: 1表示两个样本标签相同
        label_mask = labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).T).float()
        label_mask.fill_diagonal_(0)  # 自身不算正例
        
        # 计算每个样本的正例对和负例对
        pos_mask = label_mask
        neg_mask = 1 - label_mask
        
        # 计算损失
        pos_pairs = torch.sum(sim_matrix * pos_mask, dim=1)
        neg_pairs = torch.sum(sim_matrix * neg_mask, dim=1)
        
        # 避免数值不稳定性
        eps = 1e-8
        loss = -torch.log((pos_pairs / (pos_pairs + neg_pairs + eps)) + eps)
        return loss.mean()    
class CenterBasedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, alpha: float = 0.5, beta: float = 0.5, device: str = 'cuda'):
        super(CenterBasedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha  # 类内聚集权重
        self.beta = beta    # 类间分离权重
        self.device = device
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        输入:
            features: 样本特征向量，形状为 [batch_size, feature_dim]
            labels: 样本标签，形状为 [batch_size]
        """
        batch_size = features.shape[0]
        
        # 计算类中心
        unique_labels = torch.unique(labels)
        num_classes = len(unique_labels)
        centers = []
        
        for label in unique_labels:
            # 筛选当前类的所有样本
            class_mask = (labels == label).float().unsqueeze(1)
            class_features = features * class_mask
            # 计算类中心（特征均值）
            class_center = torch.sum(class_features, dim=0) / (torch.sum(class_mask) + 1e-8)
            centers.append(class_center)
        
        centers = torch.stack(centers).to(self.device)  # [num_classes, feature_dim]
        
        # -----------------------
        # 1. 计算原始对比损失
        # -----------------------
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        sim_matrix = torch.exp(sim_matrix)
        
        # 创建标签掩码
        label_mask = labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).T).float()
        label_mask.fill_diagonal_(0)  # 自身不算正例
        pos_mask = label_mask  # 同类样本（不含自身）
        neg_mask = 1 - label_mask  # 不同类样本
        # 计算正例和负例相似度之和
        pos_pairs = torch.sum(sim_matrix * pos_mask, dim=1)
        neg_pairs = torch.sum(sim_matrix * neg_mask, dim=1)
        
        # 避免数值不稳定性
        eps = 1e-8
        contrastive_loss = -torch.log((pos_pairs / (pos_pairs + neg_pairs + eps)) + eps)
        
        # -----------------------
        # 2. 计算类内聚集损失
        # -----------------------
        intra_loss = torch.zeros(batch_size, device=self.device)
        
        for i, label in enumerate(labels):
            # 找到当前样本的类中心索引
            center_idx = (unique_labels == label).nonzero(as_tuple=True)[0][0]
            # 计算样本到其类中心的距离（欧氏距离平方）
            distance = torch.sum((features[i] - centers[center_idx]) ** 2)
            intra_loss[i] = distance
        
        # -----------------------
        # 3. 计算类间分离损失
        # -----------------------
        # 计算所有类中心之间的距离
        center_distances = []
        for i in range(num_classes):
            for j in range(i+1, num_classes):
                # 计算类中心之间的欧氏距离
                dist = torch.sum((centers[i] - centers[j]) ** 2)
                center_distances.append(dist)
        
        # 类间分离损失取最小距离的负值（最大化最小距离）
        if len(center_distances) > 0:
            min_center_distance = torch.min(torch.stack(center_distances))
            inter_loss = -min_center_distance
        else:
            inter_loss = torch.tensor(0.0, device=self.device)
        
        # -----------------------
        # 合并三种损失
        # -----------------------
        total_loss = contrastive_loss.mean() + \
                     self.alpha * intra_loss.mean() + \
                     self.beta * inter_loss
        
        return total_loss
import torch

def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    计算两个概率分布之间的KL散度（Kullback-Leibler Divergence）。
    
    参数:
        p (torch.Tensor): 第一个概率分布，形状为[batch_size, num_classes]
        q (torch.Tensor): 第二个概率分布，形状与p相同
        eps (float): 用于数值稳定性的小常数，避免对数计算时出现零值
    
    返回:
        torch.Tensor: 每个样本的KL散度值，形状为[batch_size]
    
    数学公式:
        KL(p || q) = Σ p(x) * log(p(x)/q(x))
    """
    # 确保概率分布已经归一化（即总和接近1）
    p = p / (p.sum(dim=-1, keepdim=True) + eps)
    q = q / (q.sum(dim=-1, keepdim=True) + eps)
    
    # 计算KL散度
    kl = torch.sum(p * torch.log(p / (q + eps) + eps), dim=-1)
    
    # 确保结果非负（由于数值稳定性可能会有微小负值）
    kl = torch.clamp(kl, min=0.0)
    
    return kl

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    计算两个概率分布之间的JS散度（Jensen-Shannon Divergence）。
    
    参数:
        p (torch.Tensor): 第一个概率分布，形状为[batch_size, num_classes]
        q (torch.Tensor): 第二个概率分布，形状与p相同
        eps (float): 用于数值稳定性的小常数
    
    返回:
        torch.Tensor: 每个样本的JS散度值，形状为[batch_size]
    
    数学公式:
        JS(p || q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)
        其中 m = 0.5 * (p + q)
    """
    m = 0.5 * (p + q)
    js = 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)
    return js

