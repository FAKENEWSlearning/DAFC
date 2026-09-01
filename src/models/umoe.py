import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union

# --- 基础组件解耦 ---

class VanillaMLP(nn.Module):
    """标准的 MLP 层，用于共享专家"""
    def __init__(self, hidden_size, intermediate_size, hidden_act="silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        # 兼容不同版本的激活函数获取
        if isinstance(hidden_act, str):
            from transformers.activations import ACT2FN
            self.act_fn = ACT2FN[hidden_act]
        else:
            self.act_fn = hidden_act

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class AddAuxiliaryLoss(torch.autograd.Function):
    """将辅助损失注入梯度流"""
    @staticmethod
    def forward(ctx, x, loss):
        ctx.save_for_backward(loss)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (loss,) = ctx.saved_tensors
        return grad_output, torch.ones_like(loss)

# --- UMoE 核心路由 ---

class UMoERouter(nn.Module):
    def __init__(
        self, 
        hidden_size: int,
        n_routed_experts: int,
        topk: int,
        aux_loss_alpha: float = 0.001,
        score_func: str = "softmax",
        topk_method: str = "greedy",
        n_group: int = 1,
        topk_group: int = 1,
        routed_scaling_factor: float = 1.0
    ):
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.topk = topk
        self.aux_loss_alpha = aux_loss_alpha
        self.score_func = score_func
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        
        self.weight = nn.Parameter(torch.empty((n_routed_experts, hidden_size)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, h)
        
        # 计算路由分数
        logits = F.linear(hidden_states_flat.float(), self.weight.float(), None)
        if self.score_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = logits # 可扩展其他 scoring
            
        # Top-K 选择逻辑
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(scores, k=self.topk, dim=-1, sorted=False)
        elif self.topk_method == "group_limited_greedy":
            # 简化版的分组路由逻辑
            group_scores = scores.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            group_max_score = group_scores.max(dim=-1).values
            group_idx = torch.topk(group_max_score, k=self.topk_group, dim=-1, sorted=False)[1]
            mask = torch.zeros_like(group_max_score).scatter_(1, group_idx, 1)
            scores_masked = scores.view(-1, self.n_group, -1) * mask.unsqueeze(-1)
            topk_weight, topk_idx = torch.topk(scores_masked.view(bsz*seq_len, -1), k=self.topk, dim=-1, sorted=False)

        # 归一化与缩放
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weight = topk_weight * self.routed_scaling_factor

        # 辅助损失 (Load Balancing)
        aux_loss = None
        if self.training and self.aux_loss_alpha > 0:
            # 经典的 MoE 负载均衡损失计算
            count = torch.zeros(self.n_routed_experts, device=hidden_states.device)
            count.scatter_add_(0, topk_idx.view(-1), torch.ones_like(topk_idx.view(-1), dtype=torch.float))
            fi = count / (bsz * seq_len * self.topk)
            Pi = scores.mean(dim=0)
            aux_loss = self.aux_loss_alpha * (fi * Pi).sum() * self.n_routed_experts

        return topk_idx, topk_weight.to(hidden_states.dtype), aux_loss

# --- UMoE 主模块 ---

class UMoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        
        # 1. 路由层
        self.gate = UMoERouter(
            hidden_size=config.hidden_size,
            n_routed_experts=config.n_routed_experts,
            topk=config.num_experts_per_tok,
            aux_loss_alpha=getattr(config, "aux_loss_alpha", 0.001),
            n_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1)
        )
        
        # 2. 共享专家 (可选)
        self.shared_experts = None
        if getattr(config, "n_shared_experts", 0) > 0:
            self.shared_experts = VanillaMLP(
                config.hidden_size, 
                config.moe_intermediate_size * config.n_shared_experts,
                config.hidden_act
            )

        # 3. 路由专家 (这里演示标准的专家集合，可替换为 GroupedGEMM 版本)
        self.experts = nn.ModuleList([
            VanillaMLP(config.hidden_size, config.moe_intermediate_size, config.hidden_act)
            for _ in range(config.n_routed_experts)
        ])

    def forward(self, hidden_states):
        identity = hidden_states
        bsz, seq_len, h = hidden_states.shape
        
        # 计算路由
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        
        # 计算共享专家
        shared_output = self.shared_experts(hidden_states) if self.shared_experts else 0
        
        # 计算路由专家 (标准循环实现，适合小规模或非 GroupedGEMM 环境)
        # 如果需要极致性能，可在此处集成你上传的 grouped_gemm_util
        final_hidden_states = torch.zeros_like(hidden_states)
        flat_hidden = hidden_states.view(-1, h)
        
        for i, expert in enumerate(self.experts):
            # 找到选择了当前专家的 token 索引和对应的 top-k 位置
            mask = (topk_idx == i)
            if not mask.any():
                continue
            
            token_indices, topk_positions = torch.where(mask)
            expert_out = expert(flat_hidden[token_indices])
            
            # 加权累加
            weight = topk_weight[token_indices, topk_positions].unsqueeze(-1)
            final_hidden_states.view(-1, h).index_add_(0, token_indices, expert_out * weight)

        out = final_hidden_states + shared_output
        
        # 注入辅助损失梯度
        if aux_loss is not None:
            out = AddAuxiliaryLoss.apply(out, aux_loss)
            
        return out
from types import SimpleNamespace



# 定义极简配置
config = SimpleNamespace(
    hidden_size=512,
    moe_intermediate_size=1024,
    n_routed_experts=8,
    num_experts_per_tok=2,
    n_shared_experts=1,
    hidden_act="silu"
)

# 初始化并运行
model = UMoELayer(config)
x = torch.randn(2, 16, 512) # [batch, seq, hidden]
output = model(x)

print(output.shape) # torch.Size([2, 16, 512])