# Adaptive Cross Attention
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from src.models.imoe import IMOE

# 多模态注意力机制
class multimodal_attention(nn.Module):
    def __init__(self, attention_dropout=0.5):
        super(multimodal_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.matmul(q, k.transpose(-2, -1))
        if scale:
            attention = attention * scale
        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        attention = torch.matmul(attention, v)
        return attention

# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=256, num_heads=8, dropout=0.5):
        super(MultiHeadAttention, self).__init__()
        self.model_dim = model_dim
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(1, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(1, self.dim_per_head * num_heads, bias=False)
        self.linear_q = nn.Linear(1, self.dim_per_head * num_heads, bias=False)
        self.dot_product_attention = multimodal_attention(dropout)
        self.linear_final = nn.Linear(model_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        residual = query
        query = query.unsqueeze(-1)
        key = key.unsqueeze(-1)
        value = value.unsqueeze(-1)
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)
        key = key.view(-1, self.num_heads, self.model_dim, self.dim_per_head)
        value = value.view(-1, self.num_heads, self.model_dim, self.dim_per_head)
        query = query.view(-1, self.num_heads, self.model_dim, self.dim_per_head)
        #print("key.size(-1)", key.size(-1))
        #print("self.num_heads", self.num_heads)
        scale = (key.size(-1) // self.num_heads)**-0.5
        attention = self.dot_product_attention(query, key, value, scale, attn_mask)
        attention = attention.view(-1, self.model_dim, self.dim_per_head * self.num_heads)
        output = self.linear_final(attention).squeeze(-1)
        output = self.dropout(output)
        output = self.layer_norm(residual + output)
        return output

# 位置感知前馈网络
class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=256, ffn_dim=2048, dropout=0.5):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x
        x = self.w2(F.relu(self.w1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x

# 多模态融合层
class improved_cross_attention_with_moe(nn.Module):
    def __init__(self, model_dim=256, num_heads=8, ffn_dim=2048, dropout=0.5):
        super(improved_cross_attention_with_moe, self).__init__()
        self.attention_1 = MultiHeadAttention(model_dim, num_heads, dropout)
        self.attention_2 = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward_1 = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        self.feed_forward_2 = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        self.fusion_linear = nn.Linear(model_dim * 2, model_dim)
        self.dropout = nn.Dropout(0.5)
        self.IMOE = IMOE(
            ds_inputsize=model_dim,
            input_size=1,
            output_size=1,
            num_experts=8,
            hidden_size=model_dim,
            noisy_gating=True,
            k=4,
            trainingmode=True
        )
    def forward(self, image_output, text_output, attn_mask=None):
        output_1 = self.attention_1(image_output, text_output, text_output, attn_mask)
        output_2 = self.attention_2(text_output, image_output, image_output, attn_mask)
        output_1 = self.feed_forward_1(output_1)
        output_2 = self.feed_forward_2(output_2)
        #print("output1", output_1.shape)
        #print("output2", output_2.shape)

        output, loss_aca = self.IMOE(output_1, output_2)
        output_1 = torch.mul(output_1, output)
        output_2 = torch.mul(output_2, output)
        
        output = torch.cat([output_1, output_2], dim=1)
        output = self.fusion_linear(output)
        output = self.dropout(output)
        return output, loss_aca

import torch

# 假设上述所有类（multimodal_attention, MultiHeadAttention, 等）已定义
# 假设 IMOE 类已正确实现（若未实现，可参考文末的简易模拟版）

def run_aca_demo():
    # --------------------------
    # 1. 配置模型参数
    # --------------------------
    model_dim = 256       # 特征维度（需与 MultiHeadAttention 中一致）
    num_heads = 8         # 多头注意力头数
    ffn_dim = 2048        # 前馈网络隐藏层维度
    dropout = 0.5         # dropout 概率
    batch_size = 4        # 批次大小（可自定义）

    # --------------------------
    # 2. 初始化多模态融合层（ACA 核心）
    # --------------------------
    aca_layer = improved_cross_attention_with_moe(
        model_dim=model_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout
    )

    # --------------------------
    # 3. 构造输入数据（图像和文本特征）
    # --------------------------
    # 输入特征维度：[batch_size, model_dim]
    # 这里用随机张量模拟图像和文本的特征（实际应用中应替换为真实特征）
    image_features = torch.randn(batch_size, model_dim)  # 模拟图像特征
    text_features = torch.randn(batch_size, model_dim)   # 模拟文本特征

    # --------------------------
    # 4. 前向传播（不跟踪梯度）
    # --------------------------
    aca_layer.eval()  # 切换到评估模式（避免 dropout 等训练阶段操作的影响）
    with torch.no_grad():  # 不计算梯度，节省内存
        # 调用 ACA 层进行融合
        fused_output = aca_layer(image_features, text_features)

    # --------------------------
    # 5. 输出结果信息
    # --------------------------
    print("前向传播完成！")
    print(f"输入图像特征维度: {image_features.shape}")
    print(f"输入文本特征维度: {text_features.shape}")
    print(f"ACA 输出特征维度: {fused_output.shape}")  # 应与 [batch_size, model_dim] 一致


if __name__ == "__main__":
    # 运行 ACA 前向传播
    run_aca_demo()