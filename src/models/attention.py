import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNormProxy(nn.Module):
    """ 
    支持 (B, C, H, W) 的 LayerNorm，在 C 维度归一化 
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1) # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2) # [B, C, H, W]
        return x

class MultiScaleBranch(nn.Module):
    """
    单源多尺度分支：
    输入一个 x，内部通过不同的卷积核提取不同感受野的特征
    """
    def __init__(self, dim, kernel_size):
        super().__init__()
        # 使用分组卷积(Groups=dim)降低参数量，模拟大核注意力
        pad = kernel_size // 2
        
        # 针对 HxW 或者 1xN 的通用条形分解
        # 分解 1: 水平方向 (捕捉序列上下文 或 图像宽度特征)
        self.conv_w = nn.Conv2d(dim, dim, kernel_size=(1, kernel_size), padding=(0, pad), groups=dim, bias=False)
        # 分解 2: 垂直方向 (捕捉图像高度特征，如果输入是1D序列，此处充当Channel混合或保留)
        self.conv_h = nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1), padding=(pad, 0), groups=dim, bias=False)
        
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        # 并行处理：既看水平也看垂直，或者叠加
        x_w = self.conv_w(x)
        x_h = self.conv_h(x)
        return self.act(self.bn(x_w + x_h))

class SelectiveFusion(nn.Module):
    """
    动态门控融合：自适应决定每个尺度的权重
    """
    def __init__(self, dim, num_branches=3):
        super().__init__()
        self.dim = dim
        self.num_branches = num_branches
        # 全局上下文描述子
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 降维再升维，计算权重
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(dim, num_branches * dim, 1, bias=False)
        )

    def forward(self, branches):
        # branches: List of tensors
        batch_size = branches[0].shape[0]
        
        # 1. 聚合特征计算全局描述
        feat_sum = sum(branches)
        gap = self.avg_pool(feat_sum) # [B, C, 1, 1]
        
        # 2. 计算注意力权重 [B, branches*C, 1, 1]
        weights = self.fc(gap)
        weights = weights.view(batch_size, self.num_branches, self.dim, 1, 1)
        weights = F.softmax(weights, dim=1) # 在分支维度归一化
        
        # 3. 加权求和
        out = 0
        for i, branch in enumerate(branches):
            out += branch * weights[:, i, :, :, :]
        return out

class MSAA_Single(nn.Module):
    def __init__(self, channels, factor=4.0):
        super(MSAA_Single, self).__init__()
        # 内部处理维度 (Bottleneck structure)
        mid_channels = max(int(channels // factor), 16)
        
        # 1. 降维映射 (Pointwise Conv)
        self.stem = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            LayerNormProxy(mid_channels),
            nn.GELU()
        )
        
        # 2. 多尺度生成模块 (从单一输入生成多尺度特征)
        # Branch 0: 只有 1x1 卷积 (保留原始特征/恒等映射)
        self.branch0 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU()
        )
        
        # Branch 1: 小感受野 (3x3 / 3x1) - 关注局部细节
        self.branch1 = MultiScaleBranch(mid_channels, kernel_size=3)
        
        # Branch 2: 大感受野 (11x11 / 11x1) - 关注全局语义/长距离依赖
        # 这里使用 Strip Convolution 的特性，即使是 1D 序列也能捕捉很远的依赖
        self.branch2 = MultiScaleBranch(mid_channels, kernel_size=11)
        
        # 3. 动态融合
        self.fusion = SelectiveFusion(mid_channels, num_branches=3)
        
        # 4. 升维输出 (Mix-FFN 结构增强特征表达)
        self.output_proj = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels * 4, 1),
            nn.GELU(),
            # Depthwise Conv 增强空间交互
            nn.Conv2d(mid_channels * 4, mid_channels * 4, 3, padding=1, groups=mid_channels * 4),
            nn.GELU(),
            nn.Conv2d(mid_channels * 4, channels, 1)
        )
        
    def forward(self, x):
        """
        Input: x (B, N, C) e.g., (4, 128, 128)
        """
        B, N, C = x.shape
        
        # --- 1. Shape Adapter (适配器) ---
        # 自动判断是否能开平方，不能则当作 1D 序列处理
        # H_sqrt = int(math.sqrt(N))
        # if H_sqrt * H_sqrt == N:
        #     H, W = H_sqrt, H_sqrt
        # else:
        #     # 针对 N=128 这种情况，将其视为 1x128 的特征图
        #     H, W = 1, N
        H, W = 1, N
        # (B, N, C) -> (B, C, N) -> (B, C, H, W)
        x_img = x.permute(0, 2, 1).reshape(B, C, H, W)
        
        # --- 2. Main Processing ---
        # 降维
        x_stem = self.stem(x_img)
        
        # 多尺度特征提取 (Internal Multi-Scale)
        b0 = self.branch0(x_stem) # Identity-like
        b1 = self.branch1(x_stem) # Local
        b2 = self.branch2(x_stem) # Global/Long-range
        
        # 动态融合
        x_fused = self.fusion([b0, b1, b2])
        
        # 输出投影
        out = self.output_proj(x_fused)
        
        # 残差连接
        out = out + x_img

        # --- 3. Output Adapter ---
        # (B, C, H, W) -> (B, N, C)
        out = out.flatten(2).permute(0, 2, 1)
        
        return out

if __name__ == '__main__':
    # 模拟输入 (Batch=4, N=128, C=128)
    x = torch.randn(4, 128, 128) 
    
    # 实例化模型
    model = MSAA_Single(channels=128) 
    
    output = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 检查参数量 (轻量化设计)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {num_params}")

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# class AddAuxiliaryLoss(torch.autograd.Function):
#     """将辅助损失注入梯度流"""
#     @staticmethod
#     def forward(ctx, x, loss):
#         ctx.save_for_backward(loss)
#         return x

#     @staticmethod
#     def backward(ctx, grad_output):
#         (loss,) = ctx.saved_tensors
#         return grad_output, torch.ones_like(loss)
# class Orthogonal_BiMGRIA_Attention(nn.Module):
#     """
#     基于正交解耦的双向多粒度残差冲突感知注意力（最终可运行版本）
#     Orth-BiMGRIA: Orthogonal Disentanglement Bi-directional Multi-Granularity Residual Incongruity Attention
#     """
#     def __init__(self, head_dim, num_heads, dropout=0.1):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = head_dim
#         self.scale = self.head_dim ** -0.5  # 64^-0.5 ≈ 0.125
#         self.model_dim = self.head_dim * self.num_heads
#         print("Orthogonal_BiMGRIA_Attention model_dim:", self.model_dim)
#         # 线性投影层（保持原设计）
#         self.w_a = nn.Linear(self.model_dim, self.model_dim)
#         self.w_b = nn.Linear(self.model_dim, self.model_dim)
        
#         # 特征解耦编码器
#         self.shared_encoder = nn.Sequential(
#             nn.Linear(self.model_dim, self.model_dim), 
#             nn.LayerNorm(self.model_dim), 
#             nn.GELU()
#         )
#         self.conflict_encoder = nn.Sequential(
#             nn.Linear(self.model_dim, self.model_dim), 
#             nn.LayerNorm(self.model_dim), 
#             nn.GELU()
#         )
        
#         # 动态融合门控
#         self.fusion_gate = nn.Sequential(
#             nn.Linear(self.model_dim * 2, self.model_dim),
#             nn.Sigmoid()
#         )
#         self.out_proj = nn.Linear(self.model_dim, self.model_dim)
#         self.norm = nn.LayerNorm(self.model_dim)
#         self.dropout = nn.Dropout(dropout) 
#     def forward(self, x_a, x_b):
#         """
#         x_a: [B, H, D] 模态A特征（文本）
#         x_b: [B, H, D] 模态B特征（图像）
#         返回：融合特征, 正交损失, 共享特征, 冲突特征
#         """
#         B, H, D = x_a.shape  # B=8, H=8, D=64
#         #print("x_a:",x_a.shape)
#         flat_a = x_a.reshape(B, -1)  # [8, 8*64=512]
#         flat_b = x_b.reshape(B, -1)  # [8, 512]

#         # 1. 多粒度特征投影（输出[B, H, D]）
#         proj_a = self.w_a(flat_a).reshape(B, H, D)  # [8,8,64]
#         proj_b = self.w_b(flat_b).reshape(B, H, D)  # [8,8,64]
#         #print(proj_a.shape)
#         #print(proj_b.shape)
#         # ====================== 核心修复：正确计算相似度矩阵   ======================
#         # 使用einsum直接计算每个Head内的D×D交叉相似度矩阵
#         # 'bhd,bhe->bhde' 解释：
#         # - b: batch维度，h: head维度，d/e: 每个head的特征维度
#         # - 对每个batch、每个head，计算d×e的相似度矩阵（D=E=64）
#         sim_ab = torch.einsum('bhd,bhe->bhde', proj_a, proj_b) * self.scale  # [8,8,64,64]
#         sim_ba = torch.einsum('bhd,bhe->bhde', proj_b, proj_a) * self.scale  # [8,8,64,64]
        
#         # 对称协同相关性矩阵（双向Softmax交集）
#         S_ab = F.softmax(sim_ab, dim=-1)  # 在最后一维（e）做Softmax
#         S_ba = F.softmax(sim_ba, dim=-1)
#         S = S_ab * S_ba  # [8,8,64,64] （总元素数=8×8×64×64=262144，维度匹配）
#         # =======================================================================
        
#         # 3. 特征解耦：共享特征 + 冲突特征
#         # 共享特征：用相关性矩阵重构另一模态的特征s
#         shared_a = torch.einsum('bhde,bhe->bhd', S, proj_a)  # [8,8,64]
#         shared_b = torch.einsum('bhde,bhe->bhd', S.transpose(-1,-2), proj_b)  # [8,8,64]
        
#         # 编码共享/冲突特征
#         shared_feat = self.shared_encoder((shared_a + shared_b).reshape(B, -1))  # [8,512]
#         conflict_a = proj_a - shared_a  # A中无法被B解释的部分
#         conflict_b = proj_b - shared_b  # B中无法被A解释的部分
#         conflict_feat = self.conflict_encoder((conflict_a + conflict_b).reshape(B, -1))  # [8,512]
        
#         # 4. 正交损失计算
#         # orth_loss = F.cosine_similarity(shared_feat, conflict_feat, dim=-1).mean()
#         # orth_loss = (1 - torch.abs(F.cosine_similarity(shared_feat, conflict_feat, dim=-1))).mean()
#         orth_loss = self.hsic_loss(shared_feat, conflict_feat)
#         s_a = shared_a.reshape(shared_a.size(0), -1)
#         s_b = shared_b.reshape(shared_b.size(0), -1)
#         loss_share = 1 - F.cosine_similarity(s_a, s_b, dim=-1).mean()
#         # 5. 动态门控融合
#         concat_feat = torch.cat([shared_feat, conflict_feat], dim=-1)  # [8,1024]
#         gate_weight = self.fusion_gate(concat_feat)  # [8,512]
#         fusion_feat = gate_weight * shared_feat + (1 - gate_weight) * conflict_feat
        
#         # 6. 输出投影 + 残差连接
#         out = self.out_proj(fusion_feat)  # [8,512]
#         out = self.dropout(out)
#         out = self.norm(out + (flat_a + flat_b) / 2)  # 残差连接
#         #out = out + (flat_a + flat_b) / 2
#         out = out.reshape(B, H, D)
        
#         return out, orth_loss, loss_share #, shared_feat, conflict_feat
        
#     def hsic_loss(self, x, y):
#         """
#         希尔伯特-施密特独立性准则 (HSIC) - 修复设备不匹配问题
#         衡量两个分布 x, y 的统计独立性。
#         """
#         x = x.view(x.size(0), -1)
#         y = y.view(y.size(0), -1)
#         m = x.size(0)
        
#         # ========== 核心修复：让H矩阵和输入张量在同一设备上 ==========
#         # 获取输入张量的设备（如cuda:0）
#         device = x.device
#         # 计算核矩阵 (RBF Kernel)
#         K = torch.mm(x, x.t())
#         L = torch.mm(y, y.t())
        
#         # 中心化矩阵 H = I - 1/m，指定设备为输入张量的设备
#         H = torch.eye(m, device=device) - 1.0 / m * torch.ones((m, m), device=device)
        
#         # HSIC(x, y) = Tr(K H L H) / (m-1)^2
#         KH = torch.mm(K, H)
#         LH = torch.mm(L, H)
#         hsic = torch.trace(torch.mm(KH, LH)) / ((m - 1) ** 2)
#         return hsic
#     def pairwise_distances(self, x):
#         # 计算样本间的 L2 距离平方
#         instances_norm = torch.sum(x**2, -1).reshape((-1, 1))
#         dist = instances_norm + instances_norm.t() - 2 * torch.mm(x, x.t())
#         return torch.clamp(dist, min=0.0)
#     def hsic_loss1(self, x, y):
#         """
#         基于 RBF 核的高级 HSIC 损失函数
#         """
#         m = x.size(0)
#         device = x.device
        
#         # 将特征拉平
#         x = x.view(m, -1)
#         y = y.view(m, -1)
    
#         # 1. 定义 RBF 核计算函数 (使用中位数启发式自动设置 sigma)
#         def rbf_kernel(mat):
#             # 计算两两之间的平方欧式距离: ||x-y||^2 = ||x||^2 + ||y||^2 - 2xy^T
#             dist_sq = torch.cdist(mat, mat, p=2)**2
#             # Sigma 取距离平方的中位数（Median Heuristic），确保核函数的灵敏度
#             sigma = torch.median(dist_sq)
#             if sigma < 1e-7: sigma = 1e-7 # 防止除以 0
#             return torch.exp(-dist_sq / (2 * sigma))
    
#         # 2. 计算核矩阵 K 和 L
#         K = rbf_kernel(x)
#         L = rbf_kernel(y)
    
#         # 3. 创建中心化矩阵 H 并保证在同一设备
#         H = torch.eye(m, device=device) - (1.0 / m) * torch.ones((m, m), device=device)
    
#         # 4. 计算 HSIC: Tr(KHLH)
#         # 注意：为了数值稳定，先计算 KH 和 LH
#         KH = torch.mm(K, H)
#         LH = torch.mm(L, H)
        
#         # 最终迹计算
#         hsic = torch.trace(torch.mm(KH, LH)) / ((m - 1) ** 2)
#         return hsic
#     def rbf_kernel(self, x, sigma=None):
#         dist = self.pairwise_distances(x)
#         if sigma is None:
#             sigma = torch.median(dist) # 中位数启发式
#         return torch.exp(-dist / (2 * sigma + 1e-8))
# def orthogonal_constraint_loss(shared_feat, conflict_feat):
#     """强制共享特征与冲突特征正交（余弦相似度平方均值）"""
#     cos_sim = F.cosine_similarity(shared_feat, conflict_feat, dim=-1)
#     return (cos_sim ** 2).mean()
# # def hsic_loss(self, x, y):
# #     """
# #     基于 RBF 核的高级 HSIC 损失函数
# #     """
# #     m = x.size(0)
# #     device = x.device
    
# #     # 将特征拉平
# #     x = x.view(m, -1)
# #     y = y.view(m, -1)

# #     # 1. 定义 RBF 核计算函数 (使用中位数启发式自动设置 sigma)
# #     def rbf_kernel(mat):
# #         # 计算两两之间的平方欧式距离: ||x-y||^2 = ||x||^2 + ||y||^2 - 2xy^T
# #         dist_sq = torch.cdist(mat, mat, p=2)**2
# #         # Sigma 取距离平方的中位数（Median Heuristic），确保核函数的灵敏度
# #         sigma = torch.median(dist_sq)
# #         if sigma < 1e-7: sigma = 1e-7 # 防止除以 0
# #         return torch.exp(-dist_sq / (2 * sigma))

# #     # 2. 计算核矩阵 K 和 L
# #     K = rbf_kernel(x)
# #     L = rbf_kernel(y)

# #     # 3. 创建中心化矩阵 H 并保证在同一设备
# #     H = torch.eye(m, device=device) - (1.0 / m) * torch.ones((m, m), device=device)

# #     # 4. 计算 HSIC: Tr(KHLH)
# #     # 注意：为了数值稳定，先计算 KH 和 LH
# #     KH = torch.mm(K, H)
# #     LH = torch.mm(L, H)
    
# #     # 最终迹计算
# #     hsic = torch.trace(torch.mm(KH, LH)) / ((m - 1) ** 2)
# #     return hsic
# # ====================== 最终可运行测试案例 ======================
# if __name__ == "__main__":
#     # 1. 测试参数配置（严格保证dim=num_heads×head_dim）
#     batch_size = 8
#     num_heads = 8
#     head_dim = 64
#     dim = num_heads * head_dim  # 512
#     dropout = 0.1
    
#     # 2. 构造模拟输入（文本/图像模态特征）
#     torch.manual_seed(42)  # 固定随机种子，结果可复现
#     x_text = torch.randn(batch_size, num_heads, head_dim)  # [8,8,64]
#     x_image = torch.randn(batch_size, num_heads, head_dim) # [8,8,64]
    
#     # 3. 初始化模型
#     model = Orthogonal_BiMGRIA_Attention(
#         dim=dim,
#         num_heads=num_heads,
#         dropout=dropout
#     )
    
#     # 4. 前向传播测试（无维度错误）
#     print("===== 最终修复版 - 维度测试 =====")
#     fusion_feat, orth_loss, shared_feat, conflict_feat = model(x_text, x_image)
    
#     # 打印各维度验证
#     print(f"输入文本特征维度: {x_text.shape}          → 预期 [8,8,64]")
#     print(f"输入图像特征维度: {x_image.shape}         → 预期 [8,8,64]")
#     print(f"融合输出特征维度: {fusion_feat.shape}     → 预期 [8,512]")
#     print(f"共享特征维度: {shared_feat.shape}         → 预期 [8,512]")
#     print(f"冲突特征维度: {conflict_feat.shape}       → 预期 [8,512]")
#     print(f"正交损失值: {orth_loss.item():.4f}        → 正常数值（无报错）")
    
#     # 5. 正交损失计算验证
#     orth_loss_external = orthogonal_constraint_loss(shared_feat, conflict_feat)
#     print(f"\n外部计算正交损失值: {orth_loss_external.item():.4f} → 预期≈(正交损失值)²")
    
#     # 6. 模拟训练步骤（验证梯度可回传）
#     print("\n===== 训练梯度测试 =====")
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#     criterion_cls = nn.CrossEntropyLoss()
    
#     # 单步训练
#     optimizer.zero_grad()
#     fusion_feat, orth_loss, shared_feat, conflict_feat = model(x_text, x_image)
#     # 模拟分类任务
#     cls_logits = nn.Linear(dim, 2)(fusion_feat)
#     labels = torch.randint(0, 2, (batch_size,))
#     loss_cls = criterion_cls(cls_logits, labels)
#     loss_orth = orthogonal_constraint_loss(shared_feat, conflict_feat)
#     total_loss = loss_cls + 0.3 * loss_orth
#     # 反向传播（验证无梯度错误）
#     total_loss.backward()
#     optimizer.step()
    
#     print("✅ 前向传播+反向传播均无错误，模型可正常训练！")
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AddAuxiliaryLoss(torch.autograd.Function):
    """
    将辅助损失注入梯度流。
    在 forward 时原样返回特征，在 backward 时将 loss 的梯度加到反向传播中。
    """
    @staticmethod
    def forward(ctx, x, loss):
        # 保存 loss 用于 backward
        ctx.save_for_backward(loss)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (loss,) = ctx.saved_tensors
        # grad_output 是主任务传回的梯度
        # torch.ones_like(loss) 是为辅助损失提供的梯度起点
        return grad_output, torch.ones_like(loss)

class Orthogonal_BiMGRIA_Attention(nn.Module):
    def __init__(self, head_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.model_dim = self.head_dim * self.num_heads
        
        # 线性投影层
        self.w_a = nn.Linear(self.model_dim, self.model_dim)
        self.w_b = nn.Linear(self.model_dim, self.model_dim)
        
        # 特征解耦编码器
        self.shared_encoder = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim), 
            nn.LayerNorm(self.model_dim), 
            nn.GELU()
        )
        self.conflict_encoder = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim), 
            nn.LayerNorm(self.model_dim), 
            nn.GELU()
        )
        
        # 动态融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.model_dim * 2, self.model_dim),
            nn.Sigmoid()
        )
        self.out_proj = nn.Linear(self.model_dim, self.model_dim)
        self.norm = nn.LayerNorm(self.model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_a, x_b):
        B, H, D = x_a.shape 
        flat_a = x_a.reshape(B, -1)
        flat_b = x_b.reshape(B, -1)

        # 1. 多粒度特征投影
        proj_a = self.w_a(flat_a).reshape(B, H, D)
        proj_b = self.w_b(flat_b).reshape(B, H, D)

        # 2. 计算对称协同相关性矩阵
        sim_ab = torch.einsum('bhd,bhe->bhde', proj_a, proj_b) * self.scale
        sim_ba = torch.einsum('bhd,bhe->bhde', proj_b, proj_a) * self.scale
        S = F.softmax(sim_ab, dim=-1) * F.softmax(sim_ba, dim=-1)
        
        # 3. 特征解耦：共享特征 + 冲突特征
        shared_a = torch.einsum('bhde,bhe->bhd', S, proj_a)
        shared_b = torch.einsum('bhde,bhe->bhd', S.transpose(-1,-2), proj_b)
        
        shared_feat = self.shared_encoder((shared_a + shared_b).reshape(B, -1))
        conflict_a = proj_a - shared_a
        conflict_b = proj_b - shared_b
        conflict_feat = self.conflict_encoder((conflict_a + conflict_b).reshape(B, -1))
        
        # 4. 计算辅助损失
        # (1) 独立性损失 HSIC (希望 shared 和 conflict 尽可能独立)
        orth_loss = self.hsic_loss(shared_feat, conflict_feat)
        # (2) 共享一致性损失 (希望 s_a 和 s_b 尽可能相似)
        s_a = shared_a.reshape(B, -1)
        s_b = shared_b.reshape(B, -1)
        loss_share = 1 - F.cosine_similarity(s_a, s_b, dim=-1).mean()
        
        # 合并辅助损失 (可以根据需要调整系数)
        total_aux_loss = orth_loss + 0.5 * loss_share

        # 5. 动态门控融合
        concat_feat = torch.cat([shared_feat, conflict_feat], dim=-1)
        gate_weight = self.fusion_gate(concat_feat)
        fusion_feat = gate_weight * shared_feat + (1 - gate_weight) * conflict_feat
        
        # 6. 输出投影 + 残差连接
        out = self.out_proj(fusion_feat)
        out = self.dropout(out)
        out = self.norm(out + (flat_a + flat_b) / 2)
        out = out.reshape(B, H, D)

        # ====================== 关键修正：注入梯度流 ======================
        # 如果在训练模式，使用 AddAuxiliaryLoss 钩子
        if self.training:
            out = AddAuxiliaryLoss.apply(out, total_aux_loss)
        # ===============================================================

        return out, orth_loss, loss_share # 依然返回 loss 供监控，但 backward 已经自动化

    def hsic_loss(self, x, y):
        device = x.device
        m = x.size(0)
        K = torch.mm(x, x.t())
        L = torch.mm(y, y.t())
        H = torch.eye(m, device=device) - 1.0 / m * torch.ones((m, m), device=device)
        KH = torch.mm(K, H)
        LH = torch.mm(L, H)
        return torch.trace(torch.mm(KH, LH)) / ((m - 1) ** 2)

# ====================== 测试代码 ======================
if __name__ == "__main__":
    batch_size, num_heads, head_dim = 8, 8, 64
    x_text = torch.randn(batch_size, num_heads, head_dim)
    x_image = torch.randn(batch_size, num_heads, head_dim)
    
    model = Orthogonal_BiMGRIA_Attention(head_dim=head_dim, num_heads=num_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 模拟训练步骤
    model.train()
    optimizer.zero_grad()
    
    # 前向传播：此时 output 内部已经携带了 total_aux_loss 的梯度路径
    output, aux_loss_val = model(x_text, x_image)
    
    # 模拟主任务损失（如分类）
    main_loss = output.mean() ** 2 
    
    # 只对 main_loss 调用 backward
    # 由于 AddAuxiliaryLoss 的存在，aux_loss 会自动产生梯度更新模型
    main_loss.backward()
    
    optimizer.step()
    
    print("✅ 注入成功！辅助损失已自动参与反向传播。")
    print(f"当前监控到的辅助损失值: {aux_loss_val.item():.6f}")

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoHAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            shared_head=0,
            routed_head=0,
            head_dim=None,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        assert shared_head + routed_head <= num_heads, "共享头 + 动态头不能超过总头数！"
        self.num_heads = num_heads
        
        if head_dim is None:
            self.head_dim = dim // num_heads
        else:
            self.head_dim = head_dim
        
        self.scale = self.head_dim ** -0.5
        # self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, (self.head_dim * self.num_heads) * 3, bias=qkv_bias)
    
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        
        self.proj_drop = nn.Dropout(proj_drop)

        self.shared_head = shared_head
        self.routed_head = routed_head
        
        # if self.routed_head > 0:
        self.wg = torch.nn.Linear(dim, num_heads - shared_head, bias=False)
        if self.shared_head > 0:
            self.wg_0 = torch.nn.Linear(dim, 2, bias=False)

        if self.shared_head > 1:
            self.wg_1 = torch.nn.Linear(dim, shared_head, bias=False)

        # 新增投影层
        self.q_proj = nn.Linear(dim, self.head_dim * self.num_heads, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, self.head_dim * self.num_heads, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, self.head_dim * self.num_heads, bias=qkv_bias)



    def forward(self, x1, y1):
        x = x1.clone()
        x.requires_grad_(True)
        y = y1.clone()
        y.requires_grad_(True)
        B, N, C = x.shape
        _, _, C_y = y.shape

        assert C == C_y, "X/Y的嵌入维度必须一致"
        # print("x 版本 before t:", x._version)
        _x = x.reshape(B * N, C)#.clone()
        
        # print("_x 版本 before t:", _x._version)
        if self.routed_head > 0:
            # print("_x 版本 before _x:", _x._version)
            logits = self.wg(_x)
            # print("_x 版本 before _x:", _x._version)
            gates = F.softmax(logits, dim=1)

            num_tokens, num_experts = gates.shape
            _, indices = torch.topk(gates, k=self.routed_head, dim=1)
            mask = F.one_hot(indices, num_classes=num_experts).sum(dim=1)

            me = gates.mean(dim=0)  # 专家选择概率的均值
            ce = mask.float().mean(dim=0)  # 专家被选中的频率
            moh_load_balance_loss = torch.mean(me * ce) * num_experts * num_experts
            

            routed_head_gates = gates * mask
            denom_s = torch.sum(routed_head_gates, dim=1, keepdim=True)
            denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
            routed_head_gates = routed_head_gates / denom_s
            routed_head_gates = routed_head_gates.reshape(B, N, -1) * self.routed_head
        
        #print("x:{}", x.shape)
        # 生成Q/K/V
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(y).view(B, N, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(y).view(B, N, self.num_heads, self.head_dim).transpose(1,2)

        # 归一化处理
        q = self.q_norm(q)
        k = self.k_norm(k)
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # print("qkv:",qkv.shape)
        # q, k, v = qkv.unbind(0)
        # print("q:,{} k:{}, v:{}", q.shape, k.shape, v.shape)
        # q, k = self.q_norm(q), self.k_norm(k)

        # scale_dot_product_attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v    
        
        if self.routed_head > 0:
            x = x.transpose(1, 2)

            if self.shared_head > 0:
                shared_head_weight = self.wg_1(_x)
                shared_head_gates = F.softmax(shared_head_weight, dim=1).reshape(B, N, -1) * self.shared_head

                weight_0 = self.wg_0(_x)
                weight_0 = F.softmax(weight_0, dim=1).reshape(B, N, 2) * 2
        
                shared_head_gates = torch.einsum("bn,bne->bne", weight_0[:,:,0], shared_head_gates)
                routed_head_gates = torch.einsum("bn,bne->bne", weight_0[:,:,1], routed_head_gates)
                
                masked_gates = torch.cat([shared_head_gates, routed_head_gates], dim=2)
            else:
                masked_gates = routed_head_gates

            x = torch.einsum("bne,bned->bned", masked_gates, x)
            x = x.reshape(B, N, self.head_dim * self.num_heads)
        else:
            shared_head_weight = self.wg_1(_x)
            masked_gates = F.softmax(shared_head_weight, dim=1).reshape(B, N, -1) * self.shared_head
            x = x.transpose(1, 2)

            x = torch.einsum("bne,bned->bned", masked_gates, x)
            x = x.reshape(B, N, self.head_dim * self.num_heads)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, moh_load_balance_loss

if __name__ == '__main__':
    moh_attn = MoHAttention(
        dim=512,
        num_heads=8,
        shared_head=0,
        routed_head=3,
        qk_norm=True
    )

    # 输入数据 (batch_size=2, seq_len=10, embed_dim=512)
    x = torch.randn(2, 1, 512)
    y = torch.randn(2, 1, 512)
    # 前向计算
    output, moh_loss = moh_attn(x,y)
    print(output.shape)