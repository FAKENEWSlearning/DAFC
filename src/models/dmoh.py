import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionGate(nn.Module):
    def __init__(self, dim, num_experts, head_dim):
        super().__init__()
        self.num_experts = num_experts
        self.head_dim = head_dim
        # print("head_dim",head_dim)
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, num_experts * head_dim)
        self.k_proj = nn.Linear(dim, num_experts * head_dim)
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)

        # 修正：将 Sequential 拆开，以便我们可以操作 logits
        self.gate_fc_linear = nn.Linear(num_experts * head_dim, num_experts)
        
        self.base_threshold = nn.Parameter(torch.full((num_experts,), 0.2))
        self.temperature = nn.Parameter(torch.tensor(2.0))

    def forward(self, x, y, router_probs, remaining_mask):
        B, N, _ = x.shape
        epsilon = 1e-8

        # 1. 计算不确定性
        entropy = -torch.sum(router_probs * torch.log(router_probs + epsilon), dim=-1, keepdim=True)
        max_entropy = torch.log(torch.tensor(float(self.num_experts)))
        norm_entropy = entropy / (max_entropy + epsilon)

        # 2. 跨模态注意力计算
        q = self.q_proj(x).view(B, N, self.num_experts, self.head_dim).transpose(1, 2)
        k = self.k_proj(y).view(B, N, self.num_experts, self.head_dim).transpose(1, 2)
        q = self.q_norm(q); k = self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn, dim=-1) 
        context = (attn_weights @ k).transpose(1, 2).reshape(B, N, -1)
        
        # 3. 注入掩码的核心逻辑
        gate_logits = self.gate_fc_linear(context)
        # 将已经被选中的专家在 Logits 层面屏蔽 (赋予极小值)
        # 这样经过 Sigmoid 后，对应的 multi_gates 就会接近 0
        masked_logits = gate_logits.masked_fill(remaining_mask == 0, -1e9)
        raw_gate_values = torch.sigmoid(masked_logits)

        # 4. 自适应稀疏化
        adaptive_threshold = self.base_threshold * (1.0 - norm_entropy)
        diff = raw_gate_values - adaptive_threshold
        soft_mask = torch.sigmoid(diff * torch.clamp(self.temperature, min=0.1, max=10.0))
        
        multi_gates = raw_gate_values * soft_mask
        return multi_gates, norm_entropy

class AddAuxiliaryLoss(torch.autograd.Function):
    """
    将辅助损失注入梯度流。
    在 forward 时原样返回特征 x，但会在 backward 时自动计算并传播 loss 的梯度。
    """
    @staticmethod
    def forward(ctx, x, loss):
        # 确保 loss 是标量
        ctx.save_for_backward(loss)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (loss,) = ctx.saved_tensors
        # grad_output: 来自后续层的梯度 (与 x 形状相同)
        # 第二个返回值对应 loss 的梯度，设为 1.0 即可触发 loss 的反向传播
        return grad_output, torch.ones_like(loss)

class MoHAttention(nn.Module):
    def __init__(self, dim, num_heads=8, shared_head=0, routed_head=1, head_dim=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim else dim // num_heads
        self.shared_head = shared_head
        self.routed_head = routed_head
        self.num_experts = num_heads - shared_head

        # 投影层定义
        self.q_proj = nn.Linear(self.head_dim, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.head_dim, self.num_heads * self.head_dim)
        self.wl = nn.Linear(self.head_dim, self.num_experts, bias=False)
        
        # 系数融合：4个分支 (Shared, Routed, Dynamic, Identity)
        self.wc_0 = nn.Linear(self.head_dim + 1, 3, bias=False)

        self.proj_s = nn.Linear(self.head_dim * self.shared_head, self.head_dim) if shared_head > 0 else None
        self.proj_r = nn.Linear(self.head_dim * self.num_experts, self.head_dim)
        self.proj_d = nn.Linear(self.head_dim * self.num_experts, self.head_dim)

        self.mha_gate = MultiHeadAttentionGate(self.head_dim, self.num_experts, self.head_dim)
        self.dynamic_layer_norm = nn.LayerNorm(self.head_dim)
        self.dynamic_dropout = nn.Dropout(p=0.1)
        self.final_norm = nn.LayerNorm(self.head_dim)
        self.proj_x = nn.Linear(self.head_dim * self.num_heads, self.head_dim)
        self.residual_proj = nn.Linear(self.head_dim, self.head_dim)
        
    def forward(self, x_in, y_in):
        B, N, C = x_in.shape

        # 1. 基础路由与掩码生成
        gates = F.softmax(self.wl(x_in), dim=-1)
        topk_mask = torch.zeros_like(gates).scatter_(2, torch.topk(gates, k=self.routed_head, dim=-1)[1], 1.0)
        remaining_mask = 1.0 - topk_mask
  
        # --- 计算辅助损失 ---
        me = gates.mean(dim=(0, 1))  # 修正：在 batch 和 seq 维度上求均值
        ce = topk_mask.mean(dim=(0, 1))
        load = me * ce  # 每个专家的实际负载 (E,)
        moh_load_balance_loss = torch.mean(me * ce) * (self.num_experts ** 2)
        # moh_load_balance_loss = torch.sum((load - 1/self.num_experts) ** 2) * self.num_experts
            
        # 2. 计算动态门控
        multi_gates, uncertainty = self.mha_gate(x_in, y_in, gates, remaining_mask)

        # 3. 门控合并
        routed_gates = gates * topk_mask
        dynamic_gates = gates * remaining_mask * multi_gates
        # print("x_in.shape",x_in.shape)
        # 4. 注意力特征提取
        q = self.q_proj(x_in).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(y_in).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        x_out = (F.softmax((q @ k.transpose(-2, -1)) * (self.head_dim**-0.5), dim=-1) @ k).transpose(1, 2)
        expert_outputs = x_out[:, :, self.shared_head:]
        share_outputs = x_out[:, :, :self.shared_head]
        # 5. 系数计算与聚合
        coeffs = F.softmax(self.wc_0(torch.cat([x_in, uncertainty], dim=-1)), dim=-1)
        res_r_outputs = expert_outputs * routed_gates.unsqueeze(-1)
        res_d_outputs = expert_outputs * dynamic_gates.unsqueeze(-1)

        x_out = torch.cat([share_outputs* coeffs[:, :, 0:1].unsqueeze(dim=-2), res_r_outputs* coeffs[:, :, 1:2].unsqueeze(dim=-2) + res_d_outputs* coeffs[:, :, 2:3].unsqueeze(dim=-2)],dim=-2)

        
        # x_in * coeffs[:, :, 3:4].unsqueeze(dim=-2)
        # 原始输出投影
        out = self.proj_x(x_out.reshape(B, N, -1))
        
        # ========== 核心修改：添加残差连接 ==========
        # 先适配x_in的维度（确保和out维度一致）
 
        #x_in_adapted = self.residual_proj(x_in * coeffs[:, :, 3:4])
        # 残差相加
        out = out #+ x_in_adapted
        out = self.final_norm(out)
        out = AddAuxiliaryLoss.apply(out, moh_load_balance_loss)
        
        return out, moh_load_balance_loss

def test_moh_fallback_behavior():
    # 1. 基础配置
    dim = 128
    num_heads = 8
    shared_head = 2
    routed_head = 1
    batch_size = 2
    seq_len = 4

    # 初始化模型并切换到训练模式
    model = MoHAttention(
        dim=dim, 
        num_heads=num_heads, 
        shared_head=shared_head, 
        routed_head=routed_head
    )
    model.train() # 必须是 train 模式

    # 构造输入，并开启梯度追踪
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    y = torch.randn(batch_size, seq_len, dim)

    print("\n" + "="*50)
    print("开始单元测试：极端不确定性与梯度流验证")
    print("="*50)

    # 2. 模拟极端不确定性：让路由器权重归零
    # 注意：修改参数必须在 no_grad 块下进行
    with torch.no_grad():
        model.wl.weight.zero_()
        model.mha_gate.base_threshold.fill_(0.9) # 设高阈值，抑制专家激活

    # 3. 前向传播 (不要使用 no_grad!)
    output, aux_loss = model(x, y)
    
    # 4. 验证计算图连通性
    if output.grad_fn is None:
        print("❌ 错误：输出张量没有 grad_fn，梯度链条断裂！")
        return

    # 5. 执行反向传播
    # 计算一个标量 loss
    loss = output.sum() + aux_loss
    loss.backward()

    # 6. 检查各个参数的梯度情况
    grad_status = {
        "wl (Router)": model.wl.weight.grad is not None,
        "mha_gate (Dynamic)": model.mha_gate.gate_fc_linear.weight.grad is not None,
        "wc_0 (Coeffs)": model.wc_0.weight.grad is not None
    }

    print("\n--- 梯度流状态检查 ---")
    all_ok = True
    for name, status in grad_status.items():
        print(f"{name:20}: {'✅ OK' if status else '❌ No Grad'}")
        if not status: all_ok = False

    # 7. 打印不确定性下的系数分布
    with torch.no_grad():
        # 通过一个 forward 外部逻辑看看 coeffs 怎么分的
        # 注意此时 uncertainty 应该很大 (因为 wl 被置零了)
        dummy_input = torch.cat([x, torch.ones(batch_size, seq_len, 1)], dim=-1)
        coeffs = torch.softmax(model.wc_0(dummy_input), dim=-1)
        
        print("\n--- 高不确定性下的分支权重平均值 ---")
        c_names = ["Shared", "Routed", "Dynamic", "Identity(残差)"]
        avg_coeffs = coeffs.mean(dim=(0, 1))
        for i, name in enumerate(c_names):
            print(f"  {name:15}: {avg_coeffs[i].item():.4f}")

    if all_ok:
        print("\n🎉 单元测试通过：模型逻辑健壮，梯度流正常！")
    else:
        print("\n⚠️ 单元测试未完全通过，请检查代码逻辑。")

if __name__ == "__main__":
    test_moh_fallback_behavior()