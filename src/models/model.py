import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import tqdm
import torchvision.models as models
from transformers import BertModel, BertTokenizer, SwinModel
from src.models.dmoh import MoHAttention
from src.losses.losses import js_divergence, kl_divergence, SupervisedContrastiveLoss, info_nce_loss, TripletLoss
from src.models.attention import Orthogonal_BiMGRIA_Attention
from src.models.imoe import IMOE
import cn_clip.clip as cp # 需要安装: pip install cn-clip
from transformers import AutoModel
import clip
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import torch.nn.functional as F
from transformers import AutoModel, ChineseCLIPProcessor, ChineseCLIPModel
import cn_clip.clip as clipcn
from cn_clip.clip import load_from_name, available_models
from src.models.umoe import UMoELayer
from types import SimpleNamespace
from src.models.attention import MSAA_Single
# 配置映射：根据数据集自动选择最佳预训练权重
PRETRAINED_MAP = {
    "weibo_dataset": {
        "text": "./chinese-roberta-wwm-ext",# "hfl/chinese-roberta-wwm-ext"
        "clip_model": "./chinese-clip-vit-large-patch14", # "OFA-Sys/chinese-clip-vit-large-patch14"中文 CLIP 规格
        "image_dim": 2048,  # ResNet50 输出
        "is_chinese": True
    },
    "twitter_dataset": {# xlm-roberta-base
        "text": "./roberta-base", # 专门针对 Twitter 优化vinai/bertweet-base
        "clip_model": "ViT-B/32", # 原生 CLIP 规格
        "image_dim": 2048,
        "is_chinese": False
    },
    "gossipcop_dataset": {
        "text": "./roberta-base",# xlm-roberta-base
        "clip_model": "ViT-B/32",
        "image_dim": 2048,
        "is_chinese": False
    },
    "politifact_dataset": {
        "text": "./roberta-base",#xlm-roberta-base
        "clip_model": "ViT-B/32",
        "image_dim": 2048,
        "is_chinese": False
    }
}

class TextModel(nn.Module):
    def __init__(self, dataset_name):
        super().__init__()
        model_name = PRETRAINED_MAP.get(dataset_name, {}).get("text", "bert-base-uncased")
        # 使用 AutoModel 自动兼容不同的模型架构 (BERT, RoBERTa, BERTweet)
        print(model_name)
        self.textmodel = AutoModel.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.textmodel(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids# if token_type_ids is not None else None
        )
        #print(outputs)
        # 统一使用 pooler_output (经过池化层的向量) 或 last_hidden_state[:, 0, :]
        return outputs.last_hidden_state[:, 0, :]

class ImageModel(nn.Module):
    def __init__(self, use_vit=False):
        super().__init__()
        if use_vit:
            # 方案 A: 使用视觉 Transformer
            self.model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")
            self.feature_dim = 768
        else:
            # 方案 B: 使用更高精度的 ResNet50 V2
            from torchvision.models import resnet50, ResNet50_Weights
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.feature_dim = self.model.fc.in_features
            self.model.fc = nn.Identity()

    def forward(self, images):
        out = self.model(images)
        # 如果是 ViT，取其 pooler_output (CLS token)
        if hasattr(out, 'pooler_output'):
            return out.pooler_output
        return out

 
class ChineseCLIPExtractor(nn.Module):
    def __init__(self, dataset_name):
        super().__init__()
        config = PRETRAINED_MAP.get(dataset_name, {})
        self.is_chinese = config.get("is_chinese", False)
        self.clip_model_name = config.get("clip_model", "ViT-B/32")
        
        if self.is_chinese:
            # 使用 Transformers 加载中文 CLIP
            #self.model = ChineseCLIPModel.from_pretrained(self.clip_model_name)
            #self.processor = ChineseCLIPProcessor.from_pretrained(self.clip_model_name)
            self.model, self.preprocess = load_from_name("ViT-B-16", download_root='./', use_modelscope=True)
        else:
            # 英文环境下保持 OpenAI CLIP
            self.model, self.preprocess = clip.load(self.clip_model_name)
            
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image, text):
        if self.is_chinese:
            # image 和 text 在这里假设已经是处理好的 tensor
            # 或者直接调用 model 的 get 方法
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
        else:
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
        
        # 归一化
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        return image_features, text_features


def load_pre_models(dataset, modality):
    if modality == 'text':
        # model = TextModel(dataset, pretraining)
        model = TextModel(dataset)
    elif modality == 'image':
        #model = ImageModel(dataset, pretraining)
        model = ImageModel()
    else:
        raise ValueError('ERROR! modality must be text or image.')
    if torch.cuda.is_available():
        model = model.cuda()
    return model

class Model(nn.Module):
    def __init__(self, dataset, input_dim=1024, hidden_dim=512, output_dim=386):
        super().__init__()
        # 1. 加载基础 预训练模型
        self.imagemodel = load_pre_models(dataset, 'image')
        self.textmodel = load_pre_models(dataset, 'text')

        # 2. 加载 CLIP 提取器并动态获取维度
        self.clip_extractor = ChineseCLIPExtractor(dataset)
        
        # 动态检测 CLIP 维度 (核心修改)
        # 中文 Large 通常是 768，英文 ViT-B/32 是 512
        with torch.no_grad():
            # 探测文本特征维度
            config = PRETRAINED_MAP.get(dataset, {})
            self.is_chinese = config.get("is_chinese", False)
            
            if self.is_chinese:
                self.clip_dim = 512#self.clip_extractor.model.config.projection_dim
            else:
                self.clip_dim = 512#self.clip_extractor.model.visual.output_dim

        # 冻结参数
        for param in self.imagemodel.parameters(): param.requires_grad = False#False
        for param in self.textmodel.parameters(): param.requires_grad = False#False
        for param in self.clip_extractor.parameters(): param.requires_grad = False#False

        self.class_num = 2
        self.model_dim = 64

        # ====================== 自适应线性层 ======================
        # 根据 self.clip_dim 自动调整输入维度
        self.num_heads = 4
        self.head_dim = self.model_dim // self.num_heads
        print("head_dim",self.head_dim)
        self.clip_text_layers = nn.Sequential(
            nn.Linear(self.clip_dim, self.model_dim), 
            nn.BatchNorm1d(self.model_dim), 
            nn.LeakyReLU(0.5),
            nn.Dropout(0.5),
        )
        self.clip_image_layers = nn.Sequential(
            nn.Linear(self.clip_dim, self.model_dim), 
            nn.BatchNorm1d(self.model_dim), 
            nn.LeakyReLU(0.5),
            nn.Dropout(0.5),
        )

        # 原始特征层 (RoBERTa 768 / ResNet 2048)
        self.text_layers = nn.Sequential(
            nn.Linear(768, self.model_dim), # RoBERTa 输出通常是 768
            nn.BatchNorm1d(self.model_dim), 
            nn.LeakyReLU(0.5),
            nn.Dropout(0.5),
        )
        
        self.image_layers = nn.Sequential(
            nn.Linear(2048, self.model_dim), # ResNet50 是 2048， ViT 是 768
            nn.BatchNorm1d(self.model_dim),
            nn.LeakyReLU(0.5),
            nn.Dropout(0.5),
        )
        
        # BLIP 层及其它融合层 (假设 BLIP 输入为 768)
        self.blip_text_layers = nn.Sequential(
            nn.Linear(512, self.model_dim), 
            nn.BatchNorm1d(self.model_dim), 
            nn.LeakyReLU(0.5),
            nn.Dropout(0.5),
        )
        self.blip_image_layers = nn.Sequential(
            nn.Linear(512, self.model_dim), 
            nn.BatchNorm1d(self.model_dim), 
            nn.LeakyReLU(0.5),
            nn.Dropout(0.5),
        )

        # ====================== 特征融合MOH =====================
        self.MoH_multi1 = MoHAttention(
            dim=self.model_dim, 
            num_heads=self.num_heads, 
            shared_head=1, 
            routed_head=3
        )
        self.MoH_multi2 = MoHAttention(
            dim=self.model_dim, 
            num_heads=self.num_heads, 
            shared_head=1, 
            routed_head=3
        )
        self.MoH_text = MoHAttention(
            dim=self.model_dim,
            num_heads=self.num_heads,
            shared_head=1,
            routed_head=3
        )
        self.MoH_image = MoHAttention(
            dim=self.model_dim,
            num_heads=self.num_heads,
            shared_head=1,
            routed_head=3
        )
        self.MoH_blip = MoHAttention(
            dim=self.model_dim,
            num_heads=self.num_heads,
            shared_head=1,
            routed_head=3
        )
        # ===== t =========
        self.t_alpha = nn.Parameter(torch.zeros(1))
        self.t_beta = nn.Parameter(torch.ones(1))
        self.t_gamma = nn.Parameter(torch.ones(1))
        self.t_delta = nn.Parameter(torch.zeros(1))
        self.to_t = nn.Linear(self.head_dim, self.head_dim)
        self.t_ffn = nn.Linear(self.head_dim, self.head_dim)

        # ===== blip t =====
        self.blip_t_alpha = nn.Parameter(torch.zeros(1))
        self.blip_t_beta = nn.Parameter(torch.ones(1))
        self.blip_t_gamma = nn.Parameter(torch.ones(1))
        self.blip_t_delta = nn.Parameter(torch.zeros(1))
        self.to_blip_t = nn.Linear(self.head_dim, self.head_dim)
        self.blip_t_ffn = nn.Linear(self.head_dim, self.head_dim)
        
        # ===== v =====
        self.v_alpha = nn.Parameter(torch.zeros(1))
        self.v_beta = nn.Parameter(torch.ones(1))
        self.v_gamma = nn.Parameter(torch.ones(1))
        self.v_delta = nn.Parameter(torch.zeros(1))
        self.to_v = nn.Linear(self.head_dim, self.head_dim)
        self.v_ffn = nn.Linear(self.head_dim, self.head_dim)

        # ===== blip v =====
        self.blip_v_alpha = nn.Parameter(torch.zeros(1))
        self.blip_v_beta = nn.Parameter(torch.ones(1))
        self.blip_v_gamma = nn.Parameter(torch.ones(1))
        self.blip_v_delta = nn.Parameter(torch.zeros(1))
        self.to_blip_v = nn.Linear(self.head_dim, self.head_dim)
        self.blip_v_ffn = nn.Linear(self.head_dim, self.head_dim)

        # ====================== 交叉注意力层 ======================

        self.MGRIA_tv = Orthogonal_BiMGRIA_Attention(
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            dropout=0.5
        )
        self.MGRIA_blip = Orthogonal_BiMGRIA_Attention(
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            dropout=0.5
        ) 
        self.MGRIA_multi = Orthogonal_BiMGRIA_Attention(
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            dropout=0.5
        )
        # ====================== 输出层 ======================
        # self.ori_layer = nn.Sequential(
        #     nn.Linear(self.model_dim*3, self.head_dim), 
        #     nn.BatchNorm1d(self.head_dim), 
        #     #nn.ReLU()
        #     nn.LeakyReLU(0.1),
        #     nn.Dropout(0.5),
        #     nn.Linear(self.head_dim, 2)  # 输出2维（对应标签0/1）
        # )
        # ====================== 改进输出层 ======================
        self.ori_layer = nn.Sequential(
            nn.Linear(self.model_dim*3, self.model_dim//2), 
            nn.BatchNorm1d(self.model_dim//2), 
            # nn.ReLU(),
            # nn.Tanh(),
            nn.LeakyReLU(0.5),
            nn.Dropout(0.5),
            nn.Linear(self.model_dim//2, 2)  # 输出2维（对应标签0/1）
        )
 
        self.prod_transform = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.LeakyReLU(0.5),
            nn.Dropout(0.5),
            nn.Linear(self.model_dim, self.model_dim//2)  # 降维到32
        )
        self.gate_abs = nn.Sequential(
            nn.Linear(self.model_dim//2, 1),  # 32→1，输出权重
            nn.Sigmoid()  # 权重归一化到0-1
        )
        self.gate_prod = nn.Sequential(
            nn.Linear(self.model_dim//2, 1),
            nn.Sigmoid()
        )
        # 4. 最终融合层（拼接主特征+abs+prod后降维）

        self.config = SimpleNamespace(
            hidden_size=self.head_dim,
            moe_intermediate_size=16,
            n_routed_experts=8,
            num_experts_per_tok=4,
            n_shared_experts=2,
            hidden_act="silu"
        )
 
        self.ot = MMD_Global_Alignment()
        self.attention = MultiHeadAttentionFusion(feature_dim=self.model_dim,
                                        num_heads = 8)
        self.ot_local1 = OT_Local_Alignment()
        self.ot_local2 = OT_Local_Alignment()
        self.ot_local3 = OT_Local_Alignment()

        self.mssa_text = MSAA_Single(channels=self.head_dim)
        self.mssa_image = MSAA_Single(channels=self.head_dim)
        self.mssa_blip = MSAA_Single(channels=self.head_dim)
        pool_type="mean"
        self.du = DualBranchModelForSamplePred(d_model=self.head_dim, num_classes=2, pool_type=pool_type)
        in_dim = self.head_dim if pool_type != "mean_max" else 2*self.head_dim
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, self.model_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.model_dim, 2)
        )
        self.w1 = nn.Parameter(torch.tensor(1.0/3))  # t_self权重
        self.w2 = nn.Parameter(torch.tensor(1.0/3))  # v_self权重
        self.w3 = nn.Parameter(torch.tensor(1.0/3))  # blip_self权重
        # ====================== 改进输出层 ======================
    def forward(self, images, input_ids, attention_mask,token_type_ids, pre_images, pre_text):
        # ============================================================================================
        # 获取当前实际的 Batch Size (防止最后一个 batch 数量不足)
        batch_size = pre_images.size(0)

        # 转换维度: (B, model_dim) -> (B, num_heads, head_dim)
        # blip_v = blip_v.view(batch_size, self.num_heads, self.head_dim)
        blip_v = pre_images
        blip_t = pre_text
        
        blip_v = self.blip_image_layers(blip_v)
        blip_t = self.blip_text_layers(blip_t) 
        
        blip_v = blip_v.view(batch_size, self.num_heads, self.head_dim)
        blip_t = blip_t.view(batch_size, self.num_heads, self.head_dim)
        # =======================================================================
        # resnet
        v = self.imagemodel(images)
        v = self.image_layers(v)
        v = v.view(batch_size, self.num_heads, self.head_dim)
        t = self.textmodel(input_ids, attention_mask ,token_type_ids)
        t = self.text_layers(t)
        t = t.view(batch_size, self.num_heads, self.head_dim)
        # ot1= self.ot(blip_v, blip_t)
        
        # t_moh, t_moh_loss = self.MoH_multi(t, blip_t)
        # v_moh, v_moh_loss = self.MoH_multi(v, blip_v)

        # t_self, t_self_loss = self.MoH_text(t, t)
        t =self.mssa_text(t)
        v = self.mssa_image(v)
        # blip_mssa = self.mssa_blip(blip_self)
        t_self, t_self_loss = self.MoH_text(t, t)
        if self.training:
            t_self = AddAuxiliaryLoss.apply(t_self, t_self_loss)
        # t_self_linear = self.to_t(t_self)
        # t_self_prime = self.t_alpha * t_self + self.t_beta * t_self_linear
        # t_self_ffn_out = self.t_ffn(t_self_prime)
        # t_self_out = self.t_gamma * t_self_prime + self.t_delta * t_self_ffn_out

        # v_self, v_self_loss = self.MoH_image(v, v)
        v_self, v_self_loss = self.MoH_image(v, v)
        if self.training:
            v_self = AddAuxiliaryLoss.apply(v_self, v_self_loss)
        # v_self_linear = self.to_v(v_self)
        # v_self_prime = self.v_alpha * v_self + self.v_beta * v_self_linear
        # v_self_ffn_out = self.v_ffn(v_self_prime)
        # v_self_out = self.v_gamma * v_self_prime + self.v_delta * v_self_ffn_out

        # blip_t_self, blip_t_self_loss = self.MoH_text(blip_t, blip_t)
        
        # blip_t_self_linear = self.to_blip_t(blip_t_self)
        # blip_t_self_prime = self.blip_t_alpha * blip_t_self + self.blip_t_beta * blip_t_self_linear
        # blip_t_self_ffn_out = self.blip_t_ffn(blip_t_self_prime)
        # blip_t_self_out = self.blip_t_gamma * blip_t_self_prime + self.blip_t_delta * blip_t_self_ffn_out
        

        # blip_v_self, blip_v_self_loss = self.MoH_image(blip_v, blip_v)
        
        # blip_v_self_linear = self.to_blip_v(blip_v_self)
        # blip_v_self_prime = self.blip_v_alpha * blip_v_self + self.blip_v_beta * blip_v_self_linear
        
        # blip_v_self_ffn_out = self.blip_v_ffn(blip_v_self_prime)
        # blip_v_self_out = self.blip_v_gamma * blip_v_self_prime + self.blip_v_delta * blip_v_self_ffn_out

        blip_self, blip_self_loss = self.MoH_blip(blip_t, blip_v)
        if self.training:
            blip_self = AddAuxiliaryLoss.apply(blip_self, blip_self_loss)
        # ot1= self.ot(t_self, blip_t_self)
        # ot2 = self.ot(v_self, blip_v_self)
        
        # t_moh, t_moh_loss = self.MoH_multi(t, blip_t)
        # v_moh, v_moh_loss = self.MoH_multi(v, blip_v)
        # blip_tv_moh, blip_tv_moh_loss = self.MoH_multi1(blip_t,blip_v)
        # tv_moh, tv_moh_loss = self.MoH_multi2(t,v)
        # tv_moh, tv_moh_loss = self.MoH_multi(blip_t,blip_v)
        # ot3 = self.ot(blip_t, blip_v)
         # ot3 = self.ot(blip_t_self, blip_v_self)ot_local1===============================
        # tv, local_loss_tv = self.ot_local1(tv_moh, blip_tv_moh)
        # # tv, local_loss_tv  = self.ot_local1(blip_tv_moh,tv_moh)
        # ttt, local_loss_ttt = self.ot_local2(t_self, blip_t_self)
        # vvv, local_loss_vvv = self.ot_local3(v_self, blip_v_self)
        # blip_v, local_loss_v = self.ot_local1(blip_v_self_out,tv_moh)
        # ============================================================================================
        # tv_moh,orth_loss_tv, loss_share_tv = self.MGRIA_tv(t_moh, v_moh)
        # blip,orth_loss_blip, loss_share_blip = self.MGRIA_blip(blip_t_self_out, blip_v_self_out)
        # tv,orth_loss_blip, loss_share_blip = self.MGRIA_blip(t_self_out, v_self_out)

        # fused_feature = torch.cat([t_mssa.view(batch_size,-1), v_mssa.view(batch_size,-1),blip_mssa.view(batch_size,-1)],dim=-1)
        # # fused_feature = torch.cat([tv_moh.view(batch_size,-1), blip_t.view(batch_size,-1), blip_v.view(batch_size,-1) ], dim=-1)
        # #print(fused_feature.shape)
        # # ===== 6. 分类输出 + 损失计算（原有逻辑适配）=====
        # out = self.ori_layer(fused_feature) 
        # t_mssa =self.mssa_text(t_self)
        # v_mssa = self.mssa_image(v_self)
        # blip_mssa = self.mssa_blip(blip_self)
        # out, total_U = self.du(t_mssa, v_mssa, blip_mssa)
        # t_mssa =self.mssa_text(t_self)
        # v_mssa = self.mssa_image(v_self)
        # blip_mssa = self.mssa_blip(blip_self)
        # U, total_U = self.du(t_self, v_self, blip_self)
        # if self.training:
            # U = AddAuxiliaryLoss.apply(U, -total_U)
        # print(U.shape)
        # out = self.mssa_text(U)
        # if self.pool_type == "mean":
        #     return feat.mean(dim=1)  # 均值池化（最常用）
        # elif self.pool_type == "max":
        #     return feat.max(dim=1)[0]  # 最大池化
        # elif self.pool_type == "mean_max":
        #     mean_feat = feat.mean(dim=1)
        #     max_feat = feat.max(dim=1)[0]
        #     return torch.cat([mean_feat, max_feat], dim=-1)  # 均值+最大（需调整分类头维度）
        # U_t = U * t_self       # U ⊙ t_self (B×L×D)
        # U_v = U * v_self       # U ⊙ v_self (B×L×D)
        # U_blip = U * blip_self # U ⊙ blip_self (B×L×D)
        
        # 加权求和（权重为可学习参数）
    
        final_feat = self.w1 * t_self + self.w2 * v_self + self.w3 * blip_self
        
        out = self.classifier(final_feat.max(dim=1)[0])
        # print(out.shape)
        # print(out)
        # if self.training:
        #     # out = AddAuxiliaryLoss.apply(out, blip_t_self_loss)
        #     # out = AddAuxiliaryLoss.apply(out, blip_v_self_loss)
        #     out = AddAuxiliaryLoss.apply(out, t_self_loss)
        #     out = AddAuxiliaryLoss.apply(out, v_self_loss)
        #     out = AddAuxiliaryLoss.apply(out, blip_self_loss)
            
            # out = AddAuxiliaryLoss.apply(out, ot1)
            # out = AddAuxiliaryLoss.apply(out, ot2)
            # out = AddAuxiliaryLoss.apply(out, blip_tv_moh_loss)
            # out = AddAuxiliaryLoss.apply(out, tv_moh_loss)
            
            # out = AddAuxiliaryLoss.apply(out, local_loss_tv)
            # out = AddAuxiliaryLoss.apply(out, local_loss_ttt)
            # out = AddAuxiliaryLoss.apply(out, local_loss_vvv)
            #local_loss_t
            #============================================================================================

        # MOH loss
        # moh_loss = t_moh_loss + v_moh_loss + t_self_loss + v_self_loss + blip_t_self_loss + blip_v_self_loss # + multi_loss
        moh_loss =t_self_loss #blip_t_self_loss + blip_v_self_loss
        orth_loss =t_self_loss#blip_t_self_loss #orth_loss_tv + orth_loss_blip# + orth_loss_multi
        share_loss =t_self_loss#blip_t_self_loss# loss_share_tv + loss_share_blip# + loss_share_multi
        return out, orth_loss, share_loss, moh_loss
        # return out, orth_loss_tv + orth_loss_blip + orth_loss_multi, loss_share_tv + loss_share_blip + loss_share_multi

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== 1. 对齐分支（无改动，维度B×L×D） =====================
class AlignmentBranch(nn.Module):
    def __init__(self, d_model=512, context_window=8):
        super().__init__()
        self.d_model = d_model
        self.context_window = context_window

        self.semantic_mask_generator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.ones(2*context_window+1)/(2*context_window+1))
        self.beta = nn.Parameter(torch.ones(2*context_window+1)/(2*context_window+1))

    def get_context_neighbors(self, feat):
        B, L, D = feat.shape
        padded_feat = F.pad(feat, (0, 0, self.context_window, self.context_window), mode="constant", value=0)
        neighbors = [padded_feat[:, i:i+2*self.context_window+1, :].unsqueeze(1) for i in range(L)]
        return torch.cat(neighbors, dim=1)  # (B, L, 2w+1, D)

    def context_similarity(self, q_feat, k_feat):
        base_sim = torch.matmul(q_feat, k_feat.transpose(-1, -2))
        q_neighbors = self.get_context_neighbors(q_feat)
        k_neighbors = self.get_context_neighbors(k_feat)
        q_context = (q_neighbors * self.alpha.view(1,1,-1,1)).sum(dim=2)
        k_context = (k_neighbors * self.beta.view(1,1,-1,1)).sum(dim=2)
        context_sim = torch.matmul(q_context, k_context.transpose(-1, -2))
        return torch.sigmoid(base_sim + context_sim)

    def forward(self, T, V):
        M_Q = self.semantic_mask_generator(T)
        M_K = self.semantic_mask_generator(V)
        Q = T * M_Q
        K = V * M_K
        align_sim = self.context_similarity(Q, K)
        attn_weights = F.softmax(align_sim / torch.sqrt(torch.tensor(self.d_model, device=T.device, dtype=torch.float32)), dim=-1)
        align_feat = torch.matmul(attn_weights, V)
        return align_feat, align_sim

# ===================== 2. 冲突分支（简化为一维语义冲突） =====================
class ConflictBranch(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.d_model = d_model
        # 仅保留语义冲突的可学习权重（删除时间/空间/事件权重）
        self.lambda1 = nn.Parameter(torch.tensor(1.0))  # 语义冲突权重（初始值调整为1.0）
        self.A = nn.Parameter(torch.tensor(1.0))
        self.weight_func = nn.Linear(2*d_model, 1)

    def single_dimension_conflict(self, t_expand, v_expand):
        """
        简化为一维语义冲突计算
        Args:
            t_expand: 扩展后的文本特征 (B, L, L, D)
            v_expand: 扩展后的图像特征 (B, L, L, D)
        Returns:
            conf_score: 语义冲突得分 (B, L, L)
        """
        # 仅保留语义冲突（余弦距离：1 - 余弦相似度）
        semantic_diff = 1 - F.cosine_similarity(t_expand, v_expand, dim=-1)
        # 语义冲突权重加权（lambda1为可学习参数）
        conf_score = self.lambda1 * semantic_diff
        return conf_score

    def forward(self, T, V, align_sim):
        B, L, D = T.shape
        t_expand = T.unsqueeze(2).repeat(1,1,L,1)
        v_expand = V.unsqueeze(1).repeat(1,L,1,1)
        
        # 调用简化后的一维冲突计算函数（替换原multi_dimension_conflict）
        conf_score = self.single_dimension_conflict(t_expand, v_expand)
        weight = self.weight_func(torch.cat([t_expand, v_expand], dim=-1)).squeeze(-1)
        weight = F.softmax(weight, dim=-1)
        conflict_score = (conf_score * weight).sum(dim=[1,2]) / (L*L)
        
        inverse_sim = self.A - align_sim
        conflict_attn = F.softmax(inverse_sim / torch.sqrt(torch.tensor(D, device=T.device, dtype=torch.float32)), dim=-1)
        conflict_feat = torch.matmul(conflict_attn, V)
        return conflict_feat, conflict_score.unsqueeze(1)

# ===================== 3. 博弈融合模块（无改动） =====================
class GameTheoryFusionWithM(nn.Module):
    def __init__(self, d_model=512, pool_type="mean"):
        super().__init__()
        self.d_model = d_model
        self.pool_type = pool_type  # 池化方式：mean/max/mean_max
        
        # 收益函数参数
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.delta = nn.Parameter(torch.tensor(0.5))
        self.eta = nn.Parameter(torch.tensor(1.0))
        self.theta = nn.Parameter(torch.tensor(0.5))
        
        # 门控融合层
        self.gate_fusion = nn.Sequential(
            nn.Linear(4*d_model, d_model),
            nn.Sigmoid()
        )
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def sequence_pooling(self, feat):
        """
        序列特征池化：压缩L维度，得到样本级特征 (B×D)
        Args:
            feat: 序列特征 (B×L×D)
        Returns:
            pooled_feat: 样本级特征 (B×D)
        """
        if self.pool_type == "mean":
            return feat.mean(dim=1)  # 均值池化（最常用）
        elif self.pool_type == "max":
            return feat.max(dim=1)[0]  # 最大池化
        elif self.pool_type == "mean_max":
            mean_feat = feat.mean(dim=1)
            max_feat = feat.max(dim=1)[0]
            return torch.cat([mean_feat, max_feat], dim=-1)  # 均值+最大（需调整分类头维度）
        else:
            raise ValueError(f"不支持的池化方式: {self.pool_type}")

    def mutual_information(self, a, b):
        a_mean = self.sequence_pooling(a)
        b_mean = self.sequence_pooling(b)
        a_centered = a_mean - a_mean.mean(dim=0)
        b_centered = b_mean - b_mean.mean(dim=0)
        cov = torch.matmul(a_centered.T, b_centered) / a_mean.shape[0]
        return torch.trace(cov).abs()

    def divergence(self, a, b):
        a_dist = F.softmax(self.sequence_pooling(a), dim=-1) + 1e-8
        b_dist = F.softmax(self.sequence_pooling(b), dim=-1) + 1e-8
        return F.kl_div(a_dist.log(), b_dist, reduction="batchmean")

    def reward_function(self, align_feat, conflict_feat, conflict_score):
        consistency = F.cosine_similarity(align_feat, align_feat.mean(dim=1, keepdim=True), dim=-1).mean(dim=1)
        U_A = self.alpha * consistency - self.beta * conflict_score.squeeze(-1)
        redundancy = self.divergence(align_feat, conflict_feat)
        U_C = self.gamma * conflict_score.squeeze(-1) - self.delta * redundancy
        mi = self.mutual_information(align_feat, conflict_feat)
        div = self.divergence(align_feat, conflict_feat)
        U_J = self.eta * mi - self.theta * div
        return (U_A + U_C).mean() + U_J

    def forward(self, align_feat, conflict_feat, T, M):
        B, L, D = align_feat.shape
        # 计算总收益
        conflict_score = F.cosine_similarity(align_feat, conflict_feat, dim=-1).mean(dim=1, keepdim=True)
        total_U = self.reward_function(align_feat, conflict_feat, conflict_score)
        # 动态门控融合（序列级）
        gate_input = torch.cat([align_feat, conflict_feat, T, M], dim=-1)
        gate = self.gate_fusion(gate_input)
        fused_feat = gate * align_feat + (1 - gate) * conflict_feat
        # 序列级融合 → 保留序列维度输出
        fused_feat = self.final_fusion(torch.cat([fused_feat, M], dim=-1))  # (B×L×D)
        
        return fused_feat, total_U

# ===================== 4. 整体模型（无改动） =====================
class DualBranchModelForSamplePred(nn.Module):
    """
    输入：T/V/M 均为 (B×L×D)
    输出：fused_feat (B×L×D)、total_U（标量）
    """
    def __init__(self, d_model=512, num_classes=2, pool_type="mean"):
        super().__init__()
        self.alignment_branch = AlignmentBranch(d_model=d_model)
        self.conflict_branch = ConflictBranch(d_model=d_model)
        self.gt_fusion = GameTheoryFusionWithM(d_model=d_model, pool_type=pool_type)
        
        # 分类头输入维度适配（若需样本级预测，可取消注释并启用pooling）
        # in_dim = d_model if pool_type != "mean_max" else 2*d_model
        # self.classifier = nn.Sequential(
        #     nn.Linear(in_dim, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(256, num_classes)
        # )

    def forward(self, T, V, M):
        # 双分支计算（序列级）
        align_feat, align_sim = self.alignment_branch(T, V)
        conflict_feat, conflict_score = self.conflict_branch(T, V, align_sim)
        # 融合（保留序列维度）
        fused_feat, total_U = self.gt_fusion(align_feat, conflict_feat, T, M)
        
        # 若需样本级预测，可添加以下代码：
        # pooled_feat = self.gt_fusion.sequence_pooling(fused_feat)
        # logits = self.classifier(pooled_feat)
        # return logits, total_U, pooled_feat
        
        return fused_feat, total_U
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 1. 维度对齐层 (如果输入的三个特征维度不一致，可以在这里统一)
        # 这里假设输入都是 feature_dim
        
        # 2. 多头自注意力
        # batch_first=True 使得输入形状为 (Batch, Seq_len, Dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # 3. 前馈网络 (FFN) - 增加非线性映射能力
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        # 4. 归一化层
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # 5. 聚合层：将 3 个 Token 压回 1 个特征向量
        # 使用可学习的权重对三个交互后的 Token 进行加权求和
        self.combine = nn.Parameter(torch.ones(3))

    def forward(self, f1, f2, f3):
        """
        f1, f2, f3 形状均需为 (B, 64)
        """
        # Step 1: 构造序列 (Batch, Seq_len=3, Dim=64)
        x = torch.stack([f1, f2, f3], dim=1)
        
        # Step 2: 多头注意力交互
        # attn_output 包含三个经过彼此交互后的 Token
        attn_output, attn_weights = self.mha(x, x, x)
        x = self.norm1(x + attn_output) # 残差连接
        
        # Step 3: 前馈网络提纯
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output) # 残差连接
        
        # Step 4: 加权聚合 (B, 3, D) -> (B, D)
        # 这里的 weights 会根据训练学习出哪种特征组合更可靠
        weights = F.softmax(self.combine, dim=0)
        fused_feature = (x[:, 0, :] * weights[0] + 
                         x[:, 1, :] * weights[1] + 
                         x[:, 2, :] * weights[2])
        
        return fused_feature
import torch
import torch.nn as nn
import torch.nn.functional as F

class MMD_Global_Alignment(nn.Module):
    def __init__(self, kernel_type='gaussian', sigma=1.0):
        super(MMD_Global_Alignment, self).__init__()
        self.kernel_type = kernel_type
        self.sigma = sigma  # 对应论文中的带宽参数 sigma

    def gaussian_kernel(self, x, y):
        """
        实现公式 (9): k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
        """
        # x: [B, N, D], y: [B, M, D]
        # 计算成对欧式距离矩阵
        x_size = x.size(1)
        y_size = y.size(1)
        dim = x.size(2)

        x_expanded = x.unsqueeze(2)  # [B, N, 1, D]
        y_expanded = y.unsqueeze(1)  # [B, 1, M, D]

        # 计算 ||x - y||^2
        dist = torch.pow(x_expanded - y_expanded, 2).sum(dim=3)
        
        # 计算高斯核
        return torch.exp(-dist / (2 * self.sigma**2))

    def forward(self, x, y):
        """
        输入:
        x: 模态A特征 [Batch, T1, Dim]
        y: 模态B特征 [Batch, T2, Dim]
        
        输出:
        mmd_loss: 全局分布差异损失 (公式 8)
        """
        batch_size = x.size(0)
        T1 = x.size(1)
        T2 = y.size(1)

        # 1. 计算三个核矩阵 (Kernel Matrices)
        k_xx = self.gaussian_kernel(x, x)  # [B, T1, T1]
        k_yy = self.gaussian_kernel(y, y)  # [B, T2, T2]
        k_xy = self.gaussian_kernel(x, y)  # [B, T1, T2]

        # 2. 实现公式 (8): 
        # MMD^2 = 1/T1^2 * sum(k_xx) + 1/T2^2 * sum(k_yy) - 2/(T1*T2) * sum(k_xy)
        
        # 为了数值稳定，对每个样本分别求均值再求 Batch 平均
        term_xx = k_xx.sum(dim=(1, 2)) / (T1 * T1)
        term_yy = k_yy.sum(dim=(1, 2)) / (T2 * T2)
        term_xy = k_xy.sum(dim=(1, 2)) / (T1 * T2)

        mmd_squared = term_xx + term_yy - 2 * term_xy
        
        # 确保损失非负（由于浮点误差可能产生极小的负数）
        mmd_loss = torch.relu(mmd_squared).mean()

        return mmd_loss

class OT_Local_Alignment(nn.Module):
    def __init__(self, eps=0.1, max_iter=100):
        super(OT_Local_Alignment, self).__init__()
        self.eps = eps  # 正则化系数
        self.max_iter = max_iter

    def forward(self, x_a, x_b):
        """
        输入:
        x_a: 模态A的特征 [Batch, Seq_len, Dim]
        x_b: 模态B的特征 [Batch, Seq_len, Dim]
        
        输出:
        aligned_x_a: 对齐后的模态A特征
        ot_loss: 最优传输损失
        """
        # 1. 计算代价矩阵 Cost Matrix (通常使用余弦距离)
        # 归一化特征以计算余弦相似度
        x_a_norm = F.normalize(x_a, p=2, dim=-1)
        x_b_norm = F.normalize(x_b, p=2, dim=-1)
        
        # Cost = 1 - Cosine Similarity
        # [Batch, Seq_len, Seq_len]
        C = 1 - torch.bmm(x_a_norm, x_b_norm.transpose(1, 2))
        
        # 2. Sinkhorn 算法求解 Transport Plan (T)
        # 初始化对偶向量 u, v
        batch_size, n, m = C.shape
        u = torch.zeros(batch_size, n, device=x_a.device)
        v = torch.zeros(batch_size, m, device=x_a.device)
        
        # 简单的 Sinkhorn 迭代 (Log-domain 更加数值稳定，这里演示标准版本)
        # K = exp(-C / eps)
        K = torch.exp(-C / self.eps)
        
        # 假设均匀分布 (uniform marginals)
        a = torch.ones(batch_size, n, device=x_a.device) / n
        b = torch.ones(batch_size, m, device=x_a.device) / m
        
        for _ in range(self.max_iter):
            # u = a / (K @ v)
            # v = b / (K.T @ u)
            v = b / (torch.bmm(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2) + 1e-9)
            u = a / (torch.bmm(K, v.unsqueeze(2)).squeeze(2) + 1e-9)
            
        # 计算 Transport Plan T = diag(u) * K * diag(v)
        T = torch.diag_embed(u) @ K @ torch.diag_embed(v)
        
        # 3. 计算 OT Loss (Transport Cost)
        # Loss = sum(T * C)
        ot_loss = torch.sum(T * C, dim=(1, 2)).mean()
        
        # 4. 根据 Plan 重构特征 (Feature Warping/Alignment)
        # aligned_x_a = T * x_b (将 B 的信息根据对应关系传给 A)
        # 注意：这里常将 T 归一化为注意力权重使用
        aligned_x_a = torch.bmm(T, x_b) * n # 缩放因子，保持量级
        
        return aligned_x_a, ot_loss
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
def load_model(dataset=None):
    model = Model(dataset)
    if torch.cuda.is_available():
        model = model.cuda()
        
    return model

if __name__ == '__main__':
    # ============================================================================
    # input_resolution = model.visual.input_resolution
    # context_length = model.context_length
    # vocab_size = model.vocab_size
    # print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    # print("Input resolution:", input_resolution)
    # print("Context length:", context_length)
    # print("Vocab size:", vocab_size)
    model = load_model(dataset="weibo_dataset")
    print(model)