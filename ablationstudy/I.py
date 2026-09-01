import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import tqdm
import torchvision.models as models
from transformers import BertModel, BertTokenizer, SwinModel
from src.models.attention import MoHAttention
import clip
from src.models.imoe import MoEGlobal
# [已移除] from DDPMmodel import DDPM  # DDPMmodel.py 缺失,当前代码仅注释引用 DDPM
from src.losses.losses import js_divergence, kl_divergence, SupervisedContrastiveLoss, info_nce_loss, TripletLoss
from src.models.aca import improved_cross_attention_with_moe
from src.models.imoe import IMOE

class TextModel(nn.Module):
    def __init__(self, dataset,pretraining=False):
        super().__init__()
        # from safetensors.torch import load_file
 
        # # 加载模型时指定 safetensors 路径
        # model_path = "./bert-base-chinese"
        # state_dict = load_file(f"{model_path}/pytorch_model.bin")
        # self.textmodel = BertModel.from_pretrained(model_path, state_dict=state_dict)
        self.dataset = dataset
        if self.dataset == "weibo_dataset":
            model_name = './bert-base-chinese'
        elif self.dataset == "twitter_dataset":
            model_name = './bert-base-uncased'
        self.textmodel = BertModel.from_pretrained(model_name)#, local_files_only=True
        #self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # self.textmodel.eval()
        # for param in self.textmodel.parameters():
        #     param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids):
        # text : [batch_size, token_num, dim]
        #self.text_ids = torch.tensor(encodes["input_ids"])
        #self.attention_mask = torch.tensor(encodes["attention_mask"])
        #self.token_type_ids = torch.tensor(encodes["token_type_ids"])
        # input_ids = text[0]
        # attention_mask = text[1]
        # token_type_ids = text[2]
        last_hidden_state = self.textmodel(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        cls_emb = last_hidden_state[:, 0, :]  # cls_emb: [batch_size, dim]
        #token_emb = last_hidden_state[:, 1:-1, :]  # tokens_emb: [batch_size, MAX_LENGTH, dim]
        return cls_emb#, token_emb

class ImageModel(nn.Module):
    def __init__(self, dataset, pretraining):
        super().__init__()
        # self.imagemodel = models.vgg16(pretrained=True)
        # new_classifier = self.imagemodel.classifier[:6]
        # self.imagemodel.classifier = new_classifier
        
        # 加载预训练ResNet18
        self.imagemodel = models.resnet50(pretrained=True)
        # 移除最后一层全连接层（fc层）
        # 原始结构：avgpool → flatten → fc(512→1000)
        self.imagemodel.fc = nn.Identity()  # 用恒等映射替换fc层


        # self.imagemodel = SwinModel.from_pretrained("./swin-base-patch4-window7-224")
        
        # 冻结所有参数（可选）
        # self.imagemodel.eval()
        for param in self.imagemodel.parameters():
            param.requires_grad = False
    def forward(self, images):
        # images : [batch_size, 3, 224, 224]
        out = self.imagemodel(images)#.detach()
        # out = self.imagemodel(images).pooler_output
        return out

class CLIPFeatureExtractor(nn.Module):
    def __init__(self, model_name = "./vit_b/ViT-B-32.pt"):
        super().__init__()
        self.model, _ = clip.load(model_name) # clip.load(model_name, device="cpu")
        # self.token = clip.tokenize()
        # 
    def forward(self, image,text):
        image_features = self.model.encode_image(image)
        # text : [batch_size, 77]
        # text = clip.tokenize(text)
        text_features = self.model.encode_text(text)#.detach()
        
        # 512 特征维度
        
        return image_features,text_features

def load_pre_models(dataset, pretraining, modality):
    if modality == 'text':
        model = TextModel(dataset, pretraining)
    elif modality == 'image':
        model = ImageModel(dataset, pretraining)
    else:
        raise ValueError('ERROR! modality must be text or image.')
    if torch.cuda.is_available():
        model = model.cuda()
    return model

# class Reasoning(nn.Module):
#     def __init__(self, model_dim = 128):
#         super(Reasoning, self).__init__()
#         self.imoe = IMOE(
#                     ds_inputsize = model_dim*2, # 
#                     input_size = 1,
#                     output_size = 1,
#                     num_experts = 8,
#                     hidden_size = model_dim*2,
#                     noisy_gating = True,
#                     k = 4,
#                     trainingmode = True
#                     )
#         self.f_layers = nn.Sequential(
#             nn.Linear(model_dim * 4, model_dim),
#             nn.ReLU(),
#         )
#     def forward(self, t, i, m):
#         n = torch.cat([t, i], dim=1)
        
#         f1 = torch.abs(t - i)
#         f2 = torch.mul(t, i)
#         #print("n shape", n.shape)
#         #print("m shape", m.shape)
#         f3, loss_reason = self.imoe(n,m)
#         f4 = torch.cat([m, n], dim=1)
#         # f4 = self.f_layers(f4)
#         return torch.cat([f1, f2, f3, f4], dim=1), loss_reason

class Reasoning(nn.Module):
    def __init__(self, model_dim = 128):
        super(Reasoning, self).__init__()
        self.model_dim = model_dim
        self.imoe = IMOE(
                    ds_inputsize = self.model_dim, # 
                    input_size = 1,
                    output_size = 1,
                    num_experts = 8,
                    hidden_size = self.model_dim,
                    noisy_gating = True,
                    k = 4,
                    trainingmode = True
                    )
        
        self.f_layers = nn.Sequential(
            nn.Linear(self.model_dim * 8, self.model_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.MoH1 = MoHAttention(dim=self.model_dim, num_heads=8, shared_head=2, routed_head=1, qk_norm=True)
        self.MoH2 = MoHAttention(dim=self.model_dim, num_heads=8, shared_head=2, routed_head=2, qk_norm=True)
    def forward(self, t, i, m):
        n = torch.cat([t, i], dim=1)
        
        t0 = t.unsqueeze(dim=1)
        i0 = i.unsqueeze(dim=1)
        m0 = m.unsqueeze(dim=1)
        f1, loss1 = self.MoH1(t0, i0)
        f2, loss2 = self.MoH1(m0, m0)
        #print("n shape", n.shape)
        #print("m shape", m.shape)
        f10 = f1.squeeze(dim=1)
        f20 = f2.squeeze(dim=1)

        f3, loss_reason1 = self.imoe(t,i)
        f4, loss_reason2 = self.imoe(m,m)
        f5 = torch.cat([m, n], dim=1)
        # f4 = self.f_layers(f4)
        # print("f4", f4.shape)
        # print("f5", f5.shape)
        return torch.cat([f10, f20, f3, f4, f5], dim=1), loss_reason1

class Model(nn.Module):
    def __init__(self, dataset, input_dim=1024, hidden_dim=512, output_dim=386):
        super().__init__()
        self.imagemodel = load_pre_models(dataset, 'imagenet', 'image')
        self.textmodel = load_pre_models(dataset, 'chinese', 'text')
        # 冻结预训练模型参数
        for param in self.imagemodel.parameters():
            param.requires_grad = False
        for param in self.textmodel.parameters():
            param.requires_grad = False

        self.class_num = 2 # class_num
        self.model_dim = 64
        
        # clip for semantic emb
        self.clip_text_layers = nn.Sequential(
            nn.Linear(512, self.model_dim), 
            nn.BatchNorm1d(self.model_dim), 
            #nn.ReLU()
            nn.LeakyReLU(0.1)
        )
        self.clip_image_layers = nn.Sequential(
            nn.Linear(512, self.model_dim), 
            nn.BatchNorm1d(self.model_dim), 
            #nn.ReLU()
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
        )
        # Linear for semantic emb
        self.text_layers = nn.Sequential(
            nn.Linear(768, self.model_dim), 
            nn.BatchNorm1d(self.model_dim), 
            #nn.ReLU()
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
        )
        self.blip_text_layers = nn.Sequential(
            nn.Linear(768, self.model_dim), 
            nn.BatchNorm1d(self.model_dim), 
            #nn.ReLU()
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
        )
        self.image_layers = nn.Sequential(
            nn.Linear(2048, self.model_dim),
            nn.BatchNorm1d(self.model_dim),
            #nn.ReLU()
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
        )
        self.blip_image_layers = nn.Sequential(
            nn.Linear(768, self.model_dim), 
            nn.BatchNorm1d(self.model_dim), 
            #nn.ReLU()
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
        )
        self.blip_layer = nn.Sequential(
            nn.Linear(self.model_dim*2, self.model_dim), 
            nn.BatchNorm1d(self.model_dim), 
            #nn.ReLU()
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
        )
        self.ori_layer = nn.Sequential(
            nn.Linear(self.model_dim*2, self.model_dim), 
            nn.BatchNorm1d(self.model_dim), 
            #nn.ReLU()
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
        )

        # 文本 bert 和 blip 特征融合
        self.MoH_text = MoHAttention(dim=self.model_dim, num_heads=8,shared_head=3, routed_head=1, qk_norm=True)
        # 图像 resnet 和 blip 特征融合
        self.MoH_image = MoHAttention(dim=self.model_dim, num_heads=8, shared_head=3, routed_head=1, qk_norm=True)
        # # 文本 bert 和 resnet 特征融合
        # self.MoH_text_image = MoHAttention(dim=128, num_heads=8, shared_head=0, routed_head=3, qk_norm=True)
        # # 文本 bert 和 图像blip 特征融合
        # self.MoH_text_image_blip = MoHAttention(dim=128, num_heads=8, shared_head=0, routed_head=3, qk_norm=True)
        # # 文本 blip 和 resnet 特征融合
        # self.MoH_blip_text_image = MoHAttention(dim=128, num_heads=8, shared_head=0, routed_head=3, qk_norm=True)
        # # 文本 blip 和 图像 blip 特征融合
        # self.MoH_blip_text_image_blip = MoHAttention(dim=128, num_heads=8, shared_head=0, routed_head=3, qk_norm=True)
        # 文本-图像
        self.MoH_text_image = MoHAttention(dim=self.model_dim, num_heads=8, shared_head=3, routed_head=1, qk_norm=True)
        # 输入图像和文本
        self.MoH_image_text = MoHAttention(dim=self.model_dim, num_heads=8, shared_head=3, routed_head=1, qk_norm=True)
        
        # self.iAFF = iAFF_1D(channels=1)

        # self.ddpm = DDPM(input_size = self.model_dim, num_units=self.model_dim, num_steps=100, nhead=4)
        self.imoe_layer = improved_cross_attention_with_moe(
        model_dim=self.model_dim,
        num_heads=1,
        ffn_dim=self.model_dim//4,
        dropout=0.5
    )
        self.imoe_multi_layer = improved_cross_attention_with_moe(
        model_dim=self.model_dim,
        num_heads=1,
        ffn_dim=self.model_dim//4,
        dropout=0.5
    )
        self.reason = Reasoning(
            self.model_dim
        )
        self.out = nn.Linear(self.model_dim*7, self.class_num)
        self.bn_layer_out = nn.BatchNorm1d(self.class_num)
        self.dropout = nn.Dropout(0.5)
        self.tripletloss = TripletLoss()

        self.i1 = nn.Sequential(
            nn.Linear(self.model_dim*2, self.model_dim), 
            nn.BatchNorm1d(self.model_dim), 
            #nn.ReLU()
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
        )
        self.i2 = nn.Sequential(
            nn.Linear(self.model_dim*2, self.model_dim), 
            nn.BatchNorm1d(self.model_dim), 
            #nn.ReLU()
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
        )
        self.i3 = nn.Sequential(
            nn.Linear(self.model_dim*2, self.model_dim), 
            nn.BatchNorm1d(self.model_dim), 
            #nn.ReLU()
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
        )
    def forward(self, images, input_ids, attention_mask,token_type_ids, pre_images, pre_text, labels, feature_mode="clip"):
        # ============================================================================================
        blip_v = pre_images # 768
        blip_t = pre_text # 768ss
        blip_v = self.blip_image_layers(blip_v) 
        # blip_v shape torch.Size([256, 768])
        blip_t = self.blip_text_layers(blip_t) 
        # blip_t shape torch.Size([256, 768])
        # resnet
       # print("image shape", images.shape)
        v = self.imagemodel(images)
       # print("image shape", v.shape)
        v = self.image_layers(v) # v shape torch.Size([256, 128])
        #print("v shape", v.shape)
        # batch_size_v = images.shape[0]
        t = self.textmodel(input_ids, attention_mask ,token_type_ids)
        t = self.text_layers(t) # t shape torch.Size([256, 128])
        # ============================================================================================
        # MoHAttention feature processing
        blip_v_f = blip_v.unsqueeze(dim=1)
        blip_t_f = blip_t.unsqueeze(dim=1)

        t_f = t.unsqueeze(dim=1) # t_f shape: torch.Size([256, 1, 128])
        v_f = v.unsqueeze(dim=1) 
        # ============================================================================================
        ### 文本特征处理 MoHAttention
        # 文本 bert 和 blip 特征融合
        #text_text_blip, mohloss_text_text_blip = self.MoH_text(t_f, blip_t_f)
        text_text_blip, mohloss_text_text_blip = self.MoH_text(t_f, t_f)
        t_f0 = t_f + text_text_blip # t_f shape: torch.Size([256, 1, 128])
        t_f0 = t_f0.squeeze(dim=1)
        #blip_text_text, mohloss_blip_text_text = self.MoH_text(blip_t_f, t_f)
        blip_text_text, mohloss_blip_text_text = self.MoH_text(blip_t_f, blip_t_f)
        blip_t_f0 = blip_t_f + blip_text_text
        blip_t_f0 = blip_t_f0.squeeze(dim=1)
        ### 图像特征处理 MoHAttention
        ## 图像 resnet 和 blip 特征融合
        # image_image_blip, mohloss_image_image_blip = self.MoH_image(v_f, blip_v_f)
        image_image_blip, mohloss_image_image_blip = self.MoH_image(v_f, v_f)
        v_f0 = v_f + image_image_blip
        v_f0 = v_f0.squeeze(dim=1)

        #blip_image_image, mohloss_blip_image_image = self.MoH_image(blip_v_f, v_f)
        blip_image_image, mohloss_blip_image_image = self.MoH_image(blip_v_f, blip_v_f)
        # print("blip_v_f",blip_v_f.shape)
        # print("blip_image_image", blip_image_image.shape)
        blip_v_f0 = blip_v_f + blip_image_image
        # print("blip_v_f0", blip_v_f0.shape)
        blip_v_f0 = blip_v_f0.squeeze(dim=1)
        
        ### 基于预训练模型一侧的多模态交互
        ## blip交互
        blip_text_image, mohloss_blip_text_image = \
                                self.MoH_text_image(blip_t_f, blip_v_f)
        
        blip_t_i_m0 = torch.mul(blip_t_f.squeeze(dim=1), blip_t_f0) + blip_text_image.squeeze(dim=1)
        #blip_t_i_m0 = blip_t_i_m0.squeeze(dim=1)
        
        blip_image_text, mohloss_blip_image_text = \
                                self.MoH_text_image(blip_v_f, blip_t_f)
        # blip_image_text, mohloss_blip_image_text = \
        #                        self.MoH_image_text(blip_v_f, blip_t_f)
        # print("blip_v_f", blip_v_f.shape)
        # print("blip_v_f0", blip_v_f0.shape)
        #print("torch.mul(blip_v_f, blip_v_f0)", torch.mul(blip_v_f.squeeze(dim=1), blip_v_f0).shape)
        #print("blip_image_text", blip_image_text.shape)
        blip_i_t_m0 = torch.mul(blip_v_f.squeeze(dim=1), blip_v_f0) + blip_image_text.squeeze(dim=1)
        #blip_i_t_m0 = blip_i_t_m0.squeeze(dim=1)

        #print("blip_t_i_m0", blip_t_i_m0.shape)
        #print("blip_i_t_m0", blip_i_t_m0.shape)

        blip_t_i = self.blip_layer(torch.cat([blip_t_i_m0, blip_i_t_m0], dim=1))

        ## pretrained 交互
        # print("t_f shape", t_f.shape)
        text_image, mohloss_text_image = self.MoH_image_text(t_f, v_f)
        # print("text_image shape", text_image.shape)MoH_image_text

        t_i_m0 = torch.mul(t_f.squeeze(dim=1), t_f0) + text_image.squeeze(dim=1)
       # t_i_m0 = t_i_m0.squeeze(dim=1)
        # image_text, mohloss_image_text = self.MoH_image_text(v_f, t_f)
        image_text, mohloss_image_text = self.MoH_image_text(v_f, t_f)
        i_t_m0 = torch.mul(v_f.squeeze(dim=1), v_f0) + image_text.squeeze(dim=1)
     #   i_t_m0 = i_t_m0.squeeze(dim=1)

        t_i = self.ori_layer(torch.cat([t_i_m0, i_t_m0], dim=1))
        #print("t_f0",t_f0.shape)
        #print("blip_t_f0",blip_t_f0.shape)
        
        t_loss = self.tripletloss(t_f0, blip_t_f0, blip_v_f0)
        v_loss = self.tripletloss(v_f0, blip_v_f0, blip_t_f0)
        t_i_loss = self.tripletloss(t_f0, t_i, v_f0)
        blip_t_i_loss = self.tripletloss(blip_t_f0, blip_t_i, blip_v_f0)
        # trip_loss = t_loss + v_loss + t_i_loss + blip_t_i_loss
        #moe_text, loss_aca_text = self.imoe_layer(t_f0, blip_t_f0)
        #moe_text = moe_text# + t_f0 + blip_t_f0
        a = torch.cat([t_f0, blip_t_f0], dim=1)
        moe_text = self.i1(a)
        # moe_image, loss_aca_image = self.imoe_layer(v_f0, blip_v_f0)
        # moe_image = moe_image# + v_f0 + blip_v_f0
        b = torch.cat([v_f0, blip_v_f0], dim=1)
        moe_image = self.i2(b)
        #moe_multi, loss_aca_multi = self.imoe_multi_layer(blip_t_i, t_i)
        #moe_multi = moe_multi# + blip_t_i + t_i
        c = torch.cat([blip_t_i, t_i], dim=1)
        moe_multi = self.i3(c)
        f, loss_reason = self.reason(moe_text, moe_image, moe_multi)
        out = self.out(f)
        out = self.bn_layer_out(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = F.softmax(out, dim=1)
        # mohloss_text_text_blip (MoH_text)
        # mohloss_blip_text_text (MoH_text)
        # mohloss_image_image_blip (MoH_image)
        # mohloss_blip_image_image (MoH_image)
        # mohloss_blip_text_image (MoH_text_image)
        # mohloss_text_image (MoH_text_image)
        # mohloss_blip_image_text (MoH_image_text)
        # mohloss_image_text (MoH_image_text)
        
        # loss_aca_text (imoe_layer)
        # loss_aca_image (imoe_layer)
        # loss_aca_multi (imoe_multi_layer)
        loss_aca_multi = 0
        loss_aca_text = 0
        loss_aca_image = 0
        return out, (mohloss_text_text_blip, mohloss_blip_text_text), \
                (mohloss_image_image_blip, mohloss_blip_image_image), \
                (mohloss_blip_text_image, mohloss_text_image), \
                (mohloss_blip_image_text, mohloss_image_text), \
                 (loss_aca_text, loss_aca_image),  loss_aca_multi, (t_loss, v_loss, t_i_loss, blip_t_i_loss)
# (t_loss, v_loss, t_i_loss, blip_t_i_loss)
# trip_loss = t_loss + v_loss + t_i_loss + blip_t_i_loss
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
    model = load_model()
    print(model)