import os
import warnings
import torch
import torch.utils.data as data
from src.data.datasets import weibo_dataset_filter
# from src.data.datasets import twitter_dataset_filter
from src.data.datasets import twitter_dataset_filter
from src.data.datasets import gossipcop_dataset_filter
from src.data.datasets import politifact_dataset_filter
from transformers import BertTokenizer, AutoTokenizer
from torchvision import transforms
#from lavis.models import load_model_and_preprocess
from torchvision.transforms.functional import InterpolationMode
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import clip as open_clip  # OpenAI CLIP
from PIL import Image
import cn_clip.clip as clipcn
# 抑制 fairscale 的弃用警告（可选，只是清理输出）
warnings.filterwarnings("ignore", category=FutureWarning, module="fairscale")
from cn_clip.clip import load_from_name, available_models
# 保持你之前的 PRETRAINED_MAP 配置
CLIP_CONFIG = {
    "weibo_dataset": {"model": "ViT-B-16", "is_chinese": True},
    "twitter_dataset": {"model": "ViT-B/32", "is_chinese": False},
    "gossipcop_dataset": {"model": "ViT-B/32", "is_chinese": False},
    "politifact_dataset": {"model": "ViT-B/32", "is_chinese": False}
}

PRETRAINED_MAP = {
    "weibo_dataset": {
        "text": "./chinese-roberta-wwm-ext",
        "clip_model": "ViT-B-16",
        "image_dim": 2048,
        "is_chinese": True
    },
    "twitter_dataset": {
        "text": "./roberta-base",
        "clip_model": "ViT-B/32",
        "image_dim": 2048,
        "is_chinese": False
    },
    "gossipcop_dataset": {
        "text": "./roberta-base",
        "clip_model": "ViT-B/32",
        "image_dim": 2048,
        "is_chinese": False
    },
    "politifact_dataset": {
        "text": "./roberta-base",
        "clip_model": "ViT-B/32",
        "image_dim": 2048,
        "is_chinese": False
    },
}

vit_img_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073), 
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])

class PreDataset(data.Dataset):
    def __init__(self, dataset, data_type, dtype=torch.float32):
        self.dataset = dataset
        self.data_type = data_type
        self.dtype = dtype  # 统一数据类型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}, Data type: {self.dtype}")
        
        # 1. 基础图像变换 (用于 ResNet)
        self.resnet_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 训练阶段的ResNet变换（增加随机增强）
        self.resnet_transform_train = transforms.Compose([
            transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),  # 先放大到256x256
            transforms.RandomCrop((224, 224)),  # 随机裁剪（替代固定中心裁剪）
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转（50%概率）
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 随机亮度/对比度/饱和度调整
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet归一化
        ])
        
        # 测试/推理阶段的变换（保持稳定，移除随机操作）
        self.resnet_transform_test = transforms.Compose([
            transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop((224, 224)),  # 固定中心裁剪
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.save_path = f'/root/autodl-tmp/data/{dataset}/{data_type}_clip_checkpoint.pt'
        
        if not os.path.exists(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path))

        if not os.path.exists(self.save_path):
            # 加载原始数据逻辑
            if self.dataset == 'weibo_dataset':
                self.images_list, self.texts_list, self.text_image_ids, self.labels = weibo_dataset_filter(dataset, data_type)
                self.is_chinese = True
                self.clip_model_path = "ViT-B-16"
            elif self.dataset == 'twitter_dataset':
                self.images_list, self.texts_list, self.labels = twitter_dataset_filter(dataset, data_type)
                self.is_chinese = False
                self.clip_model_path = "ViT-B/32"
            elif self.dataset == 'gossipcop_dataset':
                self.images_list, self.texts_list, self.labels = gossipcop_dataset_filter(dataset, data_type)
                self.is_schinese = False
                self.clip_model_path = "ViT-B/32"
            elif self.dataset == 'politifact_dataset':
                self.images_list, self.texts_list, self.labels = politifact_dataset_filter(dataset, data_type)
                self.is_chinese = False
                self.clip_model_path = "ViT-B/32"
            else:
                raise ValueError(f"Unsupported dataset: {dataset}")

            # 提取 ResNet 图像张量并统一数据类型
            if self.data_type == "train":
                self.resnet_features_image = torch.stack([self.resnet_transform_train(img).to(self.dtype) for img in self.images_list], dim=0)
            else:
                self.resnet_features_image = torch.stack([self.resnet_transform_test(img).to(self.dtype) for img in self.images_list], dim=0)

            # ================= CLIP 特征提取模块 =================
            clip_info = CLIP_CONFIG.get(dataset)
            self.clip_features_image = []
            self.clip_features_text = []

            if clip_info["is_chinese"]:
                # 加载中文 CLIP (Transformers 库)
                print(clip_info["model"])
                clip_model, preprocess = load_from_name(clip_info["model"], download_root='./', use_modelscope=True)
                #clip_model = ChineseCLIPModel.from_pretrained(clip_info["model"])#.to(self.device)
                # 手动设置模型数据类型
                #clip_model = clip_model#.to(self.dtype)
                #clip_processor = ChineseCLIPProcessor.from_pretrained(clip_info["model"])
                
                for idx, (img, txt) in enumerate(zip(self.images_list, self.texts_list)):
                    # 图像处理
                    img_input = preprocess(img).unsqueeze(0).to(self.device)
                    #img_input = {k: v.to(self.dtype) for k, v in img_input.items()}
                    #img_input = {k: v  for k,   v in img_input.items()}
                    # 文本处理
                   # print(txt)
                    txt_input = clipcn.tokenize([txt]).to(self.device)
                    #txt_input = {k: v.to(self.dtype) if v.dtype != torch.long else v for k, v in txt_input.items()}
                    #print(txt_input)
                    with torch.no_grad():
                        img_feat = clip_model.encode_image(img_input)#.to(self.dtype)
                        txt_feat = clip_model.encode_text(txt_input)#.to(self.dtype)
                        self.clip_features_image.append(img_feat)
                        self.clip_features_text.append(txt_feat)
            else:
                # 加载英文 CLIP (OpenAI 库) - 修复：移除dtype参数，手动设置类型
                clip_model, clip_preprocess = open_clip.load(clip_info["model"], device=self.device)
                # 手动将模型转换为指定数据类型
                clip_model = clip_model#.to(self.dtype)
                
                for idx, (img, txt) in enumerate(zip(self.images_list, self.texts_list)):
                    img_input = clip_preprocess(img).unsqueeze(0).to(self.device)#.to(self.dtype)
                    txt_input = open_clip.tokenize([txt], truncate=True).to(self.device)  # token是long类型，不需要转换
                    
                    with torch.no_grad():
                        img_feat = clip_model.encode_image(img_input).to(self.dtype)
                        txt_feat = clip_model.encode_text(txt_input).to(self.dtype)
                        self.clip_features_image.append(img_feat)
                        self.clip_features_text.append(txt_feat)

            self.clip_features_image = torch.cat(self.clip_features_image, dim=0).to(self.dtype)
            self.clip_features_text = torch.cat(self.clip_features_text, dim=0)#.to(self.dtype)

            # ================= BERT Tokenize =================
            bert_model = PRETRAINED_MAP.get(self.dataset, {}).get("text", "bert-base-uncased")
            print(f"Using BERT model: {bert_model}")
            tokenizer = AutoTokenizer.from_pretrained(bert_model)
            encodes = tokenizer(self.texts_list, padding='max_length', truncation=True,max_length=130, return_tensors="pt")

            # ========== 关键修复：初始化实例属性 ==========
            # 无论是否缓存，都初始化这些属性
            self.text_ids = encodes["input_ids"]
            self.attention_mask = encodes["attention_mask"]#.to(self.dtype)
            self.token_type_ids = encodes.get("token_type_ids", torch.zeros_like(encodes["input_ids"]))#.to(self.dtype)
            self.labels = torch.tensor(self.labels, dtype=torch.long)
            
            # 统一所有张量的数据类型
            checkpoint = {
                'text_ids': self.text_ids,
                'attention_mask': self.attention_mask,
                'token_type_ids': self.token_type_ids,
                'resnet_features_image': self.resnet_features_image,
                'labels': self.labels,
                'clip_features_image': self.clip_features_image,
                'clip_features_text': self.clip_features_text
            }
            torch.save(checkpoint, self.save_path)
        else:
            print(f"===> Loading cached CLIP features from {self.save_path}")
            checkpoint = torch.load(self.save_path, map_location='cpu')
            
            # 加载时统一转换为指定数据类型
            self.text_ids = checkpoint['text_ids']
            self.attention_mask = checkpoint['attention_mask'] 
            self.token_type_ids = checkpoint['token_type_ids']
            self.resnet_features_image = checkpoint['resnet_features_image'].to(self.dtype)
            self.labels = checkpoint['labels']  # 标签保持long类型
            self.clip_features_image = checkpoint['clip_features_image'].to(self.dtype)
            self.clip_features_text = checkpoint['clip_features_text'].to(self.dtype)

        self.length = len(self.labels)

    def __getitem__(self, item):
        # 延迟到获取数据时再移到GPU，避免提前占用显存和类型问题
        text_data = (
            self.text_ids[item].to(self.device),
            self.attention_mask[item].to(self.device),
            self.token_type_ids[item].to(self.device)
        )
        
        image_feat = self.resnet_features_image[item].to(self.device)
        clip_text_feat = self.clip_features_text[item].to(self.device)
        clip_image_feat = self.clip_features_image[item].to(self.device)
        label = self.labels[item].to(self.device)
        
        return text_data, image_feat, clip_text_feat, clip_image_feat, label

    def __len__(self):
        return self.length

def load_data(dataset, batch_size, dtype=torch.float32):
    """
    加载数据集，支持指定数据类型
    
    Args:
        dataset: 数据集名称
        batch_size: 批次大小
        dtype: 数据类型 (默认torch.float32，若使用FP16则传torch.float16)
    """
    train_data = PreDataset(dataset, 'train', dtype=dtype)
    test_data = PreDataset(dataset, 'test', dtype=dtype)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        #pin_memory=True  # 加速GPU数据传输
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        #pin_memory=True
    )

    return train_loader, test_loader

if __name__ == "__main__":
    # 如果你要使用FP16，将dtype改为torch.float16
    train_data = PreDataset("weibo_dataset", 'train', dtype=torch.float32)
    print(f"Dataset length: {len(train_data)}")
    # 测试一个样本
    sample = train_data[0]
    print(f"Text data shape: {sample[0][0].shape}")
    print(f"ResNet feature dtype: {sample[1].dtype}")
    print(f"CLIP text feature dtype: {sample[2].dtype}")
    print(f"CLIP image feature dtype: {sample[3].dtype}")
    print(f"Label dtype: {sample[4].dtype}")