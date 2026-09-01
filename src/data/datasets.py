import os
import glob
import cv2 as cv
from PIL import Image
import numpy as np
import pickle
from src.utils import text_filter_chinese, text_filter_english, image_transform, image_read
from tqdm import tqdm
from collections import Counter

def get_weibo_data(data_dir, data_type):
    """
    inputs : file_list(文本txt)
    
    intermediate variable:
        文本 id : text_id
        图像 id : image_id
        文本内容 : post_text
        行数计数 : count
        标签 label : label
        数量统计 : text_index_range

    """
    if data_type not in ['train', 'test']:
        raise ValueError('ERROR! data type must be train or test.')
    rumor_txt = '{}/tweets/{}_rumor.txt'.format(data_dir, data_type)
    nonrumor_txt = '{}/tweets/{}_nonrumor.txt'.format(data_dir, data_type)
    rumor_images = os.listdir('{}/rumor_images/'.format(data_dir))
    nonrumor_images = os.listdir('{}/nonrumor_images/'.format(data_dir))

    #tweet_ids = []
    images_list = []
    texts_list = []
    text_image_ids = []
    labels = []

    with open(rumor_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total_lines = len(lines)
        total_groups = total_lines // 3
        print("Processing {} Groups...".format(total_groups))
        # 进度条显示处理的组数（每组3行）
        for group_idx in tqdm(range(total_groups), desc="Processing rumor Groups", total=total_groups):
            # 计算当前组的起始和结束索引（避免越界）
            start = group_idx * 3
            end = min(start + 3, total_lines)
            current_group = lines[start:end]
            
            tweet_id = current_group[0].split('|')[0]  # 提取tweet_id
            image_ids_group = current_group[1].split('|')  # 提取image_id列表
            rumor_content = current_group[2].strip()  # 提取文本内容
            rumor_content = text_filter_chinese(rumor_content)  # 过滤无效文本
            if not rumor_content:
                continue  # 跳过无有效文本的组

            for image in image_ids_group:
                img_name = image.split('/')[-1]
                if img_name in rumor_images:
                    #print(img_name)
                    images_list.append(image_read('{}/rumor_images/{}'.format(data_dir, img_name)))
                    # images_list.append(image_transform('{}/rumor_images/{}'.format(data_dir, img_name)))
                    texts_list.append(rumor_content)
                    text_image_ids.append('{}|{}'.format(tweet_id, img_name.split('.')[0]))
                    labels.append(0) # [0, 1]
                    break

           # tweet_ids.append(tweet_id)

    with open(nonrumor_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total_lines = len(lines)
        total_groups = total_lines // 3
        # 进度条显示处理的组数（每组3行）
        for group_idx in tqdm(range(total_groups), desc="Processing non rumor Groups", total=total_groups):
            # 计算当前组的起始和结束索引（避免越界）
            start = group_idx * 3
            end = min(start + 3, total_lines)
            current_group = lines[start:end]
            
            tweet_id = current_group[0].split('|')[0]
            image_ids_group = current_group[1].split('|')
            norumor_content = current_group[2].strip()
            norumor_content = text_filter_chinese(norumor_content)
            if norumor_content:
                for image in image_ids_group:
                    img_name = image.split('/')[-1]
                    if img_name in nonrumor_images:
                        images_list.append(image_read('{}/nonrumor_images/{}'.format(data_dir, img_name)))
                        # images_list.append(image_transform('{}/nonrumor_images/{}'.format(data_dir, img_name)))
                        texts_list.append(norumor_content)
                        text_image_ids.append('{}|{}'.format(tweet_id, img_name.split('.')[0]))
                        labels.append(1) # [0, 1]
                        break
    label_counts = Counter(labels)
    print(f"\n{data_type}数据集标签分布：")
    print(f"谣言（label=0）数量：{label_counts.get(0)}")
    print(f"非谣言（label=1）数量：{label_counts.get(1)}")
    print(f"总样本数：{len(labels)}")
    return  images_list, texts_list, text_image_ids, labels


def weibo_dataset_filter(dataset_name, data_type):

    if dataset_name == 'weibo_dataset':
        images_list, texts_list, text_image_ids, labels = get_weibo_data('./data/weibo_dataset', data_type)

    # elif dataset_name == 'twitter':
    #     text_lists, image_lists, labels, text_image_ids = get_twitter_matrix(data_type)
    # elif dataset_name == 'twitter':
    #     text_lists, image_lists, labels, text_image_ids = get_twitter_matrix(data_type)

    else:
        raise ValueError('ERROR! Dataset must be weibo or twitter!')

    return images_list, texts_list, text_image_ids, labels

if __name__ == '__main__':
    data_dir = './data/weibo_dataset/'
    images_list, texts_list, text_image_ids, labels = get_weibo_data(data_dir, 'train')
    print("for the train dataset:")
    print("the length of tweet_id_list:{%d}" % (len(images_list)))
    print("the length of texts_list:{%d}" % (len(texts_list)))
    print("the length of text_image_ids:{%d}"%(len(text_image_ids)))
    train_save_path = os.path.join(data_dir, 'processed', 'train_data.pkl')
    os.makedirs(os.path.dirname(train_save_path), exist_ok=True)  # 自动创建保存目录
    train_data ={"images_list": images_list, 
                 "texts_list": texts_list,
                 "text_image_ids":text_image_ids,
                 "labels":labels
                }
    with open(train_save_path, 'wb') as f:
        pickle.dump(train_data, f)
    
    print("Processing test data...")
    images_list, texts_list, text_image_ids, labels = get_weibo_data(data_dir, 'test')
    print("for the train dataset:")
    print("the length of images_list:{%d}" % (len(images_list)))
    print("the length of texts_list:{%d}" % (len(texts_list)))
    print("the length of text_image_ids:{%d}"%(len(text_image_ids)))
    print("the length of labels:{%d}"%(len(labels)))
    test_save_path = os.path.join(data_dir, 'processed', 'test_data.pkl')
    os.makedirs(os.path.dirname(test_save_path), exist_ok=True)  # 自动创建保存目录
    test_data ={"images_list": images_list, 
                 "texts_list": texts_list,
                 "text_image_ids":text_image_ids,
                 "labels":labels
                 }

    with open(test_save_path, 'wb') as f:
        pickle.dump(test_data, f)
    print("Done.")



import os
import glob
import cv2 as cv
from PIL import Image
import numpy as np
import re
import pickle
import jieba
from src.utils import text_filter_chinese, text_filter_english, image_transform, image_read
from tqdm import tqdm
from collections import Counter


import os
import editdistance
from tqdm import tqdm
import string
import nltk

# 下载必要的nltk资源（首次运行执行一次即可）
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 初始化英文停用词表（用于文本归一化，提升去重/相似判断准确性）
stop_words = set(stopwords.words('english'))

# ---------------------- 辅助函数：文本预处理+去重+相似过滤 ----------------------
def text_normalize(text):
    """文本归一化：统一格式，消除表面相似性（如大小写、标点、停用词差异）"""
    if not text or len(text.strip()) == 0:
        return ""
    
    # 1. 转小写 + 移除标点 + 移除数字
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    text = ''.join([c for c in text if not c.isdigit()])
    
    # 2. 分词 + 移除停用词 + 过滤过短词
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    
    # 3. 拼接回文本（用于后续去重/相似判断）
    return ' '.join(tokens)

def remove_duplicate_samples(texts, images, labels, ids):
    """移除完全重复的样本（按“归一化文本+图片ID”去重，避免漏删）"""
    seen = set()  # 存储已出现的(归一化文本, 图片ID)组合
    unique_texts, unique_images, unique_labels, unique_ids = [], [], [], []
    
    for t, img, lbl, idx in zip(texts, images, labels, ids):
        # 对文本做归一化，再结合图片ID作为去重键
        norm_text = text_normalize(t)
        img_id = idx.split('|')[1]  # 提取图片ID部分
        key = (norm_text, img_id)
        
        if key not in seen:
            seen.add(key)
            unique_texts.append(t)
            unique_images.append(img)
            unique_labels.append(lbl)
            unique_ids.append(idx)
    
    print(f"去重完成：原始样本数 {len(texts)} → 去重后 {len(unique_texts)}")
    return unique_texts, unique_images, unique_labels, unique_ids

def filter_similar_samples(texts, images, labels, ids, similarity_threshold=0.8):
    """过滤高度相似的样本（基于归一化编辑距离）
    :param similarity_threshold: 归一化编辑距离阈值，<阈值则判定为相似（0=完全相同，1=完全不同）
    """
    filtered_texts, filtered_images, filtered_labels, filtered_ids = [], [], [], []
    
    for i, t1 in enumerate(texts):
        # 跳过过短文本（无过滤意义）
        norm_t1 = text_normalize(t1)
        if len(norm_t1.split()) < 3:
            continue
        
        # 与已保留的样本比较相似度，仅保留差异足够大的样本
        is_similar = False
        for t2 in filtered_texts:
            norm_t2 = text_normalize(t2)
            if len(norm_t1) == 0 or len(norm_t2) == 0:
                continue
            
            # 计算归一化编辑距离
            dist = editdistance.eval(norm_t1, norm_t2) / max(len(norm_t1), len(norm_t2))
            if dist < similarity_threshold:
                is_similar = True
                break
        
        if not is_similar:
            filtered_texts.append(t1)
            filtered_images.append(images[i])
            filtered_labels.append(labels[i])
            filtered_ids.append(ids[i])
    
    print(f"相似样本过滤完成：过滤前 {len(texts)} → 过滤后 {len(filtered_texts)}")
    return filtered_texts, filtered_images, filtered_labels, filtered_ids

# ---------------------- 原函数（集成去重+相似过滤） ----------------------
def get_twitter_data(data_dir, data_type):
    text_lists = []  # [train_number]
    image_lists = []  # [train_num]
    labels = []  # [train_num, 2]
    label_dict = {'fake': 0, 'real': 1}
    text_image_ids = []

    # 路径配置
    if data_type == 'train':
        tweets = open('{}/devset/posts.txt'.format(data_dir), 'r', encoding="utf-8").readlines()[1:]
        image_index = 3
        image_dirs = '{}/devset/images/'.format(data_dir)
        image_files = list(filter(lambda x: not x.endswith('.txt'), os.listdir(image_dirs)))
        image_name = [image_file.split('.')[0] for image_file in image_files]
    elif data_type == 'test':
        tweets = open('{}/testset/posts_groundtruth.txt'.format(data_dir), 'r', encoding="utf-8").readlines()[1:]
        image_index = 4
        image_dirs = '{}/testset/images/'.format(data_dir)
        image_files = list(filter(lambda x: not x.endswith('.txt'), os.listdir(image_dirs)))
        image_name = [image_file.split('.')[0] for image_file in image_files]
    else:
        raise ValueError('data type must be train or test!')

    # 原始数据读取（保留原逻辑，修正重复break问题）
    for lines in tqdm(tweets):
        args = lines.strip().split('\t')
        tweet_id = args[0]
        for img in args[image_index].split(','):
            if img in image_name:
                # 读取图片（需确保image_read函数已定义）
                image_lists.append(image_read('{}/{}'.format(image_dirs, image_files[image_name.index(img)])))
                labels.append(label_dict[args[-1]])
                tweet_text = args[1]
                tweet_text = text_filter_english(tweet_text)  # 保留原有文本过滤逻辑
                text_lists.append(tweet_text)
                text_image_ids.append('{}|{}'.format(tweet_id, img))
                break  # 修正：原代码有重复的break，删除多余的那一个
    
    # ---------------------- 仅对训练集做去重+相似过滤 ----------------------
    if data_type == 'train':
        # 1. 移除完全重复样本
        # text_lists, image_lists, labels, text_image_ids = remove_duplicate_samples(
        #     text_lists, image_lists, labels, text_image_ids
        # )
        # 2. 过滤高度相似样本（阈值可根据数据调整）
        text_lists, image_lists, labels, text_image_ids = filter_similar_samples(
            text_lists, image_lists, labels, text_image_ids, similarity_threshold=0.7
        )
    # 测试集不做过滤，保证评估的公平性
    
    # 验证样本数量一致性
    assert len(text_lists) == len(image_lists) == len(labels) == len(text_image_ids)
    print(f'最终{data_type}集样本数：{len(labels)}')
    return text_lists, image_lists, labels


def twitter_dataset_filter(dataset_name, data_type):

    if dataset_name == 'twitter_dataset':
        texts_list,images_list, labels = get_twitter_data('./data/twitter_dataset', data_type)

    else:
        raise ValueError('ERROR! Dataset must be weibo or twitter!')

    return images_list, texts_list, labels


if __name__ == '__main__':
    data_dir = './data/twitter_dataset/'
    images_list, texts_list, labels = get_twitter_data(data_dir, 'train')
    print("for the train dataset:")
    print("the length of tweet_id_list:{%d}" % (len(images_list)))
    print("the length of texts_list:{%d}" % (len(texts_list)))
    train_save_path = os.path.join(data_dir, 'processed', 'train_data.pkl')
    os.makedirs(os.path.dirname(train_save_path), exist_ok=True)  # 自动创建保存目录
    train_data ={"images_list": images_list, 
                 "texts_list": texts_list,
                 "labels":labels
                }
    # with open(train_save_path, 'wb') as f:
    #     pickle.dump(train_data, f)
    
    print("Processing test data...")
    images_list, texts_list, labels = get_twitter_data(data_dir, 'test')
    print("for the train dataset:")
    print("the length of images_list:{%d}" % (len(images_list)))
    print("the length of texts_list:{%d}" % (len(texts_list)))
    print("the length of labels:{%d}"%(len(labels)))
    test_save_path = os.path.join(data_dir, 'processed', 'test_data.pkl')
    os.makedirs(os.path.dirname(test_save_path), exist_ok=True)  # 自动创建保存目录
    test_data ={"images_list": images_list, 
                 "texts_list": texts_list,
                 "labels":labels
                 }

    # with open(test_save_path, 'wb') as f:
    #     pickle.dump(test_data, f)
    print("Done.")

import os
import pickle
from src.utils import text_filter_chinese, text_filter_english, image_transform, image_read
from tqdm import tqdm


def get_gossipcop_data(data_dir, data_type):
    """
    加载并处理gossipcop数据集的文本和图像信息
    :param data_dir: 数据集根目录（如 ./data/gossipcop_dataset/）
    :param data_type: 数据类型，可选 'train' / 'test'
    :return: 图像路径列表、文本列表、标签列表
    """
    text_lists = []
    image_lists = []
    labels = []
    label_dict = {'fake': 0, 'real': 1}

    # 1. 定义基础路径（修复路径拼接错误）
    tweet_file = os.path.join(data_dir, f"gossipcop_{data_type}_tweets.txt")
    image_dir = os.path.join(data_dir, "images")  # 训练/测试图像统一在images目录下
    
    # 2. 校验文件/目录是否存在
    if not os.path.exists(tweet_file):
        raise FileNotFoundError(f"文本文件不存在：{tweet_file}")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"图像目录不存在：{image_dir}")

    # 3. 读取文本文件（使用with语句保证文件正确关闭）
    with open(tweet_file, 'r', encoding="utf-8") as f:
        tweets = f.readlines()
    print(f"加载 {data_type} 数据，共 {len(tweets)} 行。")

    # 4. 加载图像列表（过滤txt文件，提取无后缀的图像名）
    image_files = [f for f in os.listdir(image_dir) if not f.endswith('.txt')]
    image_names = [os.path.splitext(f)[0] for f in image_files]
    print(f"图像目录中共有 {len(image_files)} 张图像。")


    # 6. 遍历处理每一行数据
    for line in tqdm(tweets, desc=f"Processing {data_type} data"):
        line = line.strip()
        #print(line)
        if not line:  # 跳过空行
            continue
        
        # 修复分隔符：原数据用|分隔，替换原代码的\t
        args = line.split('|')
        # 字段不足时跳过（避免索引越界）
        
        image_id = args[0]
        label_str = args[1]
        tweet_text = args[2]# if len(args) > 2 else ""
        tweet_title = args[3] if len(args) > 3 else ""
        # 标签处理：兼容数字(0/1)和文本(fake/real)
        try:
            label = int(label_str) if label_str.isdigit() else label_dict[label_str.lower()]
        except (KeyError, ValueError):
            print(f"警告：无效标签 {label_str}，跳过行：{image_id}")
            continue
        
        # 文本过滤（调用utils中的英文过滤函数）
        clean_text = text_filter_english(tweet_title+tweet_text)
        
        # 匹配图像（取第一个存在的图像）
        image_matched = False


        if image_id in image_names:
            # 拼接完整图像路径
            img_filename = image_files[image_names.index(image_id)]

            img_path = os.path.join(image_dir, img_filename)
            #print(f"匹配图像：{img_path} 对应推文ID：{clean_text[:30]}...label: {label}")
            image_lists.append(image_read(img_path))
            text_lists.append(clean_text)
            labels.append(label)
            image_matched = True
            #break
        
        if not image_matched:
            print(f"警告：无匹配图像，跳过行：{clean_text[:30]}...")

    # 校验数据长度一致性
    assert len(text_lists) == len(image_lists) == len(labels), \
        "文本/图像/标签列表长度不一致！"

    return image_lists, text_lists, labels


def gossipcop_dataset_filter(dataset_name, data_type):
    """
    保留原函数：数据集过滤入口，仅支持gossipcop_dataset
    :param dataset_name: 数据集名称（仅支持gossipcop_dataset）
    :param data_type: 数据类型（train/test）
    :return: 图像列表、文本列表、标签列表
    """
    if dataset_name == 'gossipcop_dataset':
        # 调用处理函数，传入正确的数据集根目录
        images_list, texts_list, labels = get_gossipcop_data('./data/gossipcop_dataset', data_type)
    else:
        raise ValueError('ERROR! Dataset must be gossipcop_dataset!')  # 修正原错误提示

    return images_list, texts_list, labels


def save_processed_data(data_dir, data_type, data):
    """
    封装保存逻辑，减少重复代码
    :param data_dir: 数据集根目录
    :param data_type: 数据类型（train/test）
    :param data: 待保存的字典数据
    """
    save_path = os.path.join(data_dir, 'processed', f'{data_type}_data.pkl')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 自动创建目录
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"{data_type}数据已保存至：{save_path}")


if __name__ == '__main__':
    # 数据集根目录
    DATA_DIR = './data/gossipcop_dataset/'
    
    # ========== 处理训练集（可选择调用gossic_dataset_filter或直接调用get_gossipcop_data） ==========
    print("开始处理训练集...")
    # 方式1：调用保留的gossipcop_dataset_filter函数（推荐，符合原代码逻辑）
    train_images, train_texts, train_labels = gossipcop_dataset_filter('gossipcop_dataset', 'train')
    # 方式2：直接调用get_gossipcop_data（等效）
    # train_images, train_texts, train_labels = get_gossipcop_data(DATA_DIR, 'train')
    
    print(f"\n训练集统计：")
    print(f"图像列表长度：{len(train_images)}")
    print(f"文本列表长度：{len(train_texts)}")
    print(f"标签列表长度：{len(train_labels)}")
    
    # 保存训练集
    train_data = {
        "images_list": train_images,
        "texts_list": train_texts,
        "labels": train_labels
    }
    save_processed_data(DATA_DIR, 'train', train_data)

    # ========== 处理测试集 ==========
    print("\n开始处理测试集...")
    # 调用保留的gossipcop_dataset_filter函数
    test_images, test_texts, test_labels = gossipcop_dataset_filter('gossipcop_dataset', 'test')
    
    print(f"\n测试集统计：")
    print(f"图像列表长度：{len(test_images)}")
    print(f"文本列表长度：{len(test_texts)}")
    print(f"标签列表长度：{len(test_labels)}")
    
    # 保存测试集
    test_data = {
        "images_list": test_images,
        "texts_list": test_texts,
        "labels": test_labels
    }
    save_processed_data(DATA_DIR, 'test', test_data)

    print("\n所有数据处理完成！")

import os
import pickle
from src.utils import text_filter_chinese, text_filter_english, image_transform, image_read
from tqdm import tqdm


def get_politifact_data(data_dir, data_type):
    """
    加载并处理politifact数据集的文本和图像信息
    :param data_dir: 数据集根目录（如 ./data/politifact_dataset/）
    :param data_type: 数据类型，可选 'train' / 'test'
    :return: 图像路径列表、文本列表、标签列表
    """
    text_lists = []
    image_lists = []
    labels = []
    label_dict = {'fake': 0, 'real': 1}

    # 1. 定义基础路径（修复路径拼接错误）
    tweet_file = os.path.join(data_dir, f"{data_type}_tweets.txt")
    image_dir = os.path.join(data_dir, "images")  # 训练/测试图像统一在images目录下
    
    # 2. 校验文件/目录是否存在
    if not os.path.exists(tweet_file):
        raise FileNotFoundError(f"文本文件不存在：{tweet_file}")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"图像目录不存在：{image_dir}")

    # 3. 读取文本文件（使用with语句保证文件正确关闭）
    with open(tweet_file, 'r', encoding="utf-8") as f:
        tweets = f.readlines()
    print(f"加载 {data_type} 数据，共 {len(tweets)} 行。")

    # 4. 加载图像列表（过滤txt文件，提取无后缀的图像名）
    image_files = [f for f in os.listdir(image_dir) if not f.endswith('.txt')]
    image_names = [os.path.splitext(f)[0] for f in image_files]
    print(f"图像目录中共有 {len(image_files)} 张图像。")


    # 6. 遍历处理每一行数据
    for line in tqdm(tweets, desc=f"Processing {data_type} data"):
        line = line.strip()
        #print(line)
        if not line:  # 跳过空行
            continue
        
        # 修复分隔符：原数据用|分隔，替换原代码的\t
        args = line.split('|')
        # 字段不足时跳过（避免索引越界）
        
        image_id = args[0]
        label_str = args[1]
        tweet_text = args[2]# if len(args) > 2 else ""
        tweet_title = args[3] if len(args) > 3 else ""
        # 标签处理：兼容数字(0/1)和文本(fake/real)
        try:
            label = int(label_str) if label_str.isdigit() else label_dict[label_str.lower()]
        except (KeyError, ValueError):
            print(f"警告：无效标签 {label_str}，跳过行：{image_id}")
            continue
        
        # 文本过滤（调用utils中的英文过滤函数）
        clean_text = text_filter_english(tweet_title+tweet_text)
        
        # 匹配图像（取第一个存在的图像）
        image_matched = False


        if image_id in image_names:
            # 拼接完整图像路径
            img_filename = image_files[image_names.index(image_id)]

            img_path = os.path.join(image_dir, img_filename)
            #print(f"匹配图像：{img_path} 对应推文ID：{clean_text[:30]}...label: {label}")
            
            image_lists.append(image_read(img_path))
            text_lists.append(clean_text)
            labels.append(label)
            image_matched = True
            #break
        
        if not image_matched:
            print(f"警告：无匹配图像，跳过行：{clean_text[:30]}...")

    # 校验数据长度一致性
    assert len(text_lists) == len(image_lists) == len(labels), \
        "文本/图像/标签列表长度不一致！"

    return image_lists, text_lists, labels


def politifact_dataset_filter(dataset_name, data_type):
    """
    保留原函数：数据集过滤入口，仅支持politifact_dataset
    :param dataset_name: 数据集名称（仅支持politifact_dataset）
    :param data_type: 数据类型（train/test）
    :return: 图像列表、文本列表、标签列表
    """
    if dataset_name == 'politifact_dataset':
        # 调用处理函数，传入正确的数据集根目录
        images_list, texts_list, labels = get_politifact_data('./data/politifact_dataset', data_type)
    else:
        raise ValueError('ERROR! Dataset must be politifact_dataset!')  # 修正原错误提示

    return images_list, texts_list, labels


if __name__ == '__main__':
    # 数据集根目录
    DATA_DIR = './data/politifact_dataset/'
    
    # ========== 处理训练集（可选择调用politifact_dataset_filter或直接调用get_politifact_data） ==========
    print("开始处理训练集...")
    # 方式1：调用保留的politifact_dataset_filter函数（推荐，符合原代码逻辑）
    train_images, train_texts, train_labels = politifact_dataset_filter('politifact_dataset', 'train')
    # 方式2：直接调用get_politifact_data（等效）
    # train_images, train_texts, train_labels = get_politifact_data(DATA_DIR, 'train')
    
    print(f"\n训练集统计：")
    print(f"图像列表长度：{len(train_images)}")
    print(f"文本列表长度：{len(train_texts)}")
    print(f"标签列表长度：{len(train_labels)}")
    
    # 保存训练集
    train_data = {
        "images_list": train_images,
        "texts_list": train_texts,
        "labels": train_labels
    }
    save_processed_data(DATA_DIR, 'train', train_data)

    # ========== 处理测试集 ==========
    print("\n开始处理测试集...")
    # 调用保留的politifact_dataset_filter函数
    test_images, test_texts, test_labels = politifact_dataset_filter('politifact_dataset', 'test')
    
    print(f"\n测试集统计：")
    print(f"图像列表长度：{len(test_images)}")
    print(f"文本列表长度：{len(test_texts)}")
    print(f"标签列表长度：{len(test_labels)}")
    
    # 保存测试集
    test_data = {
        "images_list": test_images,
        "texts_list": test_texts,
        "labels": test_labels
    }
    save_processed_data(DATA_DIR, 'test', test_data)

    print("\n所有数据处理完成！")