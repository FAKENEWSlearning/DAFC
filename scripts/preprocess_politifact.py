import os
import json
from PIL import Image
import re
# 配置路径
# 根目录包含 gossipcop_fake 和 gossipcop_real
DATASET_ROOT = r'C:\Users\admin\Desktop\Mcode\data\FakeNewsNet_Dataset'
# 图像存储目录
IMAGES_DIR = './data/politifact_images'
# 输出文件
OUTPUT_FILE = './data/politifact_dataset/politifact_news_data_cleaned.txt'
def is_image_readable(img_path):
    """
    检查图像文件是否可以正常读取
    """
    try:
        with Image.open(img_path) as img:
            img.verify()  # 验证文件完整性，不解码像素数据，速度快
        # 重新打开以进行更深入的检查（可选，确保能真正加载像素）
        with Image.open(img_path) as img:
            img.load() 
        return True
    except Exception:
        return False
def process_and_save():
    count = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        # 1. 遍历类别文件夹 (politifact_fake, politifact_real)
        for category_folder in os.listdir(DATASET_ROOT):
            if 'pol' not in category_folder:
                continue
            
            # 从文件夹名提取类别 (如 'fake' 或 'real')
            category = category_folder.split('_')[1]
            category_path = os.path.join(DATASET_ROOT, category_folder)
            
            # 2. 遍历每个新闻样本文件夹
            for news_folder in os.listdir(category_path):
                news_path = os.path.join(category_path, news_folder)
                json_path = os.path.join(news_path, 'news_article.json')
                
                if not os.path.exists(json_path):
                    continue
                
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # 提取核心字段
                        title = data.get("title", "").strip()
                        text = data.get("text", "").strip()
                        top_img_url = re.search(r'\d+', news_folder).group() if re.search(r'\d+', news_folder) else None
                        
                        # --- 过滤逻辑 ---
                        
                        # A. 如果没有标题或文本，跳过该样本
                        if not title or not text:
                            continue
                        
                        # B. 提取图像名称并验证是否存在于指定目录下
                        if not top_img_url:
                            continue
                        image_name = os.path.basename(top_img_url)
                        image_path = os.path.join(IMAGES_DIR, image_name) + ".jpg"
                        
                        if not os.path.exists(image_path):
                            print(f"图像不存在，跳过样本：{image_path}")
                            continue
                        if not is_image_readable(image_path):
                            print(f"跳过损坏图像: {image_path}")
                            continue
                        # --- 数据清洗 ---
                        # 移除文本和标题中的换行符和分隔符 |，确保输出为单行
                        clean_title = title.replace('\n', ' ').replace('\r', ' ').replace('|', ' ')
                        clean_text = text.replace('\n', ' ').replace('\r', ' ').replace('|', ' ')
                        
                        # 3. 按行存储：image_name|category|title|text
                        line = f"{image_name}|{category}|{clean_title}|{clean_text}\n"
                        outfile.write(line)
                        count += 1
                        
                except Exception as e:
                    print(f"处理文件夹 {news_folder} 时出错: {e}")

    print(f"处理完成！共成功提取 {count} 条样本并存入 {OUTPUT_FILE}。")

def split_dataset():
    """
    将清洗后的数据集划分为训练集和测试集，并保存为 my_train.txt 和 my_test.txt
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # 1. 读取清洗后的总表
    # 假设格式为 image_name|category|title|text
    df = pd.read_csv(OUTPUT_FILE, sep='|', names=['image', 'label', 'title', 'text'])

    # 2. 划分数据集 (80% 训练, 20% 测试)
    # stratify=df['label'] 确保类别比例平衡
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=40, 
        stratify=df['label']
    )

    # 3. 保存为新的文件
    train_df.to_csv('./data/politifact_dataset/train_tweets.txt', sep='|', index=False, header=False)
    test_df.to_csv('./data/politifact_dataset/test_tweets.txt', sep='|', index=False, header=False)

    print(f"训练集样本数: {len(train_df)}")
    print(f"测试集样本数: {len(test_df)}")
if __name__ == "__main__":
    #process_and_save()
    split_dataset()