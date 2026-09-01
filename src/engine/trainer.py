import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
from sklearn.manifold import TSNE  # 导入TSNE库
import torch.nn.functional as F

# 设置中文字体（避免图表中文乱码）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 英文环境
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文Windows环境
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def plot_tsne_2d(features, true_labels, save_path, title="TSNE Visualization"):
    """
    t-SNE 二维可视化函数：仅保留类别点+长方形方框，无任何坐标轴
    Args:
        features: 高维特征数组 (n_samples, n_features)
        true_labels: 真实标签数组 (n_samples,)，0=real，1=fake
        save_path: 图表保存路径
        title: 图表标题
    """
    # 1. t-SNE 降维（n_components=2 降维到二维）
    tsne = TSNE(
        n_components=2,  # 降维到2维
        random_state=42,  # 固定随机种子，结果可复现
        perplexity=50,    # 调整为50，更接近参考图的分布密度
        learning_rate=200, # 学习率
        init='pca'        # 用PCA初始化，结果更稳定
    )
    features_2d = tsne.fit_transform(features)

    # 2. 按真实标签分组
    real_mask = (true_labels == 0)   # 真实标签0=real（蓝色）
    fake_mask = (true_labels == 1)   # 真实标签1=fake（红色）

    # 3. 绘图（仅保留类别点+长方形方框）
    plt.figure(figsize=(12, 9), dpi=300)  # 高分辨率画布
    
    # 绘制real样本（蓝色）
    plt.scatter(
        features_2d[real_mask, 0], features_2d[real_mask, 1],
        label="real",
        color="#6495ED",  # 柔和的蓝色，接近参考图
        alpha=0.8,
        s=60,
        edgecolors='none'
    )
    
    # 绘制fake样本（红色）
    plt.scatter(
        features_2d[fake_mask, 0], features_2d[fake_mask, 1],
        label="fake",
        color="#FF6347",  # 柔和的红色，接近参考图
        alpha=0.8,
        s=60,
        edgecolors='none'
    )

    # 4. 核心修改：仅保留长方形方框，完全隐藏坐标轴
    plt.title("")
    plt.xlabel("")  # 清空x轴标签
    plt.ylabel("")  # 清空y轴标签
    plt.xticks([])  # 隐藏x轴刻度
    plt.yticks([])  # 隐藏y轴刻度
    
    # 显示所有边框，形成完整的长方形方框
    ax = plt.gca()
    ax.spines['top'].set_visible(True)    # 显示上边框
    ax.spines['right'].set_visible(True)  # 显示右边框
    ax.spines['bottom'].set_visible(True) # 显示下边框
    ax.spines['left'].set_visible(True)   # 显示左边框
    # 设置方框线条样式
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    
    # 图例（保留类别标识）
    plt.legend(
        loc="upper right",
        fontsize=20,
        markerscale=1.2,
        frameon=True,
        facecolor='white',
        edgecolor='gray'
    )
    
    # 5. 保存高清晰图表
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight', format='png')
    plt.close()  # 关闭画布，释放内存
    print(f"\n高清晰TSNE图表已保存至: {save_path}")

def train_test(model, train_loader, val_loader, optimizer, criterion, epochs, dataset):
    """
    训练与验证函数（新增：仅保留类别+长方形方框的t-SNE 可视化 + 打印最佳准确率下错误样本的logits_pred）
    Args:
        model: 待训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        criterion: 损失函数
        epochs: 训练轮数
        dataset: 数据集（预留参数，未使用）
    Returns:
        model: 训练后的最佳模型
    """
    # 移至GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 早停参数
    patience = 500
    best_acc = 0.0
    best_loss = np.inf
    counter = 0  # 早停计数器
    # 2. 定义手动调整规则（可自定义）
    INIT_LR = 1e-3            # 初始学习率
    DECAY_FACTOR = 0.5        # 衰减因子（每次×0.5）
    DECAY_STEP = 10           # 每10个epoch衰减一次
    MIN_LR = 1e-6             # 学习率下限
    ADJUST_START_EPOCH = 30   # 从第30个epoch后开始调整

    # 新增：用于保存最佳模型时的错误样本信息
    best_val_errors = None
    # 新增：TSNE 图表保存目录
    tsne_save_dir = "/root/tsne_plots"
    os.makedirs(tsne_save_dir, exist_ok=True)  # 确保目录存在

    for epoch in range(epochs):
        # -------------------------- 训练阶段 --------------------------
        model.train()
        # 主预测结果收集
        y_true_train = []
        y_pred_train = []
        # 损失统计
        train_running_total_loss = 0.0  # 总损失（分类损失+模型内部损失）
        train_pre_loss = 0.0  # 仅分类损失
        train_aux_loss = 0.0
        train_share_loss = 0.0
        train_moh_loss = 0.0
        # 准确率统计
        train_running_correct = 0
        train_num = 0

        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
            optimizer.zero_grad()  # 清空梯度

            # 解析数据
            text_ids, images, pre_texts, pre_images, labels = data

            # 处理文本输入（text_ids为元组：(input_ids, attention_mask, token_type_ids)）
            input_ids = text_ids[0].to(device)
            attention_mask = text_ids[1].to(device)
            token_type_ids = text_ids[2].to(device)

            # 处理其他输入
            images = images.to(device)
            pre_images = pre_images.to(device)
            pre_texts = pre_texts.to(device)
            labels = labels.to(device)

            # 模型前向传播
            logits_final, orth_loss, share_loss, moh_loss = model(
                images, input_ids, attention_mask, token_type_ids,
                pre_images, pre_texts 
            )

            # 处理预测值维度（挤压batch_size,1 -> batch_size）
            logits_pred = logits_final.squeeze(1)
            # 计算各损失
            pre_loss = criterion(logits_pred, labels.long())

            # 总损失 = 分类损失 + 模型内部损失 + 中间预测损失（加权）
            orth_coff = 0.5
            share_coff = 0.5
            moh_coff = 0.01
            total_loss = pre_loss  # 暂时关闭其他损失项

            # 反向传播与优化
            total_loss.backward()
            optimizer.step()

            # 统计损失
            train_running_total_loss += total_loss.item() * labels.size(0)
            train_pre_loss += pre_loss.item() * labels.size(0)
            train_aux_loss += orth_coff * orth_loss.item() * labels.size(0)
            train_share_loss += share_coff * share_loss.item() * labels.size(0)
            train_moh_loss += moh_coff * moh_loss.item() * labels.size(0)
            
            pred_label = logits_pred.argmax(dim=1)
            train_running_correct += pred_label.eq(labels).sum().item()
            train_num += labels.size(0)

            # 收集标签和预测结果（转CPU并转numpy）
            labels_np = labels.cpu().numpy()
            y_true_train.append(labels_np)
            y_pred_train.append(logits_pred.detach().cpu().numpy())

        y_pred_train = np.concatenate(y_pred_train)
        y_true_train = np.concatenate(y_true_train)

        # 计算训练集平均指标
        train_avg_loss = train_running_total_loss / train_num
        train_avg_pre_loss = train_pre_loss / train_num
        train_avg_aux_loss = train_aux_loss / train_num
        train_avg_share_loss = train_share_loss / train_num
        train_avg_moh_loss = train_moh_loss / train_num
        train_acc = train_running_correct / train_num

        # 训练集指标打印
        y_pred = np.argmax(y_pred_train, axis=1)
        y_true_np = y_true_train
        y_pred_np = y_pred
        print(f"\n===== Epoch {epoch+1} 训练集指标 =====")
        print(f"拼接后 y_pred_train 维度：{y_pred_train.shape}")
        print(f"拼接后 y_true_train 维度：{y_true_train.shape}")
        print("y_true", y_true_np.shape)
        print("y_pred", y_pred_np.shape)
        acc = accuracy_score(y_true_np, y_pred_np)
        print(f"整体准确率（Accuracy）: {acc:.4f}")
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_np, 
            y_pred_np, 
            labels=[0, 1],
            zero_division=0
        )
        print("\n每个类别的详细指标：")
        print(f"类别0 - 精确率(Pre): {precision[0]:.4f}, 召回率(Recall): {recall[0]:.4f}, F1: {f1[0]:.4f}")
        print(f"类别1 - 精确率(Pre): {precision[1]:.4f}, 召回率(Recall): {recall[1]:.4f}, F1: {f1[1]:.4f}")
        print("\n完整分类报告：")
        print(classification_report(y_true_np, y_pred_np, target_names=["类别0", "类别1"]))
        print("train_avg_loss:", train_avg_loss)
        print("train_avg_pre_loss", train_avg_pre_loss)
        print("train_avg_aux_loss", train_avg_aux_loss)
        print("train_avg_share_loss:", train_avg_share_loss)
        print("train_avg_moh_loss:", train_avg_moh_loss)

        # -------------------------- 验证阶段 --------------------------
        model.eval()
        # 主预测结果收集
        y_true_val = []
        y_pred_val = []
        # 新增：收集每个样本的logits_pred、真实标签、预测标签（用于分析错误样本）
        val_all_logits = []    # 保存所有样本的logits_pred
        val_all_true = []      # 保存所有样本的真实标签
        val_all_pred = []      # 保存所有样本的预测标签
        val_all_indices = []
        # 损失和准确率统计
        val_running_total_loss = 0.0
        val_running_pre_loss = 0.0
        val_running_aux_loss = 0.0
        val_running_share_loss = 0.0
        val_running_moh_loss = 0.0
        val_running_correct = 0
        val_num = 0
        val_sample_idx = 0  # 关键：验证集全局样本索引计数器（从0开始，可改为1以符合人类习惯）

        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")):
                # 解析数据（与训练阶段一致）
                text_ids, images, pre_texts, pre_images, labels = data
                input_ids = text_ids[0].to(device)
                attention_mask = text_ids[1].to(device)
                token_type_ids = text_ids[2].to(device)
                images = images.to(device)
                pre_images = pre_images.to(device)
                pre_texts = pre_texts.to(device)
                labels = labels.to(device)
                batch_size = labels.size(0) 
                # 模型前向传播
                logits_final, orth_loss, share_loss, moh_loss = model(
                    images, input_ids, attention_mask, token_type_ids,
                    pre_images, pre_texts
                )

                # 处理维度
                logits_pred = logits_final.squeeze(1)

                # 计算损失
                pre_loss = criterion(logits_pred, labels.long())
                orth_coff = 0.5
                share_coff = 0.5
                moh_coff = 0.01
                total_loss = pre_loss + orth_coff * orth_loss + share_coff * share_loss + moh_coff * moh_loss

                # 统计损失
                val_running_total_loss += total_loss.item() * labels.size(0)
                val_running_pre_loss += pre_loss.item() * labels.size(0)
                val_running_aux_loss += orth_coff * orth_loss.item() * labels.size(0)
                val_running_share_loss += share_coff * share_loss.item() * labels.size(0)
                val_running_moh_loss += moh_coff * moh_loss.item() * labels.size(0)
                threshold = 0.55
                # 计算预测结果
                pred_label = logits_pred.argmax(dim=1)
                    
                val_running_correct += pred_label.eq(labels).sum().item()
                val_num += labels.size(0)

                # 收集结果
                labels_np = labels.cpu().numpy()
                y_true_val.append(labels_np)
                y_pred_val.append(logits_pred.detach().cpu().numpy())
                current_batch_indices = list(range(val_sample_idx, val_sample_idx + batch_size))  # 当前batch的全局索引
                val_all_indices.extend(current_batch_indices)  # 追加全局索引
                # 新增：收集每个样本的logits、真实标签、预测标签
                val_all_logits.extend(logits_pred.detach().cpu().numpy())  # 按样本追加
                val_all_true.extend(labels_np)                            # 真实标签
                val_all_pred.extend(pred_label.cpu().numpy())             # 预测标签
                val_sample_idx += batch_size
        val_avg_loss = val_running_total_loss / val_num
        val_avg_pre_loss = val_running_pre_loss / val_num
        val_avg_aux_loss = val_running_aux_loss / val_num
        val_avg_share_loss = val_running_share_loss / val_num
        val_avg_moh_loss = val_running_moh_loss / val_num
        # 转换为numpy数组
        y_pred_val = np.concatenate(y_pred_val)
        y_true_val = np.concatenate(y_true_val)
        val_all_indices = np.array(val_all_indices)  # 形状：[val_num,]
        val_all_logits = np.array(val_all_logits)    # 形状：[val_num, 2]
        val_all_true = np.array(val_all_true)        # 形状：[val_num,]
        val_all_pred = np.array(val_all_pred)        # 形状：[val_num,]

        # 计算验证集指标
        y_pred = np.argmax(y_pred_val, axis=1)
        y_true_np = y_true_val
        y_pred_np = y_pred
        val_acc = accuracy_score(y_true_np, y_pred_np)

        # 验证集指标打印
        print(f"\n===== Epoch {epoch+1} 验证集指标 =====")
        print(f"拼接后 y_pred_val 维度：{y_pred_val.shape}")
        print(f"拼接后 y_true_val 维度：{y_true_val.shape}")
        print("y_true", y_true_np.shape)
        print("y_pred", y_pred_np.shape)
        print(f"整体准确率（Accuracy）: {val_acc:.4f}")
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_np, 
            y_pred_np, 
            labels=[0, 1],
            zero_division=0
        )
        print("\n每个类别的详细指标：")
        print(f"类别0 - 精确率(Pre): {precision[0]:.4f}, 召回率(Recall): {recall[0]:.4f}, F1: {f1[0]:.4f}")
        print(f"类别1 - 精确率(Pre): {precision[1]:.4f}, 召回率(Recall): {recall[1]:.4f}, F1: {f1[1]:.4f}")
        print("\n完整分类报告：")
        print(classification_report(y_true_np, y_pred_np, target_names=["类别0", "类别1"]))
        print("val_avg_loss", val_avg_loss)
        print("val_avg_pre_loss", val_avg_pre_loss)
        print("val_avg_aux_loss", val_avg_aux_loss)
        print("val_avg_share_loss", val_avg_share_loss)
        print("val_avg_moh_loss", val_avg_moh_loss)

        # 学习率调整
        current_lr = optimizer.param_groups[0]['lr']
        if epoch >= ADJUST_START_EPOCH:
            decay_times = (epoch - ADJUST_START_EPOCH) // DECAY_STEP
            new_lr = max(INIT_LR * (DECAY_FACTOR ** decay_times), MIN_LR)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        # -------------------------- 早停与模型保存 --------------------------
        if val_acc > best_acc or (val_acc == best_acc and val_avg_loss < best_loss):
            best_acc = val_acc
            best_loss = val_avg_loss
            best_model_wts = model.state_dict()
            counter = 0
            # 筛选错误样本（预测标签 != 真实标签）
            error_mask = (val_all_pred != val_all_true)  # 布尔掩码：True表示错误样本
            error_indices = val_all_indices[error_mask]  # 错误样本的全局索引
            error_logits = val_all_logits[error_mask]    # 错误样本的logits_pred
            error_true = val_all_true[error_mask]        # 错误样本的真实标签
            error_pred = val_all_pred[error_mask]        # 错误样本的预测标签

            # 保存错误样本信息
            best_val_errors = {
                "global_indices": error_indices,  # 错误样本在验证集中的全局索引
                "logits_pred": error_logits,
                "true_labels": error_true,
                "pred_labels": error_pred
            }

            # 打印错误样本详情（含全局索引）
            print(f"\n===== 最佳模型（Epoch {epoch+1}）错误样本分析 =====")
            print(f"最佳验证集准确率: {best_acc:.4f}")
            print(f"最佳验证集平均损失: {best_loss:.4f}")
            print(f"验证集总样本数: {val_num}")
            print(f"错误样本数: {len(error_indices)}")
            print("\n错误样本详情（全局索引 | logits_pred | 真实标签 | 预测标签）：")
            for idx, (global_idx, logits, true_lab, pred_lab) in enumerate(zip(
                error_indices, error_logits, error_true, error_pred
            )):
                # 注：全局索引从0开始，若想从1开始，可改为 global_idx + 1
                print(f"错误样本 {idx+1}: 全局索引={global_idx} | logits_pred={logits.round(4)} | 真实标签={true_lab} | 预测标签={pred_lab}")

            # ====================== 新增：仅保留类别+长方形方框的t-SNE 可视化 ======================
            tsne_save_path = os.path.join(tsne_save_dir, f"best_tsne_epoch_{epoch+1}.png")
            plot_tsne_2d(
                features=val_all_logits,          # 验证集所有样本的logits（2维特征）
                true_labels=val_all_true,         # 真实标签（0=real，1=fake）
                save_path=tsne_save_path,
                title=f"TSNE - Best Model (Epoch {epoch+1}, Acc {best_acc:.4f})"
            )

            # 保存最佳模型
            save_dir = "/root/"
            save_path = os.path.join(save_dir, "best_model.pth")
            torch.save(best_model_wts, save_path)
            print(f"\nBest model saved at epoch {epoch+1} (Acc: {best_acc:.4f}, Loss: {best_loss:.4f})")
        else:
            counter += 1
            print(f"Early stopping counter: {counter}/{patience}")
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    print(f'\nTraining completed. Best Val Acc: {best_acc:.4f}, Best Val Loss: {best_loss:.4f}')
    
    # 可选：返回最佳模型的错误样本信息（方便后续分析）
    return model  # , best_val_errors