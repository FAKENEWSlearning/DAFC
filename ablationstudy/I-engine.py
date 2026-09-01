import torch
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import precision_recall_fscore_support

# 假设模型实例为model，需要先拆分参数组

    
def train_test(model, train_loader, val_loader, criterion, epochs, dataset):
    model.train()
    # 启用异常检测
    torch.autograd.set_detect_anomaly(True)
    patience = 30
    best_acc = 0
    best_loss = np.inf
    
    # 定义损失权重（可根据需要调整）
    moh_weight = 0.1  # MoHAttention损失权重
    imoe_weight = 0.1  # IMOE层损失权重

    # 定义参数组
    param_groups = {
        'moh_text': {'params': model.MoH_text.parameters()},
        'moh_image': {'params': model.MoH_image.parameters()},
        'moh_text_image': {'params': model.MoH_text_image.parameters()},
        'moh_image_text': {'params': model.MoH_image_text.parameters()},
        'imoe_layer_imoe1': {'params': model.imoe_layer.IMOE.moe1.parameters()},
        'imoe_layer_imoe2': {'params': model.imoe_layer.IMOE.moe2.parameters()},
        'imoe_multi_layer_imoe1': {'params': model.imoe_multi_layer.IMOE.moe1.parameters()},
        'imoe_multi_layer_imoe2': {'params': model.imoe_multi_layer.IMOE.moe2.parameters()},
        'reason_imoe1': {'params': model.reason.imoe.moe1.parameters()},
        'reason_imoe2': {'params': model.reason.imoe.moe2.parameters()},
    }

    # , 'weight_decay' = 0.001
    # 为每个参数组创建优化器
    lrx = 1e-3
    optimizers = {
        'moh_text': torch.optim.Adam(param_groups['moh_text']['params'], lr=lrx, weight_decay = 0.001),
        'moh_image': torch.optim.Adam(param_groups['moh_image']['params'], lr=lrx, weight_decay = 0.001),
        'moh_text_image': torch.optim.Adam(param_groups['moh_text_image']['params'], lr=lrx, weight_decay = 0.001),
        'moh_image_text': torch.optim.Adam(param_groups['moh_image_text']['params'], lr=lrx, weight_decay = 0.001),
        'imoe_layer_imoe1': torch.optim.Adam(param_groups['imoe_layer_imoe1']['params'], lr=lrx, weight_decay = 0.001),
        'imoe_layer_imoe2': torch.optim.Adam(param_groups['imoe_layer_imoe2']['params'], lr=lrx, weight_decay = 0.001),
        'imoe_multi_layer_imoe1': torch.optim.Adam(param_groups['imoe_multi_layer_imoe1']['params'], lr=lrx, weight_decay = 0.001),
        'imoe_multi_layer_imoe2': torch.optim.Adam(param_groups['imoe_multi_layer_imoe2']['params'], lr=lrx, weight_decay = 0.001),
        'reason_imoe1': torch.optim.Adam(param_groups['reason_imoe1']['params'], lr=lrx, weight_decay = 0.001),
        'reason_imoe2': torch.optim.Adam(param_groups['reason_imoe2']['params'], lr=lrx, weight_decay = 0.001),
        'all':  torch.optim.Adam(model.parameters(), lr=lrx, weight_decay = 0.01),
    }

    for epoch in range(epochs):
        y_true = []
        y_pred = []
        train_running_loss = 0.0
        train_running_correct = 0
        train_num = 0


        train_running_task_loss = 0.0
        train_running_loss_moh_text1 = 0.0
        train_running_loss_moh_text2 = 0.0
        train_running_loss_moh_image1 = 0.0
        train_running_loss_moh_image2 = 0.0
        train_running_loss_text_image1 = 0.0
        train_running_loss_text_image2 = 0.0
        train_running_loss_image_text1 = 0.0
        train_running_loss_image_text2 = 0.0
        train_running_loss_aca_text1 = 0.0
        train_running_loss_aca_text2 = 0.0
        train_running_loss_aca_image1 = 0.0
        train_running_loss_aca_image2 = 0.0
        train_running_loss_imoe_multi1 = 0.0
        train_running_loss_imoe_multi2 = 0.0
        train_running_loss_reason_imoe1 = 0.0
        train_running_loss_reason_imoe2 = 0.0
        for i, data in enumerate(tqdm(train_loader)):
            for opt_name in optimizers:
                # print("opt_name:", opt_name)
                optimizers[opt_name].zero_grad()

            text_ids, images, pre_texts, pre_images, labels = data
            input_ids = text_ids[0].cuda()
            attention_mask = text_ids[1].cuda()
            token_type_ids = text_ids[2].cuda()
            out, loss_moh_text, loss_moh_image, loss_text_image, loss_image_text, loss_imoe, loss_imoe_multi, loss_reason_imoe = model(
                images.cuda(), input_ids, attention_mask, token_type_ids, 
                pre_images.cuda(), pre_texts.cuda(),labels.cuda(), feature_mode="clip"
            )
            # # 冻结特征提取层
            # for param in model.parameters():
            #     param.requires_grad = False
            
            # # 解冻特定模块
            # for param in model.MoH_text.parameters():
            #     param.requires_grad = True
            # for param in model.MoH_image.parameters():
            #     param.requires_grad = True
            task_loss = criterion(out, labels.cuda())
             
            # 1. 使用分类损失优化分类相关参数
            #print("task_loss", task_loss)
            task_loss.backward(retain_graph=True)
            
            # mohloss_text_text_blip (MoH_text)
            # mohloss_blip_text_text (MoH_text)
            loss_moh_text[0].backward(retain_graph=True)
            loss_moh_text[1].backward(retain_graph=True)

            # mohloss_image_image_blip (MoH_image)
            # mohloss_blip_image_image (MoH_image)
            loss_moh_image[0].backward(retain_graph=True)
            loss_moh_image[1].backward(retain_graph=True)#

            # mohloss_blip_text_image (MoH_text_image)
            # mohloss_text_image (MoH_text_image)
            loss_text_image[0].backward(retain_graph=True)
            loss_text_image[1].backward(retain_graph=True)

            # mohloss_blip_image_text (MoH_image_text)
            # mohloss_image_text (MoH_image_text)
            loss_image_text[0].backward(retain_graph=True)
            loss_image_text[1].backward(retain_graph=True)


            loss_reason_imoe[0].backward(retain_graph=True)
            loss_reason_imoe[1].backward(retain_graph=True)
            loss_reason_imoe[2].backward(retain_graph=True)
            loss_reason_imoe[3].backward()
            
            # 5. 各优化器更新对应参数
            # for opt in optimizers.values():
            #     opt.step()
            optimizers['all'].step()
            # optimizers['moh_text'].step()
            # # optimizers['moh_text'].step()
            # optimizers['moh_image'].step()
            # # optimizers['moh_image'].step()
            # optimizers['moh_text_image'].step()
            # # optimizers['moh_text_image'].step()
            # optimizers['moh_image_text'].step()
            # # optimizers['moh_image_text'].step()
            # optimizers['imoe_layer_imoe1'].step()
            # optimizers['imoe_layer_imoe2'].step()
            # # optimizers['imoe_layer'].step()
            # optimizers['imoe_multi_layer_imoe1'].step()
            # optimizers['imoe_multi_layer_imoe2'].step()
            # optimizers['reason_imoe1'].step()
            # optimizers['reason_imoe2'].step()
            
            # optimizers['all'].zero_grad()
            # optimizers['moh_text'].zero_grad()
            #optimizers['moh_text'].zero_grad()
            # optimizers['moh_image'].zero_grad()
            # optimizers['moh_image'].zero_grad()
            # optimizers['moh_text_image'].zero_grad()
            # optimizers['moh_text_image'].zero_grad()
            # optimizers['moh_image_text'].zero_grad()
            # optimizers['moh_image_text'].zero_grad()
            # optimizers['imoe_layer'].zero_grad()
            # optimizers['imoe_layer'].zero_grad()
            # optimizers['imoe_multi_layer'].zero_grad()
            # optimizers['reason_imoe'].zero_grad()
            # 计算总损失（仅用于记录，不用于优化）
            # 1
            train_running_task_loss += task_loss.item() * labels.size(0)
            # 2
            train_running_loss_moh_text1 += loss_moh_text[0].item() * labels.size(0)
            train_running_loss_moh_text2 += loss_moh_text[1].item() * labels.size(0)
            # 4
            train_running_loss_moh_image1 += loss_moh_image[0].item() * labels.size(0)
            train_running_loss_moh_image2 += loss_moh_image[1].item() * labels.size(0)
            # 6
            train_running_loss_text_image1 += loss_text_image[0].item() * labels.size(0)
            train_running_loss_text_image2 += loss_text_image[1].item() * labels.size(0)
            # 8
            train_running_loss_image_text1 += loss_image_text[0].item() * labels.size(0)
            train_running_loss_image_text2 += loss_image_text[1].item() * labels.size(0)

            train_running_loss_reason_imoe1 += loss_reason_imoe[0].item() * labels.size(0) + loss_reason_imoe[2].item() * labels.size(0)

            train_running_loss_reason_imoe2 += loss_reason_imoe[1].item() * labels.size(0) + loss_reason_imoe[3].item() * labels.size(0)
        
            pred = out.argmax(dim=1)
            train_running_correct += pred.eq(labels).sum().item()
            train_num += labels.size(0)
   
            # 收集标签和预测结果（用于后续计算PRF）
            y_true.extend(labels.cpu().numpy())  # 注意：若标签已在GPU，需迁移到CPU
            y_pred.extend(pred.cpu().numpy())
            
        # --------------------------
        # 计算平均损失（关键！）
        # --------------------------
        train_loss = train_running_task_loss / train_num
        train_acc = train_running_correct / train_num
        
        # 各子损失平均值
        loss_moh_text_avg = [
            train_running_loss_moh_text1 / train_num,
            train_running_loss_moh_text2 / train_num
        ]
        loss_moh_image_avg = [
            train_running_loss_moh_image1 / train_num,
            train_running_loss_moh_image2 / train_num
        ]
        loss_text_image_avg = [
            train_running_loss_text_image1 / train_num,
            train_running_loss_text_image2 / train_num
        ]
        loss_image_text_avg = [
            train_running_loss_image_text1 / train_num,
            train_running_loss_image_text2 / train_num
        ]

        loss_reason_imoe_avg = [
            train_running_loss_reason_imoe1 / train_num,
            train_running_loss_reason_imoe2 / train_num,
        ]
        # --------------------------
        # 格式化输出（清晰易读）
        # --------------------------
        print(f'\nEpoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

        
        # 计算精确率、召回率、F1分数和支持度（二分类）
        pres, recalls, f1s, supports = precision_recall_fscore_support(
            y_true, y_pred, labels=[1, 0], average=None  # 分别计算类别1和类别0的指标
        )
        # 输出PRF结果
        print("real news:==================================")
        print("Class 0 - Precision: %.3f, Recall: %.3f, F1: %.3f, Support: %d" 
              % (pres[0], recalls[0], f1s[0], supports[0]))
        print("fake news:==================================")
        print("Class 1 - Precision: %.3f, Recall: %.3f, F1: %.3f, Support: %d" 
              % (pres[1], recalls[1], f1s[1], supports[1]))
        model.eval()
        test_running_loss = 0.0
        test_running_correct = 0
        test_num = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader)):
                text_ids, images, pre_texts, pre_images, labels = data
                input_ids = text_ids[0].cuda()
                attention_mask = text_ids[1].cuda()
                token_type_ids = text_ids[2].cuda()
                out, loss_moh_text, loss_moh_image, loss_text_image, loss_image_text, loss_imoe, loss_imoe_multi, loss_reason_imoe = model(
                images.cuda(), input_ids, attention_mask, token_type_ids, 
                pre_images.cuda(), pre_texts.cuda(),  labels.cuda(), feature_mode="clip"
            )
                # acc = accuracy_score(labels.cpu().numpy(), out.cpu().numpy())
                loss = criterion(out, labels)
                test_running_loss += loss.item() * labels.size(0)
                pred = out.argmax(dim=1)
                test_running_correct += pred.eq(labels).sum().item()
                test_num += labels.size(0)
                
               # 收集标签和预测结果（用于后续计算PRF）
                y_true.extend(labels.cpu().numpy())  # 注意：若标签已在GPU，需迁移到CPU
                y_pred.extend(pred.cpu().numpy())
                
        test_loss = test_running_loss / test_num
        test_acc = test_running_correct / test_num
        print('Epoch: {}, Test Loss: {:.4f}, Test Acc: {:.4f}'.format(epoch, test_loss, test_acc))
        # 计算精确率、召回率、F1分数和支持度（二分类）
        pres, recalls, f1s, supports = precision_recall_fscore_support(
            y_true, y_pred, labels=[1, 0], average=None  # 分别计算类别1和类别0的指标
        )
        # 输出PRF结果
        print("real news:==================================")
        print("Class 0 - Precision: %.3f, Recall: %.3f, F1: %.3f, Support: %d" 
              % (pres[0], recalls[0], f1s[0], supports[0]))
        print("fake news:==================================")
        print("Class 1 - Precision: %.3f, Recall: %.3f, F1: %.3f, Support: %d" 
              % (pres[1], recalls[1], f1s[1], supports[1]))

    # 早停和保存最佳模型
        if best_acc < test_acc:
            best_acc = test_acc
            best_loss = test_loss
            best_model_wts = model.state_dict()  
            counter = 0  
            # 保存模型
            save_dir = "/root/autodl-tmp/checkpoints/"
            os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
            save_path = os.path.join(save_dir, f"./best_model_epoch_acc.pth")
            save_path = os.path.join(save_dir, f"./best_model_epoch_acc.pth")
            torch.save(best_model_wts, save_path)  
        else:  
            counter += 1  
            if counter >= patience:  
                print(f'Early stopping at epoch {epoch}')
                print('Training stop in epoch {}.'.format(epoch-patience))  
                print(f'Best test accuracy: {best_acc:.4f}, Best test loss: {best_loss:.4f}')
                
                break  
    print(f'Best test accuracy: {best_acc:.4f}, Best test loss: {best_loss:.4f}')    
    # 加载最佳模型权重  
    model.load_state_dict(best_model_wts)  