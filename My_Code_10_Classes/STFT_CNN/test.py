import argparse
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from experiment.exp_resnet import ExpResNet
from utils.helpers import load_config

def main():
    # 1. 设置参数解析器
    parser = argparse.ArgumentParser(description='为 CWRU 数据集测试 ResNet 模型')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    args = parser.parse_args()

    # 2. 加载配置
    config = load_config(args.config)
    print("配置加载成功。")

    # 3. 初始化实验（主要用于获取模型和数据）
    exp = ExpResNet(config)
    
    # 4. 加载训练好的最佳模型
    best_model_path = f"{config['train']['checkpoint_dir']}/best_model.pth"
    exp.model.load_state_dict(torch.load(best_model_path, weights_only=True))
    print(f"模型已从 {best_model_path} 加载")

    # 5. 在测试集上评估模型
    exp.model.eval()
    all_preds = []
    all_labels = []

    print("正在测试集上评估...")
    with torch.no_grad():
        for inputs, labels in exp.test_loader:
            inputs, labels = inputs.to(exp.device), labels.to(exp.device)
            outputs = exp.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 6. 打印分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds))

    # 7. 绘制并保存混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    # 保存混淆矩阵图像
    cm_path = f"{config['train']['checkpoint_dir']}/confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"\n混淆矩阵已保存至 {cm_path}")
    # plt.show() # 如果希望在运行时直接显示图像，可以取消此行注释

if __name__ == '__main__':
    main()
