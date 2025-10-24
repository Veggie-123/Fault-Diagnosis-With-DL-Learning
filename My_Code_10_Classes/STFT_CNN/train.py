import argparse
import os
import torch
from experiment.exp_resnet import ExpResNet
from utils.helpers import load_config, save_config

def main():
    # 1. 设置参数解析器
    parser = argparse.ArgumentParser(description='为 CWRU 数据集训练 ResNet 模型')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    args = parser.parse_args()

    # 2. 加载配置
    config = load_config(args.config)
    print("配置加载成功。")

    # 3. 初始化实验
    exp = ExpResNet(config)
    
    # 4. 设置保存模型的路径
    checkpoint_dir = config['train']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')

    # 5. 开始训练循环
    best_val_acc = 0.0
    num_epochs = config['train']['num_epochs']

    print(f"开始在设备 {exp.device} 上进行 {num_epochs} 轮训练")

    for epoch in range(num_epochs):
        print(f"\n--- 第 {epoch + 1}/{num_epochs} 轮 ---")
        
        # 训练一个 epoch
        train_loss, train_acc = exp.train_one_epoch()
        print(f"第 {epoch + 1} 轮 | 训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f}")
        
        # 验证
        val_loss, val_acc = exp.validate()
        print(f"第 {epoch + 1} 轮 | 验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}")

        # 如果得到更好的验证准确率，则保存模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(exp.model.state_dict(), best_model_path)
            print(f"新的最佳模型已保存至 {best_model_path}，准确率: {best_val_acc:.4f}")
    
    # 保存最终的配置（如果需要）
    final_config_path = os.path.join(checkpoint_dir, 'final_config.yaml')
    save_config(config, final_config_path)
    print(f"\n训练完成。最佳验证准确率: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()
