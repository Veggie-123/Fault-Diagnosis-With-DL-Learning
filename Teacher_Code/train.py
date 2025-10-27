from utils.helpers import load_config
from experiment.exp_vgg import Experiment_VGG
import torch
import argparse

# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 有GPU先用GPU训练

if __name__ == '__main__':
    # 加载配置文件-参数
    config = load_config("configs/config.yaml")

    parser = argparse.ArgumentParser(description='PyTorch VGG Training')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train (default: 20)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--num_classes', type=int, default=10, help='分类数 (default: 10)')

    args = parser.parse_args()
    # 用命令行参数覆盖配置文件参数
    config['training']['batch_size'] = args.batch_size
    config['training']['epochs'] = args.epochs
    config['training']['learning_rate'] = args.learning_rate
    config['model']['num_classes'] = args.num_classes

    # 初始化实验对象
    exp = Experiment_VGG(config)  # set experiments

    print('>>>>>>>start training : >>>>>>>>>>>>>>>>>>>>>>>>>>')
    # 开始训练
    exp.train(device)