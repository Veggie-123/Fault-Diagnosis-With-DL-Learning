from models.vgg import VGGModel
from utils.data_loader import SpectrogramDataset
from experiment.exp_basic import Exp_Basic
from torchvision import transforms
from joblib import dump, load
from utils.tools import adjust_learning_rate
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
import copy
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['simhei'] # 添加中文字体为黑体
plt.rcParams["axes.unicode_minus"] = False
import warnings
warnings.filterwarnings('ignore')


class Experiment_VGG(Exp_Basic):
    # 构造函数
    def __init__(self, config):
        super(Experiment_VGG, self).__init__(config)

    # 构建模型
    def _build_model(self):
        model =VGGModel(
            self.config['model']['conv_archs'],
            self.config['model']['num_classes'],
            self.config['model']['input_channels']
        )
        return model

    # 获取数据：
    def _get_data(self, flag):
        # 定义数据预处理
        transform = transforms.Compose([
            transforms.Resize((self.config['data']['image_size'], self.config['data']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 通用图像的标准化
        ])

        # 数据加载
        train_dataset = SpectrogramDataset(
            path=os.path.join(self.config['training']['img_data_path'], flag),
            transform=transform
        )

        # 创建数据加载器
        data_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=self.config['training']['shuffle'],
            drop_last=self.config['training']['drop_last'],
            num_workers=self.config['training']['num_workers']
        )

        return data_loader

    # 选择优化器：
    def _select_optimizer(self):
        model_optim = torch.optim.Adam(self.model.parameters(), lr=self.config['training']['learning_rate'])
        return model_optim

    # 选择损失函数：
    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    # Xavier方法 初始化网络参数，最开始没有初始化一直训练不起来。
    def init_normal(self, m):
        if isinstance(m, torch.nn.Linear):
            # Xavier 初始化
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Conv2d):
            # Xavier 初始化
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    # 训练：
    def train(self, device):
        # 获取训练数据、验证数据的 DataLoader
        train_loader = self._get_data(flag='train')
        val_loader = self._get_data(flag='val')

        # 设置模型检查点的保存路径
        path = self.config['training']['best_model_path']
        if not os.path.exists(path):
            os.makedirs(path)  # # 如果路径不存在，则创建路径

        # 记录当前时间
        time_now = time.time()

        # 选择优化器
        model_optim = self._select_optimizer()
        # 选择损失函数
        criterion = self._select_criterion()

        # 记录训练、验证、测试集损失，用于画图
        train_loss_history = []  # 记录在训练集上每个epoch的loss的变化情况
        train_acc_history = []  # 记录在训练集上每个epoch的准确率的变化情况
        val_loss_history = []  # 记录在验证集上每个epoch的loss的变化情况
        val_acc_history = []  # 记录在验证集上每个epoch的准确率的变化情况

        self.model = self.model.to(device)
        # 参数初始化
        self.model.apply(self.init_normal)  # 确保初始化在训练前进行
        # 最高准确率  最佳模型
        best_accuracy = 0.0
        best_model = self.model

        # 训练若干个 epoch
        epochs = self.config['training']['epochs']
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')

            # === 训练阶段 ===
            # 将模型设置为训练模式
            self.model.train()
            train_loss_epoch = 0.0  # 保存当前epoch的loss和
            train_correct_epoch = 0  # 保存当前epoch的正确个数和

            for seq, labels in train_loader:
                seq, labels = seq.to(device), labels.to(device)

                # 每次更新参数前都梯度归零和初始化
                model_optim.zero_grad()
                # 前向传播
                y_pred = self.model(seq)
                # 损失计算
                loss = criterion(y_pred, labels)

                # 统计指标
                train_loss_epoch += loss.item() * seq.size(0)
                # 得到预测的类别
                predicted_labels = torch.argmax(y_pred, dim=1)
                # 与真实标签进行比较，计算预测正确的样本数量  # 计算当前batch预测正确个数
                train_correct_epoch += (predicted_labels == labels).sum().item()

                # 反向传播和参数更新
                loss.backward()
                model_optim.step()

            # 计算准确率
            train_Accuracy = train_correct_epoch / len(train_loader.dataset)
            train_loss = train_loss_epoch / len(train_loader.dataset)
            train_acc_history.append(train_Accuracy)
            train_loss_history.append(train_loss)

            # === 验证阶段 ===
            # 将模型设置为评估模式
            self.model.eval()
            val_loss_epoch = 0.0  # 保存当前epoch的loss和
            valcorrect_epoch = 0  # 保存当前epoch的正确个数和

            with torch.no_grad():
                for data, label in val_loader:
                    data, label = data.to(device), label.to(device)
                    pre = self.model(data)
                    # 损失计算
                    loss = criterion(pre, label)
                    # 统计指标
                    val_loss_epoch += loss.item() * data.size(0)
                    # 得到预测的类别
                    predicted_labels = torch.argmax(pre, dim=1)
                    # 与真实标签进行比较，计算预测正确的样本数量  # 计算当前batch预测正确个数
                    valcorrect_epoch += (predicted_labels == label).sum().item()

            val_accuracy = valcorrect_epoch / len(val_loader.dataset)
            val_loss = val_loss_epoch / len(val_loader.dataset)
            val_acc_history.append(val_accuracy)
            val_loss_history.append(val_loss)

            # 如果当前模型的准确率优于之前的最佳准确率，则更新最佳模型
            # 保存当前最优模型参数
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                # 深拷贝模型参数
                best_model = copy.deepcopy(self.model)# 更新最佳模型的参数

            # 打印日志
            print(f"Train Loss: {train_loss:.4f} Acc: {train_Accuracy:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_accuracy:.4f}")
            print('-' * 10)

            # 调整学习率
            adjust_learning_rate(model_optim, epoch + 1, self.config)

        # 保存最佳模型参数
        best_model_path = path + '/' + 'checkpoint.pth'
        torch.save(best_model.state_dict(), best_model_path)

        print("best_accuracy :", best_accuracy)
        print(f'\nDuration: {time.time() - time_now:.0f} seconds')

        # 可视化
        # 创建训练损失图  保存 训练过程可视化图片在文件中
        plt.figure(figsize=(14, 7), dpi=100)  # dpi 越大  图片分辨率越高，写论文的话 一般建议300以上设置
        plt.plot(train_loss_history, color='b', label='train_loss')
        plt.plot(train_acc_history, color='g', label='train_acc')
        plt.plot(val_loss_history, color='y', label='val_loss')
        plt.plot(val_acc_history, color='r', label='val_acc')
        plt.legend(fontsize=12)
        plt.title('Visualization of model training process', fontsize=16)
        plt.show()  # 显示
        # 保存训练图
        # plt.savefig('Train_Val', dpi=100)

        return best_model  # 返回训练好的模型

    # 测试：
    def test(self, device):
        test_loader = self._get_data(flag='test')

        # 加载模型参数
        path = self.config['training']['best_model_path']
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.model = self.model.to(device)

        true_labels = []  # 存储类别标签
        predicted_labels = []  # 存储预测的标签

        # 将模型设置为评估模式
        self.model.eval()
        with torch.no_grad():
            for test_data, test_label in test_loader:
                true_labels.extend(test_label.tolist())
                test_data = test_data.to(device)
                test_output = self.model(test_data)
                predicted = torch.argmax(test_output, dim=1)
                predicted_labels.extend(predicted.tolist())

        # 计算每一类的分类准确率
        report = classification_report(true_labels, predicted_labels, digits=4)
        print(report)

        # 保存结果
        folder_path = self.config['training']['predicted_folder_path']
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)
        np.save(folder_path + 'true.npy', true_labels)
        np.save(folder_path + 'pred.npy', predicted_labels)

        return


