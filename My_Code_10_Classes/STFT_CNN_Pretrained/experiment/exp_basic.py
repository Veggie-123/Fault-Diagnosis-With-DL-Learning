import torch
import os
from tqdm import tqdm

class ExpBasic:
    def __init__(self, config):
        """
        通用的实验基类构造函数。
        - 保存配置
        - 确定设备
        - 调用子类必须实现的抽象方法来构建模型、获取数据等
        """
        self.config = config
        self.device = self._get_device()
        
        # 以下方法需要在子类中被具体实现
        self.model = self._build_model().to(self.device)
        self.train_loader, self.val_loader, self.test_loader = self._get_data()
        self.optimizer = self._get_optimizer()
        self.criterion = self._get_criterion()

    def _get_device(self):
        """获取计算设备"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self):
        """
        构建模型的占位符方法。
        必须在子类中被重写 (override)。
        """
        raise NotImplementedError("`_build_model()` must be implemented in the subclass.")

    def _get_data(self):
        """
        获取数据加载器的占位符方法。
        必须在子类中被重写 (override)。
        """
        raise NotImplementedError("`_get_data()` must be implemented in the subclass.")

    def _get_optimizer(self):
        """
        获取优化器的占位符方法。
        必须在子类中被重写 (override)。
        """
        raise NotImplementedError("`_get_optimizer()` must be implemented in the subclass.")

    def _get_criterion(self):
        """
        获取损失函数的占位符方法。
        必须在子类中被重写 (override)。
        """
        raise NotImplementedError("`_get_criterion()` must be implemented in the subclass.")

    def train_one_epoch(self):
        """
        通用的训练循环。
        这个方法对于任何模型和数据都是一样的。
        """
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        progress_bar = tqdm(self.train_loader, desc="训练中")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=loss.item(), accuracy=f"{(predicted == labels).sum().item()/labels.size(0):.4f}")
        
        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        return epoch_loss, epoch_acc

    def validate(self):
        """
        通用的验证循环。
        """
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        return epoch_loss, epoch_acc
