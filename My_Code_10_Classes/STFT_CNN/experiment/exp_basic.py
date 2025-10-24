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
        # 将模型设置为训练模式。这会启用 Dropout 和 BatchNorm 等在训练时需要但在评估时不需要的层。
        self.model.train()

        # 初始化用于累计一个 epoch 内总损失的变量
        running_loss = 0.0
        # 初始化用于累计一个 epoch 内预测正确的样本数的变量
        correct_predictions = 0
        # 初始化用于累计一个 epoch 内总样本数的变量
        total_samples = 0
        # 使用 tqdm 库为训练数据加载器创建一个进度条，方便在终端直观地看到训练进度
        progress_bar = tqdm(self.train_loader, desc="训练中")

        # 遍历训练数据加载器中的每一个数据批次(batch)
        for inputs, labels in progress_bar:
            # 将输入数据(inputs)和真实标签(labels)移动到指定的计算设备（例如 GPU 或 CPU）
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # 在计算新的梯度之前，清除优化器中所有模型参数的旧梯度
            self.optimizer.zero_grad()
            # 前向传播：将输入数据传入模型，得到模型的预测输出
            outputs = self.model(inputs)
            # 计算损失：使用预设的损失函数(criterion)计算模型预测输出和真实标签之间的差距
            loss = self.criterion(outputs, labels)
            # 反向传播：根据计算出的损失，计算损失相对于模型所有可学习参数的梯度
            loss.backward()
            # 更新权重：优化器(optimizer)根据反向传播计算出的梯度来更新模型的权重
            self.optimizer.step()

            # 累计当前批次的损失。loss.item()是该批次样本的平均损失，乘以批次大小(inputs.size(0))得到批次总损失
            running_loss += loss.item() * inputs.size(0)
            # 从模型的输出中，沿着第一个维度（批次维度）找到最大值的索引，作为模型的预测类别
            _, predicted = torch.max(outputs.data, 1)
            # 累计处理过的样本总数
            total_samples += labels.size(0)
            # 计算当前批次中预测正确的样本数量，并累加
            correct_predictions += (predicted == labels).sum().item()
            # 在 tqdm 进度条的末尾更新额外信息，显示当前批次的实时损失和准确率
            progress_bar.set_postfix(loss=loss.item(), accuracy=f"{(predicted == labels).sum().item()/labels.size(0):.4f}")
        
        # 计算并返回整个训练周期（epoch）的平均损失
        epoch_loss = running_loss / total_samples
        # 计算并返回整个训练周期（epoch）的平均准确率
        epoch_acc = correct_predictions / total_samples
        return epoch_loss, epoch_acc

    def validate(self):
        """
        通用的验证循环。
        """
        # 将模型设置为评估模式。这会禁用 Dropout 等在训练时使用但在评估时不需要的层。
        self.model.eval()
        # 初始化用于累计验证过程中的总损失
        running_loss = 0.0
        # 初始化用于累计预测正确的样本数
        correct_predictions = 0
        # 初始化用于累计总样本数
        total_samples = 0

        # 使用 torch.no_grad() 上下文管理器，禁用梯度计算，因为在验证阶段我们不需要更新模型权重。
        # 这样做可以减少不必要的计算，节省内存和计算资源。
        with torch.no_grad():
            # 遍历验证数据加载器中的每一个数据批次
            for inputs, labels in self.val_loader:
                # 将输入数据和真实标签移动到指定的计算设备
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # 前向传播：将输入数据传入模型，得到预测输出
                outputs = self.model(inputs)
                # 计算损失
                loss = self.criterion(outputs, labels)
                # 累计当前批次的损失
                running_loss += loss.item() * inputs.size(0)
                # 获取预测类别
                _, predicted = torch.max(outputs.data, 1)
                # 累计样本总数
                total_samples += labels.size(0)
                # 累计预测正确的样本数
                correct_predictions += (predicted == labels).sum().item()

        # 计算并返回整个验证集上的平均损失
        epoch_loss = running_loss / total_samples
        # 计算并返回整个验证集上的平均准确率
        epoch_acc = correct_predictions / total_samples
        return epoch_loss, epoch_acc
