import torch.nn as nn
from torchvision import models

def ResNet18(num_classes, pretrained=False):
    """
    构建一个 ResNet18 模型。
    
    参数:
        num_classes (int): 输出分类的数量。
        pretrained (bool): 如果为 True，则加载在 ImageNet 上预训练的权重。
    
    返回:
        model (nn.Module): 配置好的 ResNet18 模型。
    """
    if pretrained:
        # 加载预训练的 ResNet18 模型
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 修改第一个卷积层以接受单通道（灰度）输入
        # 预训练模型期望3通道RGB输入，我们需要调整
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 替换最后一个全连接层
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        # 加载一个没有预训练权重的标准 ResNet18 模型
        model = models.resnet18(weights=None)
        
        # 修改第一个卷积层以接受单通道（灰度）输入
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 替换最后一个全连接层
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    return model