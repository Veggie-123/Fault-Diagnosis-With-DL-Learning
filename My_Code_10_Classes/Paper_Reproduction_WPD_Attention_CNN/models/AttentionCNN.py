import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    通道注意力模块 (Channel Attention Module)
    基于Squeeze-and-Excitation (SE) 机制实现
    """
    def __init__(self, in_channels, reduction_ratio=16):
        """
        参数:
            in_channels (int): 输入特征图的通道数
            reduction_ratio (int): 降维比例，用于控制全连接层的大小
        """
        super(ChannelAttention, self).__init__()
        
        # 全局平均池化，将 (B, C, H, W) 压缩为 (B, C, 1, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
            
        # 第一个全连接层：降维
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
                
        # 第二个全连接层：升维
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
                
        # Sigmoid激活函数，输出0-1的权重
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入特征图 [batch_size, channels, height, width]
        返回:
            加权后的特征图
        """
        # 保存原始输入
        batch_size, channels, _, _ = x.size()
        
        # 1. 全局平均池化：(B, C, H, W) -> (B, C, 1, 1)
        y = self.global_avg_pool(x)
        
        # 2. 展平：(B, C, 1, 1) -> (B, C)
        y = y.view(batch_size, channels)
        
        # 3. 第一个全连接层 + ReLU激活
        y = F.relu(self.fc1(y))
        
        # 4. 第二个全连接层 + Sigmoid激活，得到权重
        y = self.sigmoid(self.fc2(y))
        
        # 5. 重塑权重：(B, C) -> (B, C, 1, 1)
        y = y.view(batch_size, channels, 1, 1)
        
        # 6. 将权重应用到原始特征图上
        return x * y

class ConvBlock(nn.Module):
    """
    基础卷积块，包含卷积、批标准化、激活函数和注意力机制
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        
        # 卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        # 批标准化
        self.bn = nn.BatchNorm2d(out_channels)
        
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)
        
        # 通道注意力模块
        self.attention = ChannelAttention(out_channels)
    
    def forward(self, x):
        # 卷积 -> 批标准化 -> 激活
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        
        # 应用注意力机制
        out = self.attention(out)
        
        return out

class AttentionCNN(nn.Module):
    """
    主要的注意力CNN模型
    用于处理小波包分解生成的时频图像进行故障分类
    """
    def __init__(self, num_classes=10, input_channels=1):
        """
        参数:
            num_classes (int): 分类的类别数量
            input_channels (int): 输入图像的通道数 (小波包分解图像为1通道)
        """
        super(AttentionCNN, self).__init__()
        
        # 第一层：输入层 -> 64通道
        self.conv1 = ConvBlock(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 第二层：64 -> 128通道
        self.conv2 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)
        
        # 第三层：128 -> 256通道
        self.conv3 = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)
        
        # 第四层：256 -> 512通道
        self.conv4 = ConvBlock(256, 512, kernel_size=3, stride=2, padding=1)
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 全连接层
        self.fc = nn.Linear(512, num_classes)
        
        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入的时频图像 [batch_size, channels, height, width]
        返回:
            分类预测结果 [batch_size, num_classes]
        """
        # 第一层卷积块 + 最大池化
        x = self.conv1(x)
        x = self.maxpool1(x)
        
        # 依次通过各层卷积块
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # Dropout + 全连接层
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# 测试函数
if __name__ == "__main__":
    # 创建一个测试模型
    model = AttentionCNN(num_classes=10)
      # 创建一个测试输入 (batch_size=4, channels=1, height=224, width=224)
    test_input = torch.randn(4, 1, 224, 224)
    
    # 前向传播测试
    output = model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print("模型测试通过！")
