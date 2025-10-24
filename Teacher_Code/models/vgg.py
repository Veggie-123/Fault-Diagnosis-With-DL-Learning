import torch
import torch.nn as nn

class VGGModel(nn.Module):
    def __init__(self, conv_archs, num_classes, input_channels=3):
        super().__init__()
        # CNN参数初始化
        self.conv_archs = conv_archs  # 网络结构
        self.input_channels = input_channels  # 输入通道数
        self.features = self.make_layers()  # 卷积层配置

        # 使用自适应平均池化，将特征图缩放到1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层配置
        # 这里假设最后一个卷积层的输出通道数是分类器的输入大小
        self.classifier = nn.Linear(conv_archs[-1][-1], num_classes)

    # CNN卷积池化结构
    def make_layers(self):
        layers = []
        for (num_convs, out_channels) in self.conv_archs:
            for _ in range(num_convs):
                layers.append(nn.Conv2d(self.input_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                self.input_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 这里会使宽高减半
        return nn.Sequential(*layers)

    def forward(self, x):
        # x 的形状应该是 [batch_size, input_channels, height, width]
        features = self.features(x)
        x = self.avgpool(features)
        flat_tensor = x.view(x.size(0), -1)  # 展平成 [batch_size, num_features]
        output = self.classifier(flat_tensor)
        return output

