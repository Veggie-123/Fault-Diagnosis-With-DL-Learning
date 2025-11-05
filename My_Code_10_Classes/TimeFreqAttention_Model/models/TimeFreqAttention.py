"""
基于时频融合和注意力机制的深度学习模型

模型架构：
1. 一维CNN层：融合频带特征（提取不同频带之间的关联信息）
2. 双向GRU层：融合时序特征（捕获信号在时间维度的变化）
3. 注意力机制：动态加权融合时序特征（自适应地关注重要时刻）
4. 全连接分类层：输出故障类别

数据流：
原始信号 → 小波包分解 → 系数矩阵(16×1024) → CNN → GRU → Attention → 分类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """
    注意力机制层
    
    功能：对BiGRU输出的每个时间步的特征进行动态加权，自动学习哪些时刻更重要
    
    工作原理：
    1. 计算每个时间步的注意力分数（通过全连接层）
    2. 使用softmax归一化为权重（所有权重之和为1）
    3. 使用权重对时序特征进行加权求和，得到最终的特征向量
    
    类比：就像人类在听音乐时，会重点关注某些重要的部分（如副歌），而忽略不重要的部分
    """
    def __init__(self, hidden_dim):
        """
        初始化注意力层
        
        参数:
            hidden_dim: 输入特征维度（双向GRU的输出维度，通常是 gru_hidden_dim * 2）
                       例如：如果GRU隐藏层维度是128，双向后是256
        """
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 全连接层：将每个时间步的特征映射为一个标量（注意力分数）
        # 输入维度: hidden_dim (例如 256)
        # 输出维度: 1 (注意力分数)
        # 这个分数反映了该时间步的重要性
        self.attention_fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, gru_output):
        """
        前向传播：计算注意力权重并加权求和
        
        参数:
            gru_output: 双向GRU的输出
                       形状: (batch_size, seq_len, hidden_dim*2)
                       例如: (32, 512, 256)
                       含义: 32个样本，512个时间步，每个时间步256维特征
        
        返回:
            weighted_output: 加权后的特征向量
                            形状: (batch_size, hidden_dim*2)
                            例如: (32, 256)
                            含义: 每个样本得到一个256维的加权特征向量
            
            attention_weights: 注意力权重（用于可视化分析）
                              形状: (batch_size, seq_len)
                              例如: (32, 512)
                              含义: 每个样本在512个时间步上的注意力分布
        """
        # 步骤1: 计算每个时间步的注意力分数
        # 对每个时间步的特征，通过全连接层得到注意力分数
        attention_scores = self.attention_fc(gru_output)  
        # 形状: (batch_size, seq_len, 1)
        # 例如: (32, 512, 1)
        
        # 移除最后一个维度（从3维变为2维）
        attention_scores = attention_scores.squeeze(-1)    
        # 形状: (batch_size, seq_len)
        # 例如: (32, 512)
        
        # 步骤2: 使用softmax归一化为权重
        # softmax确保所有权重之和为1，且每个权重都在[0,1]之间
        # dim=1 表示在时间步维度上进行归一化
        attention_weights = F.softmax(attention_scores, dim=1)  
        # 形状: (batch_size, seq_len)
        # 例如: (32, 512)
        # 每个样本的512个权重之和 = 1.0
        
        # 步骤3: 对权重进行维度扩展以便广播
        # 将权重从 (batch_size, seq_len) 扩展为 (batch_size, seq_len, 1)
        # 这样可以与 gru_output (batch_size, seq_len, hidden_dim) 进行广播相乘
        attention_weights_expanded = attention_weights.unsqueeze(-1)  
        # 形状: (batch_size, seq_len, 1)
        # 例如: (32, 512, 1)
        
        # 步骤4: 加权求和
        # gru_output * attention_weights_expanded: 
        #   - 形状: (batch_size, seq_len, hidden_dim)
        #   - 每个时间步的特征都被对应的注意力权重缩放
        # torch.sum(..., dim=1): 
        #   - 在时间步维度上求和，得到最终的加权特征向量
        weighted_output = torch.sum(
            gru_output * attention_weights_expanded, 
            dim=1
        )  
        # 形状: (batch_size, hidden_dim*2)
        # 例如: (32, 256)
        
        return weighted_output, attention_weights


class TimeFreqAttentionModel(nn.Module):
    """
    基于时频融合和注意力机制的故障诊断模型
    
    模型结构（按数据流顺序）：
    1. 一维CNN层：融合频带特征
       - 通过卷积核扫描不同频带，提取频带间的关联信息
       - 使用BatchNorm和ReLU激活
       - 池化降低时间维度
    
    2. 双向GRU层：融合时序特征
       - 前向和反向扫描，捕获前后时刻的依赖关系
       - 多层GRU学习更复杂的时序模式
    
    3. 注意力层：动态加权融合
       - 自动学习哪些时间步更重要
       - 对时序特征进行加权求和
    
    4. 全连接分类层：输出故障类别
       - 将特征映射到类别空间
    """
    def __init__(self, input_channels, num_classes=10, 
                 cnn_out_channels=64, gru_hidden_dim=128, 
                 dropout_rate=0.3):
        """
        初始化模型
        
        参数:
            input_channels: 输入通道数（小波包分解后的频带数）
                            例如：max_level=4时，频带数=2^4=16
                            对应输入形状: (batch_size, 16, seq_len)
            
            num_classes: 分类类别数
                         例如：10个故障类别（正常+9种故障）
            
            cnn_out_channels: CNN输出通道数
                              例如：64，表示提取了64个不同的频带特征
            
            gru_hidden_dim: GRU隐藏层维度（单方向）
                            例如：128，双向后是256
            
            dropout_rate: Dropout比率（防止过拟合）
                          例如：0.3，表示30%的神经元会被随机置零
        """
        super(TimeFreqAttentionModel, self).__init__()
        
        # ==================== 1. 一维CNN层 - 频带特征融合 ====================
        # 第一层卷积：从input_channels个频带提取特征
        # 输入: (batch_size, input_channels, seq_len) 例如 (32, 16, 1024)
        # 输出: (batch_size, cnn_out_channels, seq_len) 例如 (32, 64, 1024)
        self.cnn1 = nn.Conv1d(
            in_channels=input_channels,      # 输入通道数（频带数）
            out_channels=cnn_out_channels,   # 输出通道数（特征数）
            kernel_size=3,                   # 卷积核大小（3个时间步）
            padding=1                        # 填充，保持输出长度不变
        )
        # BatchNorm：归一化，加速训练并提高稳定性
        self.bn1 = nn.BatchNorm1d(cnn_out_channels)
        
        # 第二层卷积：进一步提取特征
        # 输入: (batch_size, cnn_out_channels, seq_len) 例如 (32, 64, 1024)
        # 输出: (batch_size, cnn_out_channels, seq_len) 例如 (32, 64, 1024)
        self.cnn2 = nn.Conv1d(
            in_channels=cnn_out_channels,     # 输入通道数
            out_channels=cnn_out_channels,   # 输出通道数
            kernel_size=3,                  # 卷积核大小
            padding=1                        # 填充
        )
        self.bn2 = nn.BatchNorm1d(cnn_out_channels)
        
        # 最大池化：降低时间维度，减少计算量
        # kernel_size=2, stride=2: 将时间长度减半
        # 例如: (32, 64, 1024) -> (32, 64, 512)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # ==================== 2. 双向GRU层 - 时序特征融合 ====================
        # GRU需要输入格式: (batch_size, seq_len, features)
        # CNN输出的特征维度是 cnn_out_channels
        self.bigru = nn.GRU(
            input_size=cnn_out_channels,    # 输入特征维度（64）
            hidden_size=gru_hidden_dim,      # 隐藏层维度（128）
            num_layers=2,                    # GRU层数（2层，可以学习更复杂的模式）
            batch_first=True,                # 输入格式为 (batch, seq, features)
            bidirectional=True,              # 双向GRU（同时向前和向后扫描）
            dropout=dropout_rate if 2 > 1 else 0  # 多层时使用dropout
        )
        # 注意：双向GRU的输出维度是 hidden_dim * 2
        # 例如：hidden_dim=128，双向后输出维度是256
        gru_output_dim = gru_hidden_dim * 2
        
        # ==================== 3. 注意力层 ====================
        # 对GRU输出的时序特征进行动态加权
        self.attention = AttentionLayer(gru_output_dim)
        
        # Dropout层：防止过拟合
        self.dropout = nn.Dropout(dropout_rate)
        
        # ==================== 4. 全连接分类层 ====================
        # 第一层：将特征维度映射到256
        self.fc1 = nn.Linear(gru_output_dim, 256)
        # 第二层：将256维特征映射到类别数
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        """
        前向传播：将输入的小波包系数矩阵转换为分类结果
        
        参数:
            x: 输入张量
               形状: (batch_size, input_channels, seq_len)
               例如: (32, 16, 1024)
               含义: 32个样本，每个样本是16个频带×1024个时间点的小波包系数矩阵
        
        返回:
            output: 分类输出（未经过softmax的logits）
                    形状: (batch_size, num_classes)
                    例如: (32, 10)
                    含义: 每个样本对10个类别的得分
            
            attention_weights: 注意力权重（可选，用于可视化）
                              形状: (batch_size, seq_len)
                              例如: (32, 512)
                              含义: 每个样本在时序上的注意力分布
        """
        # ==================== 1. CNN层 - 频带特征融合 ====================
        # 输入: x 形状为 (batch_size, input_channels, seq_len)
        # 例如: (32, 16, 1024)
        
        # 第一层卷积 + BatchNorm + ReLU
        x = self.cnn1(x)      # (32, 16, 1024) -> (32, 64, 1024)
        x = self.bn1(x)       # BatchNorm归一化
        x = F.relu(x)         # ReLU激活函数（将负值置零）
        
        # 第二层卷积 + BatchNorm + ReLU
        x = self.cnn2(x)      # (32, 64, 1024) -> (32, 64, 1024)
        x = self.bn2(x)       # BatchNorm归一化
        x = F.relu(x)         # ReLU激活函数
        
        # 最大池化：降低时间维度
        x = self.pool(x)      # (32, 64, 1024) -> (32, 64, 512)
        
        # ==================== 2. 准备GRU输入 ====================
        # GRU需要输入格式: (batch_size, seq_len, features)
        # 当前x是: (batch_size, features, seq_len)
        # 需要转置: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)  
        # (32, 64, 512) -> (32, 512, 64)
        # 含义: 32个样本，512个时间步，每个时间步64维特征
        
        # ==================== 3. 双向GRU层 - 时序特征融合 ====================
        # 双向扫描：同时向前和向后处理时序
        # 输出: (batch_size, seq_len, gru_hidden_dim*2)
        gru_output, _ = self.bigru(x)  
        # (32, 512, 64) -> (32, 512, 256)
        # 含义: 32个样本，512个时间步，每个时间步256维特征（双向）
        # 注意：第二个返回值是最后一个时间步的隐藏状态，这里不使用
        
        # ==================== 4. 注意力层 - 动态加权融合 ====================
        # 对每个时间步的特征进行加权求和
        # 自动学习哪些时间步更重要
        weighted_features, attention_weights = self.attention(gru_output)  
        # weighted_features: (32, 512, 256) -> (32, 256)
        # attention_weights: (32, 512) 用于可视化
        
        # Dropout：随机置零30%的神经元（训练时）
        weighted_features = self.dropout(weighted_features)
        
        # ==================== 5. 全连接层 - 分类 ====================
        # 第一层：256维 -> 256维
        x = F.relu(self.fc1(weighted_features))  # (32, 256) -> (32, 256)
        x = self.dropout(x)                      # Dropout
        
        # 第二层：256维 -> 10维（类别数）
        output = self.fc2(x)  # (32, 256) -> (32, 10)
        
        # 注意：output是logits（未经过softmax），在训练时loss函数会自动处理
        return output, attention_weights


def create_model(input_channels, num_classes=10, **kwargs):
    """
    创建模型的便捷函数
    
    参数:
        input_channels: 输入通道数（小波包分解后的频带数）
                        例如：max_level=4时，频带数=2^4=16
        
        num_classes: 分类类别数
                     例如：10个故障类别
        
        **kwargs: 其他模型参数
                  - cnn_out_channels: CNN输出通道数（默认64）
                  - gru_hidden_dim: GRU隐藏层维度（默认128）
                  - dropout_rate: Dropout比率（默认0.3）
    
    返回:
        model: TimeFreqAttentionModel实例
               已初始化的模型，可以直接用于训练
    
    使用示例:
        model = create_model(
            input_channels=16,
            num_classes=10,
            cnn_out_channels=64,
            gru_hidden_dim=128,
            dropout_rate=0.3
        )
    """
    model = TimeFreqAttentionModel(
        input_channels=input_channels,
        num_classes=num_classes,
        **kwargs
    )
    return model