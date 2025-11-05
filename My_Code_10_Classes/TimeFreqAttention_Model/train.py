"""
训练脚本 - 基于时频融合 + BiGRU + 注意力 的故障诊断模型

固定流程（与 make_img_dataset.py 思路一致）：
- 本脚本不做任何预处理/分解，只“读取”由 make_wpt_dataset.py 预先生成的结果
- 目录结构固定：data/WPTarrays/{train,val}/index.csv + .npy
- .npy 文件内是小波包系数矩阵（形状：(频带数, 时间点数) = (2^L, time_steps)）
- index.csv 包含两列：file,label；file 对应同目录下的 .npy 文件名

整体结构：
1) 读取配置（目录、超参）
2) 读取 train/val 数据集（简单的 .npy 索引Dataset）
3) 创建模型（CNN1d → BiGRU → Attention → FC）
4) 训练循环（显示进度、记录曲线、保存最优权重）
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 配置matplotlib支持中文（图像显示标签为中文时不乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from models.TimeFreqAttention_clean import create_model  # 模型：CNN + BiGRU + Attention
from utils.helpers import load_config              # 读取 YAML 配置

# ==================== 简单的 .npy 索引数据集 ====================
class NpyIndexDataset(Dataset):
	"""
	读取预计算的 WPT 数据：
	- split_dir 下应包含 index.csv（两列：file,label）
	- 每一行 file 指向同目录下的一个 .npy 文件
	- .npy 内容为 (bands, time) 的浮点矩阵（例如 (16, 1024)）
	- __getitem__ 返回 (Tensor[bands, time], int_label)
	"""
	def __init__(self, split_dir: str):
		index_path = os.path.join(split_dir, 'index.csv')
		if not os.path.exists(index_path):
			raise FileNotFoundError(f'未找到索引文件: {index_path}，请先运行 make_wpt_dataset.py')
		self.split_dir = split_dir
		self.index_df = pd.read_csv(index_path)
		if 'file' not in self.index_df.columns or 'label' not in self.index_df.columns:
			raise ValueError('index.csv 需包含列: file,label')

	def __len__(self):
		return len(self.index_df)

	def __getitem__(self, idx):
		row = self.index_df.iloc[idx]
		path = os.path.join(self.split_dir, str(row['file']))
		# 加载单个样本（小波包系数矩阵）
		arr = np.load(path)               # numpy.ndarray, 形状：(bands, time)
		tensor = torch.from_numpy(arr).float()
		label = int(row['label'])
		return tensor, label

# ==================== 加载配置 ====================
config = load_config('configs/config.yaml')

data_config = config['data']
train_config = config['train']
model_config = config['model']

# 训练超参数
batch_size = train_config['batch_size']        # 每批次样本量（例如 32）
num_epochs = train_config['num_epochs']        # 训练轮数（例如 50）
learning_rate = train_config['learning_rate']  # 学习率（例如 1e-3）
checkpoint_dir = train_config['checkpoint_dir']# 权重与曲线的输出目录
num_classes = model_config['num_classes']      # 分类类别数（例如 10）

# 设备选择（优先 GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备:{device}')

# 确保输出目录存在
os.makedirs(checkpoint_dir, exist_ok=True)
print(f'模型将保存至: {checkpoint_dir}')

# ==================== 加载预计算数据 ====================
# 说明：与 make_img_dataset.py 一致，读取固定目录结构即可
root = data_config.get('wpt_output_dir', '../data/WPTarrays')
train_dir = os.path.join(root, 'train')
val_dir = os.path.join(root, 'val')

print(f"从预计算目录读取: {root}")
train_dataset = NpyIndexDataset(train_dir)
val_dataset = NpyIndexDataset(val_dir)

# DataLoader 负责批量化与打乱（训练集 shuffle=True）
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 获取输入通道数（即频带数 = 2^L）；从一个样本读取其第一维
sample_matrix, _ = train_dataset[0]
input_channels = sample_matrix.shape[0]
print(f"输入通道数（频带数）: {input_channels}")

# ==================== 创建模型 ====================
# 模型由：CNN1d（融合频带特征）→ BiGRU（融合时序）→ 注意力（动态加权）→ 全连接分类
model = create_model(
	input_channels=input_channels,
	num_classes=num_classes,
	cnn_out_channels=model_config['cnn_out_channels'],
	gru_hidden_dim=model_config['gru_hidden_dim'],
	dropout_rate=model_config['dropout_rate']
).to(device)

# 交叉熵损失 + Adam 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
# ==================== 训练/验证函数 ====================
def train_one_epoch(model, train_loader, criterion, optimizer, device):
	"""
	训练一个 epoch：
	- 前向 → 计算损失 → 反向传播 → 参数更新
	- 统计并返回平均损失与准确率（%）
	"""
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0
	progress_bar = tqdm(train_loader, desc='训练中')
	for wpt_matrices, labels in progress_bar:
		# wpt_matrices: (batch, bands, time) 例如 (32, 16, 1024)
		# labels: (batch,) 分类标签
		wpt_matrices = wpt_matrices.to(device)
		labels = labels.to(device)
		# 清零梯度（防止累积）
		optimizer.zero_grad()
		# 前向传播（模型内部：CNN→BiGRU→Attention→FC）
		outputs, _ = model(wpt_matrices)      # outputs: (batch, num_classes)
		loss = criterion(outputs, labels)      # 交叉熵
		# 反向传播 + 更新
		loss.backward()
		optimizer.step()
		# 统计本 batch 的损失与正确数
		running_loss += loss.item()
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
		progress_bar.set_postfix({'loss': loss.item()})
	# 汇总该 epoch 指标
	epoch_loss = running_loss / len(train_loader)
	epoch_acc = 100 * correct / total
	return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
	"""
	验证：不做反向传播，仅前向与统计。
	返回平均验证损失与准确率（%）。
	"""
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0
	with torch.no_grad():
		progress_bar = tqdm(val_loader, desc='验证中')
		for wpt_matrices, labels in progress_bar:
			wpt_matrices = wpt_matrices.to(device)
			labels = labels.to(device)
			outputs, _ = model(wpt_matrices)
			loss = criterion(outputs, labels)
			running_loss += loss.item()
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			progress_bar.set_postfix({'loss': loss.item()})
	epoch_loss = running_loss / len(val_loader)
	epoch_acc = 100 * correct / total
	return epoch_loss, epoch_acc

# ==================== 训练主循环 ====================
print("\n开始训练...\n")
print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"Batch Size: {batch_size}")
print(f"总Epochs: {num_epochs}\n")

best_val_acc = 0.0

# 历史指标，用于绘制曲线
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

# 实时曲线（可选）
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Training Progress')

def update_plot():
	"""更新训练与验证曲线（对交互式环境友好）。"""
	epochs = range(1, len(history['train_loss']) + 1)
	ax1.clear(); ax2.clear()
	ax1.plot(epochs, history['train_loss'], label='train_loss')
	ax1.plot(epochs, history['val_loss'], label='val_loss')
	ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True)
	ax2.plot(epochs, history['train_acc'], label='train_acc')
	ax2.plot(epochs, history['val_acc'], label='val_acc')
	ax2.set_xlabel('Epoch'); ax2.set_ylabel('Acc (%)'); ax2.legend(); ax2.grid(True)
	fig.tight_layout()
	try:
		fig.canvas.draw(); fig.canvas.flush_events()
	except Exception:
		pass

for epoch in range(num_epochs):
	print(f"\n{'='*50}")
	print(f"Epoch [{epoch+1}/{num_epochs}]")
	print(f"{'='*50}")
	# 训练一个 epoch
	train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
	# 在验证集上评估
	val_loss, val_acc = validate(model, val_loader, criterion, device)
	scheduler.step(val_loss)
    
	current_lr = optimizer.param_groups[0]['lr']
	print(f"\n训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
	print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
	print(f"当前学习率: {current_lr:.6f}")
	# 记录并更新曲线
	history['train_loss'].append(train_loss)
	history['train_acc'].append(train_acc)
	history['val_loss'].append(val_loss)
	history['val_acc'].append(val_acc)
	update_plot()
	# 保存最佳模型（按验证准确率）
	if val_acc > best_val_acc:
		best_val_acc = val_acc
		best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
		torch.save(model.state_dict(), best_model_path)
		print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")

print("\n" + "="*50)
print("训练完成!")
print(f"最佳验证准确率: {best_val_acc:.2f}%")
print("="*50)

# 保存最终曲线图
final_curve_path = os.path.join(checkpoint_dir, 'training_curves.png')
fig.savefig(final_curve_path, dpi=150)
plt.ioff()
try:
	plt.close(fig)
except Exception:
	pass