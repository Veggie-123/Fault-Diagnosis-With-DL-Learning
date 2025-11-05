"""
测试脚本 - 固定从预计算 WPT 数据集中读取

固定流程：
- make_wpt_dataset.py 已经输出 data/WPTarrays/test/index.csv + .npy
- 本脚本只做三件事：
  1) 从预计算目录读取测试集（不做分解）
  2) 构建与训练时一致的模型并加载 best_model.pth
  3) 在测试集上评估（总体指标 + 分类报告 + 混淆矩阵 + 各类别准确率）
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

# 中文显示配置（避免中文标签乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from models.TimeFreqAttention_clean import create_model
from utils.helpers import load_config

# 与 train.py 一致的简易 .npy 数据集读取器
class NpyIndexDataset(Dataset):
	"""读取 data/WPTarrays/{split}/index.csv 与对应的 .npy 样本。"""
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
		arr = np.load(path)               # (bands, time)
		tensor = torch.from_numpy(arr).float()
		label = int(row['label'])
		return tensor, label


def test_model(model, test_loader, criterion, device):
	"""
	在测试集上评估：
	- 不进行反向传播，仅前向计算、统计损失/准确率
	- 返回测试损失、准确率、预测列表、标签列表、概率（softmax）
	"""
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0
	all_preds = []
	all_labels = []
	all_probs = []
	with torch.no_grad():
		progress_bar = tqdm(test_loader, desc='测试中')
		for wpt_matrices, labels in progress_bar:
			wpt_matrices = wpt_matrices.to(device)
			labels = labels.to(device)
			outputs, attention_weights = model(wpt_matrices)  # outputs: (batch, num_classes)
			loss = criterion(outputs, labels)
			running_loss += loss.item()
			probs = torch.softmax(outputs, dim=1)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			# 记录结果以便后续生成报告与混淆矩阵
			all_preds.extend(predicted.cpu().numpy())
			all_labels.extend(labels.cpu().numpy())
			all_probs.extend(probs.cpu().numpy())
			progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%'})
	# 汇总整体指标
	test_loss = running_loss / len(test_loader)
	test_acc = 100 * correct / total
	return test_loss, test_acc, all_preds, all_labels, all_probs


def main():
	# 1) 读取配置
	config = load_config('configs/config.yaml')
	data_config = config['data']
	model_config = config['model']
	train_config = config['train']
	test_config = config['test']

	# 2) 设备
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f'使用设备: {device}')

	# 3) 权重路径检查
	checkpoint_dir = train_config['checkpoint_dir']
	model_path = os.path.join(checkpoint_dir, 'best_model.pth')
	if not os.path.exists(model_path):
		print(f"错误: 模型文件不存在 {model_path}")
		print("请先运行 train.py 进行训练")
		return

	# 4) 加载测试集（固定目录结构）
	wpt_root = data_config.get('wpt_output_dir', '../data/WPTarrays')
	test_dir = os.path.join(wpt_root, 'test')
	print(f"从预计算目录读取测试集: {test_dir}")
	dataset = NpyIndexDataset(test_dir)
	test_loader = DataLoader(dataset, batch_size=test_config['batch_size'], shuffle=False)
	# 输入通道数（频带数）
	sample_matrix, _ = dataset[0]
	input_channels = sample_matrix.shape[0]
	print(f"输入通道数（频带数）: {input_channels}")

	# 5) 创建模型并加载权重
	model = create_model(
		input_channels=input_channels,
		num_classes=model_config['num_classes'],
		cnn_out_channels=model_config['cnn_out_channels'],
		gru_hidden_dim=model_config['gru_hidden_dim'],
		dropout_rate=model_config['dropout_rate']
	).to(device)
	print(f"\n加载模型权重: {model_path}")
	model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
	print("模型加载完成")

	# 6) 在测试集上评估
	criterion = nn.CrossEntropyLoss()
	print("\n开始测试...\n")
	test_loss, test_acc, test_preds, test_labels, test_probs = test_model(model, test_loader, criterion, device)
	print("\n" + "="*50)
	print("测试结果")
	print("="*50)
	print(f"测试损失: {test_loss:.4f}")
	print(f"测试准确率: {test_acc:.2f}%")
	print("="*50)

	# 7) 生成分类报告（Precision/Recall/F1/Support）
	class_names = [f'Class {i}' for i in range(model_config['num_classes'])]
	report = classification_report(test_labels, test_preds, target_names=class_names)
	print("\n分类报告:")
	print(report)
	report_path = os.path.join(checkpoint_dir, 'classification_report.txt')
	with open(report_path, 'w', encoding='utf-8') as f:
		f.write("测试集分类报告\n")
		f.write("="*50 + "\n")
		f.write(f"测试准确率: {test_acc:.2f}%\n")
		f.write(f"测试损失: {test_loss:.4f}\n")
		f.write("="*50 + "\n\n")
		f.write(report)
	print(f"\n分类报告已保存至: {report_path}")

	# 8) 绘制混淆矩阵
	cm = confusion_matrix(test_labels, test_preds)
	plt.figure(figsize=(10, 8))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
	plt.xlabel('预测标签')
	plt.ylabel('真实标签')
	plt.title('测试集混淆矩阵')
	plt.tight_layout()
	cm_path = os.path.join(checkpoint_dir, 'confusion_matrix.png')
	plt.savefig(cm_path, dpi=150)
	print(f"混淆矩阵已保存至: {cm_path}")
	plt.close()

	# 9) 各类别准确率（便于发现难分类类别）
	print("\n各类别准确率:")
	for i in range(model_config['num_classes']):
		class_mask = np.array(test_labels) == i
		if np.sum(class_mask) > 0:
			class_acc = np.sum(np.array(test_preds)[class_mask] == i) / np.sum(class_mask) * 100
			print(f"  Class {i}: {class_acc:.2f}%")


if __name__ == '__main__':
	main()
