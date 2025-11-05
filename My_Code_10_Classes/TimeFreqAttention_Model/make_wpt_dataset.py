# 预计算小波包分解(WPT)数据集，并保存为 .npy，便于后续直接读取
# 使用方式（典型三步）：
#   1) 先用 mat_to_csv.py 生成 samples_data.csv（每行=一个样本窗口，最后一列=label）
#   2) cd TimeFreqAttention_Model && python make_wpt_dataset.py
#      本脚本会：读取 CSV → 按比例划分 → 对每条样本做 WPT → 保存为 .npy + 生成 index.csv
#   3) python train.py / python test.py 即可直接读取 WPT 结果，跳过在线分解
#
# 读取配置：configs/config.yaml
#   依赖键：
#     data.csv_file_path     # CSV路径
#     data.split_ratios      # 数据集划分比例 [train, val, test]
#     wavelet.wavelet_type   # 小波基（如 'db4'）
#     wavelet.max_level      # 分解层数（L=4 → 2^L=16个频带）
#     data.wpt_output_dir    # 预计算输出目录（默认 ../data/WPTarrays）
#
# 产出目录结构（示例）：
#   ../data/WPTarrays/
#     train/
#       index.csv            # 两列：file,label
#       0_3.npy              # 一个样本：形状 (bands, time) = (2^L, time_steps)
#       1_7.npy
#       ...
#     val/
#       index.csv
#       ...
#     test/
#       index.csv
#       ...

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pywt

from utils.helpers import load_config
from matplotlib import MatplotlibDeprecationWarning

# 忽略特定的 Matplotlib 弃用警告，保证控制台整洁
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


def wavelet_packet_decomposition(signal: np.ndarray, wavelet: str = 'db4', max_level: int = 4) -> np.ndarray:
	"""
	将一维信号做小波包分解，返回 (频带数, 时间点数) 的系数矩阵。
	
	核心概念（零基础友好）：
	- 小波包 = 一组“互不重叠的窄频带滤波器” + 分层细分策略
	- 第 L 层会产生 2^L 个“频带通道”；每个通道都有一条“随时间变化的系数序列”
	- 把 2^L 条序列按行堆叠，就得到二维矩阵：行=频带（低→高），列=时间点
	
	参数：
	- signal: 一维 numpy 数组，长度为 time_steps（例如 1024）
	- wavelet: 小波基，如 'db4'（Daubechies 4 阶），兼顾时频分辨
	- max_level: 分解层数，决定频带数=2^L（L=3→8，L=4→16）
	
	实现细节：
	- 通过 pywt.WaveletPacket 建立分解树；get_level(L, order='freq') 取最底层叶节点路径并按频率排序
	- 逐个叶节点读取 data（即该频带的时间系数序列）
	- 因边界处理影响，各序列长度可能略有差异 → 以最长序列为准，统一右侧补零，形成矩阵
	
	返回：
	- coeff_matrix: 形状 (2^L, T)，dtype=float32；T≈time_steps
	"""
	# 建立小波包分解树（包含 0..L 层各节点）
	wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=max_level)
	# 取最底层所有叶节点（频带），按频率从低到高排列
	leaf_nodes = [node.path for node in wp.get_level(max_level, order='freq')]
	# 收集每个频带的时间序列系数
	coeffs_list = [wp[path].data for path in leaf_nodes]
	# 找到最长序列长度，便于对齐
	max_len = max(len(c) for c in coeffs_list)
	# 初始化矩阵：行=频带数，列=最大时间长度
	coeff_matrix = np.zeros((len(coeffs_list), max_len), dtype=np.float32)
	# 将每条序列复制到矩阵相应行（不足补 0，超出截断）
	for i, c in enumerate(coeffs_list):
		l = min(len(c), max_len)
		coeff_matrix[i, :l] = c[:l]
	return coeff_matrix


def split_datasets_by_ratio(X, y, split_rate, random_state=42):
	"""
	按比例划分训练/验证/测试集（与 make_img_dataset.py 同策略）。
	
	输入：
	- X: (N, time_steps) 的二维数组，每行一个样本窗口
	- y: (N,) 的标签数组
	- split_rate: [train_ratio, val_ratio, test_ratio]，总和必须为 1
	
	输出：
	- X_train, y_train, X_val, y_val, X_test, y_test
	"""
	assert abs(sum(split_rate) - 1) < 1e-9, "split_rate 的总和必须为 1"

	train_ratio = split_rate[0]
	val_ratio = split_rate[1]
	test_ratio = split_rate[2]

	# 第一步：先切出训练集
	X_train, X_temp, y_train, y_temp = train_test_split(
		X, y, test_size=(1 - train_ratio), random_state=random_state, stratify=y
	)
	# 第二步：在剩余部分里按 val:test 比例划分
	test_size_in_temp = test_ratio / (val_ratio + test_ratio)
	X_val, X_test, y_val, y_test = train_test_split(
		X_temp, y_temp, test_size=test_size_in_temp, random_state=random_state, stratify=y_temp
	)

	print(f"训练集大小: {X_train.shape[0]}")
	print(f"验证集大小: {X_val.shape[0]}")
	print(f"测试集大小: {X_test.shape[0]}")

	return X_train, y_train, X_val, y_val, X_test, y_test


def ensure_dir(path: str):
	"""若目录不存在则创建（幂等）。"""
	if not os.path.exists(path):
		os.makedirs(path)


def save_wpt_arrays(data: np.ndarray, labels: np.ndarray, out_dir: str, wavelet: str, level: int):
	"""
	将一批 1D 信号做小波包分解，并保存结果：
	- 每个样本生成一个 (bands, time) 的 .npy 文件，文件名格式：{行索引}_{label}.npy
	- 同时生成 index.csv（两列：file,label）便于后续 DataLoader 读取
	
	输入：
	- data: (N_samples, time_steps) 的二维数组
	- labels: (N_samples,) 的标签数组
	- out_dir: 输出目录，例如 ../data/WPTarrays/train
	- wavelet/level: 小波基与分解层数
	"""
	ensure_dir(out_dir)
	index_rows = []
	for i, signal in enumerate(data):
		# 单条样本做 WPT（核心计算）
		wpt_matrix = wavelet_packet_decomposition(signal.astype(np.float32), wavelet=wavelet, max_level=level)
		# 构建文件名并保存 .npy
		fname = f"{i}_{labels[i]}.npy"
		fpath = os.path.join(out_dir, fname)
		np.save(fpath, wpt_matrix)
		# 记录索引（后续 DataLoader 读取）
		index_rows.append({"file": fname, "label": int(labels[i])})
		# 打印进度（每 200 条）
		if (i + 1) % 200 == 0:
			print(f"  已处理 {i + 1}/{len(data)}")
	# 写出索引 CSV（UTF-8）
	index_df = pd.DataFrame(index_rows)
	index_df.to_csv(os.path.join(out_dir, "index.csv"), index=False, encoding='utf-8')
	print(f"已保存索引: {os.path.join(out_dir, 'index.csv')}")


def main():
	"""
	主流程（与图像预处理脚本风格一致）：
	1) 读取配置（CSV路径、划分比例、小波参数、输出目录）
	2) 加载 CSV：X 为信号，y 为标签
	3) 按比例划分 train/val/test
	4) 分别生成对应目录下的 .npy 与 index.csv
	"""
	# 1) 读取配置
	config = load_config("configs/config.yaml")
	csv_path = config['data']['csv_file_path']
	split_ratios = config['data']['split_ratios']
	wavelet = config['wavelet']['wavelet_type']
	level = int(config['wavelet']['max_level'])
	wpt_output_dir = config['data'].get('wpt_output_dir', '../data/WPTarrays')

	# 2) 加载 CSV
	print("加载 CSV 数据...")
	samples_df = pd.read_csv(csv_path)
	# 前 N-1 列是时间点，最后一列是 label（与 mat_to_csv.py 生成格式一致）
	X = samples_df.iloc[:, :-1].values.astype(np.float32)
	y = samples_df.iloc[:, -1].values.astype(int)
	print(f"总样本数: {X.shape[0]}, 每样本长度: {X.shape[1]}")

	# 3) 划分数据集（复现实验：固定随机种子）
	X_train, y_train, X_val, y_val, X_test, y_test = split_datasets_by_ratio(X, y, split_ratios)

	# 4) 分别生成 train/val/test 的 WPT 输出
	train_dir = os.path.join(wpt_output_dir, 'train')
	val_dir = os.path.join(wpt_output_dir, 'val')
	test_dir = os.path.join(wpt_output_dir, 'test')

	print("\n生成训练集 WPT 数组...")
	save_wpt_arrays(X_train, y_train, train_dir, wavelet, level)
	print("生成验证集 WPT 数组...")
	save_wpt_arrays(X_val, y_val, val_dir, wavelet, level)
	print("生成测试集 WPT 数组...")
	save_wpt_arrays(X_test, y_test, test_dir, wavelet, level)

	print("\nWPT 预计算完成！")
	print(f"输出目录: {os.path.abspath(wpt_output_dir)}")
	print("结构：train/val/test 目录下为 .npy 样本与 index.csv")
	print("下次训练可直接从这些 .npy 读取，跳过小波包分解。")


if __name__ == '__main__':
	main()
