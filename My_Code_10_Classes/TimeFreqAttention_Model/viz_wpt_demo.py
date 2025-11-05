import os
import numpy as np
import matplotlib.pyplot as plt
import pywt

from utils.wavelet_transform import wavelet_packet_decomposition
from utils.helpers import load_config

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_synthetic_signal(length: int, fs: int) -> np.ndarray:
	"""
	生成一个便于理解的小示例信号：
	- 0~1/3段：10 Hz 低频正弦
	- 1/3~2/3段：120 Hz 中频正弦
	- 2/3~1段：1500 Hz 高频正弦
	叠加少量噪声
	"""
	t = np.arange(length) / fs
	seg = length // 3
	sig = np.zeros_like(t)
	# 低频段
	sig[:seg] = 1.0 * np.sin(2 * np.pi * 10 * t[:seg])
	# 中频段
	sig[seg:2*seg] = 0.8 * np.sin(2 * np.pi * 120 * t[seg:2*seg])
	# 高频段
	sig[2*seg:] = 0.6 * np.sin(2 * np.pi * 1500 * t[2*seg:])
	# 噪声
	sig += 0.05 * np.random.randn(length)
	return sig.astype(np.float32)


def plot_wpt_heatmap(coeff_matrix: np.ndarray, wavelet: str, level: int, save_dir: str):
	os.makedirs(save_dir, exist_ok=True)
	plt.figure(figsize=(10, 4))
	plt.imshow(coeff_matrix, aspect='auto', origin='lower', cmap='viridis')
	plt.colorbar(label='系数幅值')
	plt.xlabel('时间点')
	plt.ylabel('频带索引 (0=低频 → 高频)')
	plt.title(f'小波包系数热力图 | wavelet={wavelet}, level={level}, bands={coeff_matrix.shape[0]}')
	path = os.path.join(save_dir, 'wpt_heatmap.png')
	plt.tight_layout()
	plt.savefig(path, dpi=150)
	plt.close()
	print(f'已保存: {path}')


def plot_wpt_bands_grid(coeff_matrix: np.ndarray, wavelet: str, level: int, save_dir: str):
	os.makedirs(save_dir, exist_ok=True)
	n_bands, T = coeff_matrix.shape
	# 自动决定网格：优先正方形
	cols = int(np.ceil(np.sqrt(n_bands)))
	rows = int(np.ceil(n_bands / cols))
	fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2), sharex=True)
	axes = np.array(axes).reshape(rows, cols)
	for i in range(rows * cols):
		ax = axes.flat[i]
		if i < n_bands:
			ax.plot(coeff_matrix[i], lw=0.8)
			ax.set_title(f'Band {i}')
			ax.grid(True, alpha=0.3)
		else:
			ax.axis('off')
	for r in range(rows):
		axes[r, 0].set_ylabel('幅值')
	for c in range(cols):
		axes[-1, c].set_xlabel('时间点')
	fig.suptitle(f'各频带时间曲线 | wavelet={wavelet}, level={level}', y=1.02)
	fig.tight_layout()
	path = os.path.join(save_dir, 'wpt_bands_grid.png')
	fig.savefig(path, dpi=150, bbox_inches='tight')
	plt.close(fig)
	print(f'已保存: {path}')


def main():
	# 优先从配置读取一些默认参数
	config = load_config('configs/config.yaml')
	fs = int(config['data'].get('sampling_frequency', 12000))
	level = int(config['wavelet'].get('max_level', 4))
	wavelet = str(config['wavelet'].get('wavelet_type', 'db4'))
	segment_len = int(config['data'].get('time_steps', 1024))

	print(f'使用配置: fs={fs}, level={level}, wavelet={wavelet}, segment_len={segment_len}')

	# 生成一个合成信号（直观且稳定），也可改为读取真实信号
	signal = generate_synthetic_signal(length=segment_len, fs=fs)

	# 小波包分解
	coeff_matrix = wavelet_packet_decomposition(signal, wavelet=wavelet, max_level=level)
	print(f'WPT 矩阵形状: {coeff_matrix.shape} (频带 × 时间)')

	# 规范化到相似量级，便于可视化（可选）	
	# 防止某个频带幅值过大导致热力图对比度太低
	if coeff_matrix.std() > 0:
		coeff_matrix = (coeff_matrix - coeff_matrix.mean()) / (coeff_matrix.std() + 1e-6)

	save_dir = os.path.join('figs')
	plot_wpt_heatmap(coeff_matrix, wavelet, level, save_dir)
	plot_wpt_bands_grid(coeff_matrix, wavelet, level, save_dir)

	print('\n说明:')
	print('1) 热力图的纵轴是频带索引（0=最低频，越大越高频），横轴是时间点。')
	print('2) 16条子图分别展示每个频带的时间曲线，你会看到哪一段时间哪个频段更强。')
	print('3) 你可以在 configs/config.yaml 中改 wavelet/max_level 观察频带数变化。')
	print('4) 想看真实数据，将本脚本中 generate_synthetic_signal 替换为从 CSV/Mat 读取的一段信号。')


if __name__ == '__main__':
	main()
