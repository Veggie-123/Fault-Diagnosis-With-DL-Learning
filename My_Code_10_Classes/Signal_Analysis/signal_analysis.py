import scipy.io
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.helpers import load_config
from scipy import signal as sig

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def plot_graph(x_data, y_data, title, xlabel, ylabel, save_name, xlim=None):
    """一个用于绘制通用图形的函数"""
    plt.figure(figsize=(15, 5))
    plt.plot(x_data, y_data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim:
        plt.xlim(xlim)
    plt.grid(True)
    plt.savefig(os.path.join(save_path, save_name), dpi=300, bbox_inches='tight')
    plt.show()

def plot_subplot(ax, x_data, y_data, title, xlabel, ylabel, xlim=None, annotations=None):
    """在一个指定的子画板(ax)上绘制图形的通用函数"""
    ax.plot(x_data, y_data)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    if xlim:
        ax.set_xlim(xlim)
    if annotations:
        for ann in annotations:
            ax.axvline(x=ann['x'], color=ann['color'], linestyle=ann['style'], label=ann['label'])
        ax.legend()

# 一、加载数据

# 加载文件
config = load_config('configs/config.yaml')
data_config = config.get('data', {})
data_path = data_config['data_path']  # 原始文件路径
save_path = data_config['save_path'] # 文件保存地址

file_name = data_config['file_name'] # 目标信号名
file_path = os.path.join(data_path,file_name) # 故障信号地址
mat_data = scipy.io.loadmat(file_path) # 加载故障文件

# 带通滤波参数加载
fft_order = data_config['fft_order']
low_cut = data_config['low_cut']
high_cut = data_config['high_cut']


# 寻找目标信号
signal = None # 信号数据
key = ''  # 信号键值
for key_ in mat_data.keys():
    if 'DE_time' in key_:
        key = key_
        signal = mat_data[key_].flatten()

if signal is not None:
    # print('驱动端信号键名：',key)
    # print('原始信号信号类型：',type(signal))
    print('原始信号形状：',signal.shape)
else:
    print('未找到原始信号')


# 二、可视化信号

# 获取时域信号
fs = data_config['sampling_frequency'] #采样频率
time = np.linspace(0,len(signal)/fs,len(signal))

# 验证
# print(time)
# print(len(time))

# 绘制时域信号
plot_graph(time, signal, '原始时域图', '时间', '幅度', '原始时域图.png',xlim=(0,0.1))


# 三、初始频谱图分析

# 计算振幅
fft_amplitudes = (abs(np.fft.fft(signal))/len(signal))[:len(signal)//2]
# 计算频率轴
fft_frequencies = (np.fft.fftfreq(len(signal),1/fs))[:len(signal)//2]
# 绘制频域信号
plot_graph(fft_frequencies, fft_amplitudes, '原始频域图', '频率', '振幅', '原始频域图.png')


# 四、带通滤波
b, a = sig.butter(fft_order, [low_cut, high_cut], btype = 'bandpass', fs = fs)
signal_filtered = sig.filtfilt(b, a, signal)
print("带通滤波信号的形状:", signal_filtered.shape)

# 绘制时域信号
plot_graph(time, signal_filtered, '带通滤波时域图', '时间', '幅度', '带通滤波时域图.png',xlim=(0,0.1))

# 计算振幅
fft_amplitudes_filtered = (abs(np.fft.fft(signal_filtered))/len(signal_filtered))[:len(signal_filtered)//2]
# 计算频率轴
fft_frequencies_filtered = (np.fft.fftfreq(len(signal_filtered),1/fs))[:len(signal_filtered)//2]
plot_graph(fft_frequencies_filtered, fft_amplitudes_filtered, '带通滤波频域图', '频率', '振幅', '带通滤波频域图.png')


# 五、希尔伯特变换+包络谱分析
signal_analytic  = sig.hilbert(signal_filtered) # 希尔伯特变换
signal_envelope = np.abs(signal_analytic) # 包络谱分析
envelope_no_dc = signal_envelope - np.mean(signal_envelope)
print("包络信号的形状:", signal_envelope.shape)

plt.figure(figsize=(15,5))
plt.plot(time, signal_filtered, alpha=0.6, label='滤波信号')
plt.plot(time,signal_envelope, color='red', label='包络信号')
plt.title('带通滤波信号与包络')
plt.xlabel('时间')
plt.ylabel('振幅')
plt.xlim(0, 0.1)
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(save_path, '包络谱时域图.png'),dpi=300,bbox_inches='tight')
plt.show()

# 计算振幅
fft_amplitudes_envelope = (abs(np.fft.fft(envelope_no_dc))/len(envelope_no_dc))[:len(envelope_no_dc)//2]
# 计算频率轴
fft_frequencies_envelope = (np.fft.fftfreq(len(envelope_no_dc),1/fs))[:len(envelope_no_dc)//2]
plot_graph(fft_frequencies_envelope, fft_amplitudes_envelope, '包络谱频域图', '频率', '振幅', '包络谱频域图.png',xlim=(0,1000))


# 六、整合分析图
fig, axs = plt.subplots(2, 2, figsize=(20, 5))
fig.suptitle('轴承故障信号分析全流程', fontsize=16)

plot_subplot(axs[0, 0], time, signal, '1. 原始时域信号', '时间', '幅度',xlim=(0,0.1))
plot_subplot(axs[0, 1], fft_frequencies, fft_amplitudes, '2. 原始频谱图', '频率', '振幅')

# 子图3: 滤波与包络
axs[1, 0].plot(time, signal_filtered, alpha=0.6, label='滤波信号')
axs[1, 0].plot(time, signal_envelope, color='red', linewidth=2, label='包络')
axs[1, 0].set_title('3. 滤波信号与包络')
axs[1, 0].set_xlabel('时间')
axs[1, 0].set_ylabel('幅度')
axs[1, 0].set_xlim(0, 0.1)
axs[1, 0].legend()
axs[1, 0].grid(True)

# 准备子图4的注释
plot_subplot(axs[1, 1], fft_frequencies_envelope, fft_amplitudes_envelope, '4. 最终包络谱分析', '频率', '振幅',xlim=(0,500))

# 最终调整与显示
plt.tight_layout(rect=[0, 0, 1, 1]) # 调整布局，为总标题留出空间
plt.savefig(os.path.join(save_path, '分析全流程图.png'), dpi=300)
plt.show()

