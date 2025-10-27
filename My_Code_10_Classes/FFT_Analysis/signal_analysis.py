import scipy.io
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.helpers import load_config
from scipy import signal as sig

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

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
    # print('信号类型：',type(signal))
    # print('信号形状：',signal.shape)
    print('故障信号长度：',len(signal))
else:
    print('未找到故障信号')

# 二、可视化信号

# 获取时域信号
fs = data_config['sampling_frequency'] #采样频率
time = np.linspace(0,len(signal)/fs,len(signal))

# 验证
# print(time)
# print(len(time))

# 绘制时域信号
# plt.figure(figsize=(15,5))
# plt.plot(time,signal)
# plt.title('原始时域信号')
# plt.xlabel('时间')
# plt.ylabel('幅度')
# plt.grid(True) # 显示网格
# plt.savefig(os.path.join(save_path, '原始时域信号.png'),dpi=300,bbox_inches='tight')
# plt.show()

# 三、初始频谱图分析

# 计算振幅
fft_amplitudes = (abs(np.fft.fft(signal))/len(signal))[:len(signal)//2]
# 计算频率轴
fft_frequencies = (np.fft.fftfreq(len(signal),1/fs))[:len(signal)//2]
# 绘制频域信号
# plt.figure(figsize=(15,5))
# plt.plot(fft_frequencies,fft_amplitudes)
# plt.title('原始频域图')
# plt.xlabel('频率')
# plt.ylabel('振幅')
# plt.grid(True)
# plt.savefig(os.path.join(save_path, '原始频域图.png'),dpi=300,bbox_inches='tight')
# plt.show()


# 四、带通滤波
b, a = sig.butter(fft_order, [low_cut, high_cut], btype = 'bandpass', fs = fs)
signal_filtered = sig.filtfilt(b, a, signal)

# 绘制时域信号
# plt.figure(figsize=(15,5))
# plt.plot(time,signal_filtered)
# plt.title('滤波时域信号')
# plt.xlabel('时间')
# plt.ylabel('幅度')
# plt.grid(True) # 显示网格
# plt.savefig(os.path.join(save_path, '滤波时域信号.png'),dpi=300,bbox_inches='tight')
# plt.show()

print("滤波后信号长度:", len(signal_filtered))
# 计算振幅
fft_amplitudes_filtered = (abs(np.fft.fft(signal_filtered))/len(signal_filtered))[:len(signal_filtered)//2]
# 计算频率轴
fft_frequencies_filtered = (np.fft.fftfreq(len(signal_filtered),1/fs))[:len(signal_filtered)//2]
# plt.figure(figsize=(15,5))
# plt.plot(fft_frequencies_filtered,fft_amplitudes_filtered)
# plt.title('滤波频域图')
# plt.xlabel('频率')
# plt.ylabel('振幅')
# plt.grid(True)
# plt.savefig(os.path.join(save_path, '滤波频域图.png'),dpi=300,bbox_inches='tight')
# plt.show()
