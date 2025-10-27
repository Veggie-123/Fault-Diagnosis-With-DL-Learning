import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 设置参数和文件路径 ---

# CWRU数据的采样频率 (Hz)
fs = 12000 

# .mat文件所在的目录
# 我们用os.path.join来构建路径，这样更规范
data_dir = '../data/matfiles'

# 我们先分析一个有代表性的故障文件：内圈、0.007英寸损伤
file_name = 'IR007_0.mat'
file_path = os.path.join(data_dir, file_name)

# --- 2. 加载数据 ---

# 使用scipy.io.loadmat加载.mat文件
mat_data = scipy.io.loadmat(file_path)

# --- 3. 提取驱动端(DE)振动信号 ---

# .mat文件加载后是一个字典，我们需要找到存储信号的那个键(key)
# CWRU数据的键名通常是 'X..._DE_time' 的格式
# 为了找到正确的键名，我们可以先打印出所有的键
print(f"文件 '{file_name}' 中包含的键: {mat_data.keys()}")

# 从经验来看，文件名和数据键名有一个对应关系，我们先用一个假设的键名
# 稍后你可以根据上面打印出的实际键名来修改它
data_key = [key for key in mat_data.keys() if 'DE_time' in key][0]
signal = mat_data[data_key].flatten() # 使用.flatten()确保它是一个一维数组

print(f"成功加载信号，键名为: '{data_key}'")
print(f"信号长度: {len(signal)} 个数据点")
