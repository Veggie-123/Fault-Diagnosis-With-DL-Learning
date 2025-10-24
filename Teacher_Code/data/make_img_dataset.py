from utils.helpers import load_config
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.signal import stft
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

# 数据集分层抽样 划分数据集
def split_datasets(data_file_csv, split_rate = [0.7,0.2,0.1], random_state=42):
    '''
        分割数据集为训练集、验证集和测试集。

        参数:
        - data_file_csv: str, 数据集的CSV文件路径，数据应为2000行x1025列，最后一列为标签。
        - split_rate: list of float, 默认是 [0.7, 0.2, 0.1]，表示训练集、验证集和测试集的比例。
        - random_state: int, 默认是42，控制随机数生成以便复现。

        返回:
        - X_train, y_train: 训练集数据和标签
        - X_val, y_val: 验证集数据和标签
        - X_test, y_test: 测试集数据和标签
    '''
    # 确保 split_rate 的和为 1
    # 使用一个小容忍范围来处理浮点数精度问题
    assert abs(sum(split_rate) - 1) < 1e-9, "split_rate 的总和必须为 1"

    # 1.读取数据
    samples_data = pd.read_csv(data_file_csv)

    # 2.将数据和标签分开
    X = samples_data.iloc[:, :-1].values  # 数据部分
    y = samples_data.iloc[:, -1].values   # 标签部分

    # 3.首先划分 训练集 和 临时集（验证集 + 测试集）
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3 , random_state=42, stratify=y
    )

    # 4.然后从临时集中划分验证集和测试集
    test_size_temp = split_rate[2] / (split_rate[1] + split_rate[2])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp
    )

    # 打印各集合的大小
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")

    return  X_train, y_train, X_val, y_val, X_test, y_test


# 图片名称 对应  标签名称
# 对应版本 VGG16
# 输入：224x224的RGB 彩色图像
# 窗口大小 window_size = 64
# 重叠比例 overlap = 0.5

# 生成 STFT 时频图片
def makeTimeFrequencyImage(data, labels, folder, config):
    img_size = config['data']['image_size']

    if not os.path.exists(folder):
        os.makedirs(folder)

    for i, sample in enumerate(data):
        signal = np.array(sample)
        # 短时傅里叶变换参数
        # 采样频率 12 kHz
        fs = 12000
        # 窗口大小
        window_size = 64
        # 重叠比例
        overlap = 0.5
        # 计算重叠的样本数
        overlap_samples = int(window_size * overlap)
        # 进行短时傅里叶变换
        frequencies, times, magnitude = stft(signal, fs, nperseg=window_size, noverlap=overlap_samples)
        # 生成图片
        plt.pcolormesh(times, frequencies, np.abs(magnitude), shading='gouraud')
        plt.axis('off')  # 设置图像坐标轴不可见
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # 调整子图的位置和间距，将子图充满整个图像
        plt.margins(0, 0)  # 设置图像的边距为0，即没有额外的空白边框。
        plt.gcf().set_size_inches(img_size / 100, img_size / 100)  # 设置图像的大小，单位为英寸
        plt.savefig(os.path.join(folder, f'{i}_{labels[i]}.png'), bbox_inches='tight', pad_inches=0)
        plt.clf()  # 避免内存溢出
        plt.close()  # 释放内存

if __name__ == '__main__':
    # 加载配置文件-参数
    config = load_config("configs/config.yaml")

    # csv 路径
    csvfile_path_name = config['data']['csvfile_path_name']
    split_rate = config['data']['split_ratios']
    # 生成数据集
    X_train, y_train, X_val, y_val, X_test, y_test = split_datasets(csvfile_path_name, split_rate)

    # 时频图像-数据集路径
    img_output_dir = config['data']['img_output_dir']

    train_path = f'{img_output_dir}/train/'
    val_path = f'{img_output_dir}/val/'
    test_path = f'{img_output_dir}/test/'

    # 生成时频图片
    makeTimeFrequencyImage(X_train, y_train, train_path, config)
    makeTimeFrequencyImage(X_val, y_val, val_path, config)
    makeTimeFrequencyImage(X_test, y_test, test_path, config)

    # 可能会出现 警告，只要没报错停止， 就等程序运行结束 就行！！！
    # 程序运行耗时约 3分钟， 慢慢等待（时长跟自己电脑设备 也有关系）
    print("时频图像生成完毕！")