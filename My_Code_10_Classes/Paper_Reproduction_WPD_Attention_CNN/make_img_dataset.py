# 导入os模块，用于处理文件和目录路径
import os
# 导入warnings模块，用于控制警告信息的显示
import warnings
# 导入matplotlib.pyplot，用于绘图
import matplotlib.pyplot as plt
# 导入numpy，用于进行高效的数值计算
import numpy as np
# 导入pandas，用于读取和处理CSV文件
import pandas as pd
# 导入pywt库，用于小波包分解
import pywt
# 从sklearn.model_selection导入train_test_split，用于划分数据集
from sklearn.model_selection import train_test_split
# 从我们自己编写的utils工具包中，导入load_config函数
from utils.helpers import load_config
# 从matplotlib导入MatplotlibDeprecationWarning，用于特定警告类型的处理
from matplotlib import MatplotlibDeprecationWarning
# 设置警告过滤器，忽略掉Matplotlib的弃用警告，保持输出整洁
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


def split_datasets(data_file_csv, split_rate, random_state=42):
    """
    本函数负责将数据集划分为训练集、验证集和测试集。

    参数:
        data_file_csv (str): 包含所有样本数据的CSV文件的路径。
        split_rate (list of float): 一个包含三个浮点数的列表，分别代表训练集、验证集和测试集的比例。
        random_state (int): 随机数种子，确保每次划分的结果都一样，便于复现实验。

    返回:
        tuple: 返回一个元组，包含了划分好的六个部分：
               (训练集特征, 训练集标签, 验证集特征, 验证集标签, 测试集特征, 测试集标签)
    """
    # 确保 split_rate 的和为 1
    assert abs(sum(split_rate) - 1) < 1e-9, "split_rate 的总和必须为 1"

    # 1.读取数据
    samples_data = pd.read_csv(data_file_csv)
    # 2.将数据和标签分开
    X = samples_data.iloc[:, :-1].values
    y = samples_data.iloc[:, -1].values

    # 3.首先划分 训练集 和 临时集（验证集 + 测试集）
    # 从配置中获取训练集的比例
    train_ratio = split_rate[0]
    # 计算验证集和测试集的总比例
    val_test_ratio = 1 - train_ratio

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_test_ratio, random_state=random_state, stratify=y
    )

    # 4.然后从临时集中划分验证集和测试集
    # 从配置中获取验证集和测试集的原始比例
    val_ratio = split_rate[1]
    test_ratio = split_rate[2]
    # 计算在临时集中，测试集应该占的比例
    test_size_in_temp = test_ratio / (val_ratio + test_ratio)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size_in_temp, random_state=random_state, stratify=y_temp
    )

    # 5.打印各个数据集的大小，方便确认
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")

    # 6.返回所有划分好的数据
    return X_train, y_train, X_val, y_val, X_test, y_test


def make_wavelet_packet_images(data, labels, folder, config):
    """
    本函数负责将信号数据通过小波包变换转换成时频图并保存。

    参数:
        data (np.ndarray): 输入的信号数据是一个numpy数组。
        labels (np.ndarray): 对应信号数据的标签。
        folder (str): 保存生成图像的文件夹路径。
        config (dict): 包含所有配置参数的字典。
    """
    # 1.从配置文件中读取相关参数
    img_size = config['data']['image_size']              # 图像尺寸
    wavelet = config['data']['wavelet_type']
    decomp_level = config['data']['decomp_level']

    # 2.检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 4.遍历数据集中的每一个信号样本
    for i, signal in enumerate(data):
        # 执行小波包分解
        coeffs = pywt.wavedec(signal, wavelet, level=decomp_level)
        # 重构信号
        reconstructed_signal = pywt.waverec(coeffs, wavelet)

        # 5.绘制时频图
        plt.figure(figsize=(img_size / 100, img_size / 100))
        plt.plot(reconstructed_signal)
        # 关闭坐标轴的显示，因为在图像识别中坐标轴是无用信息
        plt.axis('off')
        # 调整子图边距，让图像填满整个画布
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # 去除图像周围的白边
        plt.margins(0, 0)
        # 设置图像的实际大小（以英寸为单位），img_size/100是为了将像素转换为英寸
        plt.gcf().set_size_inches(img_size / 100, img_size / 100)
        
        # 6.保存图像
        # 构建图像的完整保存路径和文件名，文件名格式为 "索引_标签.png"
        image_path = os.path.join(folder, f'{i}_{labels[i]}.png')
        # 保存图像。bbox_inches='tight', pad_inches=0 确保保存的图像没有额外的白边
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        
        # 7.清理和关闭图像，防止在循环中占用过多内存
        plt.clf()     # 清除当前图形
        plt.close()   # 关闭当前窗口

if __name__ == '__main__':
    # 1. 加载配置文件 
    # 调用我们自己写的load_config函数，读取YAML配置文件
    config = load_config("configs/config.yaml")

    # 2. 划分数据集 
    # 从配置中获取CSV文件路径和划分比例
    csv_path = config['data']['csv_file_path']
    split_ratios = config['data']['split_ratios']
    # 调用split_datasets函数，得到划分好的数据
    X_train, y_train, X_val, y_val, X_test, y_test = split_datasets(csv_path, split_ratios)

    # 3. 准备图像输出目录 
    # 从配置中获取总的图像输出目录
    img_output_dir = config['data']['img_output_dir']
    # 分别构建训练集、验证集、测试集图像的保存路径
    train_path = os.path.join(img_output_dir, 'train')
    val_path = os.path.join(img_output_dir, 'val')
    test_path = os.path.join(img_output_dir, 'test')

    # 4. 循环生成各个数据集的时频图 
    print("开始生成训练集图像...")
    make_wavelet_packet_images(X_train, y_train, train_path, config)
    
    print("开始生成验证集图像...")
    make_wavelet_packet_images(X_val, y_val, val_path, config)
    
    print("开始生成测试集图像...")
    make_wavelet_packet_images(X_test, y_test, test_path, config)

    # 5. 任务完成提示 
    print("时频图像生成完毕！")
