from scipy.io import loadmat
import pandas as pd
from utils.helpers import load_config

# 分类 mat文件
file_names = ['0_0.mat', '7_1.mat', '7_2.mat', '7_3.mat', '14_1.mat', '14_2.mat', '14_3.mat', '21_1.mat',
              '21_2.mat', '21_3.mat']

# 采用驱动端数据
data_columns = ['X097_DE_time', 'X105_DE_time', 'X118_DE_time', 'X130_DE_time', 'X169_DE_time',
                'X185_DE_time', 'X197_DE_time', 'X209_DE_time', 'X222_DE_time', 'X234_DE_time']

columns_name = ['de_normal', 'de_7_inner', 'de_7_ball', 'de_7_outer', 'de_14_inner', 'de_14_ball', 'de_14_outer',
                'de_21_inner', 'de_21_ball', 'de_21_outer']


def split_data_with_overlap(data, time_steps, lable, overlap_ratio=0.5):
    """
        data:要切分的时间序列数据,可以是一个一维数组或列表。
        time_steps:切分的时间步长,表示每个样本包含的连续时间步数。
        lable: 表示切分数据对应 类别标签
        overlap_ratio:前后帧切分时的重叠率,取值范围为 0 到 1,表示重叠的比例。
    """
    stride = int(time_steps * (1 - overlap_ratio))  # 计算步幅
    samples = (len(data) - time_steps) // stride + 1  # 计算样本数
    # 用于存储生成的数据
    Clasiffy_dataFrame = pd.DataFrame(columns=[x for x in range(time_steps + 1)])

    data_list = []
    for i in range(samples):
        start_idx = i * stride
        end_idx = start_idx + time_steps
        temp_data = data[start_idx:end_idx].tolist()
        temp_data.append(lable)  # 对应哪一类
        data_list.append(temp_data)
    Clasiffy_dataFrame = pd.DataFrame(data_list, columns=Clasiffy_dataFrame.columns)
    return Clasiffy_dataFrame


# 数据集的制作
def make_samples(config):
    # 从配置字典中获取数据路径
    data_path = config['data']['data_path']  # 访问嵌套字典值
    truncated_value = config['data']['truncated_value']
    time_steps = config['data']['time_steps']
    overlap_ratio =  config['data']['overlap_ratio']

    # 用于存储生成的数据# 10个样本集合
    samples_data = pd.DataFrame(columns=[x for x in range(time_steps + 1)])
    # 记录类别标签
    label = 0

    for index in range(len(file_names)):
        # 读取MAT文件
        data = loadmat(f'{data_path}/{file_names[index]}')
        dataList = data[data_columns[index]].reshape(-1)
        # 截断数据集
        dataList = dataList[:truncated_value]
        # 划分样本点  window = 1024  overlap_ratio = 0.5  samples = 2000 每个类有200个样本
        split_data = split_data_with_overlap(dataList, time_steps, label, overlap_ratio)
        label += 1  # 类别标签递增
        samples_data = pd.concat([samples_data, split_data])

    # 随机打乱样本顺序
    # 打乱索引并重置索引
    samples_data = samples_data.sample(frac=1).reset_index(drop=True)

    return samples_data

if __name__ == '__main__':
    # 加载配置文件-参数
    config = load_config("configs/config.yaml")
    # 数据集制作
    samples_data = make_samples(config)

    # csv 输出目录
    csvfile_path_name = config['data']['csvfile_path_name']

    # 保存数据
    samples_data.to_csv(csvfile_path_name, index=False)

    print("数据形状-信息：")
    print(samples_data.shape)
    print(samples_data.head())
