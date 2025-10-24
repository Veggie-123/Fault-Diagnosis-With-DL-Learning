from utils.helpers import load_config
from experiment.exp_vgg import Experiment_VGG
import torch

# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 有GPU先用GPU训练

if __name__ == '__main__':
    # 加载配置文件-参数
    config = load_config("configs/config.yaml")

    # 初始化实验对象
    exp = Experiment_VGG(config)  # set experiments

    print('>>>>>>> test : >>>>>>>>>>>>>>>>>>>>>>>>>>')
    # 开始测试
    exp.test(device)