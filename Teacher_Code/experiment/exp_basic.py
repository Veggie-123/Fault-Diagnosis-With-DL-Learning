import os
import torch

# 实验基类
# 引入了一些用于深度学习实验的基本方法和设置，模型构建,训练，验证，测试
class Exp_Basic(object):
    def __init__(self, config):
        self.config = config
        self.model = self._build_model()

    # 模型构建方法
    def _build_model(self):
        raise NotImplementedError
        return None

    def _get_data(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
