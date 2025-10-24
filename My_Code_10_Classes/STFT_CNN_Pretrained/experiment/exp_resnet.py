import torch.nn as nn
from torch.optim import Adam
import os

from experiment.exp_basic import ExpBasic 
from models.ResNet import ResNet18
from utils.data_loader import LabeledImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms

class ExpResNet(ExpBasic):
    def __init__(self, config):
        # super().__init__(config) 会调用父类 ExpBasic 的构造函数。
        # 这会运行父类中的所有初始化代码（比如 self.device 的设置），
        # 并接着调用下面被我们重写的 _build_model, _get_data 等方法。
        super(ExpResNet, self).__init__(config)

    def _build_model(self):
        """
        重写父类的占位符方法，提供具体的模型构建逻辑。
        """
        num_classes = self.config['model']['num_classes']
        use_pretrained = self.config['model']['pretrained']
        model = ResNet18(num_classes=num_classes, pretrained=use_pretrained)
        return model

    def _get_data(self):
        """
        重写父类的占位符方法，提供具体的数据加载逻辑。
        """
        data_config = self.config['data']
        train_config = self.config['train']

        img_output_dir = data_config['img_output_dir']
        image_size = data_config['image_size']
        batch_size = train_config['batch_size']

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]),
            'val': transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]),
        }

        train_dataset = LabeledImageDataset(path=os.path.join(img_output_dir, 'train'), transform=data_transforms['train'])
        val_dataset = LabeledImageDataset(path=os.path.join(img_output_dir, 'val'), transform=data_transforms['val'])
        test_dataset = LabeledImageDataset(path=os.path.join(img_output_dir, 'test'), transform=data_transforms['val'])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        return train_loader, val_loader, test_loader

    def _get_optimizer(self):
        """
        重写父类的占位符方法，提供具体的优化器。
        如果使用预训练模型，则为新层和预训练层设置不同的学习率。
        """
        learning_rate = self.config['train']['learning_rate']
        
        if self.config['model']['pretrained']:
            fine_tune_lr = self.config['train']['fine_tune_learning_rate']
            
            # 将模型的参数分为两组
            params_to_update_scratch = []  # 从头开始训练的参数 (conv1, fc)
            params_to_update_finetune = [] # 需要微调的预训练参数
            
            print("设置差异化学习率:")
            for name, param in self.model.named_parameters():
                if name.startswith('conv1') or name.startswith('fc'):
                    params_to_update_scratch.append(param)
                    print(f"  - [新层] {name} 使用学习率: {learning_rate}")
                else:
                    params_to_update_finetune.append(param)
            
            # 为两组参数配置不同的学习率
            optimizer = Adam([
                {'params': params_to_update_scratch, 'lr': learning_rate},
                {'params': params_to_update_finetune, 'lr': fine_tune_lr}
            ])
            print(f"  - [预训练层] 使用学习率: {fine_tune_lr}")
            
        else:
            # 如果不使用预训练模型，所有参数使用相同的学习率
            print(f"所有层使用统一学习率: {learning_rate}")
            optimizer = Adam(self.model.parameters(), lr=learning_rate)
            
        return optimizer

    def _get_criterion(self):
        """
        重写父类的占位符方法，提供具体的损失函数。
        """
        return nn.CrossEntropyLoss()

    # train_one_epoch 和 validate 方法现在被移到了父类 ExpBasic 中，
    # 所以我们不再需要在这里定义它们，可以直接删除。