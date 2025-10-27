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
        super(ExpResNet, self).__init__(config)

    def _build_model(self):
        """
        重写父类的占位符方法，提供具体的模型构建逻辑。
        """
        num_classes = self.config['model']['num_classes']
        model = ResNet18(num_classes=num_classes)
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
        """
        learning_rate = self.config['train']['learning_rate']
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        return optimizer

    def _get_criterion(self):
        """
        重写父类的占位符方法，提供具体的损失函数。
        """
        return nn.CrossEntropyLoss()

    # train_one_epoch 和 validate 方法现在被移到了父类 ExpBasic 中，
    # 所以我们不再需要在这里定义它们，可以直接删除。