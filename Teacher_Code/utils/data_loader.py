import os
from torch.utils.data import Dataset
from PIL import Image

# 自定义数据集加载器
class SpectrogramDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.files = os.listdir(path)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.path, self.files[idx])
        image = Image.open(img_name).convert('RGB')
        label = int(self.files[idx].split('_')[1].split('.png')[0])
        if self.transform:
            image = self.transform(image)
        return image, label
