import os
from torch.utils.data import Dataset
from PIL import Image

class LabeledImageDataset(Dataset):
    """
    通用的图像数据集类。
    从文件名 'name_label.ext' 中解析标签。
    """
    def __init__(self, path, transform=None):
        self.path = path
        self.files = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.path, self.files[idx])
        
        # 尝试以灰度图打开，如果失败则以RGB模式打开
        try:
            image = Image.open(img_name).convert('L')
        except IOError:
            image = Image.open(img_name).convert('RGB')
        
        label = int(self.files[idx].split('_')[-1].split('.')[0])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
