import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from utils.data_loader import LabeledImageDataset
from utils.helpers import load_config
from models.ResNet import ResNet18

config = load_config('configs/config.yaml')

data_config = config['data']
train_config = config['train']
model_config = config['model']

test_img_path = data_config['img_output_dir'] + '/test'

batch_size = train_config['batch_size']
num_classes = model_config['num_classes']
checkpoint_dir = train_config['checkpoint_dir']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])

test_dataset = LabeledImageDataset(path=test_img_path, transform=transform)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print(f"测试集大小：{len(test_dataset)}")
print(f"使用设备：{device}")

model = ResNet18(num_classes=num_classes).to(device)
model_path = os.path.join(checkpoint_dir, "best_model.pth")

model.load_state_dict(torch.load(model_path, weights_only=True))

model.eval()

print(f"已加载模型: {model_path}")

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='测试中'):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("测试完成!")

report = classification_report(all_labels, all_preds)
print("\n详细分类报告:")
print(report)

report_path = os.path.join(checkpoint_dir, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write(report)
print(f"\n分类报告已保存至: {report_path}")

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
cm_path = f"{config['train']['checkpoint_dir']}/confusion_matrix.png"
plt.savefig(cm_path)
print(f"\n混淆矩阵已保存至 {cm_path}")

plt.show()