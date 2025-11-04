import os
from random import shuffle
from turtle import mode
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm 
import matplotlib
import matplotlib.pyplot as plt

from utils.data_loader import LabeledImageDataset
from utils.helpers import load_config
from models.AttentionCNN import AttentionCNN

config = load_config('configs/config.yaml')

data_config = config['data']
train_config = config['train']
model_config = config['model']

train_img_path = data_config['img_output_dir'] + '/train'
val_img_path = data_config['img_output_dir'] + '/val'

batch_size = train_config['batch_size']
num_epochs = train_config['num_epochs']
learning_rate = train_config['learning_rate']
checkpoint_dir = train_config['checkpoint_dir']
num_classes = model_config['num_classes']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备:{device}')

os.makedirs(checkpoint_dir, exist_ok=True)
print(f'模型将保存至: {checkpoint_dir}')

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = LabeledImageDataset(path=train_img_path, transform=transform)
val_dataset = LabeledImageDataset(path=val_img_path, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

model = AttentionCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 添加学习率调度器 - 验证损失不下降时自动降低学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

def train_one_epoch(model, train_loader, criterion, optimizer, device): 
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc='训练中')

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='验证中')
        
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc

print("\n开始训练...\n")
print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"Batch Size: {batch_size}")
print(f"总Epochs: {num_epochs}\n")

best_val_acc = 0.0

# 指标记录
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

# 实时绘图初始化（若环境不支持可忽略）
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Training Progress')

def update_plot():
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.clear(); ax2.clear()
    ax1.plot(epochs, history['train_loss'], label='train_loss')
    ax1.plot(epochs, history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(epochs, history['train_acc'], label='train_acc')
    ax2.plot(epochs, history['val_acc'], label='val_acc')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Acc (%)'); ax2.legend(); ax2.grid(True)
    fig.tight_layout()
    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
    except Exception:
        pass

for epoch in range(num_epochs):
    print(f"\n{'='*50}")
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"{'='*50}")
    
    train_loss, train_acc = train_one_epoch(model=model, train_loader=train_loader, 
                                criterion=criterion, optimizer = optimizer, device=device)
    val_loss, val_acc = validate(model=model, val_loader=val_loader, 
                            criterion=criterion, device=device)
    
    # 改进4: 使用学习率调度器
    scheduler.step(val_loss)
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"\n训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
    print(f"当前学习率: {current_lr:.6f}")

    # 记录历史并更新曲线
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    update_plot()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
print("\n" + "="*50)
print("训练完成!")
print(f"最佳验证准确率: {best_val_acc:.2f}%")
print("="*50)

# 训练结束后，保存最终曲线图
final_curve_path = os.path.join(checkpoint_dir, 'training_curves.png')
fig.savefig(final_curve_path, dpi=150)
plt.ioff()
try:
    plt.close(fig)
except Exception:
    pass