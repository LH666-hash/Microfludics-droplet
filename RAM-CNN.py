import os
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.optim as optim
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor
import torch.nn.functional as F
import numpy as np


class GrayImageDataset(Dataset):
    def __init__(self, root_dir, image_size=(800, 512)):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.resize_transform = Resize(image_size)

        for root, dirs, files in os.walk(root_dir):
            for dir in dirs:
                total_dir = os.path.join(root, dir)
                label = dir
                for file in os.listdir(total_dir):
                    image_path = os.path.join(total_dir, file)
                    self.image_paths.append(image_path)
                    self.labels.append(int(label))  # 将标签转换为整数

    def add_salt_and_pepper_noise(self, image, salt_prob=0.01, pepper_prob=0.01):
        """向图像添加椒盐噪声"""
        noise_mask = np.random.rand(*image.shape[1:])
        salt_mask = noise_mask < salt_prob
        pepper_mask = noise_mask > (1 - pepper_prob)

        image[:, salt_mask] = 1.0  # 盐噪声
        image[:, pepper_mask] = 0.0  # 椒噪声

        return image

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        pil_image = Image.open(image_path).convert('L')  # 转换为灰度图像
        pil_image = self.resize_transform(pil_image)  # 缩放图像
        image = pil_to_tensor(pil_image)
        image = image.to(torch.float32)

        # 添加椒盐噪声
        image = self.add_salt_and_pepper_noise(image.numpy())
        image = torch.from_numpy(image)

        return image, label

    def __len__(self):
        return len(self.image_paths)


def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack([img for img in images])
    labels = torch.tensor(labels)



    return images, labels


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(10,4), stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(10,4), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=(10,4), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # (B, HW, C//8)
        key = self.key_conv(x).view(batch_size, -1, height * width)  # (B, C//8, HW)
        value = self.value_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # (B, HW, C)

        attention_scores = F.softmax(torch.bmm(query, key), dim=-1)  # (B, HW, HW)
        out = torch.bmm(attention_scores, value).permute(0, 2, 1).view(batch_size, channels, height,
                                                                       width)  # (B, C, H, W)
        out = self.gamma * out + x

        return out


class DualBranchCNN(nn.Module):
    def __init__(self):
        super(DualBranchCNN, self).__init__()

        # 大感受野分支
        self.branch_large = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(10, 4), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(4, 4, kernel_size=(10, 4), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(4, 6, kernel_size=(10, 4), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(6, 6, kernel_size=(10, 4), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            ResidualBlock(6, 6),  # 添加残差块
            SelfAttention(6)  # 添加自注意力机制
        )

        # 小感受野分支
        self.branch_small = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(4, 4, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(4, 6, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(6, 6, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            ResidualBlock(6, 6),  # 添加残差块
            SelfAttention(6)  # 添加自注意力机制
        )
        # 上采样层
        self.upsample = nn.Upsample(size=(102, 66), mode='bilinear', align_corners=True)

        # 全连接层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(75888, 10000)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10000, 1000)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1000, 100)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(100, 10)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(10, 3)

    def forward(self, x):
        x = x.float()

        # 大感受野分支
        x_large = self.branch_large(x)

        # 小感受野分支
        x_small = self.branch_small(x)
        #print(x_large.shape,x_small.shape)
        # 上采样小感受野分支的输出
        x_small = self.upsample(x_small)
        x_small = F.interpolate(x_small, size=(93, 68), mode='bilinear', align_corners=False)

        # 特征图叠加
        x = torch.cat((x_large, x_small), dim=1)
        # 全连接层
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)

        return x



def train(model, train_loader, val_loader, criterion, optimizer, epochs, patience=15):
    best_val_loss = float('inf')
    no_improvement_count = 0
    best_model_weights = None

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            labels = labels.long()  # 转换为长整型
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}')

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.long()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        val_accuracy = 100 * correct / total
        val_loss_avg = val_loss / len(val_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss_avg}, Validation Accuracy: {val_accuracy}%')
        cm = confusion_matrix(all_labels, all_predictions)
        print("Validation Confusion Matrix:")
        print(cm)

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            no_improvement_count = 0
            best_model_weights = model.state_dict()  # 记录最佳模型的权重
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(
                    f'Early stopping at epoch {epoch + 1} due to no improvement in validation loss for {patience} epochs.')
                break

    # 保存最佳模型
    torch.save(best_model_weights, 'best_model.pth')


def test(model, test_loader):
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.long()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 10
    learning_rate = 0.001
    epochs = 100

    root_dir = r"F:\lh\classfy-2"  # 确保路径在不同系统上都有效
    total_dataset = GrayImageDataset(root_dir=root_dir)
    train_dataset, test_dataset = Data.random_split(total_dataset,
                                                    lengths=[int(0.9 * len(total_dataset)),
                                                             len(total_dataset) - int(0.9 * len(total_dataset))],
                                                    generator=torch.Generator().manual_seed(0))
    train_dataset, val_dataset = Data.random_split(train_dataset,
                                                   lengths=[int(0.8 * len(train_dataset)),
                                                            len(train_dataset) - int(0.8 * len(train_dataset))],
                                                   generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = DualBranchCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, train_loader, val_loader, criterion, optimizer, epochs, patience=15)
    test(model, test_loader)