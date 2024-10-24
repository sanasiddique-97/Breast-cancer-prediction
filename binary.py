import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class ASDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.labels = [0 if 'healthy' in f else 1 for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = ASDataset(data_dir='PATH_TO_DATA', transform=transform)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.vgg16.classifier[6].in_features, 2)
        )

    def forward(self, x):
        return self.vgg16(x)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies, precisions, recalls, f1_scores = [], [], [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):
    print(f'Fold {fold + 1}/{kf.n_splits}')

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    model = BinaryClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f'Fold {fold + 1} Accuracy: {accuracy}')
    print(f'Fold {fold + 1} Precision: {precision}')
    print(f'Fold {fold + 1} Recall: {recall}')
    print(f'Fold {fold + 1} F1 Score: {f1}')

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

print('Cross-Validation Results:')
print(f'Average Accuracy: {np.mean(accuracies)}')
print(f'Average Precision: {np.mean(precisions)}')
print(f'Average Recall: {np.mean(recalls)}')
print(f'Average F1 Score: {np.mean(f1_scores)}')
