import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, Subset
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import ResNet18_Weights
import json

# Set cuDNN flags
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# Load the CSV file
csv_path = 'merged_dataset.csv'
df = pd.read_csv(csv_path)

# Custom dataset class
class CarDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['image_name']  # Using the "image_name" column
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = self.dataframe.iloc[idx]['make_id']  # Assuming the column name is "make_id"

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transforms for data augmentation and normalization
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom function to train and validate the model
def train_validate(model, criterion, optimizer, train_loader, val_loader, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    model.eval()
    validation_running_loss = 0.0
    validation_correct = 0
    validation_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            validation_total += labels.size(0)
            validation_correct += (predicted == labels).sum().item()

    val_loss = validation_running_loss / len(val_loader)
    val_accuracy = 100 * validation_correct / validation_total

    return train_loss, train_accuracy, val_loss, val_accuracy

# Function to train the model with cross-validation
def train():
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create the dataset
    dataset = CarDataset(dataframe=df, root_dir='data', transform=transform_train)

    # Cross-validation setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels = df['make_id'].values

    all_train_losses = []
    all_train_accuracies = []
    all_val_losses = []
    all_val_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"Fold {fold+1}/{skf.n_splits}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

        # Use a pre-trained ResNet18 model
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Dropout(0.7),
            nn.Linear(num_ftrs, df['make_id'].nunique())
        )
        model = model.to(device)

        # Loss function and optimizer with stronger L2 regularization
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        scaler = GradScaler()

        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        num_epochs = 20
        fold_train_losses = []
        fold_train_accuracies = []
        fold_val_losses = []
        fold_val_accuracies = []

        for epoch in range(num_epochs):
            train_loss, train_accuracy, val_loss, val_accuracy = train_validate(
                model, criterion, optimizer, train_loader, val_loader, device, scaler
            )

            fold_train_losses.append(train_loss)
            fold_train_accuracies.append(train_accuracy)
            fold_val_losses.append(val_loss)
            fold_val_accuracies.append(val_accuracy)

            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'best_car_make_classifier_fold{fold+1}.pth')
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping")
                break

        all_train_losses.append(fold_train_losses)
        all_train_accuracies.append(fold_train_accuracies)
        all_val_losses.append(fold_val_losses)
        all_val_accuracies.append(fold_val_accuracies)

    print("Finished Cross-Validation")

    # Save training and validation data to JSON file
    results = {
        'train_losses': all_train_losses,
        'train_accuracies': all_train_accuracies,
        'val_losses': all_val_losses,
        'val_accuracies': all_val_accuracies
    }

    with open('training_results.json', 'w') as f:
        json.dump(results, f)

# Function to plot the results
def plot():
    # Load the saved training results
    with open('training_results.json', 'r') as f:
        results = json.load(f)

    train_losses = results['train_losses']
    val_losses = results['val_losses']

    # Plot training and validation loss on the same plot
    plt.figure()
    for i, (fold_train_losses, fold_val_losses) in enumerate(zip(train_losses, val_losses)):
        epochs = range(1, len(fold_train_losses) + 1)
        plt.plot(epochs, fold_train_losses, label=f'Training Loss Fold {i+1}')
        plt.plot(epochs, fold_val_losses, label=f'Validation Loss Fold {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()

# Protect the entry point
if __name__ == '__main__':
    train()
    plot()
