import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from collections import Counter

# ✅ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ✅ Transformations (with RGB conversion + ImageNet normalization)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Use ResNet default input size
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Lambda(lambda img: img.convert("RGB")),  # Force RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),  # Force RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ✅ Dataset paths
train_dir = "C:/Users/User/Documents/THESIS FILES/SmartWasteSegregator_CNN/waste_dataset/train"
test_dir = "C:/Users/User/Documents/THESIS FILES/SmartWasteSegregator_CNN/waste_dataset/test"

train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_data = datasets.ImageFolder(root=test_dir, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# ✅ Compute class weights (to handle imbalance)
class_counts = Counter([label for _, label in train_data.samples])
num_samples = sum(class_counts.values())
weights = [num_samples / class_counts[i] for i in range(len(train_data.classes))]
weights = torch.tensor(weights, dtype=torch.float).to(device)

print("Class counts:", class_counts)
print("Class weights:", weights)

# ✅ Model (ResNet18 with pretrained weights)
weights_enum = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights_enum)

# Freeze all layers except the last FC
for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_data.classes))  # trainable last layer
model = model.to(device)

# ✅ Loss & Optimizer
criterion = nn.CrossEntropyLoss(weight=weights)  # use weighted loss
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # optimize only last layer
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# ✅ Training + Evaluation
def train_model(epochs=30, patience=5):
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        # Adjust LR
        scheduler.step(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        # ---- Early Stopping + Save Best ----
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_weights.pth")
            print("✅ Best model updated & saved (weights only)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹ Early stopping triggered")
                break

    # ✅ Save full model
    torch.save(model, "cnn_full.pth")
    print(f"🎯 Training complete. Best Validation Accuracy: {best_acc:.2f}%")
    print("✅ Full model saved as cnn_full.pth")


# ✅ Run training
if __name__ == "__main__":
    train_model()
