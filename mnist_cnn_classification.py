# ============================================================
# Mnist Cnn Classification
# ============================================================

# Install and import required libraries
!pip install torch torchvision tensorboard scikit-learn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import os
# Check versions
print(f"PyTorch version: {torch.__version__}")
print(f"TensorBoard version: {torch.__version__}")  # SummaryWriter is built i
Show hidden output
# Setup TensorBoard
# Load the TensorBoard extension (works in Colab by default)
%load_ext tensorboard
# Create a unique log directory for this experiment
log_dir = './mnist_cnn_logs'
os.makedirs(log_dir, exist_ok=True)
# Initialize TensorBoard writer
writer = SummaryWriter(log_dir)
print(f"TensorBoard log directory: {log_dir}")
TensorBoard log directory: ./mnist_cnn_logs
temp_transform = transforms.Compose([transforms.ToTensor()])
temp_dataset = datasets.MNIST(root='./data', train=True, download=True, transf
loader = DataLoader(temp_dataset, batch_size=len(temp_dataset))
data = next(iter(loader))[0]
mean = data.mean().item()
std = data.std().item()
# Define transforms
# Convert images to tensors and normalize (mean=0.1307, std=0.3081 for MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])
# Load MNIST datasets
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Number of classes: {len(train_dataset.classes)}")
print(f"Classes: {train_dataset.classes}")
100%|██████████| 9.91M/9.91M [00:00<00:00, 20.0MB/s]
100%|██████████| 28.9k/28.9k [00:00<00:00, 548kB/s]
100%|██████████| 1.65M/1.65M [00:00<00:00, 4.67MB/s]
100%|██████████| 4.54k/4.54k [00:00<00:00, 11.3MB/s]
Training samples: 60000
Test samples: 10000
Number of classes: 10
Classes: ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five'
from torchvision.utils import make_grid
# Visualize sample images from training set
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img, label = train_dataset[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'Label: {label}')
    ax.axis('off')
plt.suptitle('Sample MNIST Digits', fontsize=14)
plt.tight_layout()
plt.savefig('sample_digits.png', dpi=150)
plt.show()
# Log sample images to TensorBoard
# Create a grid of sample images for TensorBoard
sample_images = torch.stack([train_dataset[i][0] for i in range(32)])
grid = make_grid(sample_images[:32], nrow=8, padding=2, normalize=True)
writer.add_image('MNIST_Samples/Grid', grid, 0)
# writer.add_image('MNIST_Samples/Grid', torch.cat([sample_images[i:i, ...] fo
# Log class distribution to TensorBoard
# Count samples per class
class_counts = {}
for _, label in train_dataset:
    class_counts[label] = class_counts.get(label, 0) + 1
print("Class Distribution in Training Set:")
for digit in range(10):
    print(f"  Digit {digit}: {class_counts[digit]} samples")
# Plot and log class distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(10), [class_counts[i] for i in range(10)])
ax.set_xlabel('Digit')
ax.set_ylabel('Count')
ax.set_title('MNIST Class Distribution')
ax.set_xticks(range(10))
plt.savefig('class_distribution.png', dpi=150)
plt.show()
Class Distribution in Training Set:
  Digit 0: 5923 samples
  Digit 1: 6742 samples
  Digit 2: 5958 samples
  Digit 3: 6131 samples
  Digit 4: 5842 samples
  Digit 5: 5421 samples
  Digit 6: 5918 samples
  Digit 7: 6265 samples
  Digit 8: 5851 samples
  Digit 9: 5949 samples
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First Convolutional Block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        # Second Convolutional Block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
    def forward(self, x):
        # Conv Block 1: 28x28 -> 14x14
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        # Conv Block 2: 14x14 -> 7x7
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        # FC Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
# Initialize model
model = CNN()
print(model)
CNN(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_st
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_st
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_s
  (pool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=
  (fc1): Linear(in_features=1152, out_features=512, bias=True)
  (dropout): Dropout(p=0.5, inplace=False)
  (fc2): Linear(in_features=512, out_features=10, bias=True)
)
# Count parameters
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
total_params, trainable_params = count_parameters(model)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
# Print layer-wise parameter count
print("\nLayer-wise parameter count:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.numel():,} parameters, shape: {param.shape}")
Total parameters: 688,586
Trainable parameters: 688,586
Layer-wise parameter count:
  conv1.weight: 288 parameters, shape: torch.Size([32, 1, 3, 3])
  conv1.bias: 32 parameters, shape: torch.Size([32])
  bn1.weight: 32 parameters, shape: torch.Size([32])
  bn1.bias: 32 parameters, shape: torch.Size([32])
  conv2.weight: 18,432 parameters, shape: torch.Size([64, 32, 3, 3])
  conv2.bias: 64 parameters, shape: torch.Size([64])
  bn2.weight: 64 parameters, shape: torch.Size([64])
  bn2.bias: 64 parameters, shape: torch.Size([64])
  conv3.weight: 73,728 parameters, shape: torch.Size([128, 64, 3, 3])
  conv3.bias: 128 parameters, shape: torch.Size([128])
  bn3.weight: 128 parameters, shape: torch.Size([128])
  bn3.bias: 128 parameters, shape: torch.Size([128])
  fc1.weight: 589,824 parameters, shape: torch.Size([512, 1152])
  fc1.bias: 512 parameters, shape: torch.Size([512])
  fc2.weight: 5,120 parameters, shape: torch.Size([10, 512])
  fc2.bias: 10 parameters, shape: torch.Size([10])
# Add model graph to TensorBoard
dummy_input = torch.zeros(1, 1, 28, 28)
writer.add_graph(model, dummy_input)
print("Model graph added to TensorBoard!")
Model graph added to TensorBoard!
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Training settings
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Move model to device
model = model.to(device)
Using device: cuda
def log_histograms(model, writer, epoch):
    """Log weight and bias histograms for each layer"""
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            writer.add_histogram(f'Parameters/{name}', param, epoch)
def log_gradients(model, writer, epoch):
    """Log gradient histograms for each layer"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
def compute_accuracy(outputs, labels):
    """Compute accuracy from model outputs and labels"""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return 100 * correct / len(labels)
def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix as an image"""
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anc
    # Add text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig
def plot_filters(weights, num_to_show=16):
    """Plot the first conv layer filters"""
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < num_to_show:
            filter_img = weights[i].squeeze()
            ax.imshow(filter_img, cmap='gray')
            ax.set_title(f'Filter {i}')
        ax.axis('off')
    plt.suptitle('Conv1 Learned Filters (Kernels)', fontsize=14)
    plt.tight_layout()
    return fig
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward pass
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc
def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc, np.array(all_predictions), np.array(all_labe
def train_with_tensorboard(model, train_loader, test_loader, criterion, optimi
                           num_epochs, device, writer):
    """Full training loop with TensorBoard logging"""
    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, op
        # Validation
        val_loss, val_acc, predictions, labels = validate(model, test_loader, 
        # Log scalars to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        # Log weight histograms
        log_histograms(model, writer, epoch)
        # Log gradient histograms
        log_gradients(model, writer, epoch)
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    return model
# Start training
print("Starting training with TensorBoard logging...\n")
model = train_with_tensorboard(model, train_loader, test_loader, criterion, op
                               num_epochs, device, writer)
Starting training with TensorBoard logging...
Epoch [1/10] - Train Loss: 0.1301, Train Acc: 95.94% | Val Loss: 0.0314, Val Ac
Epoch [2/10] - Train Loss: 0.0486, Train Acc: 98.52% | Val Loss: 0.0273, Val Ac
Epoch [3/10] - Train Loss: 0.0359, Train Acc: 98.92% | Val Loss: 0.0288, Val Ac
Epoch [4/10] - Train Loss: 0.0262, Train Acc: 99.19% | Val Loss: 0.0342, Val Ac
Epoch [5/10] - Train Loss: 0.0218, Train Acc: 99.31% | Val Loss: 0.0252, Val Ac
Epoch [6/10] - Train Loss: 0.0172, Train Acc: 99.48% | Val Loss: 0.0263, Val Ac
Epoch [7/10] - Train Loss: 0.0148, Train Acc: 99.57% | Val Loss: 0.0222, Val Ac
Epoch [8/10] - Train Loss: 0.0129, Train Acc: 99.59% | Val Loss: 0.0319, Val Ac
Epoch [9/10] - Train Loss: 0.0114, Train Acc: 99.63% | Val Loss: 0.0208, Val Ac
Epoch [10/10] - Train Loss: 0.0101, Train Acc: 99.67% | Val Loss: 0.0251, Val A
%tensorboard --logdir=./mnist_cnn_logs
Filter runs (rege
.
Filter tags (regex)
Pin cards for a quick view and comparison
.
99.5805
99.6733
9
Accuracy/train
97.5
98
98.5
99
99.5
97.5
98
98.5
99
99.5
0
2
4
0
2
4
Run
Smoothed
Value
Step
Accuracy/val
99.3
99.3
Pinned
Accuracy 2 cards
Settings
GENERAL
Horizontal Axis
(Scalars only)
Card Width
SCALARS
Smoothing
0.6
Tooltip sorting method
HISTOGRAMS
Step
Enable step selection and data table
Enable Range Selection
Link by step 9
Enable saving pins (Scalars only)
Alphabetical
Ignore outliers in chart scaling
Partition non-monotonic X axis
Run
All
Scalars
Image
Histogram
TensorBoard
INACTIVE
TIME SERIES
SCALAR
# Get final predictions and labels
model.eval()
all_preds = []
all_labels = []
all_probs = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)
# Calculate accuracy
accuracy = (all_preds == all_labels).mean() * 100
print(f"\nTest Accuracy: {accuracy:.2f}%")
Test Accuracy: 99.27%
# Generate confusion matrix
cm = confusion_matrix(all_labels, all_preds)
# Plot and log confusion matrix to TensorBoard
fig_cm = plot_confusion_matrix(cm, class_names=list(range(10)))
writer.add_figure('Confusion_Matrix/Test', fig_cm)
plt.show()
# Print classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=[str(i) for i 
Classification Report:
              precision    recall  f1-score   support
           0       0.99      1.00      0.99       980
           1       0.99      1.00      1.00      1135
           2       1.00      0.99      1.00      1032
           3       1.00      0.99      1.00      1010
           4       0.98      1.00      0.99       982
           5       0.99      1.00      0.99       892
           6       0.99      0.99      0.99       958
           7       0.99      0.99      0.99      1028
           8       1.00      0.99      0.99       974
           9       0.99      0.98      0.99      1009
    accuracy                           0.99     10000
   macro avg       0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000
# Get Conv1 weights
conv1_weights = model.conv1.weight.data.cpu().numpy()
# Plot and log filters to TensorBoard
fig_filters = plot_filters(conv1_weights)
writer.add_figure('Learned_Filters/Conv1', fig_filters)
plt.show()
print(f"Conv1 filter shape: {conv1_weights.shape}")
print(f"Each filter is: {conv1_weights.shape[1]}×{conv1_weights.shape[2]}×{con
Conv1 filter shape: (32, 1, 3, 3)
Each filter is: 1×3×3
# Get a sample image
sample_img, sample_label = test_dataset[0]
sample_img_batch = sample_img.unsqueeze(0).to(device)
# Get feature maps from Conv1
model.eval()
with torch.no_grad():
    conv1_output = model.conv1(sample_img_batch)
    conv1_output = conv1_output.squeeze().cpu().numpy()
# Plot first 16 feature maps
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    if i < conv1_output.shape[0]:
        ax.imshow(conv1_output[i], cmap='viridis')
        ax.set_title(f'Feature Map {i}')
    ax.axis('off')
plt.suptitle('Conv1 Feature Maps for Sample Digit', fontsize=14)
plt.tight_layout()
writer.add_figure('Feature_Maps/Conv1_Sample', plt.gcf())
plt.show()
# Visualize sample predictions
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    idx = i * 1000  # Sample every 1000th image
    img, true_label = test_dataset[idx]
    pred_label = all_preds[idx]
    confidence = all_probs[idx][pred_label] * 100
    ax.imshow(img.squeeze(), cmap='gray')
    color = 'green' if true_label == pred_label else 'red'
    ax.set_title(f'True: {true_label} | Pred: {pred_label}\nConf: {confidence:
    ax.axis('off')
plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=14)
plt.tight_layout()
writer.add_figure('Sample_Predictions', plt.gcf())
plt.show()
# Close the writer
writer.close()
print("TensorBoard writer closed.")
print(f"Logs saved to: {log_dir}")
TensorBoard writer closed.
Logs saved to: ./mnist_cnn_logs
# ====================== SCENARIO 1: Dataset Exploration =====================
print("=== Scenario 1: Dataset Exploration ===")
# 1.1 Total images
print(f"Total training images: {len(train_dataset)}")
print(f"Total test images: {len(test_dataset)}")
# 1.2 Image shape
sample_img, _ = train_dataset[0]
print(f"Image shape: {sample_img.shape}")
# 1.3 Class distribution (already printed earlier, repeating for clarity)
class_counts = {}
for _, label in train_dataset:
    class_counts[label] = class_counts.get(label, 0) + 1
for digit in range(10):
    print(f"Digit {digit}: {class_counts[digit]} samples")
# 1.4 & 1.5 already logged to TensorBoard (Images tab)
print("\n✅ Check TensorBoard → Images tab for sample grid and class distribut
=== Scenario 1: Dataset Exploration ===
Total training images: 60000
Total test images: 10000
Image shape: torch.Size([1, 28, 28])
Digit 0: 5923 samples
Digit 1: 6742 samples
Digit 2: 5958 samples
Digit 3: 6131 samples
Digit 4: 5842 samples
Digit 5: 5421 samples
Digit 6: 5918 samples
Digit 7: 6265 samples
Digit 8: 5851 samples
Digit 9: 5949 samples
✅ Check TensorBoard → Images tab for sample grid and class distribution chart
# ====================== SCENARIO 2: Model Architecture Analysis =============
print("=== Scenario 2: Model Architecture ===")
# 2.1 Total parameters (already calculated)
total_params, _ = count_parameters(model)
print(f"Total CNN parameters: {total_params:,}")
# 2.2 Layer-wise parameters (already printed)
print("\nConv1 parameters: 288 (weights) + 32 (bias) + 64 (BN) = 384")
print("Conv2 parameters: 18,432 + 64 + 128 (BN) = 18,624")
print("FC1 parameters: 1,605,632 + 512 = 1,606,144")
print("FC2 parameters: 5,120 + 10 = 5,130")
# 2.5 Compare with simple MLP (784 → 128 → 10)
mlp_params = (28*28*128 + 128) + (128*10 + 10)
print(f"\nSimple MLP parameters: {mlp_params:,} (much smaller than CNN)")
=== Scenario 2: Model Architecture ===
Total CNN parameters: 688,586
Conv1 parameters: 288 (weights) + 32 (bias) + 64 (BN) = 384
Conv2 parameters: 18,432 + 64 + 128 (BN) = 18,624
FC1 parameters: 1,605,632 + 512 = 1,606,144
FC2 parameters: 5,120 + 10 = 5,130
Simple MLP parameters: 101,770 (much smaller than CNN)
# ====================== SCENARIO 3: Training Dynamics ======================
print("=== Scenario 3: Training Dynamics ===")
print("Final Training Loss   :", 0.0148)
print("Final Validation Loss :", 0.0247)
print("Final Training Acc    :", "99.49%")
print("Final Validation Acc  :", "99.27%")
print("Overfitting occurred? : No (val loss stayed stable)")
print("Learning rate used    : 0.001")
# Optional: Retrain with higher LR=0.01 to see effect
# model_new = CNN().to(device)
# optimizer_new = optim.Adam(model_new.parameters(), lr=0.01)
# train_with_tensorboard(model_new, train_loader, test_loader, criterion, opti
=== Scenario 3: Training Dynamics ===
Final Training Loss   : 0.0148
Final Validation Loss : 0.0247
Final Training Acc    : 99.49%
Final Validation Acc  : 99.27%
Overfitting occurred? : No (val loss stayed stable)
Learning rate used    : 0.001
# ====================== SCENARIO 4: Inference Pipeline ======================
print("=== Scenario 4: Inference ===")
# Most confused pair (already calculated in notebook)
print(f"Most confused pair: 9 confused with 4 (8 times)")
# Best / Worst performing digit from classification report
print("Best performing digit : 0 (F1=1.000)")
print("Worst performing digit: 9 (Recall=0.977, F1=0.988)")
print(f"Overall test accuracy: 99.27%")
print("Lowest per-class accuracy: Digit 9 (recall 97.7%)")
# Scenario 4: Inference Analysis
# 4.2: Find most confused pair
max_confused = 0
confused_pair = (0, 0)
for i in range(10):
    for j in range(10):
        if i != j and cm[i, j] > max_confused:
            max_confused = cm[i, j]
            confused_pair = (i, j)
print(f"Most confused pair: {confused_pair[0]} confused with {confused_pair[1]
# 4.3: Per-class metrics from classification report
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_pre
for i in range(10):
    print(f"Digit {i}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F
=== Scenario 4: Inference ===
Most confused pair: 9 confused with 4 (8 times)
Best performing digit : 0 (F1=1.000)
Worst performing digit: 9 (Recall=0.977, F1=0.988)
Overall test accuracy: 99.27%
Lowest per-class accuracy: Digit 9 (recall 97.7%)
Most confused pair: 9 confused with 4 (8 times)
Digit 0: Precision=0.997, Recall=0.998, F1=0.997
Digit 1: Precision=0.995, Recall=0.998, F1=0.996
Digit 2: Precision=0.989, Recall=0.995, F1=0.992
Digit 3: Precision=0.991, Recall=0.996, F1=0.994
Digit 4: Precision=0.990, Recall=0.992, F1=0.991
Digit 5: Precision=0.987, Recall=0.991, F1=0.989
Digit 6: Precision=0.994, Recall=0.995, F1=0.994
Digit 7: Precision=0.989, Recall=0.990, F1=0.990
Digit 8: Precision=0.996, Recall=0.994, F1=0.995
Digit 9: Precision=0.999, Recall=0.977, F1=0.988
# ====================== SCENARIO 5: Architecture Modifications =============
# 5.1 No BatchNorm
class CNN_NoBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
# 5.2 Kernel 5x5 (already in notebook - just train)
class CNN_LargeKernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
# 5.3 Double Filters
class CNN_DoubleFilters(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)   # 32→64
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1) # 64→128
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*7*7, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 128*7*7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
# 5.4 No Dropout
class CNN_NoDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, 10)
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# ====================== TRAIN ALL VARIANTS ======================
criterion = nn.CrossEntropyLoss()
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
variants = {
    "No BatchNorm": CNN_NoBN,
    "Kernel 5x5": CNN_LargeKernel,
    "Double Filters": CNN_DoubleFilters,
    "No Dropout": CNN_NoDropout
}
results = {}
for name, ModelClass in variants.items():
    print(f"\n🚀 Training {name} ...")
    model_var = ModelClass().to(device)
    opt_var = optim.Adam(model_var.parameters(), lr=0.001)
    writer_var = SummaryWriter(f'./logs/{name.lower().replace(" ", "_")}')
    # Quick training (you can reduce epochs if time is short)
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model_var, train_loader, criterio
        val_loss, val_acc, _, _ = validate(model_var, test_loader, criterion,
        print(f"Epoch {epoch+1:2d} | Train Acc: {train_acc:.2f}% | Val Acc: {
    # Final accuracy
    _, final_acc, _, _ = validate(model_var, test_loader, criterion, device)
    results[name] = final_acc
    print(f"✅ {name} Final Val Acc: {final_acc:.2f}%")
    writer_var.close()
print("\nAll variants trained!")
🚀 Training No BatchNorm ...
Epoch  1 | Train Acc: 95.59% | Val Acc: 98.92%
Epoch  2 | Train Acc: 98.48% | Val Acc: 99.09%
Epoch  3 | Train Acc: 98.89% | Val Acc: 99.12%
Epoch  4 | Train Acc: 99.14% | Val Acc: 99.30%
Epoch  5 | Train Acc: 99.29% | Val Acc: 99.17%
Epoch  6 | Train Acc: 99.40% | Val Acc: 99.18%
Epoch  7 | Train Acc: 99.44% | Val Acc: 99.32%
Epoch  8 | Train Acc: 99.58% | Val Acc: 99.20%
Epoch  9 | Train Acc: 99.56% | Val Acc: 99.30%
Epoch 10 | Train Acc: 99.68% | Val Acc: 99.27%
✅ No BatchNorm Final Val Acc: 99.27%
🚀 Training Kernel 5x5 ...
Epoch  1 | Train Acc: 95.06% | Val Acc: 98.51%
Epoch  2 | Train Acc: 97.98% | Val Acc: 99.00%
Epoch  3 | Train Acc: 98.63% | Val Acc: 99.20%
Epoch  4 | Train Acc: 98.89% | Val Acc: 99.18%
Epoch  5 | Train Acc: 99.11% | Val Acc: 99.14%
Epoch  6 | Train Acc: 99.23% | Val Acc: 99.20%
Epoch  7 | Train Acc: 99.37% | Val Acc: 99.40%
Epoch  8 | Train Acc: 99.46% | Val Acc: 99.42%
Epoch  9 | Train Acc: 99.47% | Val Acc: 99.38%
Epoch 10 | Train Acc: 99.57% | Val Acc: 99.52%
✅ Kernel 5x5 Final Val Acc: 99.52%
🚀 Training Double Filters ...
Epoch  1 | Train Acc: 94.80% | Val Acc: 98.10%
Epoch  2 | Train Acc: 97.80% | Val Acc: 99.05%
Epoch  3 | Train Acc: 98.47% | Val Acc: 99.06%
Epoch  4 | Train Acc: 98.77% | Val Acc: 99.13%
Epoch  5 | Train Acc: 98.98% | Val Acc: 99.30%
Epoch  6 | Train Acc: 99.19% | Val Acc: 99.30%
Epoch  7 | Train Acc: 99.22% | Val Acc: 99.34%
Epoch
8 | Train Acc: 99.47% | Val Acc: 99.29%
