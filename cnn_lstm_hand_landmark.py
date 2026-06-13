# ============================================================
# Cnn Lstm Hand Landmark
# ============================================================

# Install required libraries
# Install required libraries
!pip install torch torchvision tensorboard scikit-learn opencv-python mediapipe matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
from sklearn.metrics import mean_squared_error
import zipfile
print(f"PyTorch version: {torch.__version__}")
Show hidden output
# Download FreiHAND datasets
print("Downloading FreiHAND training dataset...")
!wget  https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip -O freihand_train.zip
print("Downloading FreiHAND evaluation dataset...")
!wget https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip -O freihand_eval.zip
print("Download complete!")
Show hidden output
!rm -rf ./training/
!rm -rf ./evaluation/
!rm -rf ./FreiHAND_pub_v2/
!rm -rf ./FreiHAND_pub_v2_eval/
!unzip freihand_train.zip -d FreiHAND_pub_v2
!unzip freihand_eval.zip -d FreiHAND_pub_v2_eval
Show hidden output
# Check extracted folder structure
import os
# Find the extracted folders
for item in os.listdir('.'):
    if os.path.isdir(item) and 'Frei' in item:
        print(f"Folder: {item}")
        for sub in os.listdir(item):
            print(f"  - {sub}")
Show hidden output
# Set correct paths based on extracted folders
BASE_PATH_TRAIN = 'FreiHAND_pub_v2'
BASE_PATH_EVAL = 'FreiHAND_pub_v2_eval'
TRAIN_PATH = f'{BASE_PATH_TRAIN}/training'
EVAL_PATH = f'{BASE_PATH_EVAL}/evaluation'
TRAIN_IMG_DIR = os.path.join(TRAIN_PATH, 'rgb')
TRAIN_ANNO_FILE = os.path.join(BASE_PATH_TRAIN, 'training_xyz.json')
TRAIN_K_FILE = os.path.join(BASE_PATH_TRAIN, 'training_K.json')
EVAL_IMG_DIR = os.path.join(EVAL_PATH, 'rgb')
EVAL_ANNO_FILE = os.path.join(BASE_PATH_EVAL, 'evaluation_xyz.json')
EVAL_K_FILE = os.path.join(BASE_PATH_EVAL, 'evaluation_K.json')
# Verify paths exist
print(f"Training images exist: {os.path.exists(TRAIN_IMG_DIR)}")
print(f"Training annotations exist: {os.path.exists(TRAIN_ANNO_FILE)}")
print(f"Evaluation images exist: {os.path.exists(EVAL_IMG_DIR)}")
print(f"Evaluation annotations exist: {os.path.exists(EVAL_ANNO_FILE)}")
Show hidden output
4.1 Helper Functions
4. Data Loading Functions
keyboard_arrow_down
def load_freihand_images(image_dir, num_samples=None):
    """Load images from FreiHAND dataset"""
    image_files = sorted([f for f in os.listdir(image_dir) if (f.endswith('.jpg'))])
    if num_samples is not None:
        image_files = image_files[:num_samples]
    images = []
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize((64, 64))  # Resize to 64x64
        images.append(np.array(img, dtype=np.float32) / 255.0)
    return np.array(images)
def load_freihand_landmarks(anno_file, k_file, num_samples=None):
    """Load and project 3D landmarks to 2D using camera matrix"""
    # Load 3D landmarks
    with open(anno_file, 'r') as f:
        anno_3d = json.load(f)
    anno_3d = np.array(anno_3d, dtype=np.float32)
    # Load camera matrices
    with open(k_file, 'r') as f:
        K_matrices = json.load(f)
    K_matrices = np.array(K_matrices, dtype=np.float32)
    if num_samples is not None:
        anno_3d = anno_3d[:num_samples]
        K_matrices = K_matrices[:num_samples]
    # Project 3D to 2D
    landmarks_2d = []
    for i in range(len(anno_3d)):
        xyz = anno_3d[i]  # Shape: (21, 3)
        K = K_matrices[i]  # Shape: (3, 3)
        # Project: uv = K @ xyz
        uv = np.dot(K, xyz.T).T  # (21, 3)
        uv = uv[:, :2] / uv[:, 2:3]  # Divide by depth (21, 2)
        # Normalize to [0, 1] for 64x64 image
        uv[:, 0] = uv[:, 0] / 64.0
        uv[:, 1] = uv[:, 1] / 64.0
        # Flatten: 21 landmarks × 2 coords = 42 values
        landmarks_2d.append(uv.flatten())
    return np.array(landmarks_2d)
def count_images(directory):
    """Count number of images in directory"""
    return len([f for f in os.listdir(directory) if f.endswith('.jpg')])
# Get dataset sizes
train_img_count = count_images(TRAIN_IMG_DIR)
eval_img_count = count_images(EVAL_IMG_DIR)
print(f"Training images: {train_img_count}")
print(f"Evaluation images: {eval_img_count}")
Training images: 130240
Evaluation images: 3960
4.2 Load Training and Test Data
keyboard_arrow_down
# Load training data (use subset for faster training, e.g., 5000 samples)
TRAIN_SAMPLES = 5000  # Adjust based on your scenario or requirement
print(f"Loading {TRAIN_SAMPLES} training images...")
train_images = load_freihand_images(TRAIN_IMG_DIR, TRAIN_SAMPLES)
print(f"Loading {TRAIN_SAMPLES} training landmarks...")
train_landmarks = load_freihand_landmarks(TRAIN_ANNO_FILE, TRAIN_K_FILE, TRAIN_SAMPLES)
# Load evaluation data (full set as test set)
print(f"Loading {eval_img_count} evaluation images...")
test_images = load_freihand_images(EVAL_IMG_DIR)
print(f"Loading {eval_img_count} evaluation landmarks...")
test_landmarks = load_freihand_landmarks(EVAL_ANNO_FILE, EVAL_K_FILE)
print(f"\nTraining: {train_images.shape}, {train_landmarks.shape}")
print(f"Test: {test_images.shape}, {test_landmarks.shape}")
Loading 5000 training images...
Loading 5000 training landmarks...
Loading 3960 evaluation images...
Loading 3960 evaluation landmarks...
Training: (5000, 64, 64), (5000, 42)
Test: (3960, 64, 64), (3960, 42)
4.3 PyTorch Dataset Class
keyboard_arrow_down
class FreiHandDataset(Dataset):
    """PyTorch Dataset for FreiHAND Hand Landmark Recognition"""
    def __init__(self, images, landmarks):
        self.images = images
        self.landmarks = landmarks
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        # Get image and add channel dimension (1, H, W)
        image = np.expand_dims(self.images[idx], axis=0)
        # Get landmarks
        landmarks = self.landmarks[idx]
        return torch.from_numpy(image), torch.from_numpy(landmarks)
# Create datasets
train_dataset = FreiHandDataset(train_images, train_landmarks)
test_dataset = FreiHandDataset(test_images, test_landmarks)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"\nTraining samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Landmarks per sample: {train_landmarks.shape[1] // 2} keypoints × 2 coords = {train_landmarks.shape[1]} values")
Training samples: 5000
Test samples: 3960
Landmarks per sample: 21 keypoints × 2 coords = 42 values
5. Data Visualization (Scenario 1)
keyboard_arrow_down
# Visualize sample images with landmarks
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i, ax in enumerate(axes.flat):
    if i >= len(train_images):
        break
    # Get sample
    img = train_images[i]
    landmarks = train_landmarks[i]
    # Plot image
    ax.imshow(img, cmap='gray')
    # Plot landmarks (denormalize from [0,1] to [0,64])
    for j in range(0, len(landmarks), 2):
        x = landmarks[j] * 64
        y = landmarks[j+1] * 64
        ax.plot(x, y, 'r.', markersize=3)
    ax.set_title(f'Train Sample {i+1}')
    ax.axis('off')
plt.suptitle('FreiHAND Hand Landmark Samples (21 keypoints × 2 coords)', fontsize=14)
plt.tight_layout()
plt.savefig('freihand_samples.png', dpi=150)
plt.show()
6.1 CNN + LSTM for Hand Landmark Prediction
The model consists of:
1. CNN Backbone: Extracts spatial features from input image
2. LSTM: Models patterns in the extracted features
3. Output Head: Predicts 42 values (21 landmarks × 2 coordinates)
6. Model Architecture (CNN + LSTM)
keyboard_arrow_down
class CNN_LSTM_Landmark(nn.Module):
    """CNN + LSTM model for hand landmark prediction"""
    def __init__(self, num_landmarks=21):
        super(CNN_LSTM_Landmark, self).__init__()
        self.num_landmarks = num_landmarks
        self.num_coords = num_landmarks * 2  # x, y for each landmark
        # CNN Feature Extractor
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        # After 3 pooling layers: 64 → 32 → 16 → 8
        self.fc_cnn = nn.Linear(128 * 8 * 8, 256)
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
        # Output layer
        self.fc_out = nn.Linear(128, self.num_coords)
    def forward(self, x):
        batch_size = x.size(0)
        # CNN Feature Extraction
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # Flatten
        x = x.view(batch_size, -1)
        x = F.relu(self.fc_cnn(x))
        # Prepare for LSTM
        x = x.unsqueeze(1)  # (batch, seq_len=1, features=256)
        # LSTM
        lstm_out, _ = self.lstm(x)
        # Output predictions
        output = self.fc_out(lstm_out.squeeze(1))
        return output
# Initialize model
model = CNN_LSTM_Landmark()
print(model)
# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
CNN_LSTM_Landmark(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc_cnn): Linear(in_features=8192, out_features=256, bias=True)
  (lstm): LSTM(256, 128, batch_first=True)
  (fc_out): Linear(in_features=128, out_features=42, bias=True)
)
Total parameters: 2,393,578
Trainable parameters: 2,393,578
7. TensorBoard Setup
keyboard_arrow_down
# Setup TensorBoard
%load_ext tensorboard
log_dir = './hand_landmark_logs'
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)
print(f"TensorBoard log directory: {log_dir}")
The tensorboard extension is already loaded. To reload it, use:
  %reload_ext tensorboard
TensorBoard log directory: ./hand_landmark_logs
8. Training Setup
keyboard_arrow_down
# Loss and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = model.to(device)
Using device: cuda
9. Training Loop with TensorBoard
keyboard_arrow_down
def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    for images, landmarks in loader:
        images = images.to(device)
        landmarks = landmarks.to(device)
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, landmarks)
        # Backward pass
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, landmarks in loader:
            images = images.to(device)
            landmarks = landmarks.to(device)
            outputs = model(images)
            loss = criterion(outputs, landmarks)
            total_loss += loss.item()
    return total_loss / len(loader)
def log_histograms(model, writer, epoch):
    """Log weight histograms"""
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            writer.add_histogram(f'Parameters/{name}', param, epoch)
def log_landmark_predictions(model, test_loader, writer, epoch, num_samples=4):
    """Log predicted vs actual landmarks as images"""
    model.eval()
    with torch.no_grad():
        for i, (images, landmarks) in enumerate(test_loader):
            if i >= 1:
                break
            images = images.to(device)
            outputs = model(images)
            # Plot first few samples
            for j in range(min(num_samples, images.size(0))):
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                # Original image with actual landmarks
                img = images[j].cpu().numpy().squeeze()
                actual = landmarks[j].cpu().numpy()
                axes[0].imshow(img, cmap='gray')
                for k in range(0, len(actual), 2):
                    x = actual[k] * 64
                    y = actual[k+1] * 64
                    axes[0].plot(x, y, 'g.', markersize=4)
                axes[0].set_title('Actual Landmarks (Green)')
                axes[0].axis('off')
                # Predicted landmarks
                predicted = outputs[j].cpu().numpy()
                axes[1].imshow(img, cmap='gray')
                for k in range(0, len(predicted), 2):
                    x = predicted[k] * 64
                    y = predicted[k+1] * 64
                    axes[1].plot(x, y, 'r.', markersize=4)
                axes[1].set_title('Predicted Landmarks (Red)')
                axes[1].axis('off')
                writer.add_figure(f'Predictions/Epoch_{epoch}_Sample_{j}', fig)
                plt.close()
            break
# Training loop
print("Starting training...\n")
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, test_loader, criterion, device)
    # Log to TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    # Log histograms and predictions every 5 epochs
    if epoch % 5 == 0:
        log_histograms(model, writer, epoch)
        log_landmark_predictions(model, test_loader, writer, epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
print("\nTraining complete!")
Starting training...
Epoch [1/20] - Train Loss: 0.261829, Val Loss: 0.125127
Epoch [2/20] - Train Loss: 0.096362, Val Loss: 0.110120
Epoch [3/20] - Train Loss: 0.087239, Val Loss: 0.104900
Epoch [4/20] - Train Loss: 0.080021, Val Loss: 0.101866
Epoch [5/20] - Train Loss: 0.069838, Val Loss: 0.099168
Epoch [6/20] - Train Loss: 0.064175, Val Loss: 0.102123
Epoch [7/20] - Train Loss: 0.060187, Val Loss: 0.100952
Epoch [8/20] - Train Loss: 0.056107, Val Loss: 0.109263
Epoch [9/20] - Train Loss: 0.051348, Val Loss: 0.101531
Epoch [10/20] - Train Loss: 0.046581, Val Loss: 0.097994
Epoch [11/20] - Train Loss: 0.041278, Val Loss: 0.100589
Epoch [12/20] - Train Loss: 0.036676, Val Loss: 0.110270
Epoch [13/20] - Train Loss: 0.032702, Val Loss: 0.106432
Epoch [14/20] - Train Loss: 0.027848, Val Loss: 0.102567
Epoch [15/20] - Train Loss: 0.024964, Val Loss: 0.102971
Epoch [16/20] - Train Loss: 0.022278, Val Loss: 0.103967
Epoch [17/20] - Train Loss: 0.019407, Val Loss: 0.107480
Epoch [18/20] - Train Loss: 0.017208, Val Loss: 0.114805
Epoch [19/20] - Train Loss: 0.015325, Val Loss: 0.110195
Epoch [20/20] - Train Loss: 0.013803, Val Loss: 0.113485
Training complete!
10. Evaluation
keyboard_arrow_down
# Final evaluation
model.eval()
all_predictions = []
all_targets = []
with torch.no_grad():
    for images, landmarks in test_loader:
        images = images.to(device)
        outputs = model(images)
        all_predictions.extend(outputs.cpu().numpy())
        all_targets.extend(landmarks.numpy())
all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)
# Calculate MSE
mse = mean_squared_error(all_targets, all_predictions)
print(f"\nFinal Test MSE: {mse:.6f}")
print(f"RMSE: {np.sqrt(mse):.6f}")
# Per-landmark MSE
per_landmark_mse = []
for i in range(21):
    x_mse = mean_squared_error(all_targets[:, i*2], all_predictions[:, i*2])
    y_mse = mean_squared_error(all_targets[:, i*2+1], all_predictions[:, i*2+1])
    per_landmark_mse.append((x_mse + y_mse) / 2)
print(f"\nAverage MSE per landmark: {np.mean(per_landmark_mse):.6f}")
Final Test MSE: 0.113473
RMSE: 0.336858
Average MSE per landmark: 0.113473
12. Visualize Final Predictions
keyboard_arrow_down
# Visualize final predictions
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i, ax in enumerate(axes.flat):
    if i >= len(all_predictions):
        break
    img = test_images[i]
    actual = all_targets[i]
    predicted = all_predictions[i]
    ax.imshow(img, cmap='gray')
    # Plot actual (green) and predicted (red)
    for j in range(0, len(actual), 2):
        ax.plot(actual[j] * 64, actual[j+1] * 64, 'g.', markersize=3)
        ax.plot(predicted[j] * 64, predicted[j+1] * 64, 'r.', markersize=3)
    ax.set_title(f'Test Sample {i+1}')
    ax.axis('off')
plt.suptitle('Final Predictions: Green=Actual, Red=Predicted', fontsize=14)
plt.tight_layout()
plt.savefig('final_predictions.png', dpi=150)
plt.show()
13. Close TensorBoard Writer
keyboard_arrow_down
# Close writer
writer.close()
print("TensorBoard writer closed.")
TensorBoard writer closed.
# Scenario 1: Dataset Exploration
# Your code here:
# 1.1 Total samples
print(f"Total training samples: {len(train_dataset)}")
print(f"Total test samples: {len(test_dataset)}")
# 1.2 Image shape
sample_img, _ = train_dataset[0]
print(f"Image shape: {sample_img.shape}")
# 1.3 Number of landmarks
_, sample_landmarks = train_dataset[0]
print(f"Landmark values: {len(sample_landmarks)} (should be 42)")
print(f"Number of keypoints: {len(sample_landmarks) // 2}")
Total training samples: 5000
Total test samples: 3960
Image shape: torch.Size([1, 64, 64])
Landmark values: 42 (should be 42)
Number of keypoints: 21
# Scenario 2: Model Architecture Analysis
# Your code here:
# Count parameters per layer type
cnn_params = sum(p.numel() for n, p in model.named_parameters() if 'conv' in n)
lstm_params = sum(p.numel() for n, p in model.named_parameters() if 'lstm' in n)
fc_params = sum(p.numel() for n, p in model.named_parameters() if 'fc' in n)
print(f"CNN parameters: {cnn_params:,}")
print(f"LSTM parameters: {lstm_params:,}")
print(f"FC parameters: {fc_params:,}")
print(f"Total: {cnn_params + lstm_params + fc_params:,}")
CNN parameters: 92,672
LSTM parameters: 197,632
FC parameters: 2,102,826
Total: 2,393,130
# Scenario 4: Inference Analysis
# Find highest and lowest error landmarks
error_per_landmark = np.array(per_landmark_mse)
best_idx = np.argmin(error_per_landmark)
worst_idx = np.argmax(error_per_landmark)
print(f"Best landmark (lowest MSE): Landmark {best_idx} with MSE {error_per_landmark[best_idx]:.6f}")
print(f"Worst landmark (highest MSE): Landmark {worst_idx} with MSE {error_per_landmark[worst_idx]:.6f}")
Best landmark (lowest MSE): Landmark 9 with MSE 0.037194
Worst landmark (highest MSE): Landmark 4 with MSE 0.240905
# Scenario 5: Architecture Modifications
# 5.1: CNN-only model (no LSTM)
class CNN_Only_Landmark(nn.Module):
    def __init__(self, num_landmarks=21):
        super(CNN_Only_Landmark, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 8 * 8, num_landmarks * 2)
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
# Uncomment to train CNN-only:
model_cnn = CNN_Only_Landmark().to(device)
optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=0.001)
# Train and compare...
# ====================== 5.1 CNN-ONLY MODEL ======================
class CNN_Only_Landmark(nn.Module):
    def __init__(self, num_landmarks=21):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 8 * 8, num_landmarks * 2)
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
# Train CNN-only
model_cnn = CNN_Only_Landmark().to(device)
optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=0.001)
criterion = nn.MSELoss()
print("🚀 Training CNN-Only model (20 epochs)...\n")
for epoch in range(20):
    train_loss = train_epoch(model_cnn, train_loader, criterion, optimizer_cnn, device)
    val_loss = validate(model_cnn, test_loader, criterion, device)
    print(f"Epoch [{epoch+1:2d}/20]  Train Loss: {train_loss:.6f}   Val Loss: {val_loss:.6f}")
# Final MSE
model_cnn.eval()
all_preds_cnn = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model_cnn(images)
        all_preds_cnn.extend(outputs.cpu().numpy())
all_preds_cnn = np.array(all_preds_cnn)
mse_cnn = mean_squared_error(all_targets, all_preds_cnn)
print(f"\n✅ CNN-Only Final Test MSE: {mse_cnn:.6f}")
🚀 Training CNN-Only model (20 epochs)...
Epoch [ 1/20]  Train Loss: 0.493779   Val Loss: 0.236681
Epoch [ 2/20]  Train Loss: 0.107465   Val Loss: 0.265686
Epoch [ 3/20]  Train Loss: 0.099466   Val Loss: 0.175625
Epoch [ 4/20]  Train Loss: 0.095736   Val Loss: 0.144894
Epoch [ 5/20]  Train Loss: 0.089906   Val Loss: 0.136483
Epoch [ 6/20]  Train Loss: 0.084091   Val Loss: 0.143892
Epoch [ 7/20]  Train Loss: 0.081119   Val Loss: 0.125236
Epoch [ 8/20]  Train Loss: 0.082440   Val Loss: 0.133999
Epoch [ 9/20]  Train Loss: 0.080560   Val Loss: 0.145330
Epoch [10/20]  Train Loss: 0.071224   Val Loss: 0.130298
Epoch [11/20]  Train Loss: 0.068022   Val Loss: 0.140526
Epoch [12/20]  Train Loss: 0.064471   Val Loss: 0.134012
Epoch [13/20]  Train Loss: 0.062035   Val Loss: 0.134116
Epoch [14/20]  Train Loss: 0.065397   Val Loss: 0.134332
Epoch [15/20]  Train Loss: 0.056162   Val Loss: 0.127827
Epoch [16/20]  Train Loss: 0.051322   Val Loss: 0.140220
Epoch [17/20]  Train Loss: 0.050189   Val Loss: 0.131376
Epoch [18/20]  Train Loss: 0.046228   Val Loss: 0.147054
Epoch [19/20]  Train Loss: 0.042819   Val Loss: 0.131447
Epoch [20/20]  Train Loss: 0.041471   Val Loss: 0.136145
✅ CNN-Only Final Test MSE: 0.136120
# ====================== 5.3 Kernel 5×5 Variant ======================
class CNN_LSTM_Kernel5(nn.Module):
    def __init__(self, num_landmarks=21):
        super().__init__()
        self.num_coords = num_landmarks * 2
        # Changed kernel_size=5
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc_cnn = nn.Linear(128 * 8 * 8, 256)
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.fc_out = nn.Linear(128, self.num_coords)
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_cnn(x))
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        return self.fc_out(lstm_out.squeeze(1))
# Train Kernel 5x5 model
model_k5 = CNN_LSTM_Kernel5().to(device)
optimizer_k5 = optim.Adam(model_k5.parameters(), lr=0.001)
print("🚀 Training Kernel 5×5 model (20 epochs)...\n")
for epoch in range(20):
    train_loss = train_epoch(model_k5, train_loader, criterion, optimizer_k5, device)
    val_loss = validate(model_k5, test_loader, criterion, device)
    print(f"Epoch [{epoch+1:2d}/20]  Train Loss: {train_loss:.6f}   Val Loss: {val_loss:.6f}")
# Final MSE
model_k5.eval()
all_preds_k5 = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model_k5(images)
        all_preds_k5.extend(outputs.cpu().numpy())
all_preds_k5 = np.array(all_preds_k5)
mse_k5 = mean_squared_error(all_targets, all_preds_k5)
print(f"\n✅ Kernel 5×5 Final Test MSE: {mse_k5:.6f}")
🚀 Training Kernel 5×5 model (20 epochs)...
Epoch [ 1/20]  Train Loss: 0.273425   Val Loss: 0.119227
Epoch [ 2/20]  Train Loss: 0.119472   Val Loss: 0.121858
Epoch [ 3/20]  Train Loss: 0.119496   Val Loss: 0.122011
Epoch [ 4/20]  Train Loss: 0.119585   Val Loss: 0.118648
Epoch [ 5/20]  Train Loss: 0.119724   Val Loss: 0.121697
Epoch [ 6/20]  Train Loss: 0.120045   Val Loss: 0.121812
Epoch [ 7/20]  Train Loss: 0.120139   Val Loss: 0.122534
Epoch [ 8/20]  Train Loss: 0.119972   Val Loss: 0.118155
Epoch [ 9/20]  Train Loss: 0.120541   Val Loss: 0.121887
Epoch [10/20]  Train Loss: 0.116304   Val Loss: 0.128430
Epoch [11/20]  Train Loss: 0.093486   Val Loss: 0.110124
Epoch [12/20]  Train Loss: 0.083656   Val Loss: 0.105937
Epoch [13/20]  Train Loss: 0.074354   Val Loss: 0.112243
Epoch [14/20]  Train Loss: 0.068757   Val Loss: 0.127622
Epoch [15/20]  Train Loss: 0.063970   Val Loss: 0.102807
Epoch [16/20]  Train Loss: 0.056867   Val Loss: 0.106012
Epoch [17/20]  Train Loss: 0.051828   Val Loss: 0.106580
Epoch [18/20]  Train Loss: 0.046820   Val Loss: 0.106155
Epoch [19/20]  Train Loss: 0.042537   Val Loss: 0.102140
Epoch [20/20]  Train Loss: 0.037380   Val Loss: 0.099005
✅ Kernel 5×5 Final Test MSE: 0.098974
# ====================== 5.4 Dropout Variant ======================
class CNN_LSTM_Dropout(nn.Module):
    def __init__(self, num_landmarks=21):
        super().__init__()
        self.num_coords = num_landmarks * 2
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)
        self.fc_cnn = nn.Linear(128 * 8 * 8, 256)
        self.dropout_fc = nn.Dropout(0.5)
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.fc_out = nn.Linear(128, self.num_coords)
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_cnn(x))
        x = self.dropout_fc(x)
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        return self.fc_out(lstm_out.squeeze(1))
# Train Dropout model
model_drop = CNN_LSTM_Dropout().to(device)
optimizer_drop = optim.Adam(model_drop.parameters(), lr=0.001)
print("🚀 Training Dropout model (20 epochs)...\n")
for epoch in range(20):
    train_loss = train_epoch(model_drop, train_loader, criterion, optimizer_drop, device)
    val_loss = validate(model_drop, test_loader, criterion, device)
    print(f"Epoch [{epoch+1:2d}/20]  Train Loss: {train_loss:.6f}   Val Loss: {val_loss:.6f}")
# Final MSE
model_drop.eval()
all_preds_drop = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model_drop(images)
        all_preds_drop.extend(outputs.cpu().numpy())
all_preds_drop = np.array(all_preds_drop)
mse_drop = mean_squared_error(all_targets, all_preds_drop)
print(f"\n✅ Dropout Final Test MSE: {mse_drop:.6f}")
🚀 Training Dropout model (20 epochs)...
Epoch [ 1/20]  Train Loss: 0.289364   Val Loss: 0.121146
Epoch [ 2/20]  Train Loss: 0.119867   Val Loss: 0.122290
Epoch [ 3/20]  Train Loss: 0.120227   Val Loss: 0.122278
Epoch [ 4/20]  Train Loss: 0.120137   Val Loss: 0.124001
Epoch [ 5/20]  Train Loss: 0.119877   Val Loss: 0.122762
Epoch [ 6/20]  Train Loss: 0.118968   Val Loss: 0.122506
Epoch [ 7/20]  Train Loss: 0.112916   Val Loss: 0.124580
Epoch [ 8/20]  Train Loss: 0.103927   Val Loss: 0.126528
Epoch [ 9/20]  Train Loss: 0.098264   Val Loss: 0.114180
Epoch [10/20]  Train Loss: 0.095782   Val Loss: 0.112284
Epoch [11/20]  Train Loss: 0.093241   Val Loss: 0.108584
Epoch [12/20]  Train Loss: 0.091513   Val Loss: 0.106447
Epoch [13/20]  Train Loss: 0.088660   Val Loss: 0.104766
Epoch [14/20]  Train Loss: 0.087476   Val Loss: 0.106713
Epoch [15/20]  Train Loss: 0.085310   Val Loss: 0.104481
Epoch [16/20]  Train Loss: 0.082201   Val Loss: 0.120266
Epoch [17/20]  Train Loss: 0.079874   Val Loss: 0.100657
Epoch [18/20]  Train Loss: 0.078064   Val Loss: 0.099343
Epoch [19/20]  Train Loss: 0.076865   Val Loss: 0.104651
Epoch [20/20]  Train Loss: 0.075915   Val Loss: 0.101510
✅ Dropout Final Test MSE: 0.101501
