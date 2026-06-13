# ============================================================
# Face Recognition Cnn
# ============================================================

from google.colab import drive
from google.colab import drive
drive.mount('/content/drive')
# Install required packages
!pip install torch torchvision torchaudio
!pip install mtcnn pillow opencv-python
!pip install scikit-learn
Show hidden output
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
# =============================================================================
# SECTION 2: DATASET CREATION
# =============================================================================
"""
This section handles:
1. Loading images from the dataset directory
2. Face detection using MTCNN
3. Cropping and aligning faces
4. Organizing data by person (label)
5. Creating train/test splits
The dataset should be organized as:
/content/drive/MyDrive/FaceDataset/
    ├── student_001/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── student_002/
    │   ├── img1.jpg
    │   └── ...
    └── ...
OR use a single folder with naming convention: name_001.jpg, name_002.jpg
"""
class FaceDatasetCreator:
    """
    Handles dataset creation from raw images.
    - Detects faces using MTCNN
    - Crops and aligns faces
    - Saves processed images
    """
    def __init__(self, input_dir, output_dir, target_size=(64, 64)):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_size = target_size
        self.detector = MTCNN()
    def load_image(self, image_path):
        """Load image from file"""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def detect_faces(self, image):
        """
        Detect faces using MTCNN.
        Returns list of bounding boxes and keypoints.
        """
        detections = self.detector.detect_faces(image)
        return detections
    def crop_and_resize(self, image, bounding_box):
        """
        Crop face using bounding box and resize to target size.
        bounding_box = [x, y, width, height]
        """
        x, y, w, h = bounding_box
        # Add margin around face (20%)
        margin = int(max(w, h) * 0.2)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        # Crop
        face = image[y:y+h, x:x+w]
        # Resize
        face = cv2.resize(face, self.target_size)
        return face
    def process_all_images(self):
        """
        Process all images in input directory.
        Returns: X (images), y (labels), paths (file paths)
        """
        X, y, paths = [], [], []
        class_names = []
        # Get all subdirectories (each is a person)
        persons = sorted([d for d in os.listdir(self.input_dir)
                        if os.path.isdir(os.path.join(self.input_dir, d))])
        for person_idx, person_name in enumerate(persons):
            person_dir = os.path.join(self.input_dir, person_name)
            class_names.append(person_name)
            # Get all images for this person
            images = [f for f in os.listdir(person_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img_name in images:
                img_path = os.path.join(person_dir, img_name)
                try:
                    # Load and detect faces
                    img = self.load_image(img_path)
                    detections = self.detect_faces(img)
                    if len(detections) > 0:
                        # Use the largest face
                        largest = max(detections, key=lambda d: d['box'][2] * d['box'][3])
                        face = self.crop_and_resize(img, largest['box'])
                        X.append(face)
                        y.append(person_idx)
                        paths.append(img_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        return np.array(X), np.array(y), class_names, paths
# =============================================================================
# SECTION 3: PYTORCH DATASET
# =============================================================================
"""
PyTorch Dataset class for face recognition.
Handles:
- Converting numpy arrays to tensors
- Normalizing pixel values
- Data augmentation (optional)
"""
class FaceDataset(Dataset):
    """
    Custom PyTorch Dataset for face images.
    """
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images: numpy array of face images (N, H, W, C)
            labels: numpy array of labels (N,)
            transform: optional transforms
        """
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        # Get image and label
        image = self.images[idx]
        label = self.labels[idx]
        # Convert to PIL Image
        image = Image.fromarray(image)
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
images  =[] #30 images
labels  = [] # 30 labels
self.
def get_transforms(train=True):
    """
    Get transforms for training or testing.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
# =============================================================================
# SECTION 4: CUSTOM CNN MODEL (FROM SCRATCH)
# =============================================================================
"""
Custom CNN Architecture for Face Recognition
---------------------------------------------
This is a custom CNN built from scratch (not pre-trained).
Architecture:
Input: 64×64×3 (RGB face image)
    │
    ▼
Conv1: 32 filters, 3×3 → BatchNorm → ReLU → MaxPool(2×2)
    │ Output: 32×32×32
    ▼
Conv2: 64 filters, 3×3 → BatchNorm → ReLU → MaxPool(2×2)
    │ Output: 16×16×64
    ▼
Conv3: 128 filters, 3×3 → BatchNorm → ReLU → MaxPool(2×2)
    │ Output: 8×8×128
    ▼
Conv4: 256 filters, 3×3 → BatchNorm → ReLU → MaxPool(2×2)
    │ Output: 4×4×256
    ▼
Flatten: 4×4×256 = 4096
    │
    ▼
FC1: 4096 → 512 → ReLU → Dropout(0.5)
    │
    ▼
FC2: 512 → num_classes (91)
    │
    ▼
Output: Logits for each class
"""
class FaceRecognitionCNN(nn.Module):
    """
    Custom CNN for Face Recognition (built from scratch).
    """
    def __init__(self, num_classes=91, dropout=0.5):
        super(FaceRecognitionCNN, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64×64 → 32×32
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 32×32 → 16×16
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 16×16 → 8×8
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 8×8 → 4×4
        )
        # Fully Connected Layers
        # After conv4: 256 channels × 4×4 = 4096 features
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc2 = nn.Linear(512, num_classes)
    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x: Input tensor (batch_size, 3, 64, 64)
        Returns:
            Output: Logits (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # FC layers
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    def get_embedding(self, x):
        """
        Get feature embedding (before final classification layer).
        Useful for similarity comparison.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
# =============================================================================
# SECTION 5: TRAINING
# =============================================================================
"""
Training loop for the face recognition model.
"""
def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=10, device='cuda'):
    """
    Train the face recognition model.
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs
        device: Device to train on
    Returns:
        history: Dictionary with training history
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        val_loss = running_loss / len(val_loader)
        val_acc = 100.0 * correct / total
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    return history
def create_dataset_from_directory(data_dir, test_size=0.2):
    """
    Create dataset from directory structure.
    Expected structure:
    data_dir/
        person_1/
            img1.jpg
            img2.jpg
        person_2/
            img1.jpg
            ...
    """
    dataset_creator = FaceDatasetCreator(data_dir, None, target_size=(64, 64))
    X, y, class_names, paths = dataset_creator.process_all_images()
    print(f"Total images: {len(X)}")
    print(f"Total classes: {len(class_names)}")
    print(f"Classes: {class_names[:5]}... (showing first 5)")
    return X, y, class_names
# =============================================================================
# SECTION 6: INFERENCE AND UNKNOWN DETECTION
# =============================================================================
"""
Inference pipeline for face recognition with unknown detection.
"""
class FaceRecognizer:
    """
    Face recognition inference with unknown detection.
    """
    def __init__(self, model, class_names, threshold=0.7, device='cuda'):
        """
        Args:
            model: Trained PyTorch model
            class_names: List of class names (person names)
            threshold: Confidence threshold for known face detection
            device: Device to run inference on
        """
        self.model = model
        self.class_names = class_names
        self.threshold = threshold
        self.device = device
        self.detector = MTCNN()
    def preprocess_face(self, face):
        """Preprocess face for model input"""
        face = cv2.resize(face, (64, 64))
        face = Image.fromarray(face)
        transform = get_transforms(train=False)
        face = transform(face).unsqueeze(0).to(self.device)
        return face
    def detect_and_recognize(self, image):
        """
        Detect all faces in image and recognize them.
        Args:
            image: Input image (numpy array, RGB)
        Returns:
            results: List of dicts with 'box', 'label', 'confidence'
        """
        # Detect faces
        detections = self.detector.detect_faces(image)
        results = []
        for detection in detections:
            x, y, w, h = detection['box']
            # Crop face
            face = image[y:y+h, x:x+w]
            # Preprocess
            face_tensor = self.preprocess_face(face)
            # Get prediction
            self.model.eval()
            with torch.no_grad():
                output = self.model(face_tensor)
                probs = torch.softmax(output, dim=1)
                max_prob, predicted_idx = probs.max(dim=1)
                max_prob = max_prob.item()
                predicted_idx = predicted_idx.item()
            # Check threshold for unknown detection
            if max_prob > self.threshold:
                label = self.class_names[predicted_idx]
            else:
                label = "Unknown"
            results.append({
                'box': (x, y, w, h),
                'label': label,
                'confidence': max_prob,
                'class_id': predicted_idx if label != "Unknown" else None
            })
        return results
    def draw_results(self, image, results):
        """
        Draw bounding boxes and labels on image.
        Args:
            image: Input image
            results: List of detection results
        Returns:
            image with drawn boxes and labels
        """
        img = image.copy()
        for result in results:
            x, y, w, h = result['box']
            label = result['label']
            confidence = result['confidence']
            # Draw rectangle
            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            # Draw label
            text = f"{label} ({confidence:.2f})"
            cv2.putText(img, text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img
# =============================================================================
# SECTION 7: MAIN EXECUTION
# =============================================================================
"""
Main execution block.
Students should modify these paths according to their Drive setup.
"""
# Configuration
DATA_DIR = '/content/drive/MyDrive/FaceDataset/'  # Change this!
MODEL_SAVE_PATH = '/content/drive/MyDrive/FaceRecognition/model.pth'
NUM_CLASSES = 91  # Change based on actual number of students
NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
THRESHOLD = 0.7  # For unknown detection
def main():
    # Step 1: Create/Load Dataset
    print("=" * 50)
    print("STEP 1: Loading Dataset")
    print("=" * 50)
    X, y, class_names = create_dataset_from_directory(DATA_DIR)
    print(f"\nDataset Info:")
    print(f"  - Total images: {len(X)}")
    print(f"  - Number of people: {len(class_names)}")
    print(f"  - Image shape: {X[0].shape}")
    # Step 2: Split Data
    print("\n" + "=" * 50)
    print("STEP 2: Splitting Data")
    print("=" * 50)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    # Step 3: Create DataLoaders
    print("\n" + "=" * 50)
    print("STEP 3: Creating DataLoaders")
    print("=" * 50)
    train_dataset = FaceDataset(X_train, y_train, transform=get_transforms(train=True))
    val_dataset = FaceDataset(X_val, y_val, transform=get_transforms(train=False))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    # Step 4: Create Model
    print("\n" + "=" * 50)
    print("STEP 4: Creating Model")
    print("=" * 50)
    model = FaceRecognitionCNN(num_classes=NUM_CLASSES).to(DEVICE)
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    # Step 5: Training
    print("\n" + "=" * 50)
    print("STEP 5: Training Model")
    print("=" * 50)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer,
        num_epochs=NUM_EPOCHS,
        device=DEVICE
    )
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'threshold': THRESHOLD
    }, MODEL_SAVE_PATH)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    # Step 6: Plot Training History
    print("\n" + "=" * 50)
    print("STEP 6: Training History")
    print("=" * 50)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("\nTraining completed!")
    print("=" * 50)
# =============================================================================
# SECTION 8: INFERENCE EXAMPLE
# =============================================================================
"""
Example code for running inference on new images.
"""
def run_inference(image_path):
    """
    Run inference on a single image.
    """
    # Load model
    checkpoint = torch.load(MODEL_SAVE_PATH)
    model = FaceRecognitionCNN(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    class_names = checkpoint['class_names']
    threshold = checkpoint['threshold']
    # Create recognizer
    recognizer = FaceRecognizer(model, class_names, threshold=threshold)
    # Load and process image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Detect and recognize
    results = recognizer.detect_and_recognize(image)
    # Draw results
    output_image = recognizer.draw_results(image, results)
    # Display
    plt.figure(figsize=(12, 8))
    plt.imshow(output_image)
    plt.axis('off')
    plt.title('Face Recognition Results')
    plt.show()
    # Print results
    print("\nDetection Results:")
    for i, result in enumerate(results):
        print(f"  Face {i+1}: {result['label']} (confidence: {result['confidence']:.2f})")
# =============================================================================
# SECTION 9: EXPERIMENTS TEMPLATE
# =============================================================================
"""
This section contains experiment tasks for students to understand
the model architecture and data flow.
"""
# EXPERIMENT 1: Dataset Exploration
# --------------------------------
def experiment_1():
    """
    Explore the dataset.
    Tasks:
    1. Print total number of images
    2. Print number of classes (people)
    3. Print class distribution
    4. Visualize some sample images
    """
    X, y, class_names = create_dataset_from_directory(DATA_DIR)
    print("=" * 50)
    print("EXPERIMENT 1: Dataset Exploration")
    print("=" * 50)
    # Total images
    print(f"Total images: {len(X)}")
    # Number of classes
    print(f"Number of classes: {len(class_names)}")
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass Distribution:")
    for i, (cls, count) in enumerate(zip(unique, counts)):
        print(f"  {class_names[cls]}: {count} images")
    # Visualize sample images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[i])
        ax.set_title(class_names[y[i]])
        ax.axis('off')
    plt.suptitle('Sample Images from Dataset')
    plt.show()
# EXPERIMENT 2: Model Architecture
# --------------------------------
def experiment_2():
    """
    Explore model architecture.
    Tasks:
    1. Print model summary
    2. Count parameters per layer
    3. Print input/output shapes at each layer
    """
    print("=" * 50)
    print("EXPERIMENT 2: Model Architecture")
    print("=" * 50)
    model = FaceRecognitionCNN(num_classes=NUM_CLASSES)
    # Print layers
    print("\nModel Layers:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 64, 64)
    print(f"\nInput shape: {dummy_input.shape}")
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")
# EXPERIMENT 3: Training Dynamics
# --------------------------------
def experiment_3():
    """
    Explore training dynamics.
    Tasks:
    1. Add print statements in training loop
    2. Visualize loss curves
    3. Print accuracy per epoch
    """
    # Run main() with additional prints for this experiment
    # See training loop in Section 5
    pass
# EXPERIMENT 4: Inference Pipeline
# --------------------------------
def experiment_4():
    """
    Explore inference pipeline.
    Tasks:
    1. Trace prediction flow
    2. Print confidence scores
    3. Show detection results
    """
    # Run inference and print details
    print("=" * 50)
    print("EXPERIMENT 4: Inference Pipeline")
    print("=" * 50)
    print("\nInference Flow:")
    print("  1. Load image")
    print("  2. MTCNN detects faces -> bounding boxes")
    print("  3. Crop faces from image")
    print("  4. Preprocess (resize, normalize)")
    print("  5. Forward pass through CNN")
    print("  6. Get probabilities from softmax")
    print("  7. Check threshold -> Known or Unknown")
    print("  8. Draw bounding boxes with labels")
# EXPERIMENT 5: Threshold Tuning
# --------------------------------
def experiment_5():
    """
    Tune threshold for unknown detection.
    Tasks:
    1. Try different threshold values
    2. Observe changes in predictions
    3. Find optimal threshold
    """
    print("=" * 50)
    print("EXPERIMENT 5: Threshold Tuning")
    print("=" * 50)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for thresh in thresholds:
        print(f"\nThreshold: {thresh}")
        print("-" * 30)
        # Run inference with different thresholds
        # Compare results
        # Students should observe:
        # - Lower threshold: More people recognized, but may have false positives
        # - Higher threshold: More "Unknown" labels, but more confident predictions
# =============================================================================
# RUN CODE
# =============================================================================
if __name__ == "__main__":
    # Uncomment to run training
    main()
    # Or run experiments
    experiment_1()
    experiment_2()
    experiment_4()
    experiment_5()
    print("Code ready! Run main() or individual experiments.")
