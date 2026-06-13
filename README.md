<div align="center">

# 🧠 Deep Learning Projects — PyTorch

**A hands-on collection of deep learning projects built from scratch using PyTorch.**  
Covers computer vision, time series forecasting, face recognition, object detection, and image captioning.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=flat-square&logo=pytorch)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange?style=flat-square&logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## 📁 Project Overview

| # | Project | Dataset | Model | Task |
|---|---------|---------|-------|------|
| 1 | [LSTM Stock Price Prediction](#1-lstm-stock-price-prediction) | Synthetic (GAN-generated) | LSTM | Time Series Regression |
| 2 | [CNN+LSTM Hand Landmark Detection](#2-cnnlstm-hand-landmark-detection) | FreiHAND | CNN + LSTM | Keypoint Regression |
| 3 | [MNIST CNN Classification](#3-mnist-cnn-classification) | MNIST | Custom CNN | Image Classification |
| 4 | [Face Recognition CNN](#4-face-recognition-cnn) | **Self-collected** 📷 | Custom CNN + MTCNN | Face Recognition |
| 5 | [Image Captioning](#5-image-captioning-resnet50--lstm) | Flickr8k | ResNet50 + LSTM | Caption Generation |
| 6 | [YOLOv11 Object Detection](#6-yolov11-custom-object-detection) | **Self-collected** 📷 | YOLOv11n | Object Detection |

---

## 🔍 Project Details

### 1. LSTM Stock Price Prediction
**File:** `lstm_stock_price_prediction.py`

Predicts future stock prices using an LSTM model trained on synthetic random-walk time series data.

- **Data:** 1000-point synthetic stock price series generated using Gaussian random walk
- **Model:** 2-layer LSTM → Dropout → Fully Connected
- **Input:** 10 time steps → predict next value
- **Training:** 50 epochs, Adam optimizer, MSELoss
- **Results:** Test RMSE ≈ ±7.06 price units
- **Extras:** TensorBoard logging, weight histograms, train/val loss curves

```
Input (10 steps) → LSTM(64, layers=2) → Dropout(0.2) → FC(1) → Predicted Price
```

---

### 2. CNN+LSTM Hand Landmark Detection
**File:** `cnn_lstm_hand_landmark.py`

Detects 21 hand keypoints (42 coordinates) from grayscale images using a CNN backbone fed into an LSTM.

- **Dataset:** [FreiHAND](https://lmb.informatik.uni-freiburg.de/projects/freihand/) — 130,240 training + 3,960 evaluation images
- **Preprocessing:** 3D landmarks projected to 2D using camera intrinsic matrix K, normalized to [0,1]
- **Model Architecture:**
  ```
  Input (1×64×64)
    → Conv2d(1→32) + BN + ReLU + MaxPool
    → Conv2d(32→64) + BN + ReLU + MaxPool
    → Conv2d(64→128) + BN + ReLU + MaxPool
    → FC(8192→256)
    → LSTM(256→128, seq_len=1)
    → FC(128→42)  ← 21 keypoints × (x,y)
  ```
- **Training:** 20 epochs, 5000 samples, Adam, MSELoss
- **Results:** Test MSE ≈ 0.1135 | Best landmark: #9 (MSE 0.037) | Worst: #4 (MSE 0.241)
- **Architecture variants tested:** CNN-only (MSE 0.136), Kernel 5×5 (MSE 0.099), Heavy Dropout (MSE 0.101)
- **Extras:** TensorBoard prediction overlays every 5 epochs

---

### 3. MNIST CNN Classification
**File:** `mnist_cnn_classification.py`

Classifies handwritten digits (0–9) using a 3-block CNN with BatchNorm, Dropout, and full TensorBoard integration.

- **Dataset:** MNIST — 60,000 train / 10,000 test images (28×28 grayscale)
- **Model Architecture:**
  ```
  Input (1×28×28)
    → Conv2d(1→32, k=3) + BN + ReLU + MaxPool  → 14×14
    → Conv2d(32→64, k=3) + BN + ReLU + MaxPool → 7×7
    → Conv2d(64→128, k=3) + BN + ReLU + MaxPool → 3×3
    → FC(1152→512) + ReLU + Dropout(0.5)
    → FC(512→10)
  ```
- **Total Parameters:** 688,586
- **Training:** 10 epochs, Adam lr=0.001, CrossEntropyLoss, gradient clipping
- **Results:** Test Accuracy **99.27%**
- **Most confused pair:** Digit 9 ↔ Digit 4 (8 misclassifications)
- **Architecture variants:**

  | Variant | Val Accuracy |
  |---------|-------------|
  | Baseline (BN + Dropout) | 99.27% |
  | No BatchNorm | 99.27% |
  | Kernel 5×5 | **99.52%** |
  | Double Filters | 99.29% |
  | No Dropout | 99.27% |

- **Extras:** Confusion matrix, per-class F1 scores, Conv1 filter visualization, TensorBoard model graph

---

### 4. Face Recognition CNN
**File:** `face_recognition_cnn.py`

> 📷 **Dataset self-collected** — Real face images captured manually across 91 individuals.

End-to-end face recognition pipeline with live face detection, identity classification, and unknown person rejection.

- **Dataset:** Custom — 91 people, organized per-person folder, captured in varied lighting/angles
- **Face Detection:** MTCNN (Multi-task Cascaded CNN) — detects faces + crops with 20% margin
- **Model Architecture:**
  ```
  Input (3×64×64 RGB)
    → Conv2d(3→32) + BN + ReLU + MaxPool(2)   → 32×32
    → Conv2d(32→64) + BN + ReLU + MaxPool(2)  → 16×16
    → Conv2d(64→128) + BN + ReLU + MaxPool(2) → 8×8
    → Conv2d(128→256) + BN + ReLU + MaxPool(2)→ 4×4
    → FC(4096→512) + ReLU + Dropout(0.5)
    → FC(512→91 classes)
  ```
- **Total Parameters:** ~2.1M
- **Data Augmentation:** Random horizontal flip, rotation (±10°), color jitter
- **Unknown Detection:** Softmax confidence threshold (default 0.7) — below threshold → "Unknown"
- **Train/Test Split:** 80/20 stratified
- **Inference:** Single image → MTCNN → crop → CNN → identity + confidence score

**Dataset Structure expected:**
```
FaceDataset/
├── person_name_1/
│   ├── img1.jpg
│   └── img2.jpg
├── person_name_2/
│   └── ...
```

---

### 5. Image Captioning — ResNet50 + LSTM
**File:** `image_captioning_resnet_lstm.py`

Generates natural language captions for images using a CNN encoder + RNN decoder architecture.

- **Dataset:** [Flickr8k](https://github.com/awsaf49/flickr-dataset) — 8,000 images, 5 captions each
- **Vocabulary:** 2,988 words (freq_threshold=5), special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`
- **Encoder:** ResNet50 (ImageNet pretrained, frozen) → FC(2048→256) → BatchNorm
- **Decoder:** LSTM(embed=256, hidden=256) → FC(256→vocab_size)
- **Training:** 5 epochs, Adam lr=3e-4, CrossEntropyLoss (ignores `<PAD>`), gradient clipping
- **Loss progression:** 4.71 → 4.10 → 3.90 → 3.76 → 3.65
- **Inference modes:**
  - **Greedy decoding** (argmax at each step)
  - **Temperature sampling** — controls randomness (0.5 = conservative, 1.5 = creative)
  - **Beam search** (width=3)

```
Image → ResNet50 → embed(256) → LSTM → word1 → word2 → ... → <EOS>
```

---

### 6. YOLOv11 Custom Object Detection
**File:** `yolo_object_detection.py`

> 📷 **Dataset self-collected** — 110 real-world images captured and annotated manually with 3 custom object classes.

Fine-tunes YOLOv11n on a custom dataset for real-world object detection.

- **Dataset:** Self-collected — 110 images (88 train / 22 val), COCO-style JSON annotations
- **Classes:** 3 custom categories (e.g., Car and 2 others based on collected data)
- **Annotation pipeline:** JSON → YOLO format conversion (normalized `x_center y_center width height`)
- **Model:** YOLOv11n (nano) — pretrained on COCO, fine-tuned on custom data
- **Training:** 30 epochs, imgsz=640, batch=8
- **Inference:** conf=0.25 threshold, outputs saved with bounding boxes
- **Sample results on val set:**
  ```
  IMG_3270.JPG  → 1 Car detected  (72.6ms)
  IMG_3314.JPG  → 2 Cars detected (25.0ms)
  IMG_3285.JPG  → 1 Car detected  (11.9ms)
  Avg inference speed: ~18.7ms/image
  ```

**Dataset folder structure:**
```
Dataset/
├── images/
│   ├── train/   (88 images)
│   └── val/     (22 images)
├── labels/
│   ├── train/   (auto-generated .txt)
│   └── val/     (auto-generated .txt)
└── label.json   (COCO-format annotations)
```

---

## ⚙️ Setup & Installation

```bash
# Clone the repo
git clone https://github.com/vishal-04-singh/deep-learning-pytorch-projects.git
cd deep-learning-pytorch-projects

# Install all dependencies
pip install torch torchvision torchaudio
pip install tensorboard scikit-learn matplotlib pandas numpy
pip install opencv-python mediapipe mtcnn pillow
pip install ultralytics
```

> All scripts are designed to run on **Google Colab** with GPU support.  
> Set runtime to `GPU` for faster training.

---

## 📊 Results Summary

| Project | Metric | Value |
|---------|--------|-------|
| LSTM Stock Prediction | Test RMSE | 7.06 price units |
| Hand Landmark Detection | Test MSE | 0.1135 |
| MNIST Classification | Test Accuracy | **99.27%** |
| Face Recognition | Threshold-based | 0.7 confidence |
| Image Captioning | Train Loss (5 ep) | 3.65 |
| YOLOv11 Detection | Inference Speed | ~18.7ms/image |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Framework | PyTorch 2.x |
| Computer Vision | OpenCV, PIL, torchvision |
| Face Detection | MTCNN |
| Object Detection | Ultralytics YOLOv11 |
| Visualization | Matplotlib, TensorBoard |
| Data | NumPy, Pandas, scikit-learn |
| Platform | Google Colab (GPU) |

---

## 👤 Author

**Vishal Singh**  
GitHub: [@vishal-04-singh](https://github.com/vishal-04-singh)

---

<div align="center">
⭐ If you found this helpful, consider giving it a star!
</div>
