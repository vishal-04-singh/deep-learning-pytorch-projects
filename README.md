# Deep Learning Notebooks

A collection of deep learning projects implemented in PyTorch, covering a range of computer vision and sequence modeling tasks.

## Projects

| File | Description |
|------|-------------|
| `lstm_stock_price_prediction.py` | LSTM-based time series forecasting on synthetic stock price data |
| `cnn_lstm_hand_landmark.py` | CNN + LSTM hybrid model for hand landmark detection using FreiHAND dataset |
| `mnist_cnn_classification.py` | CNN for MNIST digit classification with TensorBoard logging and architecture variants |
| `face_recognition_cnn.py` | Custom CNN for face recognition with MTCNN face detection and unknown threshold detection |
| `image_captioning_resnet_lstm.py` | Image captioning using ResNet50 encoder + LSTM decoder on Flickr8k dataset |
| `yolo_object_detection.py` | YOLOv11 fine-tuning for custom object detection with COCO-style JSON annotations |

## Tech Stack

- **Framework**: PyTorch
- **Visualization**: TensorBoard, Matplotlib
- **Datasets**: MNIST, FreiHAND, Flickr8k, custom face dataset, custom YOLO dataset
- **Models**: LSTM, CNN, CNN+LSTM, ResNet50, YOLOv11

## Setup

```bash
pip install torch torchvision tensorboard scikit-learn matplotlib pandas numpy opencv-python mediapipe mtcnn ultralytics
```

## Usage

Each script is self-contained. Run directly in Google Colab or locally:

```bash
python lstm_stock_price_prediction.py
python mnist_cnn_classification.py
# etc.
```
