# ============================================================
# Yolo Object Detection
# ============================================================

!pip install ultralytics
!pip install ultralytics
Show hidden output
import os
import json
from ultralytics import YOLO
Show hidden output
DATASET_ROOT = "/content/drive/MyDrive/Colab Notebooks/Dataset"
JSON_PATH = os.path.join(DATASET_ROOT, "label.json")
IMAGES_TRAIN = os.path.join(DATASET_ROOT, "images/train")
IMAGES_VAL = os.path.join(DATASET_ROOT, "images/val")
LABELS_ROOT = os.path.join(DATASET_ROOT, "labels")
LABELS_TRAIN = os.path.join(LABELS_ROOT, "train")
LABELS_VAL = os.path.join(LABELS_ROOT, "val")
os.makedirs(LABELS_TRAIN, exist_ok=True)
os.makedirs(LABELS_VAL, exist_ok=True)
with open(JSON_PATH) as f:
   data = json.load(f)
# Map image_id → image info
image_map = {img['id']: img for img in data['images']}
# Map category_id → class index
category_map = {}
for i, cat in enumerate(data['categories']):
   category_map[cat['id']] = i
print("Category Mapping:", category_map)
Category Mapping: {1: 0, 2: 1, 3: 2}
for folder in [LABELS_TRAIN, LABELS_VAL]:
   for f in os.listdir(folder):
       os.remove(os.path.join(folder, f))
       train_images = set(os.listdir(IMAGES_TRAIN))
val_images = set(os.listdir(IMAGES_VAL))
print("Train Images:", len(train_images))
print("Val Images:", len(val_images))
Train Images: 88
Val Images: 22
def img_to_txt(name):
    return name.replace(".jpg", ".txt").replace(".jpeg", ".txt").replace(".png", ".txt")
# Create empty label files for ALL images
for img in train_images:
    open(os.path.join(LABELS_TRAIN, img_to_txt(img)), "w").close()
for img in val_images:
    open(os.path.join(LABELS_VAL, img_to_txt(img)), "w").close()
annotated_images = set()
for ann in data['annotations']:
    img_id = ann['image_id']
    bbox = ann['bbox']
    cat_id = category_map[ann['category_id']]
    img_info = image_map[img_id]
    img_name = img_info['file_name']
    annotated_images.add(img_name)
    w, h = img_info['width'], img_info['height']
    # YOLO format
    x_center = (bbox[0] + bbox[2] / 2) / w
    y_center = (bbox[1] + bbox[3] / 2) / h
    width = bbox[2] / w
    height = bbox[3] / h
    txt_name = img_to_txt(img_name)
    if img_name in train_images:
        label_path = os.path.join(LABELS_TRAIN, txt_name)
    elif img_name in val_images:
        label_path = os.path.join(LABELS_VAL, txt_name)
    else:
        continue
    with open(label_path, "a") as f:
        f.write(f"{cat_id} {x_center} {y_center} {width} {height}\n")
print("✅ Labels written!")
print("Annotated Images:", len(annotated_images))
✅ Labels written!
Annotated Images: 110
print("Train labels:", len(os.listdir(LABELS_TRAIN)))
print("Val labels:", len(os.listdir(LABELS_VAL)))
Train labels: 88
Val labels: 22
data_yaml = f"""
path: {DATASET_ROOT}
train: images/train
val: images/val
nc: {len(category_map)}
names: {[cat['name'] for cat in data['categories']]}
"""
with open("/content/data.yaml", "w") as f:
    f.write(data_yaml)
print("✅ data.yaml created")
✅ data.yaml created
model = YOLO("yolov11n.pt")
model.train(
    data="/content/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8
)
Show hidden output
metrics = model.val()
print(metrics)
Show hidden output
#/content/runs/detect/train/weights/best.pt
from ultralytics import YOLO
# Load trained model
model = YOLO("/content/runs/detect/train/weights/best.pt")
# Run prediction on full val folder
results = model.predict(
    source="/content/drive/MyDrive/Colab Notebooks/Dataset/images/val",
    conf=0.25,      # confidence threshold
    save=True,      # save output images
    save_txt=True   # save prediction labels (optional but useful)
)
print("✅ Folder testing completed")
image 1/22 /content/drive/MyDrive/Colab Notebooks/Dataset/images/val/IMG_3270.JPG: 480x640 1 Car, 72.6ms
image 2/22 /content/drive/MyDrive/Colab Notebooks/Dataset/images/val/IMG_3278.JPG: 640x480 1 Car, 73.8ms
image 3/22 /content/drive/MyDrive/Colab Notebooks/Dataset/images/val/IMG_3285.JPG: 640x480 1 Car, 11.9ms
image 4/22 /content/drive/MyDrive/Colab Notebooks/Dataset/images/val/IMG_3286.JPG: 480x640 (no detections), 15.8ms
image 5/22 /content/drive/MyDrive/Colab Notebooks/Dataset/images/val/IMG_3314.JPG: 640x480 2 Cars, 25.0ms
image 6/22 /content/drive/MyDrive/Colab Notebooks/Dataset/images/val/IMG_3316.JPG: 640x480 (no detections), 11.9ms
image 7/22 /content/drive/MyDrive/Colab Notebooks/Dataset/images/val/IMG_3318.JPG: 640x480 (no detections), 11.8ms
image 8/22 /content/drive/MyDrive/Colab Notebooks/Dataset/images/val/IMG_3325.JPG: 640x480 (no detections), 11.8ms
image 9/22 /content/drive/MyDrive/Colab Notebooks/Dataset/images/val/IMG_3326.JPG: 640x480 (no detections), 11.9ms
image 10/22 /content/drive/MyDrive/Colab Notebooks/Dataset/images/val/IMG_3329.JPG: 640x480 1 Car, 11.8ms
image 11/22 /content/drive/MyDrive/Colab Notebooks/Dataset/images/val/IMG_3339.JPG: 640x480 1 Car, 11.8ms
image 12/22 /content/drive/MyDrive/Colab Notebooks/Dataset/images/val/IMG_3343.JPG: 640x480 (no detections), 11.8ms
image 13/22 /content/drive/MyDrive/Colab Notebooks/Dataset/images/val/IMG_3345.JPG: 640x480 (no detections), 11.8ms
image 14/22 /content/drive/MyDrive/Colab Notebooks/Dataset/images/val/IMG_3348.jpg: 640x480 (no detections), 11.8ms
image 15/22 /content/drive/MyDrive/Colab Notebooks/Dataset/images/val/IMG_3351.JPG: 480x640 (no detections), 13.5ms
image 16/22 /content/drive/MyDrive/Colab Notebooks/Dataset/images/val/IMG_3353.JPG: 480x640 (no detections), 16.2ms
image 17/22 /content/drive/MyDrive/Colab Notebooks/Dataset/images/val/IMG_3356.JPG: 640x480 (no detections), 12.7ms
image 18/22 /content/drive/MyDrive/Colab Notebooks/Dataset/images/val/IMG_3357.JPG: 480x640 (no detections), 13.4ms
image 19/22 /content/drive/MyDrive/Colab Notebooks/Dataset/images/val/IMG_3360.JPG: 640x480 (no detections), 13.2ms
image 20/22 /content/drive/MyDrive/Colab Notebooks/Dataset/images/val/IMG_3361.JPG: 640x480 1 Car, 13.5ms
image 21/22 /content/drive/MyDrive/Colab Notebooks/Dataset/images/val/IMG_3370.JPG: 640x480 (no detections), 11.8ms
image 22/22 /content/drive/MyDrive/Colab Notebooks/Dataset/images/val/IMG_3376.JPG: 640x480 (no detections), 11.9ms
Speed: 3.9ms preprocess, 18.7ms inference, 1.2ms postprocess per image at shape (1, 3, 640, 480)
Results saved to /content/runs/detect/predict
7 labels saved to /content/runs/detect/predict/labels
✅ Folder testing completed
