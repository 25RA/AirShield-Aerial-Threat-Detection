import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ---------------- CONFIG ----------------
DATASET_DIR = "video_frames"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
images = []
labels = []

for class_name in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = preprocess_input(img)

        images.append(img)
        labels.append(class_name)

images = np.array(images)
labels = np.array(labels)

print("Total samples:", len(images))

# ---------------- ENCODE LABELS ----------------
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_cat = to_categorical(labels_encoded)

joblib.dump(le, f"{MODEL_DIR}/video_label_encoder.pkl")

# ---------------- SPLIT DATA ----------------
X_train, X_val, y_train, y_val = train_test_split(
    images, labels_cat, test_size=0.2, random_state=42, stratify=labels_encoded
)

# ---------------- MODEL ----------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(labels_cat.shape[1], activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------- TRAIN ----------------
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# ---------------- SAVE ----------------
model.save(f"{MODEL_DIR}/video_mobilenet.h5")

print("✅ Video model training complete!")
