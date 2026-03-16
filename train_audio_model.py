import os
import librosa
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

DATASET_DIR = "data\\audio"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

X = []
y = []

print("Loading audio data...")

for label in os.listdir(DATASET_DIR):
    label_path = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(label_path, file)
        try:
            audio, sr = librosa.load(file_path, sr=22050)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfcc_mean = np.mean(mfcc.T, axis=0)

            X.append(mfcc_mean)
            y.append(label)

        except Exception as e:
            print("Error loading:", file_path, e)

X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} samples")

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")

# Save model & encoder
joblib.dump(model, os.path.join(MODEL_DIR, "audio_model.pkl"))
joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

print("Model saved in models/")
