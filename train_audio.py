import pandas as pd
import librosa
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Configuration
AUDIO_ROOT = os.path.join(os.getcwd(), 'data', 'audio') 
METADATA_PATH = os.path.join(AUDIO_ROOT, 'audio_metadata_shuffled.csv')

# --- NEW STEP: Create a File Path Map ---
# This dictionary will store: { "filename.wav": "full/path/to/filename.wav" }
file_path_map = {}
print("Building file map from subfolders...")

# The **/*.wav pattern recursively searches all subdirectories for .wav files
for full_path in glob.glob(os.path.join(AUDIO_ROOT, '**', '*.wav'), recursive=True):
    # The key in our metadata is just the filename, so we extract it.
    file_name = os.path.basename(full_path)
    file_path_map[file_name] = full_path

print(f"Found {len(file_path_map)} total WAV files across all subfolders.")
# ------------------------------------------

# 2. Feature Extraction Function
def extract_features(file_name):
    # --- Check 1: Find the full path from the map ---
    if file_name not in file_path_map:
        # This means the file listed in the CSV is missing from the folders
        # (This is unlikely if all files are downloaded, but good for error checking)
        print(f"Warning: File '{file_name}' not found in any subfolder.")
        return None
        
    full_path = file_path_map[file_name]
    
    try:
        # --- Check 2: Can librosa read the file? ---
        audio, sample_rate = librosa.load(full_path, res_type='kaiser_fast', sr=22050, duration=3) 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing file {file_name} at {full_path}: {e}")
        return None

# 3. Load Data and Process
try:
    metadata = pd.read_csv(METADATA_PATH)
except FileNotFoundError:
    print(f"ERROR: Metadata file not found at {METADATA_PATH}")
    exit()

print("Extracting features (this may take a few minutes)...")

# Filter the metadata to only include files we actually found in the folders
metadata = metadata[metadata['slice_file_name'].isin(file_path_map.keys())]

# We'll stick to a subset for speed, but process all the actual files we found
data_subset = metadata.sample(min(2000, len(metadata)), random_state=42) 

features = []
files_processed = 0

for index, row in data_subset.iterrows():
    f = extract_features(row['slice_file_name'])
    if f is not None:
        features.append([f, row['classID']])
        files_processed += 1

print(f"\n--- Feature Extraction Summary ---")
print(f"Total files attempted from subset: {len(data_subset)}")
print(f"Total features successfully extracted: {files_processed}")

# 4. Train Model only if data is available
if files_processed > 0:
    # Prepare Data
    X = np.array([x[0] for x in features])
    y = np.array([x[1] for x in features])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    print("Training Model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate & Save
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    joblib.dump(model, 'models/audio_drone_detector.pkl')
    print("Model saved to models/audio_drone_detector.pkl")
else:
    print("\nFATAL ERROR: Could not extract any features. Check your file paths and if the WAV files are present.")