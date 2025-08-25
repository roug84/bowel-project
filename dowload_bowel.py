# download_bowel.py
import os
import subprocess

DATASET = "robertnowak/bowel-sounds"
DATA_DIR = "bowel_dataset"

# Create target directory
os.makedirs(DATA_DIR, exist_ok=True)

print("📥 Downloading dataset from Kaggle...")
subprocess.run([
    "kaggle", "datasets", "download",
    "-d", DATASET,
    "-p", DATA_DIR
], check=True)

print("📦 Extracting dataset...")
zip_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".zip")]
if not zip_files:
    raise FileNotFoundError("No ZIP file found in the dataset directory.")

zip_path = os.path.join(DATA_DIR, zip_files[0])

import zipfile
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR)

print("✅ Dataset ready in:", DATA_DIR)
