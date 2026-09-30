from pathlib import Path
import kagglehub
import shutil

# Download dataset
path = kagglehub.dataset_download("aryashah2k/stanford-drone-dataset")
dataset_path = Path(path)

# Find all JPG files (sorted for deterministic order)
jpg_files = sorted(dataset_path.rglob("*.jpg"))
print(f"Found {len(jpg_files)} JPG files")

# Destination directory
dest_dir = Path("variety_datasets/drone/dataset")
dest_dir.mkdir(parents=True, exist_ok=True)

# Copy and rename starting from 0
for i, file_path in enumerate(jpg_files, start=0):
    new_name = f"frame_{i}.jpg"
    shutil.copy2(file_path, dest_dir / new_name)

print(f"Copied and renamed {len(jpg_files)} files starting from frame_0.jpg")