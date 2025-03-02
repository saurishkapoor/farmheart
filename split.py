import os
import shutil
import random

# Paths
preprocessed_dir = "preprocessed_images"  # Source directory
output_dir = "split_data"  # Destination directory

# Define split ratios
train_ratio = 0.7
test_ratio = 0.2
val_ratio = 0.1

# Create output directories
for split in ["train", "test", "val"]:
    for label in ["non_cvd", "cvd"]:
        os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)

def split_data():
    """Randomly splits images into train (70%), test (20%), val (10%) sets"""
    for label in ["non_cvd", "cvd"]:
        image_paths = []
        input_class_dir = os.path.join(preprocessed_dir, label)

        # Get all images in the class
        for filename in os.listdir(input_class_dir):
            image_paths.append(os.path.join(input_class_dir, filename))

        # Shuffle the images
        random.shuffle(image_paths)

        # Compute split sizes
        total_images = len(image_paths)
        train_count = int(total_images * train_ratio)
        test_count = int(total_images * test_ratio)
        val_count = total_images - train_count - test_count  # Remaining for val

        # Split images
        train_images = image_paths[:train_count]
        test_images = image_paths[train_count:train_count + test_count]
        val_images = image_paths[train_count + test_count:]

        # Move files
        for img_path in train_images:
            shutil.copy(img_path, os.path.join(output_dir, "train", label))

        for img_path in test_images:
            shutil.copy(img_path, os.path.join(output_dir, "test", label))

        for img_path in val_images:
            shutil.copy(img_path, os.path.join(output_dir, "val", label))

        print(f"Class {label} - Train: {len(train_images)}, Test: {len(test_images)}, Val: {len(val_images)}")

# Run the split function
split_data()