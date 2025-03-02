import os
import cv2
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# Paths
input_dir = "retina_healthy_unhealthy"  # Change this to your dataset path
output_dir = "preprocessed_images"
target_size = (224, 224)  # Resizing target
desired_count = 1000  # Target number of images per class

# Augmentation Setup
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

def apply_clahe(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def preprocess_and_augment_images():
    """Preprocess and augment images to balance the dataset"""
    for label in ["0", "1"]:  # 0: Non-CVD, 1: CVD
        input_class_dir = os.path.join(input_dir, label)
        output_class_dir = os.path.join(output_dir, label)
        os.makedirs(output_class_dir, exist_ok=True)

        # Load images
        images = []
        for filename in os.listdir(input_class_dir):
            img_path = os.path.join(input_class_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, target_size)
                img = apply_clahe(img)  # Apply CLAHE
                img = img.astype(np.float32) / 255.0  # Normalize
                images.append(img)

        # Save preprocessed images
        for i, img in enumerate(images):
            output_path = os.path.join(output_class_dir, f"preprocessed_{i}.jpg")
            cv2.imwrite(output_path, (img * 255).astype(np.uint8))

        # Augment images if less than desired count
        num_existing = len(images)
        images = np.array(images)
        image_count = num_existing
        gen = datagen.flow(images, batch_size=1, save_to_dir=output_class_dir, 
                           save_prefix="aug", save_format="jpg")

        while image_count < desired_count:
            next(gen)  # Generate augmented images
            image_count += 1

        print(f"Class {label} - Processed images: {num_existing}, Augmented to: {image_count}")

# Run preprocessing and augmentation
preprocess_and_augment_images()