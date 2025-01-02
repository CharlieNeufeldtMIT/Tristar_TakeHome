import os
import cv2
import numpy as np


def preprocess_images(input_dir, output_dir, img_size=(128, 128)):
    """
    Preprocesses images by resizing, normalizing, and saving them.

    Parameters:
    - input_dir (str): Path to the raw images.
    - output_dir (str): Path to save processed images.
    - img_size (tuple): Desired size (height, width) for resizing images.
    """
    os.makedirs(output_dir, exist_ok=True)

    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)
        os.makedirs(output_category_path, exist_ok=True)

        for file_name in os.listdir(category_path):
            img_path = os.path.join(category_path, file_name)
            img = cv2.imread(img_path)  # Read image
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue

            # Resize and normalize image
            img_resized = cv2.resize(img, img_size)
            img_normalized = img_resized / 255.0  # Normalize to range [0, 1]

            # Save preprocessed image
            output_path = os.path.join(output_category_path, file_name)
            cv2.imwrite(output_path, (img_normalized * 255).astype(np.uint8))  # Save back to [0, 255] for storage


def preprocess_masks(input_dir, output_dir, img_size=(128, 128)):
    """
    Preprocesses ground truth masks by resizing and saving them.

    Parameters:
    - input_dir (str): Path to the raw masks.
    - output_dir (str): Path to save processed masks.
    - img_size (tuple): Desired size (height, width) for resizing masks.
    """
    os.makedirs(output_dir, exist_ok=True)

    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)
        os.makedirs(output_category_path, exist_ok=True)

        for file_name in os.listdir(category_path):
            mask_path = os.path.join(category_path, file_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read mask in grayscale
            if mask is None:
                print(f"Warning: Could not read {mask_path}")
                continue

            # Resize mask
            mask_resized = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)

            # Save preprocessed mask
            output_path = os.path.join(output_category_path, file_name)
            cv2.imwrite(output_path, mask_resized)


if __name__ == "__main__":
    # Preprocess training images
    preprocess_images(
        input_dir="data/train",
        output_dir="data/preprocessed_train",
        img_size=(128, 128)
    )

    # Preprocess testing images
    preprocess_images(
        input_dir="data/test",
        output_dir="data/preprocessed_test",
        img_size=(128, 128)
    )

    # Preprocess ground truth masks
    preprocess_masks(
        input_dir="data/ground_truth",
        output_dir="data/preprocessed_ground_truth",
        img_size=(128, 128)
    )
