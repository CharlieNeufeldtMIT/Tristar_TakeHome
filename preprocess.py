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

    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)
        output_subdir_path = os.path.join(output_dir, subdir)
        os.makedirs(output_subdir_path, exist_ok=True)

        for file_name in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue

            # Resize and normalize image
            img_resized = cv2.resize(img, img_size)
            img_normalized = img_resized / 255.0  # Normalize to range [0, 1]

            # Save preprocessed image
            output_path = os.path.join(output_subdir_path, file_name)
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

    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)
        output_subdir_path = os.path.join(output_dir, subdir)
        os.makedirs(output_subdir_path, exist_ok=True)

        for file_name in os.listdir(subdir_path):
            mask_path = os.path.join(subdir_path, file_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read mask in grayscale
            if mask is None:
                print(f"Warning: Could not read {mask_path}")
                continue

            # Resize mask
            mask_resized = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)

            # Save preprocessed mask
            output_path = os.path.join(output_subdir_path, file_name)
            cv2.imwrite(output_path, mask_resized)


if __name__ == "__main__":
    # Iterate over parts (e.g., "bracket_white", "bracket_brown")
    parts_dir = "data/"
    for part in os.listdir(parts_dir):
        part_dir = os.path.join(parts_dir, part)
        if not os.path.isdir(part_dir):
            continue

        # Preprocess training images
        preprocess_images(
            input_dir=os.path.join(part_dir, "train"),
            output_dir=os.path.join(part_dir, "preprocessed_train"),
            img_size=(128, 128)
        )

        # Preprocess testing images
        preprocess_images(
            input_dir=os.path.join(part_dir, "test"),
            output_dir=os.path.join(part_dir, "preprocessed_test"),
            img_size=(128, 128)
        )

        # Preprocess ground truth masks
        preprocess_masks(
            input_dir=os.path.join(part_dir, "ground_truth"),
            output_dir=os.path.join(part_dir, "preprocessed_ground_truth"),
            img_size=(128, 128)
        )