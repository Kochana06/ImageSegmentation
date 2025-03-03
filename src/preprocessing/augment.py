import os
import cv2
import numpy as np
import albumentations as A

# Paths
image_dir = "data/aug_image/normal"    # Folder containing original images
mask_dir = "data/aug_masks/normal"      # Folder containing corresponding masks
output_image_dir = "data/images/normal"
output_mask_dir = "data/masks/normal"

# Create output directories if they don’t exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Define augmentations without normalization
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ElasticTransform(p=0.2, alpha=1, sigma=50, alpha_affine=50),
    A.GaussianBlur(p=0.2),
    # A.Normalize(),  # Removed for saving visible images
])

num_augmentations = 5

# Get sorted list of image and mask files
image_files = sorted(os.listdir(image_dir))
mask_files = sorted(os.listdir(mask_dir))

for idx, (img_file, mask_file) in enumerate(zip(image_files, mask_files)):
    img_path = os.path.join(image_dir, img_file)
    mask_path = os.path.join(mask_dir, mask_file)

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Skipping {img_file} due to loading error.")
        continue

    for i in range(num_augmentations):
        augmented = transform(image=image, mask=mask)
        aug_image = augmented["image"]
        aug_mask = augmented["mask"]

        # Save augmented image and mask
        aug_img_filename = f"aug_{idx}_{i}.png"
        aug_mask_filename = f"aug_{idx}_{i}.png"

        cv2.imwrite(os.path.join(output_image_dir, aug_img_filename), aug_image)
        cv2.imwrite(os.path.join(output_mask_dir, aug_mask_filename), aug_mask)
        print(f"Saved: {aug_img_filename} and {aug_mask_filename}")

print("Augmentation complete!")
