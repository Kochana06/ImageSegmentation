import os
import cv2
import torch
import numpy as np
from src.preprocessing.image_preprocess import preprocess_image
from src.models.hurnet import HURNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HURNet(num_classes=3)
state_dict = torch.load('app/hurnet_model2.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.to(device)
model.eval()

def predict(image_path, seg_filename):
    # Define upload and segmentation folders
    upload_folder = "app/static/uploads"
    seg_folder = "app/static/segmentations"

    # Ensure directories exist
    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(seg_folder, exist_ok=True)

    # Load the original image
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return None, None, None, None  # Stop execution if image is not found

    # Save uploaded image
    upload_save_path = os.path.join(upload_folder, os.path.basename(image_path))
    cv2.imwrite(upload_save_path, original_image)

    print(f"✅ Uploaded image saved at: {upload_save_path}")

    # Preprocess image for model prediction
    image = preprocess_image(image_path)

    with torch.no_grad():
        pred_mask, pred_label = model(image)  # Get predictions

    # Process segmentation mask
    pred_mask = pred_mask.squeeze().cpu().numpy()
    pred_mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255

    # Resize segmentation mask to match original image dimensions
    pred_mask_resized = cv2.resize(pred_mask_bin, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Save segmentation mask
    seg_save_path = os.path.join(seg_folder, seg_filename)
    cv2.imwrite(seg_save_path, pred_mask_resized)

    print(f"✅ Segmentation saved at: {seg_save_path}")

    # Calculate tumor size
    tumor_size_pixels = np.sum(pred_mask_resized > 0)
    pixel_to_mm_ratio = 0.05
    tumor_size_mm2 = tumor_size_pixels * (pixel_to_mm_ratio ** 2)

    # Process classification output
    class_labels = ["benign", "malignant", "normal"]
    pred_class_idx = torch.argmax(pred_label, dim=1).item()
    pred_class = class_labels[pred_class_idx]
    print("Predicted Mask Shape:", pred_mask.shape)

    return pred_class, os.path.basename(upload_save_path), os.path.basename(seg_save_path), tumor_size_pixels, tumor_size_mm2
