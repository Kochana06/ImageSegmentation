import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from glob import glob



# Function for Data Preprocessing
class BreastCancerDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=(224, 224)):  # Fixed size
        self.image_paths = sorted(glob(os.path.join(image_dir, "*/*.png")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*/*.png")))
        self.labels = [0 if "benign" in path else 1 if "malignant" in path else 2 for path in self.image_paths]
        self.img_size = img_size  # Store fixed size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)  # Resize image

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size)  # Resize mask

        label = self.labels[idx]
        
        image = transforms.ToTensor()(image)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0
        
        return image, mask, torch.tensor(label, dtype=torch.long)