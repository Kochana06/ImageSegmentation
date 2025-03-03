from models.hurnet import BreastCancerDataset

# Function to load dataset
def load_dataset(image_dir, mask_dir, img_size=(224, 224)):
    return BreastCancerDataset(image_dir, mask_dir, img_size)