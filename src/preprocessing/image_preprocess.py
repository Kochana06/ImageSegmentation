from PIL import Image
import torchvision.transforms as transforms
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_image(image_path, img_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")  # Load image
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)