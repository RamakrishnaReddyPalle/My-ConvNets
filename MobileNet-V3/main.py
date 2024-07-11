import torch
import torch.nn as nn
from mobilenetv3 import MobileNetV3
from torchsummary import summary
from PIL import Image
import torchvision.transforms as transforms

def load_image(image_path, target_size):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

if __name__ == "__main__":
    config_name = input("Enter the MobileNetV3 configuration (large/small): ").strip().lower()
    image_path = input("Enter the path to the input image: ").strip()
    rho = 0.75 # What is rho
    res = int(rho * 224)

    try:
        net = MobileNetV3(config_name)
        summary(net, (3, res, res))

        # Load and preprocess the image
        image = load_image(image_path, res)

        # Pass the image through the model
        with torch.no_grad():
            output = net(image)
        print(f"Model output shape: {output.shape}")
    except ValueError as e:
        print(e)
