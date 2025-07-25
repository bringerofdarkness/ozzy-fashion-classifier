import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import OzzyNet

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OzzyNet(num_classes=5)
model.load_state_dict(torch.load("ozzy_model.pth", map_location=device))
model.to(device)
model.eval()

# Class names in folder order
class_names = [
    'award_show_outfits',
    'bat_cape_era',
    'glam_metal_80s',
    'modern_ozzy',
    'mtv_ozzy'
]

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_and_show(image_path):
    # Load and prepare image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    # Show image with label
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_class}", fontsize=14, color='darkgreen')


    plt.tight_layout()
    plt.show()

# 🔍 Choose your test image (USE raw string or double backslashes)
test_image_path = r"F:\Fun project\Ozzy Fashion Classifier\data\mtv_ozzy\000043.jpg"
predict_and_show(test_image_path)
