import torch
from PIL import Image
from torchvision import transforms
from model import OzzyNet  # Assuming model.py is in the same folder

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OzzyNet(num_classes=5)
model.load_state_dict(torch.load("ozzy_model.pth", map_location=device))
model.to(device)
model.eval()

# Define class names in the same order as your folders
class_names = [
    'award_show_outfits',
    'bat_cape_era',
    'glam_metal_80s',
    'modern_ozzy',
    'mtv_ozzy'
]

# Image transform (must match training transform)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    """Load an image and predict its class."""
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return None

    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    predicted_class = class_names[predicted.item()]
    return predicted_class

# 🔍 Run prediction
if __name__ == "__main__":
    # CHANGE this path to your test image
    test_image_path = r"F:\Fun project\Ozzy Fashion Classifier\data\mtv_ozzy\000043.jpg"

    predicted_label = predict_image(test_image_path)
    if predicted_label:
        print(f"🎸 Predicted Class: {predicted_label}")
