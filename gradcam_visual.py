import torch
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import OzzyNet

# --------------------------
# Step 1: Setup
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OzzyNet(num_classes=5)
model.load_state_dict(torch.load("ozzy_model.pth", map_location=device))
model.to(device)
model.eval()

# Class labels
class_names = [
    'award_show_outfits',
    'bat_cape_era',
    'glam_metal_80s',
    'modern_ozzy',
    'mtv_ozzy'
]

# --------------------------
# Step 2: Hook for Grad-CAM
# --------------------------
gradients = None
activations = None

def save_gradient(grad):
    global gradients
    gradients = grad

# We'll get the last convolutional layer
def get_last_conv_layer():
    for name, module in model.model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv

# --------------------------
# Step 3: Predict and Visualize
# --------------------------
def apply_gradcam(image_path):
    global activations, gradients

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Hook into last conv layer
    last_conv = get_last_conv_layer()
    handle_fwd = last_conv.register_forward_hook(lambda m, i, o: setattr(m, 'activations', o))
    handle_bwd = last_conv.register_full_backward_hook(lambda m, g_i, g_o: setattr(m, 'gradients', g_o[0]))

    # Forward + backward
    output = model(input_tensor)
    pred_idx = torch.argmax(output).item()
    class_label = class_names[pred_idx]

    model.zero_grad()
    output[0, pred_idx].backward()

    # Get gradients and activations
    grads = last_conv.gradients[0].cpu().numpy()
    acts = last_conv.activations[0].detach().cpu().numpy()


    # Global Average Pooling
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    cam /= cam.max()

    # Prepare image
    img_raw = cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2BGR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_raw, 0.6, heatmap, 0.4, 0)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Original - {class_label}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("gradcam_result.png", dpi=300)
    plt.show()

    # Cleanup hooks
    handle_fwd.remove()
    handle_bwd.remove()

# --------------------------
# Step 4: Run it
# --------------------------
if __name__ == "__main__":
    test_image_path = r"F:\Fun project\Ozzy Fashion Classifier\data\modern_ozzy\000051.jpg"
    apply_gradcam(test_image_path)
