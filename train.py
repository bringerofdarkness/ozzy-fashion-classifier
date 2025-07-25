import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import OzzyNet
from data_loader import get_data_loaders


# Configuration
data_dir = "data"
num_epochs = 10
batch_size = 16
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, class_names = get_data_loaders(data_dir, batch_size)

# Init model
model = OzzyNet(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
# For plotting
epoch_losses = []
epoch_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_accuracy = 100 * correct / total
    epoch_losses.append(running_loss)
    epoch_accuracies.append(epoch_accuracy)
    
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {running_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%")

# Save metrics to disk
import numpy as np
np.save("train_loss.npy", np.array(epoch_losses))
np.save("train_accuracy.npy", np.array(epoch_accuracies))

# Save model
torch.save(model.state_dict(), "ozzy_model.pth")
print("✅ Model saved as ozzy_model.pth")


# Save model
torch.save(model.state_dict(), "ozzy_model.pth")
print("✅ Model saved as ozzy_model.pth")
