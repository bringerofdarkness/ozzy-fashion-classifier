import numpy as np
import matplotlib.pyplot as plt

# Load metrics
loss = np.load("train_loss.npy")
accuracy = np.load("train_accuracy.npy")
epochs = np.arange(1, len(loss) + 1)

# Plot setup
plt.figure(figsize=(10, 5))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, marker='o', color='crimson')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, marker='o', color='seagreen')
plt.title("Training Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)

# Save and show
plt.tight_layout()
plt.savefig("training_metrics.png", dpi=300)
plt.show()
