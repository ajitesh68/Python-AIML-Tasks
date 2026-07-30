import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import numpy as np

# ---------- STEP 1: DATA (Non-linear Circles) ----------
print("Generating non-linear data...")
X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)  # Classification labels must be Long (int)

# Train/Val Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- STEP 2: MODEL (2 Hidden Layers) ----------
class CircleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: Input (2 features) -> 10 neurons
        self.fc1 = nn.Linear(2, 10)
        # Layer 2: 10 neurons -> 10 neurons (Deep representation)
        self.fc2 = nn.Linear(10, 10)
        # Layer 3: 10 neurons -> Output (2 classes: 0 or 1)
        self.fc3 = nn.Linear(10, 2)
        # Activation function (Non-linearity)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Step 1: Linear -> ReLU
        x = self.relu(self.fc2(x))  # Step 2: Linear -> ReLU
        x = self.fc3(x)             # Step 3: Final Output (Logits)
        return x

model = CircleClassifier()
criterion = nn.CrossEntropyLoss()  # Classification loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ---------- STEP 3: TRAINING LOOP (With Validation) ----------
epochs = 200
for epoch in range(epochs):
    # ---- TRAIN ----
    model.train()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ---- VALIDATION ----
    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)
            # Accuracy (Kitne sahi hain?)
            _, predicted_classes = torch.max(val_pred, 1)  # index with highest value
            acc = (predicted_classes == y_val).float().mean()
            print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {acc:.4f}")

print("\n✅ Model trained successfully!")