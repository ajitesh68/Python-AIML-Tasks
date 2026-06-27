import torch
import torch.nn as nn
import torch.optim as optim
import os

# ----- STEP 1: Model Banayen aur Train Karein -----
print("Step 1: Training Model...")
torch.manual_seed(42)

# Data (Simple wala)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_true = 2 * X + 1

# Model
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train for 100 epochs
for epoch in range(100):
    pred = model(X)
    loss = criterion(pred, y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Trained Weight: {model.linear.weight.item():.2f}, Bias: {model.linear.bias.item():.2f}")

# ----- STEP 2: Model ko DISK mein Save Karein (Recipe Card) -----
print("\nStep 2: Saving Model Weights...")

# **SAHI TAREKA (Sirf Weights - State Dict)**
torch.save(model.state_dict(), 'linear_model_pt.pth')
print("Weights saved successfully in 'linear_model_pt.pth'!")

# ----- STEP 3: Python ko Restart karke Load karein (Assume fresh start) -----
print("\nStep 3: Loading Model & Predicting (Without Training!)...")

# Naya model object banayen (random weights ke saath)
loaded_model = LinearModel()

# Saved weights (numbers) ko is naye model mein daalein
loaded_model.load_state_dict(torch.load('linear_model_pt.pth'))

# **IMPORTANT**: Inference ke liye Evaluation Mode ON karein (Dropout/BatchNorm ke liye)
loaded_model.eval()

# Naya data (jo model ne kabhi nahi dekha)
new_x = torch.tensor([[5.0], [6.0]])

# Prediction (Sirf Forward, Backward nahi)
with torch.no_grad():
    predictions = loaded_model(new_x)

print(f"Input: {new_x.flatten().tolist()}")
print(f"Predictions (should be ~11 and 13): {predictions.flatten().tolist()}")
print(f"Loaded Weight: {loaded_model.linear.weight.item():.2f}, Bias: {loaded_model.linear.bias.item():.2f}")