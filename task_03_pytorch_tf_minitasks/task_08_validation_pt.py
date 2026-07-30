import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

# 1. Data Generate (200 points)
torch.manual_seed(42)
X = torch.randn(200, 1) * 10
y = 2 * X + 1 + torch.randn(200, 1) * 0.5

# 2. Dataset aur Split (Train/Val)
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))  # 160
val_size = len(dataset) - train_size  # 40
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 3. DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 4. Model, Loss, Optimizer
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 5. Training + Validation Loop
epochs = 50
for epoch in range(epochs):
    # --- TRAINING PHASE (Practice) ---
    model.train()  # Model ko batao ab practice mode hai
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)  # Batch size se multiply karke total loss nikaalo
    
    train_loss /= len(train_loader.dataset)  # Average loss per sample

    # --- VALIDATION PHASE (Exam - No Backprop) ---
    model.eval()  # Model ko batao ab exam mode hai (Dropout/BatchNorm ke liye)
    val_loss = 0.0
    with torch.no_grad():  # GRAPH BAND KARO (Memory bachao, gradients mat nikaalo)
        for batch_x, batch_y in val_loader:
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            val_loss += loss.item() * batch_x.size(0)
    val_loss /= len(val_loader.dataset)

    # 6. Print
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")