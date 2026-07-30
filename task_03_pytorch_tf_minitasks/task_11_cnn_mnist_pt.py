import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# ---------- LINE 1: DEVICE SETUP (CPU/GPU) ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- LINE 2-5: DATA TRANSFORMS (Image Preprocessing) ----------
# transforms.Compose ek pipeline hai. Isme hum ek ke baad ek transformation apply karte hain.
transform = transforms.Compose([
    # ToTensor: Image (PIL ya NumPy, range 0-255) ko PyTorch Tensor (range 0.0 to 1.0) mein badalta hai.
    # Isme shape (Height, Width, Channels) se (Channels, Height, Width) ho jaati hai.
    transforms.ToTensor(),
    # Normalize: (tensor - mean) / std. MNIST ka mean 0.1307 aur std 0.3081 hai.
    # Isse data ka center 0 ho jaata hai aur scale 1. Isse training stable hoti hai.
    # Agar normalize nahi karte, toh values 0-1 ke beech hote hain, jo chalta hai but slow convergence hota hai.
    transforms.Normalize((0.1307,), (0.3081,))
])

# ---------- LINE 6-9: LOAD DATASET (Train & Test) ----------
# MNIST: 60,000 training images, 10,000 test images. 28x28 pixels, grayscale (1 channel).
# `train=True` -> Training set download karo. `train=False` -> Test set.
# `download=True` -> Agar folder mein nahi hai toh internet se download karo.
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# ---------- LINE 10-13: DATALOADERS (Batches + Shuffle) ----------
# DataLoader dataset ko batches mein todta hai.
# `batch_size=64` -> Ek baar me 64 images feed hongi.
# `shuffle=True` -> Har epoch me data random ho jaata hai. Zaroori hai, warna model sequence yaad rakh lega.
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)  # Test me shuffle nahi karte, sirf evaluation ke liye.

# ---------- LINE 15-29: CNN MODEL CLASS (The Hero) ----------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # ---------- CONVOLUTIONAL LAYERS (Feature Extractor) ----------
        # Conv2d: Input Channels (1 - kyuki grayscale), Output Channels (32 filters), Kernel Size (3x3), Stride (1 - default).
        # 32 filters ka matlab hai ki ye 32 alag-alag features (edges, curves) find karega.
        # Input shape: (Batch, 1, 28, 28) -> Output shape: (Batch, 32, 26, 26) kyunki 3x3 filter lagane se size 2 kam ho jaata hai.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        
        # Conv2: 32 inputs se 64 outputs. Shape: (Batch, 32, 26, 26) -> (Batch, 64, 24, 24).
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        
        # MaxPool2d: Image ki size (Height, Width) ko aadha kar deta hai. (2x2 window se max value le leta hai).
        # Isse computation kam hoti hai aur overfitting bhi kam hota hai.
        # Pool ke baad shape: (Batch, 64, 12, 12) ho jaati hai (24/2 = 12).
        self.pool = nn.MaxPool2d(2, 2)
        
        # ---------- FULLY CONNECTED LAYERS (Classifier) ----------
        # Calculation: 28 -> Conv1 -> 26 -> Pool -> 13 -> Conv2 -> 11 -> Pool -> 5.
        # Final feature map size: 64 channels * 5 * 5 = 1600.
        # Yeh 1600 features ko hum 128 neurons se connect kar rahe hain.
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        
        # Dropout: Training ke time 50% neurons randomly off kar deta hai. Isse model ek particular neuron par dependent nahi ho pata, overfitting kam hoti hai.
        # Agar dropout nahi lagaya toh MNIST pe bhi overfit ho sakta hai.
        self.dropout = nn.Dropout(0.5)
        
        # Final Output layer: 128 features -> 10 classes (digits 0 se 9 tak).
        self.fc2 = nn.Linear(128, 10)

    # ---------- FORWARD PASS (Data ka flow) ----------
    def forward(self, x):
        # Conv1 -> ReLU (Non-linearity, negative values ko 0 karo) -> Pool
        x = self.pool(F.relu(self.conv1(x)))
        # Conv2 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten: 4D tensor (Batch, 64, 5, 5) ko 2D tensor (Batch, 1600) mein flatten karo.
        # view(-1, 64*5*5) - -1 ka matlab hai "batch size khud figure kar".
        x = x.view(-1, 64 * 5 * 5)
        
        # Fully Connected layers with Dropout and ReLU
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output layer me ReLU nahi lagate, kyunki CrossEntropyLoss andar se Softmax laga leta hai.
        return x

# ---------- LINE 31-35: INSTANTIATE MODEL, LOSS, OPTIMIZER ----------
model = CNN().to(device)  # Model ko GPU/CPU pe bhejo.
# CrossEntropyLoss: Classification ke liye. Andar se Softmax laga kar probability banata hai aur negative log loss calculate karta hai.
criterion = nn.CrossEntropyLoss()
# Adam optimizer: Adaptive learning rate. Isse hume learning rate manually tune nahi karni padti.
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------- LINE 37-62: TRAINING + VALIDATION LOOP ----------
epochs = 5  # MNIST simple hai, 5 epochs me hi 99% accuracy aa jaati hai.
for epoch in range(1, epochs + 1):
    # ---- TRAIN MODE ----
    model.train()
    running_loss = 0.0
    # `enumerate` se batch index (batch_idx) aur data (images, labels) milta hai.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # Data ko GPU/CPU pe le jaao.
        
        optimizer.zero_grad()          # Purane gradients zero karo (warna accumulate honge).
        output = model(data)           # Forward pass: 64 images ek saath feed karo.
        loss = criterion(output, target)  # Loss calculate karo (Prediction vs Actual).
        loss.backward()                # Backpropagation: Gradients compute karo.
        optimizer.step()               # Weights update karo.
        
        running_loss += loss.item()    # Loss ko sum karo.
    
    # ---- VALIDATION / TEST MODE (Evaluation) ----
    model.eval()  # Evaluation mode (Dropout band karo, BatchNorm freeze karo).
    correct = 0
    total = 0
    with torch.no_grad():  # CRUCIAL: Gradients compute mat karo (speed tez, memory kam).
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # torch.max returns (max_value, max_index). Humen max_index chahiye.
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)  # Batch size (e.g., 1000) add karo.
            # predicted == target -> Boolean list. Sum karo to get correct predictions.
            correct += (predicted == target).sum().item()
    
    # Loss aur Accuracy print karo.
    avg_train_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f} | Test Accuracy: {accuracy:.2f}%")

print("✅ Task 11 Completed! CNN trained on MNIST.")