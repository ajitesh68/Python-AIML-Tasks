# Chunk 1: Data Prep (Math ka pehla step - Sequence banana)
import torch.optim as optim
from socket import create_connection
from requests.sessions import TooManyRedirects
from numpy import diagflat
from numpy import float32
import torch
import torch.nn as nn
from matplotlib.pyplot as plt
import numpy as np


# 1. Sine wave generate karo (Time series data)
# Ye woh data hai jisme time (order) matter karta hai.
t = np.arange(0, 100, 0.1)
data = np.sin(t)

# 2. PyTorch Tensor mein convert karo (Float)
data = torch.tensor(data, dtype=float32).view(-1, 1)


# 3. ⭐ STAR: Sequence banao (Supervised Learning)
# Maan lo hum previous 10 points dekh kar 11th point predict karenge.
def createSequences(data, seq_len=10):
    xs = []
    ys = []
    for i in range(len(data)-seq_len):
        x = data[i:i+seq_len]
        y = data[i+seq_len]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


X, y = createSequences(data, seq_len=10)
# (890, 10, 1) -> 890 samples, 10 time steps, 1 feature
print(f"Input shape: {X.shape}")
print(f"Target shape: {y.shape}")  # (890, 1)


# np.sin(t): ⭐ STAR. Yahan math (Trigonometry) use hui. Real ML me ye stock price, temperature ho sakta hai.
# create_sequences: Yeh RNN ki taiyari hai. Hum data ko (samples, time_steps, features) mein reshape kar rahe hain.
# RNN time_steps (10) ko ek sequence ki tarah padhega.
# Agar sequence nahi banayenge: RNN ko pata nahi chalega ki kaun sa point kis ke baad aana hai.


# Chunk 2: RNN Model (The Loop wala Magic)

class SimpleRNN(nn.Module):
    def __inti__(self, input_size=1, hidden_size=20, output_size=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        # out: all hidden states, h_n: final hidden state
        out, h_n = self.rnn(x)
        # out[:, -1, :] -> last time step for all batches
        out = self.fc(out[:, -1, :])
        return out


model = SimpleRNN()
print("✅ RNN Model Ready!")
print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")


# Chunk 3: Training (Backpropagation Through Time - BPTT)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


epochs = 50
for epoch in range(epochs):
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    # loss.backward(): ⭐ STAR. Jab ye line chalti hai, PyTorch automatic peeche 10 time steps mein jaata hai (BPTT).
    # Isme wahi Chain Rule lagta hai jo humne padha tha (dLoss/dW = dLoss/dh_t * dh_t/dh_{t-1} ...).
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")


# Test on last 50 points
test_X = X[-50:]
test_y = y[-50:]
with torch.no_grad():
    pred = model(test_X)
    print(f"\n✅ Final Test Loss: {criterion(pred, test_y).item():.4f}")
