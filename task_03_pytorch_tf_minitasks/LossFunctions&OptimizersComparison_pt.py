import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Data
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_true = 2*x + 1

# 1. Model class (yahi tak class define hai)
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()    
        self.linear = nn.Linear(1, 1)  # small correction: Linear with capital L
    def forward(self, x):
        return self.linear(x)

# 2. Training function (Class ke BAHAR, 0 indentation pe)
def train_model(loss_fn, opt_name, lr=0.01, epochs=200):  # opt_name, optimizer nahi
    model = LinearModel()
    
    # Loss
    if loss_fn == 'mse':
        criterion = nn.MSELoss()
    elif loss_fn == 'mae':
        criterion = nn.L1Loss()
    
    # Optimizer
    if opt_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif opt_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    for epoch in range(epochs):
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 50 == 0:
            print(f"{loss_fn.upper()} + {opt_name.upper()} | Epoch {epoch}: loss = {loss.item():.4f}")
    
    final_weight = model.linear.weight.item()
    final_bias = model.linear.bias.item()
    return losses, final_weight, final_bias

# 3. Run combinations
combinations = [('mse','sgd'), ('mse','adam'), ('mae','sgd'), ('mae','adam')]
results = {}
for loss_fn, opt_name in combinations:
    print(f"\n--- Training with {loss_fn.upper()} + {opt_name.upper()} ---")
    losses, w, b = train_model(loss_fn, opt_name, lr=0.01, epochs=200)
    results[(loss_fn, opt_name)] = (losses, w, b)   # store full losses list for plotting

# 4. Summary
print("\n=== SUMMARY ===")
for (loss_fn, opt_name), (losses, w, b) in results.items():
    print(f"{loss_fn.upper():3} + {opt_name.upper():4} | Final Loss: {losses[-1]:.6f} | w={w:.4f}, b={b:.4f}")

# 5. Plot
plt.figure(figsize=(10,6))
for (loss_fn, opt_name), (losses, w, b) in results.items():
    plt.plot(losses, label=f"{loss_fn.upper()} + {opt_name.upper()}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Functions & Optimizers Comparison (PyTorch)")
plt.legend()
plt.grid(True)
plt.show()