import torch

# 1. Data (Task 1 revision)
x = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)  # Shape (4,1)
y_true = 2 * x + 1

# 2. Weights (Task 2 revision - requires_grad)
# Hum khud weight aur bias bana rahe hain (initially random)
w = torch.randn(1, 1, requires_grad=True)  # Random weight
b = torch.randn(1, requires_grad=True)     # Random bias

print(f"Initial Weight: {w.item():.2f}, Initial Bias: {b.item():.2f}")

# 3. Training Loop (Task 3 revision - Math + Backward)
learning_rate = 0.01
for epoch in range(100):
    # Forward Pass: y = w*x + b (Manual Math)
    y_pred = x * w + b
    
    # Loss: Mean Squared Error (Task 3 revision)
    loss = ((y_pred - y_true) ** 2).mean()
    
    # Backward Pass (Autograd - Task 4 revision)
    # PyTorch automatically computes gradients of w and b
    loss.backward()
    
    # Update weights (Manual Step - Dekho kaise hota hai)
    with torch.no_grad():  # IMPORTANT: Update karte waqt graph band karo
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
        # Zero out gradients (Task 5 revision - kyun? warna accumulate hoga)
        w.grad.zero_()
        b.grad.zero_()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, w={w.item():.2f}, b={b.item():.2f}")

print(f"\nFinal Learned: w={w.item():.2f} (Expected ~2.0), b={b.item():.2f} (Expected ~1.0)")