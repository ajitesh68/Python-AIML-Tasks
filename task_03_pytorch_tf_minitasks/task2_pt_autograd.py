# ========== TASK : AUTOGRAD (PyTorch) ==========
import torch 

x = torch.tensor(2.0, requires_grad=True)
print("x:",x)

y = x**2
print("y:",y)

y.backward()

print("dy/dx at x=2:",x.grad)
