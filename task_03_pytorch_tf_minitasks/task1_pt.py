#problem 1:-  Create a pytorch tensor and change datatype

import numpy as np
import torch 

x=torch.tensor([[1, 2, 3],
              [4, 5, 6]], dtype=torch.float32)
              

print("shape",x.shape)
print("Element (0,1):", x[0,1])

x_int = x.to(torch.int64)
print("Converted array:", x_int.dtype)


# ========== TASK 2: RANDOM TENSORS & DEVICE (PyTorch) ==========

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')
print("device", device)

rand_uniform = torch.rand(3,4)
rand_normal = torch.randn(2,5)
rand_int = torch.randint(0,10,(3,3))

print("Uniform:", rand_uniform)
print("Normal:", rand_normal)
print("Integers:", rand_int)

if torch.cuda.is_available():
    rand_uniform_gpu = rand_uniform.to(device)
    print("\nTensor on GPU:", rand_uniform_gpu.device)

    rand_uniform_cpu = rand_uniform_gpu.cpu()
    print("Back to CPU:", rand_uniform_cpu.device)
else:
    print("\nGPU not available, skipping device move.")


# ========== TASK 3: MATH OPERATIONS (PyTorch) ========== 

a = torch.tensor([[1,2],[3,4]])
b= torch.tensor([[4,5],[6,7]])

print("Element-wise addition:", (a+b))
print("Matrix multiplication:", (a@b))
print("a:\n",a)
print("b:\n",b)
print("Element Wise multiplication:", (a*b))

print("\n=== BROADCASTING (automatic stretching) ===")
print("a + 10:\n", a + 10)               # scalar broadcast
row_vec = torch.tensor([10, 20])         # shape (2,)
print("a + row_vec:\n", a + row_vec)     # (2,2) + (2,) -> (2,2)
col_vec = torch.tensor([[10], [20]])     # shape (2,1)
print("a + col_vec:\n", a + col_vec)     # (2,2) + (2,1) -> (2,2)


