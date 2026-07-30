import numpy as np 
from preprocessing import data,normalized,standardized,one_hot

print("=== Data Shape ===", data.shape)
print("\n=== First 5 rows ===")
print(data[:5])


print("\n=== Statistics (Maths) ===")
print(f"Mean: {np.mean(data[:,0]):2f}, Std: {np.std(data[:,0]):.2f}")


print("\n=== Normalized (first 2 cols, first 3 rows) ===")
print(normalized[:3, :2])

print("\n=== Standardized (first 2 cols, first 3 rows) ===")
print(standardized[:3,:2])


print("\n=== One-hot encoding shape ===", one_hot.shape) 


high_math = data[:,0]>80 
print(f"\nStudents with Maths > 80: {len(high_math)}")



