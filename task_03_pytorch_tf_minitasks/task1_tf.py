#problem 1:-  Create a tf tensor and change datatype 

from numpy import mean
import tensorflow as tf 

x = tf.constant([[1,2,3],
                [4,5,6]], dtype=tf.float32)

print("Shape:", x.shape)
print("Element (0,1):", x[0,1].numpy())

x_int = tf.cast(x, tf.int64)
print("Converted array:", x_int.numpy(), x_int.dtype)


# ========== TASK 2: RANDOM TENSORS & DEVICE (TensorFlow) ========== 

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print (f"GPU(s) available: {[gpu.name for gpu in gpus]}")
else:
    print("No GPU detected, using CPU.")

rand_uniform = tf.random.uniform(minval=0, maxval=1, shape=(3,4))
rand_normal = tf.random.normal(mean=0,stddev=1,shape=(2,3))
rand_int = tf.random.uniform(minval=0, maxval=10, shape=(3,3))

#.numpy() – Converts tensor to NumPy array for printing.
print("Random Uniform",rand_uniform.numpy())
print("Random Normal",rand_normal.numpy())# Normal distribution
print("Random Integers",rand_int.numpy())

#You don't need explicit .to(device) – TF tries to use GPU automatically.
#To check if a tensor is on GPU: tensor.device returns something like '/job:localhost/replica:0/task:0/device:GPU:0'.

if gpus:
    with tf.device('/gpu:0'):#with tf.device('/GPU:0'): – Context manager to place operations on specific device.
                             #Daily use: Rarely needed because TF auto-places, but useful for multi-GPU or forcing CPU for debugging.
        rand_gpu = tf.random.uniform(shape = (2,2))
        print("\nTensor placed on GPU:", rand_gpu.device)
        print(rand_gpu.numpy())
    rand_cpu = tf.identity(rand_gpu)#tf.identity(rand_gpu) – Creates a new tensor on CPU (moves data).
    print("Back to CPU:", rand_cpu.device)
else:
    print("GPU not available, skipping device move.")
    


# ========== TASK 3: MATH OPERATIONS (TensorFlow) ==========
import tensorflow as tf

print("=== ELEMENT-WISE OPERATIONS ===")
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])
print("a:\n", a.numpy())
print("b:\n", b.numpy())
print("a + b:\n", (a + b).numpy())
print("a * b:\n", (a * b).numpy())

print("\n=== MATRIX MULTIPLICATION ===")
c = tf.constant([[1, 2], [3, 4]])
d = tf.constant([[5, 6, 7], [8, 9, 10]])
print("c @ d:\n", (c @ d).numpy())   # or tf.matmul(c, d)

print("\n=== BROADCASTING ===")
print("a + 10:\n", (a + 10).numpy())
row_vec = tf.constant([10, 20])
print("a + row_vec:\n", (a + row_vec).numpy())
col_vec = tf.constant([[10], [20]])
print("a + col_vec:\n", (a + col_vec).numpy())

