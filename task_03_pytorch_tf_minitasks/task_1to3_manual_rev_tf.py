import tensorflow as tf

# Data
x = tf.constant([[1.0], [2.0], [3.0], [4.0]])
y_true = 2 * x + 1

# Weights (TF me Variable banana padta hai)
w = tf.Variable(tf.random.normal((1, 1)))
b = tf.Variable(tf.random.normal((1,)))

lr = 0.01
for epoch in range(100):
    with tf.GradientTape() as tape:
        y_pred = x * w + b
        loss = tf.reduce_mean((y_pred - y_true) ** 2)
    
    # Gradients nikaalo
    grads = tape.gradient(loss, [w, b])
    
    # Update karo (assign kar rahe hain)
    w.assign(w - lr * grads[0])
    b.assign(b - lr * grads[1])
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss.numpy():.4f}")

print(f"Final: w={w.numpy()[0][0]:.2f}, b={b.numpy()[0]:.2f}")