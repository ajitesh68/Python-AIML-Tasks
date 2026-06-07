import tensorflow as tf 

x = tf.Variable(2.0)
print(x.numpy())

with tf.GradientTape() as tape:
    y = x**2
    print("y = x^2:", y.numpy())

grad = tape.gradient(y,x)
print("dy/dx at x=2:", grad.numpy())


