import tensorflow as tf 

x = tf.Variable(2.0)
print(x.numpy())

with tf.GradientTape() as tape:
    y = x**2
    print("y = x^2:", y.numpy())

grad = tape.gradient(y,x)
print("dy/dx at x=2:", grad.numpy())


#tf.constant() se jo tensor banta hai wo immutable hota hai – uski value change nahi kar sakte.
#tf.Variable() se bana tensor mutable hai – uski value change kar sakte ho (jaise weights update karte ho training mein).

