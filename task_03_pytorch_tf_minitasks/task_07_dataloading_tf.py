import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense



tf.random.set_seed(42)
X = tf.random.normal((100,1))*10
y = 2*X + 1 + tf.random.normal((100,1))*0.5 


dataset = tf.data.Dataset.from_tensor_slices((X,y))


dataset = dataset.shuffle(buffer_size=100)
dataset = dataset.batch(16)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

model = Sequential([Dense(1,input_shape=(1,))])
model.compile(optimizer='sgd',loss='mse')

history=model.fit(dataset,epochs=50,verbose=0)

print(f"Finaly loss: {history.history['loss'][-1]:.4f}")