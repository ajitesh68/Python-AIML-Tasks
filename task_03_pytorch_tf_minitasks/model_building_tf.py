
import tensorflow as tf 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense 


x = tf.constant([[1.0], [2.0], [3.0], [4.0]])
y_true = 2*x + 1

model = Sequential([Dense(1, input_shape=(1,))])
#Dense(1, ...) – Is layer me 1 neuron hoga
#Input shape must be a tuple
#input_shape=(1,) – Batata hai ki ek sample mein kitne features hain.
#Yahan x ki shape (4,1) hai. Har sample me 1 feature (number).
#Tuple (1,) ka matlab: "ek feature, aur wo feature ki value ek scalar hai".
#Maan le tera data 100 images hai, har image 28x28 pixel (grayscale) – to input_shape=(28,28,1).
#Yahan 28 rows, 28 cols, 1 channel. Is tarah layer automatically 28*28*1 = 784 weights input se connect ho jayenge.
model.compile(optimizer='sgd',loss='mse')

model.fit(x,y_true,epochs=100,verbose=0)

weights = model.layers[0].get_weights()
print("Learned weight", weights[0][0][0])
print("Learned bias", weights[1][0])
# model.layers[0] – first layer (Dense).
# .get_weights() – returns list: [weights_array, bias_array]. Weights shape (1,1), bias shape (1,).
