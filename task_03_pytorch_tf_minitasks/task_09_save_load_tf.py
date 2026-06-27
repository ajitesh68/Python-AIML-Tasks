import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import os

# ----- STEP 1: Model Banayen aur Train Karein -----
print("Step 1: Training Model...")
tf.random.set_seed(42)

X = tf.constant([[1.0], [2.0], [3.0], [4.0]])
y_true = 2 * X + 1

model = Sequential([Dense(1, input_shape=(1,))])
model.compile(optimizer='sgd', loss='mse')
model.fit(X, y_true, epochs=100, verbose=0)

weights = model.layers[0].get_weights()
print(f"Trained Weight: {weights[0][0][0]:.2f}, Bias: {weights[1][0]:.2f}")

# ----- STEP 2: Model ko DISK mein Save Karein -----
print("\nStep 2: Saving Model...")
model.save('linear_model_tf.keras')  # 'keras' format is recommended over 'h5'
print("Model saved successfully in 'linear_model_tf.keras'!")

# ----- STEP 3: Load Karein aur Predict Karein (Fresh Start) -----
print("\nStep 3: Loading Model & Predicting (Without Training!)...")

# Saved model ko load karo
loaded_model = tf.keras.models.load_model('linear_model_tf.keras')

# Naya data
new_x = tf.constant([[5.0], [6.0]])

# Prediction (TF by default inference mode me hi rehta hai)
predictions = loaded_model.predict(new_x, verbose=0)

print(f"Input: {new_x.numpy().flatten().tolist()}")
print(f"Predictions (should be ~11 and 13): {predictions.flatten().tolist()}")