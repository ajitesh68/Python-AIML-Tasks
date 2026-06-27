import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# 1. Data Generate (200 points)
tf.random.set_seed(42)
X = tf.random.normal((200, 1)) * 10
y = 2 * X + 1 + tf.random.normal((200, 1)) * 0.5

# 2. Train/Val Split (Manual)
val_size = 40
train_data = tf.data.Dataset.from_tensor_slices((X[:-val_size], y[:-val_size]))
val_data = tf.data.Dataset.from_tensor_slices((X[-val_size:], y[-val_size:]))

# 3. Pipeline (Shuffle + Batch)
train_data = train_data.shuffle(buffer_size=160).batch(16)
val_data = val_data.batch(16)  # Validation me shuffle nahi karte

# 4. Model & Compile
model = Sequential()
model.add([Dense(1, input_shape=(1,))])
model.compile(optimizer='sgd', loss='mse')

# 5. Train with Validation Data (TF automatic handles eval mode)
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    verbose=0
)

# 6. Print
print("Epoch | Train Loss | Val Loss")
for i in range(0, 50, 10):
    print(f"{i+1:5d} | {history.history['loss'][i]:.4f} | {history.history['val_loss'][i]:.4f}")