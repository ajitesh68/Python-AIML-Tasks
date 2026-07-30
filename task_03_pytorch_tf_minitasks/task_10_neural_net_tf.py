import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, ReLU
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# ---------- STEP 1: DATA ----------
print("Generating non-linear data...")
X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)

# ---------- STEP 2: MODEL (2 Hidden Layers) ----------
model = Sequential([
    Dense(10, input_shape=(2,)),  # Layer 1: 2 input -> 10 neurons
    ReLU(),                       # Activation (Non-linearity)
    Dense(10),                    # Layer 2: 10 -> 10
    ReLU(),
    Dense(2, activation='softmax') # Layer 3: Output (2 classes, probability)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # Labels are integers (0,1)
              metrics=['accuracy'])

# ---------- STEP 3: TRAINING ----------
history = model.fit(X, y, epochs=200, validation_split=0.2, verbose=0)

# Print every 20 epochs
for i in range(0, 200, 20):
    print(f"Epoch {i:3d} | Train Loss: {history.history['loss'][i]:.4f} | Val Loss: {history.history['val_loss'][i]:.4f} | Val Acc: {history.history['val_accuracy'][i]:.4f}")

print("\n✅ Model trained successfully!")