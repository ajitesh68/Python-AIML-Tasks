import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist

# ---------- LINE 1-4: LOAD DATA ----------
# MNIST dataset load karo. Ye (x_train, y_train) aur (x_test, y_test) return karta hai.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ---------- LINE 6-8: RESHAPE + NORMALIZE ----------
# CNN ko input shape (Height, Width, Channels) chahiye. MNIST me channel nahi hai, toh (28,28) -> (28,28,1) karte hain.
# Normalize: Values ko 0-255 se 0-1 me laana.
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# ---------- LINE 10-19: CNN MODEL (Sequential) ----------
model = Sequential([
    # Conv2D: 32 filters, kernel size (3,3), activation ReLU.
    # input_shape: (28, 28, 1) batana zaroori hai pehli layer me.
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),  # Image size aadhi kar do.
    
    Conv2D(64, (3, 3), activation='relu'),  # 64 filters, kernel (3,3)
    MaxPooling2D((2, 2)),  # Image size aur aadhi.
    
    # Flatten: 2D feature maps ko 1D vector me badalto hai taaki Dense layer de sake.
    Flatten(),
    
    # Dense Hidden Layer: 128 neurons.
    Dense(128, activation='relu'),
    # Dropout: 50% neurons randomly off kar do (Overfitting rokne ke liye).
    Dropout(0.5),
    
    # Output Layer: 10 classes (digits 0-9). Softmax activation se probabilities ban jaati hain.
    Dense(10, activation='softmax')
])

# ---------- LINE 21-22: COMPILE MODEL ----------
# optimizer='adam' -> Adaptive learning rate.
# loss='sparse_categorical_crossentropy' -> Kyunki labels integers (0,1,2...) hain.
# metrics=['accuracy'] -> Training ke saath accuracy bhi calculate hoti rahegi.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ---------- LINE 24-26: TRAIN ----------
# batch_size=64, epochs=5.
# validation_data=(x_test, y_test) -> Har epoch ke end pe test data pe accuracy check karo.
print("Training started...")
history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test, y_test), verbose=1)

# ---------- LINE 28-30: FINAL EVALUATION ----------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Task 11 Completed! Test Accuracy: {test_acc*100:.2f}%")