import tensorflow as tf
import matplotlib.pyplot as plt
from data_loader import load_mnist
import os
import pickle

# load data
(x_train, y_train), (x_test, y_test) = load_mnist()

# proposed OCNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())

# train
history = model.fit(
    x_train, y_train,
    epochs=5,
    validation_data=(x_test, y_test),
    batch_size=64
)

# evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print("OCNN Test Accuracy:", test_acc)

os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# save model
model.save("models/ocnn_model.keras")

with open("models/ocnn_history.pkl", "wb") as f:
    pickle.dump(history.history, f)
    
# accuracy plot
plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("OCNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.savefig("figures/ocnn_accuracy.png")

# loss plot
plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("OCNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])
plt.savefig("figures/ocnn_loss.png")

plt.show()