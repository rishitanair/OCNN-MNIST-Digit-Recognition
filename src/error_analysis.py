import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_mnist

# load data
(_, _), (x_test, y_test) = load_mnist()

# load model
model = tf.keras.models.load_model("models/ocnn_model.keras")

# predictions
predictions = np.argmax(model.predict(x_test), axis=1)

# find wrong predictions
wrong_idx = np.where(predictions != y_test)[0]

plt.figure(figsize=(10,6))

for i in range(9):
    idx = wrong_idx[i]
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[idx].reshape(28,28), cmap="gray")
    plt.title(f"True:{y_test[idx]} Pred:{predictions[idx]}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("figures/misclassified_examples.png")
plt.show()