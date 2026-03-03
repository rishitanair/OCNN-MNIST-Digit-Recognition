import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from data_loader import load_mnist
import os

# load data
(_, _), (x_test, y_test) = load_mnist()

# load trained models
baseline = tf.keras.models.load_model("models/baseline_cnn.h5")
ocnn = tf.keras.models.load_model("models/ocnn_model.keras")

# predictions
baseline_pred = np.argmax(baseline.predict(x_test), axis=1)
ocnn_pred = np.argmax(ocnn.predict(x_test), axis=1)

# accuracy comparison
baseline_acc = np.mean(baseline_pred == y_test)
ocnn_acc = np.mean(ocnn_pred == y_test)

print("Baseline Accuracy:", baseline_acc)
print("OCNN Accuracy:", ocnn_acc)

os.makedirs("figures", exist_ok=True)

# comparison bar chart
plt.figure()
plt.bar(["Baseline CNN", "OCNN"], [baseline_acc, ocnn_acc])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.savefig("figures/model_comparison.png")

# confusion matrix (OCNN)
cm = confusion_matrix(y_test, ocnn_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("OCNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("figures/confusion_matrix_ocnn.png")

print("\nClassification Report (OCNN):")
print(classification_report(y_test, ocnn_pred))

plt.show()