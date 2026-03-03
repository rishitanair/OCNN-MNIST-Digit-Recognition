import pickle
import matplotlib.pyplot as plt

# load histories
with open("models/baseline_history.pkl", "rb") as f:
    baseline = pickle.load(f)

with open("models/ocnn_history.pkl", "rb") as f:
    ocnn = pickle.load(f)

# accuracy comparison
plt.figure(figsize=(8,5))

plt.plot(baseline["val_accuracy"], label="Baseline CNN")
plt.plot(ocnn["val_accuracy"], label="OCNN")

plt.title("Validation Accuracy Convergence")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()

plt.savefig("figures/convergence_comparison.png")
plt.show()