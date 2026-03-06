import matplotlib.pyplot as plt
from data_loader import load_mnist

# load dataset
(x_train, y_train), _ = load_mnist()

plt.figure(figsize=(6,6))

# show first 9 samples
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[i].reshape(28,28), cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("figures/mnist_samples.png", dpi=300)
plt.show()