import matplotlib.pyplot as plt

layers = [
    "Input\n(28x28x1)",
    "Conv2D\n32 filters",
    "BatchNorm",
    "MaxPooling",
    "Conv2D\n64 filters",
    "BatchNorm",
    "MaxPooling",
    "Flatten",
    "Dense\n128",
    "Dropout\n0.5",
    "Softmax\nOutput (10)"
]

plt.figure(figsize=(10,2))

for i, layer in enumerate(layers):
    plt.text(i, 0, layer,
             bbox=dict(boxstyle="round,pad=0.4"),
             ha='center')

plt.axis('off')
plt.savefig("figures/ocnn_architecture.png", dpi=300, bbox_inches="tight")
plt.show()