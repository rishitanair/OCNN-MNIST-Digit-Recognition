from data_loader import load_mnist

(x_train, y_train), (x_test, y_test) = load_mnist()

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)