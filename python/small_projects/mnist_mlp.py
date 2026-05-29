import numpy as np;
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from autograd import grad


# MNIST dataset loading and preprocessing
print("Loading MNIST dataset...");
X_raw, y_raw = fetch_openml('mnist_784', version=1, as_frame=False, return_X_y=True);
X = X_raw.astype(float) / 255.0;
y = y_raw.astype(int);

X_train, X_test, y_train, y_test = (arr for arr in train_test_split(X, y, test_size=0.2, random_state=42));
X_train = np.asarray(X_train); X_test = np.asarray(X_test);
y_train = np.asarray(y_train); y_test = np.asarray(y_test);

def one_hot(labels, num_classes):
    oh = np.zeros((labels.size, num_classes));
    oh[np.arange(labels.size), labels.astype(int)] = 1.0;
    return oh;

# Parameters

rng = np.random.default_rng(42);
w1 = rng.normal(size=(784, 128)) * 0.01;  # Input to hidden layer weights
w2 = rng.normal(size=(128, 10)) * 0.01;   # Hidden to output layer weights
b1 = np.zeros(128);  # Hidden layer biases
b2 = np.zeros(10);   # Output layer biases

def mlp_loss(W1, W2, B1, B2, X, y_oh):
    h = ((X @ W1) + B1).relu();
    logits = (h @ W2) + B2;
    log_probs = logits.softmax(axis=1).log();
    return -(log_probs * y_oh).sum(axis=1).mean();

print("Compiling loss function and its gradient...");

grad_loss, eval_loss = grad(mlp_loss, argnums=(0, 1, 2, 3));

print("Done compiling. Starting training...");


def accuracy(X: np.ndarray, y: np.ndarray):
    h = np.maximum(0, (X @ w1) + b1);  # ReLU activation
    logits = (h @ w2) + b2;
    return float(np.mean(np.argmax(logits, axis=1) == y));

# Training loop

lr = 0.01;
batch_size = 256;
n_epochs = 10;
N = len(X_train);

for epoch in range(n_epochs):
    perm = rng.permutation(N);
    X_shuffled = X_train[perm];
    Y_shuffled = y_train[perm];

    total_loss = 0.0;
    n_batches = 0;

    for i in range(0, N, batch_size):
        X_batch = X_shuffled[i:i + batch_size];
        y_batch = Y_shuffled[i:i + batch_size];

        y_batch_oh = one_hot(y_batch, 10);
        dw1, dw2, db1, db2 = grad_loss(w1, w2, b1, b2, X_batch, y_batch_oh);

        w1 -= lr * dw1;
        w2 -= lr * dw2;
        b1 -= lr * db1;
        b2 -= lr * db2;

        total_loss += float(eval_loss(w1, w2, b1, b2, X_batch, y_batch_oh));
        n_batches += 1;

    train_acc = accuracy(X_train, y_train);
    test_acc = accuracy(X_test, y_test);

    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss / n_batches:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}");

