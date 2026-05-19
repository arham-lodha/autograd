import numpy as np;
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from autograd import grad


# Data
X, y = fetch_california_housing(return_X_y=True);

scaler = StandardScaler();
X = scaler.fit_transform(X);
y = (y - y.mean()) / y.std();

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42);

# --- Setup and precompile MSE and its gradient ---
def mse(w, b, train_X, train_y):
    predicted = train_X @ w + b;
    error = predicted - train_y;
    return (error * error).mean();


dmse, eval_mse = grad(mse, argnums=(0, 1));

# --- Train linear regression model with gradient descent ---
lr = 0.01;
w = np.zeros(X.shape[1]);
b = np.zeros(1);

for epoch in range(1000):
    grad_w, grad_b = dmse(w, b, X_train, y_train);

    w -= lr * grad_w;
    b -= lr * grad_b;

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, MSE: {eval_mse(w, b, X_train, y_train)}");

# --- Evaluate on test set ---
test_mse = eval_mse(w, b, X_test, y_test);
print(f"Test MSE: {test_mse}");
