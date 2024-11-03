import numpy as np

# Requires data from exercise 1.5.1
from KNN_data import *
from matplotlib.pyplot import figure, plot, show, xlabel, ylabel, title, grid
from scipy.io import loadmat
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier

# This script creates predictions from multiple KNN classifiers using cross-validation

# Define the range of neighbors to evaluate (1 to 29)
L = list(range(1, 30))

# Initialize Leave-One-Out Cross-Validation
CV = model_selection.LeaveOneOut()
i = 0

# Store predictions and true labels
yhat = []
y_true = []
N = len(X)  # Number of samples

for train_index, test_index in CV.split(X, y):
    print(f"Cross-validation fold: {i + 1}/{N}")

    # Extract training and test sets for the current CV fold
    X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = y[train_index], y[test_index]

    # Fit classifiers for each k and predict the test point
    dy = []
    for l in L:
        knclassifier = KNeighborsClassifier(n_neighbors=l)
        knclassifier.fit(X_train, y_train)
        y_est = knclassifier.predict(X_test)
        dy.append(y_est[0])  # Append the scalar prediction

    yhat.append(dy)
    y_true.append(y_test[0])
    i += 1

# Convert lists to NumPy arrays for easier manipulation
yhat = np.array(yhat)       # Shape: (N, len(L))
y_true = np.array(y_true)   # Shape: (N,)

# Print the predictions matrix (optional)
#print(yhat)

# Compute accuracy for each classifier (each k)
accuracies = []
for idx, l in enumerate(L):
    accuracy = np.mean(yhat[:, idx] == y_true)
    accuracies.append(accuracy)
    print(f'Accuracy for k={l}: {accuracy * 100:.2f}%')

# Optional: Plot accuracy vs. number of neighbors
figure(figsize=(10, 6))
plot(L, np.array(accuracies) * 100, marker='o')
xlabel('Number of Neighbors (k)')
ylabel('Accuracy (%)')
title('KNN Classifier Accuracy for Different k Values')
grid(True)
show()

