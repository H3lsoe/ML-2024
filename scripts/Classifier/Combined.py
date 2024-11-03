import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from KNN_data import *  # Ensure this imports X and y appropriately
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection

# ---------------------------
# Logistic Regression Section
# ---------------------------

print("Starting Logistic Regression Evaluation with Regularization...\n")

# Define range of values for C (inverse of λ)
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
print(f"Testing Logistic Regression with C values: {C_values}\n")

# Initialize Leave-One-Out Cross-Validation
loo = LeaveOneOut()

# Initialize lists to store predictions and true labels for each C
accuracy_scores_lr = {C: [] for C in C_values}

N = len(X)  # Number of samples
i = 0

# Iterate through each fold
for train_index, test_index in loo.split(X, y):
    # Extract training and test sets for the current CV fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Iterate through each C value
    for C in C_values:
        # Initialize and train the Logistic Regression model
        model_lr = lm.LogisticRegression(C=C, max_iter=1000, solver='liblinear')  # 'liblinear' is good for small datasets
        model_lr.fit(X_train, y_train)

        # Predict the class for the test sample
        y_pred = model_lr.predict(X_test)[0]

        # Store the prediction
        accuracy_scores_lr[C].append(y_pred == y_test[0])

    i += 1
    if i % 100 == 0 or i == N:
        print(f"Processed {i}/{N} LOOCV folds.")

# Calculate accuracy for each C
accuracies_lr = {}
for C in C_values:
    accuracies_lr[C] = np.mean(accuracy_scores_lr[C])
    print(f"Accuracy for Logistic Regression with C={C}: {accuracies_lr[C] * 100:.2f}%")

# Select the best C (with highest accuracy)
best_C_lr = max(accuracies_lr, key=accuracies_lr.get)
best_accuracy_lr = accuracies_lr[best_C_lr]
best_misclass_rate_lr = 1 - best_accuracy_lr

print(f"\nBest C value for Logistic Regression: {best_C_lr} with Accuracy: {best_accuracy_lr * 100:.2f}%")
print(f"Misclassification Rate: {best_misclass_rate_lr:.3f}\n")

# ---------------------------
# K-Nearest Neighbors Section
# ---------------------------

print("Starting K-Nearest Neighbors Evaluation with LOOCV...\n")

# Define the range of neighbors to evaluate (1 to 29)
L = list(range(1, 30))

# Initialize Leave-One-Out Cross-Validation
CV = model_selection.LeaveOneOut()
i = 0

# Store predictions and true labels
yhat_knn = []
y_true_knn = []
N = len(X)  # Number of samples

for train_index, test_index in CV.split(X, y):
    # Optional: Print progress every 100 folds
    if (i + 1) % 100 == 0 or (i + 1) == N:
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

    yhat_knn.append(dy)
    y_true_knn.append(y_test[0])
    i += 1

# Convert lists to NumPy arrays for easier manipulation
yhat_knn = np.array(yhat_knn)       # Shape: (N, len(L))
y_true_knn = np.array(y_true_knn)   # Shape: (N,)

# Compute accuracy for each classifier (each k)
accuracies_knn = []
for idx, l in enumerate(L):
    accuracy = np.mean(yhat_knn[:, idx] == y_true_knn)
    accuracies_knn.append(accuracy)
    print(f'Accuracy for k={l}: {accuracy * 100:.2f}%')

# Find the best k
best_k = L[np.argmax(accuracies_knn)]
best_accuracy_knn = max(accuracies_knn)
best_misclass_rate_knn = 1 - best_accuracy_knn

print(f"\nBest k for KNN: {best_k} with Accuracy: {best_accuracy_knn * 100:.2f}%")
print(f"Misclassification Rate: {best_misclass_rate_knn:.3f}\n")

# ---------------------------
# Visualization Section
# ---------------------------

# Plotting Logistic Regression Accuracies
plt.figure(figsize=(10, 6))
plt.plot(C_values, np.array(list(accuracies_lr.values())) * 100, marker='o', label='Logistic Regression')
plt.xlabel('Inverse Regularization Strength (C)')
plt.ylabel('Accuracy (%)')
plt.title('Logistic Regression Accuracy for Different C Values')
plt.xscale('log')  # Since C spans multiple orders of magnitude
plt.grid(True)
plt.legend()
plt.show()

# Plotting KNN Accuracies
plt.figure(figsize=(10, 6))
plt.plot(L, np.array(accuracies_knn) * 100, marker='o', label='K-Nearest Neighbors')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy (%)')
plt.title('KNN Classifier Accuracy for Different k Values')
plt.grid(True)
plt.legend()
plt.show()

# Optional: Compare Logistic Regression and KNN
plt.figure(figsize=(10, 6))
plt.plot(C_values, np.array(list(accuracies_lr.values())) * 100, marker='o', label='Logistic Regression')
plt.plot(L, np.array(accuracies_knn) * 100, marker='x', label='K-Nearest Neighbors')
plt.xlabel('Parameter Value')
plt.ylabel('Accuracy (%)')
plt.title('Logistic Regression vs. KNN Accuracy')
plt.xscale('log')
plt.grid(True)
plt.legend()
plt.show()

# ---------------------------
# Final Outputs
# ---------------------------

# Display classification results for Logistic Regression with best C
print("Final Logistic Regression Model Evaluation with Best C")
print(f"Overall misclassification rate: {best_misclass_rate_lr:.3f}")

# Fit the Logistic Regression model on the entire dataset with best C for visualization
model_final_lr = lm.LogisticRegression(C=best_C_lr, max_iter=1000, solver='liblinear')
model_final_lr.fit(X, y)
y_est_white_prob_lr = model_final_lr.predict_proba(X)[:, 0]

# Plot predicted probabilities
plt.figure(figsize=(10, 6))
class0_ids = np.nonzero(y == 0)[0].tolist()
plt.plot(class0_ids, y_est_white_prob_lr[class0_ids], ".y", label="Besni")
class1_ids = np.nonzero(y == 1)[0].tolist()
plt.plot(class1_ids, y_est_white_prob_lr[class1_ids], ".r", label="Kecimen")
plt.xlabel("Data object (wine sample)")
plt.ylabel("Predicted prob. of class White")
plt.legend()
plt.ylim(-0.01, 1.5)
plt.title("Logistic Regression Predicted Probabilities (Best C)")
plt.show()

print("Ran Logistic Regression and KNN Evaluations with LOOCV")

