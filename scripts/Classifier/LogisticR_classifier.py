import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from KNN_data import *  # Ensure this imports X and y appropriately
import matplotlib.pyplot as plt

# Initialize Leave-One-Out Cross-Validation
loo = LeaveOneOut()

# Initialize lists to store predictions and true labels
yhat_lr = []
y_true_lr = []

# Initialize counter
i = 0
N = len(X)

# Iterate through each fold
for train_index, test_index in loo.split(X, y):
    print(f"Logistic Regression LOOCV fold: {i + 1}/{N}")

    # Split data into training and testing sets for the current fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Initialize and train the Logistic Regression model
    model_lr = lm.LogisticRegression(max_iter=1000)  # Increase max_iter if needed
    model_lr.fit(X_train, y_train)

    # Predict the class for the test sample
    y_pred = model_lr.predict(X_test)[0]

    # Store the prediction and the true label
    yhat_lr.append(y_pred)
    y_true_lr.append(y_test[0])

    i += 1

# Convert lists to NumPy arrays for easier manipulation
yhat_lr = np.array(yhat_lr)
y_true_lr = np.array(y_true_lr)

# Calculate overall accuracy
accuracy_lr = accuracy_score(y_true_lr, yhat_lr)
misclass_rate_lr = 1 - accuracy_lr

print("\nLogistic Regression LOOCV Results:")
print(f"Accuracy: {accuracy_lr * 100:.2f}%")
print(f"Misclassification Rate: {misclass_rate_lr:.3f}")

# (Optional) If you have KNN accuracies stored, you can compare them here
# For example:
# accuracies_knn = [...]  # Replace with your KNN accuracies list
# plt.figure(figsize=(10, 6))
# plt.plot(L, np.array(accuracies_knn) * 100, marker='o', label='KNN')
# plt.axhline(y=accuracy_lr * 100, color='r', linestyle='--', label='Logistic Regression')
# plt.xlabel('Number of Neighbors (k)')
# plt.ylabel('Accuracy (%)')
# plt.title('KNN vs. Logistic Regression Accuracy')
# plt.legend()
# plt.grid(True)
# plt.show()

# Display classification results similar to your original Logistic Regression evaluation

# (Optional) If you want to visualize probabilities as before, ensure to do it after LOOCV
# For example, you can plot the predicted probabilities for the entire dataset
model_full_lr = lm.LogisticRegression(max_iter=1000)
model_full_lr.fit(X, y)
y_est_white_prob_lr = model_full_lr.predict_proba(X)[:, 0]

f = plt.figure()
class0_ids = np.nonzero(y == 0)[0].tolist()
plt.plot(class0_ids, y_est_white_prob_lr[class0_ids], ".y", label="Besni")
class1_ids = np.nonzero(y == 1)[0].tolist()
plt.plot(class1_ids, y_est_white_prob_lr[class1_ids], ".r", label="Kecimen")
plt.xlabel("Data object (raisin samples)")
plt.ylabel("Predicted prob. of class")
plt.legend()
plt.ylim(-0.01, 1.5)
plt.title("Logistic Regression Predicted Probabilities")
plt.show()

print("Ran Logistic Regression LOOCV Evaluation")

