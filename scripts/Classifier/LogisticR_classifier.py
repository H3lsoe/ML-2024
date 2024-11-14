import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from KNN_data import *
import matplotlib.pyplot as plt
import pandas as pd

num_features = X.shape[1]
feature_names = [f"Feature {i}" for i in range(num_features)]

k = 10

kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=69)

yhat_lr = []
y_true_lr = []

i = 0
N = len(X)

lambda_value = 0.01
C_value = 1 / lambda_value  # C = 100

for train_index, test_index in kf.split(X, y):
    print(f"Logistic Regression k-Fold CV fold: {i + 1}/{k}")

    # Split data into training and testing sets for the current fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model_lr = lm.LogisticRegression(
        C=C_value,
        penalty='l2',
        solver='liblinear',
        max_iter=1000,
        random_state=42
    )
    model_lr.fit(X_train, y_train)

    y_pred = model_lr.predict(X_test)

    yhat_lr.extend(y_pred)
    y_true_lr.extend(y_test)

    i += 1

yhat_lr = np.array(yhat_lr)
y_true_lr = np.array(y_true_lr)

accuracy_lr = accuracy_score(y_true_lr, yhat_lr)
misclass_rate_lr = 1 - accuracy_lr

print("\nLogistic Regression k-Fold CV Results:")
print(f"Accuracy: {accuracy_lr * 100:.2f}%")
print(f"Misclassification Rate: {misclass_rate_lr:.3f}")

# Train the final model on the entire dataset
model_full_lr = lm.LogisticRegression(
    C=C_value,
    penalty='l2',
    solver='liblinear',
    max_iter=1000,
    random_state=42
)
model_full_lr.fit(X, y)

coefficients = model_full_lr.coef_[0]
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
})

coef_df_sorted = coef_df.sort_values(by='Abs_Coefficient', ascending=False)

print("\nFeature Importance based on Logistic Regression Coefficients:")
print(coef_df_sorted[['Feature', 'Coefficient', 'Abs_Coefficient']])

N = 10
plt.figure(figsize=(10, 6))
plt.barh(coef_df_sorted['Feature'].head(N)[::-1], coef_df_sorted['Abs_Coefficient'].head(N)[::-1], color='skyblue')
plt.xlabel('Absolute Coefficient Value')
plt.title('Logistic Regression features')
plt.grid(True)
plt.show()

y_est_white_prob_lr = model_full_lr.predict_proba(X)[:, 0]

plt.figure(figsize=(10, 6))
class0_ids = np.nonzero(y == 0)[0].tolist()
plt.plot(class0_ids, y_est_white_prob_lr[class0_ids], ".y", label="Class 0")
class1_ids = np.nonzero(y == 1)[0].tolist()
plt.plot(class1_ids, y_est_white_prob_lr[class1_ids], ".r", label="Class 1")
plt.xlabel("Data Object (Sample Index)")
plt.ylabel("Predicted Probability of Class 0")
plt.legend()
plt.ylim(-0.01, 1.05)
plt.title("Logistic Regression Predicted Probabilities (λ = 0.01)")
plt.grid(True)
plt.show()

print("Ran Logistic Regression k-Fold CV Evaluation with λ = 0.01")

