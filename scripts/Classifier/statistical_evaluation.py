# mcnemar_evaluation.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import binom
import warnings

# Suppress any warnings for cleaner output
warnings.filterwarnings('ignore')

# Load data
from KNN_data import X, y  # Ensure KNN_data.py defines X and y appropriately

# Encode labels if they are not numerical
if y.dtype == object or isinstance(y[0], str):
    le = LabelEncoder()
    y = le.fit_transform(y)

# Define hyperparameter grids
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
lr_param_grid = {'C': C_values, 'solver': ['liblinear']}  # 'liblinear' is suitable for small datasets

k_values = list(range(1, 30))
knn_param_grid = {'n_neighbors': k_values}

# Initialize cross-validators
outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store true labels and predictions
true_labels = []
predictions_logreg = []
predictions_knn = []
predictions_baseline = []

# Enumerate outer folds
for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
    print(f"\n--- Outer Fold {fold}/10 ---")

    # Split data
    X_train_outer, X_test_outer = X[train_idx], X[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]

    # -----------------------------
    # Logistic Regression - Inner CV
    # -----------------------------

    # Initialize Logistic Regression model
    lr = LogisticRegression(max_iter=1000)

    # Set up GridSearchCV for Logistic Regression
    lr_grid = GridSearchCV(estimator=lr,
                           param_grid=lr_param_grid,
                           cv=inner_cv,
                           scoring='accuracy',  # Optimize for accuracy
                           n_jobs=-1)

    # Perform Grid Search
    lr_grid.fit(X_train_outer, y_train_outer)

    # Best estimator
    best_lr = lr_grid.best_estimator_

    # Predict on outer test set
    y_pred_lr = best_lr.predict(X_test_outer)
    predictions_logreg.extend(y_pred_lr)

    # -------------------
    # K-Nearest Neighbors
    # -------------------

    # Initialize KNN model
    knn = KNeighborsClassifier()

    # Set up GridSearchCV for KNN
    knn_grid = GridSearchCV(estimator=knn,
                            param_grid=knn_param_grid,
                            cv=inner_cv,
                            scoring='accuracy',  # Optimize for accuracy
                            n_jobs=-1)

    # Perform Grid Search
    knn_grid.fit(X_train_outer, y_train_outer)

    # Best estimator
    best_knn = knn_grid.best_estimator_

    # Predict on outer test set
    y_pred_knn = best_knn.predict(X_test_outer)
    predictions_knn.extend(y_pred_knn)

    # ---------------
    # Baseline Model
    # ---------------

    # Initialize Dummy Classifier (Baseline)
    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X_train_outer, y_train_outer)

    # Predict on outer test set
    y_pred_baseline = baseline.predict(X_test_outer)
    predictions_baseline.extend(y_pred_baseline)

    # Store true labels
    true_labels.extend(y_test_outer)

def compute_contingency_table(y_true, y_pred_A, y_pred_B):
    """
    Compute the contingency table components n11, n12, n21, n22.
    """
    # Both correct
    n11 = np.sum((y_pred_A == y_true) & (y_pred_B == y_true))
    # A correct, B wrong
    n12 = np.sum((y_pred_A == y_true) & (y_pred_B != y_true))
    # A wrong, B correct
    n21 = np.sum((y_pred_A != y_true) & (y_pred_B == y_true))
    # Both wrong
    n22 = np.sum((y_pred_A != y_true) & (y_pred_B != y_true))

    return n11, n12, n21, n22

def mcnemar_test(n12, n21, alpha=0.05):
    """
    Perform McNemar's test and compute the confidence interval.
    Returns theta_hat, theta_L, theta_U, p-value
    """
    n = n12 + n21
    if n == 0:
        raise ValueError("n12 + n21 is zero; McNemar's test is not applicable.")

    theta_hat = (n12 - n21) / n

    # Confidence Interval using binomial distribution
    # Compute f and g as per user's notation
    f = n12
    g = n21

    # Compute the z-score for the given alpha
    from scipy.stats import norm
    z = norm.ppf(1 - alpha/2)

    # Standard error
    se = np.sqrt(n12 + n21)

    # Confidence interval for (n12 - n21)
    lower = (n12 - n21 - z * se) / n
    upper = (n12 - n21 + z * se) / n

    # Ensure the confidence interval is within [-1, 1]
    theta_L = max(-1, lower)
    theta_U = min(1, upper)

    # Compute p-value using binomial test
    m = min(n12, n21)
    p = binom.cdf(m, n, 0.5)
    p_value = 2 * min(p, 1 - p)  # Two-tailed test

    return theta_hat, theta_L, theta_U, p_value

# Convert lists to NumPy arrays for efficient computation
true_labels = np.array(true_labels)
predictions_logreg = np.array(predictions_logreg)
predictions_knn = np.array(predictions_knn)
predictions_baseline = np.array(predictions_baseline)

# Define pairs to compare
model_pairs = {
    'LogReg vs KNN': (predictions_logreg, predictions_knn),
    'LogReg vs Baseline': (predictions_logreg, predictions_baseline),
    'KNN vs Baseline': (predictions_knn, predictions_baseline)
}

# Initialize a list to store test results
test_results = []

for pair_name, (pred_A, pred_B) in model_pairs.items():
    print(f"\nPerforming McNemar's test for {pair_name}...")

    # Compute contingency table
    n11, n12, n21, n22 = compute_contingency_table(true_labels, pred_A, pred_B)

    print(f"Contingency Table for {pair_name}:")
    print(f"n11 (Both correct): {n11}")
    print(f"n12 (A correct, B wrong): {n12}")
    print(f"n21 (A wrong, B correct): {n21}")
    print(f"n22 (Both wrong): {n22}")

    # Only perform the test if n12 + n21 >= 5
    if (n12 + n21) < 5:
        print(f"Not enough discordant pairs for {pair_name} (n12 + n21 = {n12 + n21} < 5). Skipping test.")
        theta_hat = np.nan
        theta_L = np.nan
        theta_U = np.nan
        p_value = np.nan
    else:
        # Perform McNemar's test
        theta_hat, theta_L, theta_U, p_value = mcnemar_test(n12, n21, alpha=0.05)

    # Store the results
    test_results.append({
        'Comparison': pair_name,
        'n11': n11,
        'n12': n12,
        'n21': n21,
        'n22': n22,
        'Theta_hat': theta_hat,
        'Theta_L': theta_L,
        'Theta_U': theta_U,
        'p-value': p_value
    })

# Convert the results to a DataFrame for better visualization
test_results_df = pd.DataFrame(test_results)

print("\n=== McNemar's Test Results ===")
print(test_results_df)

