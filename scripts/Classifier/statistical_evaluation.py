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

if y.dtype == object or isinstance(y[0], str):
    le = LabelEncoder()
    y = le.fit_transform(y)

C_values = [0.001, 0.01, 0.1, 1, 10, 100]
lr_param_grid = {'C': C_values, 'solver': ['liblinear']}

k_values = list(range(1, 30))
knn_param_grid = {'n_neighbors': k_values}

outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

true_labels = []
predictions_logreg = []
predictions_knn = []
predictions_baseline = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
    print(f"\n--- Outer Fold {fold}/10 ---")

    # Split data
    X_train_outer, X_test_outer = X[train_idx], X[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]

    # -----------------------------
    # Logistic Regression - Inner CV
    # -----------------------------

    lr = LogisticRegression(max_iter=1000)

    lr_grid = GridSearchCV(estimator=lr,
                           param_grid=lr_param_grid,
                           cv=inner_cv,
                           scoring='accuracy',
                           n_jobs=-1)

    lr_grid.fit(X_train_outer, y_train_outer)

    best_lr = lr_grid.best_estimator_

    y_pred_lr = best_lr.predict(X_test_outer)
    predictions_logreg.extend(y_pred_lr)

    # -------------------
    # K-Nearest Neighbors
    # -------------------

    knn = KNeighborsClassifier()

    knn_grid = GridSearchCV(estimator=knn,
                            param_grid=knn_param_grid,
                            cv=inner_cv,
                            scoring='accuracy',  # Optimize for accuracy
                            n_jobs=-1)

    knn_grid.fit(X_train_outer, y_train_outer)

    best_knn = knn_grid.best_estimator_

    y_pred_knn = best_knn.predict(X_test_outer)
    predictions_knn.extend(y_pred_knn)

    # ---------------
    # Baseline Model
    # ---------------

    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X_train_outer, y_train_outer)

    y_pred_baseline = baseline.predict(X_test_outer)
    predictions_baseline.extend(y_pred_baseline)

    true_labels.extend(y_test_outer)

def compute_contingency_table(y_true, y_pred_A, y_pred_B):
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
    n = n12 + n21
    if n == 0:
        raise ValueError("n12 + n21 is zero; McNemar's test is not applicable.")

    theta_hat = (n12 - n21) / n

    f = n12
    g = n21

    from scipy.stats import norm
    z = norm.ppf(1 - alpha/2)

    se = np.sqrt(n12 + n21)

    lower = (n12 - n21 - z * se) / n
    upper = (n12 - n21 + z * se) / n

    theta_L = max(-1, lower)
    theta_U = min(1, upper)

    m = min(n12, n21)
    p = binom.cdf(m, n, 0.5)
    p_value = 2 * min(p, 1 - p)

    return theta_hat, theta_L, theta_U, p_value

true_labels = np.array(true_labels)
predictions_logreg = np.array(predictions_logreg)
predictions_knn = np.array(predictions_knn)
predictions_baseline = np.array(predictions_baseline)

model_pairs = {
    'LogReg vs KNN': (predictions_logreg, predictions_knn),
    'LogReg vs Baseline': (predictions_logreg, predictions_baseline),
    'KNN vs Baseline': (predictions_knn, predictions_baseline)
}

test_results = []

for pair_name, (pred_A, pred_B) in model_pairs.items():
    print(f"\nPerforming McNemar's test for {pair_name}...")

    n11, n12, n21, n22 = compute_contingency_table(true_labels, pred_A, pred_B)

    print(f"Contingency Table for {pair_name}:")
    print(f"n11 (Both correct): {n11}")
    print(f"n12 (A correct, B wrong): {n12}")
    print(f"n21 (A wrong, B correct): {n21}")
    print(f"n22 (Both wrong): {n22}")

    theta_hat, theta_L, theta_U, p_value = mcnemar_test(n12, n21, alpha=0.05)

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

test_results_df = pd.DataFrame(test_results)

print("\n=== McNemar's Test Results ===")
print(test_results_df)

