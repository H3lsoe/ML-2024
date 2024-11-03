import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score
from KNN_data import *  # Ensure this imports X and y appropriately

# Define hyperparameter grids
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
lr_param_grid = {'C': C_values, 'solver': ['liblinear']}  # 'liblinear' is suitable for small datasets

k_values = list(range(1, 30))
knn_param_grid = {'n_neighbors': k_values}

# Initialize cross-validators
outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize list to store results
results = []

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

    # Best hyperparameters
    best_C = lr_grid.best_params_['C']
    best_solver = lr_grid.best_params_['solver']

    # Best estimator
    best_lr = lr_grid.best_estimator_

    # Predict on outer test set
    y_pred_lr = best_lr.predict(X_test_outer)
    lr_accuracy = accuracy_score(y_test_outer, y_pred_lr)
    lr_error = 1 - lr_accuracy  # Compute error rate

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

    # Best hyperparameters
    best_k = knn_grid.best_params_['n_neighbors']

    # Best estimator
    best_knn = knn_grid.best_estimator_

    # Predict on outer test set
    y_pred_knn = best_knn.predict(X_test_outer)
    knn_accuracy = accuracy_score(y_test_outer, y_pred_knn)
    knn_error = 1 - knn_accuracy  # Compute error rate

    # ---------------
    # Baseline Model
    # ---------------

    # Initialize Dummy Classifier (Baseline)
    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X_train_outer, y_train_outer)

    # Predict on outer test set
    y_pred_baseline = baseline.predict(X_test_outer)
    baseline_accuracy = accuracy_score(y_test_outer, y_pred_baseline)
    baseline_error = 1 - baseline_accuracy  # Compute error rate

    # ----------------
    # Store the Result
    # ----------------

    results.append({
        'Fold': fold,
        'LogReg_C*': best_C,
        'LogReg_Etest': lr_error,
        'KNN_k*': best_k,
        'KNN_Etest': knn_error,
        'Baseline_Etest': baseline_error
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display the table
print("\n=== Nested Cross-Validation Results ===")
print(results_df)

# Calculate average error rates and most frequent hyperparameters
average_results = {
    'LogReg_Avg_C*': results_df['LogReg_C*'].mode()[0],  # Most frequent best C
    'LogReg_Avg_Etest': results_df['LogReg_Etest'].mean(),
    'KNN_Avg_k*': results_df['KNN_k*'].mode()[0],       # Most frequent best k
    'KNN_Avg_Etest': results_df['KNN_Etest'].mean(),
    'Baseline_Avg_Etest': results_df['Baseline_Etest'].mean()
}

# Convert to DataFrame for display
average_df = pd.DataFrame([average_results])

print("\n=== Average Performance ===")
print(average_df)

