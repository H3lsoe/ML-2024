import numpy as np
import xlrd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn.decomposition import PCA

# Load xls sheet with data
filename = "../data/Raisin_Dataset.xls"
doc = xlrd.open_workbook(filename).sheet_by_index(0)

# Extract attribute names (assuming the first row has the names)
attributeNames = doc.row_values(0, 0, 7)  # Extracting the attribute names from the header (first row)
# Extract class labels from the last column, then encode with integers (dict)
classLabels = doc.col_values(7, 1, doc.nrows)  # Extracting class labels from column 7, starting from row 1 (ignoring the header)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(len(classNames))))

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])
# Preallocate memory, then extract excel data to matrix X
X = np.empty((doc.nrows - 1, 7))  # 7 attribute columns, excluding the header row
for i, col_id in enumerate(range(0, 7)):
    X[:, i] = np.asarray(doc.col_values(col_id, 1, doc.nrows))  # Extracting data from each column, starting from row 1


# Compute values of N, M, and C
N = len(y)
M = len(attributeNames)
C = len(classNames)

# Subtract mean value from data
Y = (X - np.ones((N, 1)) * X.mean(axis=0)) / X.std(axis=0)

# PCA by computing SVD of Y
U, S, V = svd(Y, full_matrices=False)

# Compute variance explained by principal components
rho = (S * S) / (S * S).sum()

threshold = 0.9

pca = PCA(n_components=2)
pca.fit_transform(Y)
print(pca.explained_variance_ratio_)
print(pca.components_)
# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.show()


PC1 = U[:, 0] * S[0]
PC2 = U[:, 1] * S[1]

# Create a scatter plot
plt.figure(figsize=(10, 6))
for class_value in np.unique(y):
    plt.scatter(PC1[y == class_value], PC2[y == class_value], label=f"Class {class_value}")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - First Two Principal Components")
plt.legend()
plt.grid(True)
plt.show()
