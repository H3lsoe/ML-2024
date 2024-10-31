# exercise 2.1.1
import importlib_resources
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
import matplotlib.pyplot as plt
import xlrd



filename = "../data/Raisin_Dataset.xls"

df = pd.read_excel(filename)
df['Class'] = df['Class'].map({'Kecimen': 0, 'Besni': 1})

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(scaled_data)

components = np.arange(1, len(pca.explained_variance_ratio_) + 1)

# -------- for explained variance ratio by principal component ------------
# plt.figure(figsize=(10, 6))
# plt.bar(components, pca.explained_variance_ratio_, color='skyblue')
# plt.xlabel('Principal Component')
# plt.ylabel('Explained Variance Ratio')
# plt.title('Explained Variance Ratio by Principal Component')
# plt.xticks(components)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()

# -------- for 2d plot of PCA ------------
pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], s=100, c='blue', edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA Scatter Plot')
plt.grid(False)
plt.show()


# print(df)
# print(scaled_data)


# # Load xls sheet with data
# # filename = importlib_resources.files("dtuimldmtools").joinpath("data/nanonose.xls")
# filename = "data/Raisin_Dataset.xls"
# doc = xlrd.open_workbook(filename).sheet_by_index(0)


# # Extract attribute names (1st row, column 1 to 8)
# attributeNames = doc.row_values(0, 0, 8)

# # print(attributeNames)

# # Extract class names to python list,
# # then encode with integers (dict)
# classLabels = doc.col_values(7, 1, 901)
# classNames = sorted(set(classLabels))
# classDict = dict(zip(classNames, range(len(classNames))))

# print(classDict)

# # Extract vector y, convert to NumPy array
# y = np.asarray([classDict[value] for value in classLabels])

# print(len(y))

# # Preallocate memory, then extract excel data to matrix X
# X = np.empty((900, 8))
# for i, col_id in enumerate(range(0, 7)):
#     print("i: ", i, "col_id: ", col_id)
#     X[:, i] = np.asarray(doc.col_values(col_id, 1, 901))
# print(len(X))
# print(len(X[0]))

# print(X)

# # Compute values of N, M and C.
# N = len(y)
# M = len(attributeNames)
# C = len(classNames)

# print("Ran Exercise 2.1.1")
