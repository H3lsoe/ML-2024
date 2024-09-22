# exercise 2.1.1
import importlib_resources
import numpy as np
import xlrd




# Load xls sheet with data
# filename = importlib_resources.files("dtuimldmtools").joinpath("data/nanonose.xls")
filename = "data/Raisin_Dataset.xls"
doc = xlrd.open_workbook(filename).sheet_by_index(0)


# Extract attribute names (1st row, column 1 to 8)
attributeNames = doc.row_values(0, 0, 8)

# print(attributeNames)

# Extract class names to python list,
# then encode with integers (dict)
classLabels = doc.col_values(7, 2, 902)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(len(classNames))))

print(classDict)

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

# Preallocate memory, then extract excel data to matrix X
X = np.empty((900, 8))
for i, col_id in enumerate(range(3, 11)):
    X[:, i] = np.asarray(doc.col_values(col_id, 2, 899))
print(len(X))
print(len(X[0]))

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

print("Ran Exercise 2.1.1")
