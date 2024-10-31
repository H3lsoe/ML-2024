import importlib_resources
import numpy as np
import xlrd
import pandas as pd
import matplotlib.pyplot as plt



# Load xls sheet with data
filename = "../data/Raisin_Dataset.xls"
df = pd.read_excel(filename, sheet_name=0)

# Get header
#print(df.head())

# Get summary statistics of data
#print(df.describe())

# Make histogram for each variable
#df.hist(bins=15, figsize=(15, 10), layout=(3, 3))
#plt.tight_layout()
#plt.show()

# Make boxplot for each variable
df.boxplot()
plt.tight_layout()
plt.show()

# Correlation matrix
#df['Class'] = df['Class'].map({'Kecimen' : 0, 'Besni' : 1})
#correlation_matrix = df.corr()
#correlation_matrix_latex = df.corr().to_latex()
#print(correlation_matrix_latex)
