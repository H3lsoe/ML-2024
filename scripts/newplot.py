import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Sample DataFrame
filename = "data/Raisin_Dataset.xls"

df = pd.read_excel(filename)
df['Class'] = df['Class'].map({'Kecimen': 0, 'Besni': 1})



# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(scaled_data)
loadings = pca.components_.T  # Transpose to get loadings
loadings = loadings[:3]  # Keep only the first 3 components
print(loadings)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])

# Plotting the 3D PCA plot with feature vectors
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the principal components
ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], color='blue', alpha=0.25, s=50)

# Plotting the feature vectors
for i, (x, y, z) in enumerate(loadings):
    ax.quiver(0, 0, 0, x*10, y*10, z*10, color='red',linewidth=3, arrow_length_ratio=0.1 )
    

# Labels and title
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D PCA Plot with Vectors')
ax.grid(True)

# Show the plot
plt.show()
