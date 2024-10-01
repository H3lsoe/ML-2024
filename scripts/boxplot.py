import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as snsca

# Load your data
filename = "data/Raisin_Dataset.xls"
df = pd.read_excel(filename)

# Map the class labels
df['Class'] = df['Class'].map({'Kecimen': 0, 'Besni': 1})

# Features to scale (exclude 'Class' column)
features = ['Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'ConvexArea', 'Extent', 'Perimeter']

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

corr_matrix = scaled_data.corr()

# Display the correlation matrix
print(corr_matrix)

# # Convert the scaled data back into a DataFrame
# scaled_df = pd.DataFrame(scaled_data, columns=features)

# # Create the boxplot on the scaled data
# scaled_df.boxplot(column=features, grid=False)
# plt.title("Boxplot of scaled rasisin data")
# plt.show()

