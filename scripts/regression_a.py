import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

data = pd.read_csv("data/Raisin_Dataset.csv")

X = data.drop(['Area','Perimeter','Eccentricity','ConvexArea','Extent','Class'],axis=1)
y = data['Area']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

mse = root_mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

#print(data.head())
#print(data.info())
#print(data.describe())
#print(X.head())
#print(y)
#print('MSE:', mse)

#print('Shape of X_test:', X_test.shape)
#print(X_test)
#print('Shape of y_test:', y_test.shape)
#print('Shape of y_pred:', y_pred.shape)
X_test_np = X_test.to_numpy()
y_test_np = y_test.to_numpy()

# Set up 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot actual data points
ax.scatter(X_test_np[:, 0], X_test_np[:, 1], y_test_np, color='blue', label='Actual')

# Create grid for regression plane
x_surf, y_surf = np.meshgrid(
    np.linspace(X_test_np[:, 0].min(), X_test_np[:, 0].max(), 20),
    np.linspace(X_test_np[:, 1].min(), X_test_np[:, 1].max(), 20)
)
xy_grid = np.c_[x_surf.ravel(), y_surf.ravel()]
z_surf = model.predict(xy_grid)
z_surf = z_surf.reshape(x_surf.shape)

# Plot regression plane
ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.5, color='red')

# Customize plot
ax.set_xlabel('MajorAxisLength')
ax.set_ylabel('MinorAxisLength')
ax.set_zlabel('Area')
ax.set_title('Linear Regression Model vs Actual Data')

# Add custom legend
scatter_proxy = Line2D([0], [0], linestyle="none", marker='o', color='blue')
plane_proxy = Patch(facecolor='red', edgecolor='red', alpha=0.5)
ax.legend([scatter_proxy, plane_proxy], ['Actual', 'Predicted'])

# Show plot
plt.show()


