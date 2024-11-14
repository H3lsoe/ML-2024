import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

data = pd.read_csv("data/Raisin_Dataset.csv")
scaler = StandardScaler()
data = data.drop(['Class'],axis=1)
data_standardized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
X = data_standardized.drop(['Area','Perimeter','Eccentricity','ConvexArea','Extent'],axis=1)
y = data_standardized['Area']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
alpha = 1
model = Ridge(alpha=alpha)
model.fit(X_train, y_train)

print(model.coef_)
print(model.intercept_)








