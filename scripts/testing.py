import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import KFold
import tensorflow as tf
from scikeras.wrappers import KerasRegressor
import keras

data = pd.read_csv("data/Raisin_Dataset.csv")
scaler = StandardScaler()
data = data.drop(['Class'],axis=1)
data_standardized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
X = data_standardized.drop(['Area','Perimeter','Eccentricity','ConvexArea','Extent'],axis=1)
y = data_standardized['Area']

model_ANN = keras.Sequential()
model_ANN.add(keras.Input(shape=(2,)))
model_ANN.add(keras.layers.Dense(3,activation='linear'))
model_ANN.add(keras.layers.Dense(1, activation='linear'))
model_ANN.compile(optimizer='adam', loss='mean_squared_error')
ANN = KerasRegressor(model=model_ANN, epochs=100, batch_size=10)
mse_scores = -cross_val_score(ANN, X, y, scoring='neg_mean_squared_error', cv=10)
rmse_scores = np.sqrt(mse_scores)
avg_rmse = rmse_scores.mean()
print(avg_rmse)
#results_ANN.append(avg_rmse)