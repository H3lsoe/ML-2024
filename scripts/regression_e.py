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
from scipy.stats import t, ttest_rel

data = pd.read_csv("data/Raisin_Dataset.csv")
scaler = StandardScaler()
data = data.drop(['Class'],axis=1)
data_standardized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
X = data_standardized.drop(['Area','Perimeter','Eccentricity','ConvexArea','Extent'],axis=1)
y = data_standardized['Area']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

baseline_model = y_train.mean()


alpha = 1
model = Ridge(alpha=alpha)
model.fit(X_train, y_train)
predict_linear = model.predict(X_test)


model_ANN = keras.Sequential()
model_ANN.add(keras.Input(shape=(2,)))
model_ANN.add(keras.layers.Dense(3,activation='linear'))
model_ANN.add(keras.layers.Dense(1, activation='linear'))
model_ANN.compile(optimizer='adam', loss='mean_squared_error')
model_ANN.fit(X_train, y_train)
predict_ANN = model_ANN.predict(X_test).flatten()

loss_linear = abs(predict_linear - y_test)
loss_ANN = abs(predict_ANN - y_test)
loss_baseline_model = abs(y_test - baseline_model)


t_stat, p_value = ttest_rel(loss_baseline_model, loss_linear)

differences = np.array(loss_baseline_model) - np.array(loss_linear)
mean_diff = np.mean(differences)
std_diff = np.std(differences, ddof=1)
n = len(differences)
standard_error = std_diff / np.sqrt(n)
t_critical = t.ppf(0.975, df=n-1)
confidence_interval = (mean_diff - t_critical * standard_error, mean_diff + t_critical * standard_error)

print("t-statistic:", t_stat)
print("p-value:", p_value)
print("95% Confidence Interval for the Difference in Errors:", confidence_interval)
print(loss_linear.sum())
print(loss_ANN.sum())
print(loss_baseline_model.sum())


