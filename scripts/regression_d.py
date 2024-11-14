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


K1 = 5
K2 = 5
alpha_values = [0.01,0.1,0.25,0.8,0.1,1.2,0.5,1,1.5,2,2.5,3,4,5,6,7,8,9,10,11,12]
hidden_units = [1,2,3,4]
mse = keras.losses.MeanSquaredError()
kf = KFold(n_splits=K1,shuffle=True,random_state=42)
best_models = []
for i, (outer_train_index,outer_test_index) in enumerate(kf.split(X,y)):
    X_train = X.iloc[outer_train_index]
    y_train = y.iloc[outer_train_index]
    holder = float('inf')
    for alpha in alpha_values:
        model_Linear = Ridge(alpha=alpha)
        mse_scores = -cross_val_score(model_Linear, X_train, y_train, scoring='neg_mean_squared_error', cv=K2)
        rmse_scores = np.sqrt(mse_scores)
        avg_rmse = rmse_scores.mean()
        if avg_rmse < holder:
            s = (model_Linear,alpha,i,avg_rmse)
            holder = avg_rmse
    holder = float('inf')
    for hidden_unit in hidden_units:
        model_ANN = keras.Sequential()
        model_ANN.add(keras.Input(shape=(2,)))
        model_ANN.add(keras.layers.Dense(hidden_unit,activation='linear'))
        model_ANN.add(keras.layers.Dense(1, activation='linear'))
        model_ANN.compile(optimizer='adam', loss='mean_squared_error')
        ANN = KerasRegressor(model=model_ANN, epochs=20, batch_size=10)
        mse_scores = -cross_val_score(ANN, X_train, y_train, scoring='neg_mean_squared_error', cv=K2)
        rmse_scores = np.sqrt(mse_scores)
        avg_rmse = rmse_scores.mean()
        if avg_rmse < holder:
            l = (ANN,hidden_unit,i, avg_rmse)
            holder = avg_rmse
    _, _, y_train_baseline, y_test_baseline = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    baseline_model = y_train_baseline.mean()
    mse_value = mse(baseline_model,y_test_baseline).numpy()
    p = (baseline_model,-1,i,mse_value)


    best_models.append(s)
    best_models.append(l)
    best_models.append(p)
    
    
for best_model in best_models:
    model, parameter, outer_layer, mse_value = best_model
    print("Model: ", model, "parameter: ", parameter , "Outer_layer: ", outer_layer, "MSE_Value: ", mse_value)



        
    
        
    




