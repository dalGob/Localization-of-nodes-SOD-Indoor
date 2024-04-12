# https://realpython.com/knn-python/
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingRegressor
from sklearn.decomposition import PCA

MIN_NEIGHBOURS = 25
MAX_NEIGHBOURS = 50

irrelevant_features = ['ECoord', 'NCoord', 'FloorID', 'BuildingID', 'SceneID', 'UserID', 'PhoneID', 'SampleTimes']

output_variables = ['ECoord', 'NCoord']

train = pd.read_csv("../HCXY/Training_HCXY_All_30.csv")
test = pd.read_csv("../HCXY/Testing_HCXY_All.csv")

# Convert all values of "100" to "-100"
train = train.replace(100, -200)
test = test.replace(100, -200)


# Set the data up so ECoord and NCoord are removed from the dataset in X, and are isolated as the only columns in y.
X = train.drop(irrelevant_features, axis=1)
X_train = X.values
y = train[output_variables]
y_train = y.values

# Do the same as above on the testing data
X = test.drop(irrelevant_features, axis=1)
X_test = X.values
y = test[output_variables]
y_test = y.values

knn = KNeighborsRegressor()

bagging_model = BaggingRegressor(estimator=knn)

params = {'n_estimators': [10, 50], 'max_samples': [0.5, 0.75, 1.0]}

gridsearch = GridSearchCV(bagging_model, params, cv=5, scoring="neg_root_mean_squared_error")
gridsearch.fit(X_train, y_train)
print(gridsearch.best_params_)

test_preds_grid = gridsearch.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds_grid)
test_rmse = sqrt(test_mse)
test_R2 = r2_score(y_test, test_preds_grid)
print("\nBagging Testing RMSE: ")
print(test_rmse)
print("Bagging Testing MSE: ")
print(test_mse)
print("Badding R^2: ")
print(test_R2)


np.random.seed(7)
random_indices = np.random.choice(X_test.shape[0], size=100, replace=False)
X_test_random = X_test[random_indices]
y_pred_random = gridsearch.predict(X_test_random)

y_test_random = y_test[random_indices]

# Preparing the visualization
plt.figure(figsize=(10, 10))
cmap = plt.cm.get_cmap('viridis', 100)

for i, (true, pred) in enumerate(zip(y_test_random, y_pred_random)):
        color = cmap(i)
        plt.plot([true[0], pred[0]], [true[1], pred[1]], '-', alpha=0.6, color=color)
        plt.plot(pred[0], pred[1], 'o', markersize=5, color=color)
        plt.scatter(true[0], true[1], marker='*', s=100, color=color)
plt.plot([], [], 'o', color='gray', label='Estimated (Predicted)')
plt.scatter([], [], marker='*', s=100, color='gray', label='True')
plt.legend()
plt.title('True vs. Predicted Coordinates')
plt.xlabel('ECoord')
plt.ylabel('NCoord')
plt.grid(True)
plt.show()