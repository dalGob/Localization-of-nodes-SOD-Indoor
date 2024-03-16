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


print("Dataset HCXY")
train = pd.read_csv("./Training_CETC331.csv")
test = pd.read_csv("./Testing_CETC331.csv")

# Convert all values of "100" to "-100"
train = train.replace(100, -100)
test = test.replace(100, -100)


# Set the data up so ECoord and NCoord are removed from the dataset in X, and are isolated as the only columns in y.
X = train.drop("ECoord", axis=1)
X = X.drop("NCoord", axis=1)
X = X.drop("FloorID", axis=1)
X_train = X.values
y = train[["ECoord", "NCoord", "FloorID"]]
y_train = y.values

# Do the same as above on the testing data
X = test.drop("ECoord", axis=1)
X = X.drop("NCoord", axis=1)
X = X.drop("FloorID", axis=1)
X_test = X.values
y = test[["ECoord", "NCoord", "FloorID"]]
y_test = y.values

# Adding PCA seems to make the model much worse
# train_pca = PCA(n_components=0.9, random_state=2020)
# X_train = train_pca.fit_transform(X_train)
# test_pca = PCA(n_components=X_train.shape[1], random_state=2020)
# X_test = test_pca.fit_transform(X_test)
#
# print(X_train.shape)
# print(X_test.shape)

# Hyperparameter tuning for the number of neighbours
parameters = {"n_neighbors": range(MIN_NEIGHBOURS, MAX_NEIGHBOURS)}
gridsearch = GridSearchCV(KNeighborsRegressor(), parameters, cv=5, scoring="neg_root_mean_squared_error")
gridsearch.fit(X_train, y_train)
print("Best Hyperparameters:")
print(gridsearch.best_params_)


# See how the best hyperparameters affects the model
train_preds_grid = gridsearch.predict(X_train)
train_mse = mean_squared_error(y_train, train_preds_grid)
train_rmse = sqrt(train_mse)
train_r2 = r2_score(y_train, train_preds_grid)
test_preds_grid = gridsearch.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds_grid)
test_rmse = sqrt(test_mse)
test_r2 = r2_score(y_test, test_preds_grid)
print("\nBest Hyperparameter Training RMSE:")
print(train_rmse)
print("Best Hyperparameter Training MSE:")
print(train_mse)
print("Best Hyperparameter Training R^2:")
print(train_r2)
print("\nBest Hyperparameter Testing RMSE")
print(test_rmse)
print("Best Hyperparameter Testing MSE")
print(test_mse)
print("Best Hyperparameter Testing R^2:")
print(test_r2)


# Calculate the best hyperparameter again using a weighted average instead of a regular average
parameters = {
    "n_neighbors": range(MIN_NEIGHBOURS, MAX_NEIGHBOURS),
    "weights": ["uniform", "distance"],
}
gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
gridsearch.fit(X_train, y_train)
print("\nWeighted Average Best Parameters:")
print(gridsearch.best_params_)
test_preds_grid = gridsearch.predict(X_test)
r_squared = r2_score(y_test, test_preds_grid)
test_mse = mean_squared_error(y_test, test_preds_grid)
test_rmse = sqrt(test_mse)
print("\nWeighted Average Testing RMSE:")
print(test_rmse)
print("Weighted Average Testing MSE:")
print(test_mse)
print("R^2 Error:")
print(r_squared)


# Use bagging

best_k = gridsearch.best_params_["n_neighbors"]
best_weights = gridsearch.best_params_["weights"]
bagged_knn = KNeighborsRegressor(
    # n_neighbors=best_k, weights=best_weights
)

bagging_model = BaggingRegressor(
    estimator=bagged_knn
)

# bagged_knn = BaggingRegressor(base_estimator=bagged_knn)
params = {
    # 'base_estimator__n_neighbours': range(MIN_NEIGHBOURS, MAX_NEIGHBOURS),
    'n_estimators': [10, 50, 100],
    'max_samples': [0.5, 0.75, 1.0]
}

gridsearch = GridSearchCV(bagged_knn, params, cv=5, scoring="neg_root_mean_squared_error")
gridsearch.fit(X_train, y_train)

bagging_model.fit(X_train, y_train)
test_preds_grid = bagging_model.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds_grid)
test_rmse = sqrt(test_mse)
test_R2 = r2_score(y_test, test_preds_grid)
print("\nBagging Testing RMSE: ")
print(test_rmse)
print("Bagging Testing MSE: ")
print(test_mse)
print("Badding R^2: ")
print(test_R2)


# np.random.seed(42)
#
# random_indices = np.random.choice(X_test.shape[0], size=100, replace=False)
# X_test_random = X_test[random_indices]
#
# y_pred_random = bagging_model.predict(X_test_random)
#
# y_test_random = y_test[random_indices]
#
# y_test_random = pd.DataFrame(data=y_test_random[1:,1:],    # values
#                 index=y_test_random[1:,0],    # 1st column as index
#                 columns=y_test_random[0,1:])
#
# plt.figure(figsize=(10, 10))
# cmap = plt.cm.get_cmap('viridis', 100)
#
# for i, (true, pred) in enumerate(zip(y_test_random.values, y_pred_random)):
#         color = cmap(i)
#         plt.plot([true[0], pred[0]], [true[1], pred[1]], 'o-', markersize=5, alpha=0.6, color=color)
#
# plt.title('True vs. Predicted Coordinates')
# plt.xlabel('ECoord')
# plt.ylabel('NCoord')
# plt.grid(True)
# plt.show()

np.random.seed(42)

random_indices = np.random.choice(X_test.shape[0], size=100, replace=False)
X_test_random = X_test[random_indices]

y_pred_random = gridsearch.predict(X_test_random)

y_test_random = y_test[random_indices]

y_test_random = pd.DataFrame(data=y_test_random[1:,1:],    # values
                index=y_test_random[1:,0],    # 1st column as index
                columns=y_test_random[0,1:])

# Preparing the visualization
plt.figure(figsize=(10, 10))
cmap = plt.cm.get_cmap('viridis', 100)

for i, (true, pred) in enumerate(zip(y_test_random.values, y_pred_random)):
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