# predict house prices

# machine learning algoritm som förutspår huspriser i boston
from sklearn.datasets import load_boston
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import graphviz

boston = load_boston()
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names

data.head()
data['PRICE'] = boston.target
data.info()
data.describe()
print(boston.DESCR)

# Separate the target variable and rest of the variables using .iloc to subset the data.
X, y = data.iloc[:, :-1], data.iloc[:, -1]

# convert the dataset into an optimized data structure called Dmatrix
data_dmatrix = xgb.DMatrix(data=X, label=y)

# train and test set for cross-validation test 20% of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiate XGBoost regressor object by calling XGBRegressor class
# with hyper-parameters passed as arguments
# for classification problems, use XGBClassifier()
xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                          max_depth=5, alpha=10, n_estimators=10)

# fit the regressor to training set and make predictions on test set
# using predict and fit methods
xg_reg.fit(X_train, y_train)

preds = xg_reg.predict(X_test)

# compute rmse by invoking the mean squared error function from metrics module
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

# 3 fold cross validation model invoking XGboost cv method store in cv_results dataframe
params = {"objective": "reg:linear", 'colsample_bytree': 0.3, 'learning_rate': 0.1,
          'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=1000, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)

cv_results.head()

# print out root mean square error rmse
print((cv_results["test-rmse-mean"]).tail(1))

# train model
xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=50)

# plot model
xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()

# -----------------<>HYPERPARAMETERS<>--------------------------------------
# learning_rate: step size shrinkage used to prevent overfitting. Range is [0,1]
# max_depth: determines how deeply each tree is allowed to grow during any boosting round.
# subsample: percentage of samples used per tree. Low value can lead to underfitting.
# colsample_bytree: percentage of features used per tree. High value can lead to overfitting.
# n_estimators: number of trees you want to build.
# objective: determines the loss function to be used like reg:linear for regression problems, reg:logistic for classification problems with only decision, binary:logistic for classification problems with probability.

# K-fold cross validation
# build more robust models, where both training and original dataset are used for validation/training
# each entry only validated once
# num_boost_round: denotes the number of trees you build (analogous to n_estimators)
# metrics: tells the evaluation metrics to be watched during CV
# as_pandas: to return the results in a pandas DataFrame.
# early_stopping_rounds: finishes training of the model early if the hold-out metric ("rmse" in our case) does not improve for a given number of rounds.
# seed: for reproducibility of results.
