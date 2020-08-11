# import libraries
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# explore clean dataset (from data_clean.py)
well_header_clean.head()

# select features and target
features = ['TotalDepth','Formation','WellProfile','DBSCAN_Clusters']
target = ['TVD']
X = well_header_clean[features]
y = well_header_clean[target]

# hot encode
X = pd.get_dummies(X, 
                    columns=["Formation", "WellProfile","DBSCAN_Clusters"], 
                    prefix=["fm", "profile","dbscan_cluster"])

# split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)

# fit regressor
regr = MLPRegressor(
                    # hidden_layer_sizes=500,
                    # random_state=1,
                    # alpha=0.01,
                    # activation="logistic",
                    # max_iter=500,
                    # solver="lbfgs"
                    )
regr.fit(X_train, y_train)

# get prediction
y_prediction = regr.predict(X_test)

# get RMSE
RMSE = sqrt(mean_squared_error(y_true = y_test, y_pred = y_prediction))
print(RMSE)