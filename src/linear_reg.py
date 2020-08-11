# import libraries
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.preprocessing import PolynomialFeatures
from sklearn_pandas import DataFrameMapper

# explore clean dataset (from data_clean.py)
well_header_clean.head()

# select features and target
features = ['TotalDepth','Formation','WellProfile','DBSCAN_Clusters']
target = ['TVD']
X = well_header_clean[features]
y = well_header_clean[target]

# adding polynoial features
mapper = DataFrameMapper([
(['TotalDepth'], PolynomialFeatures(3)),
(['Formation'], None),
(['WellProfile'], None),
(['DBSCAN_Clusters'], None),
])
X = mapper.fit_transform(X)
X = pd.DataFrame(X,columns = ['index','TotalDepth',
                            'TotalDepth_poly1',
                            'TotalDepth_poly2',
                            'Formation',
                            'WellProfile','DBSCAN_Clusters'])
X.drop(columns="index",inplace=True)

# hot encode
X = pd.get_dummies(X, 
                    columns=["Formation", "WellProfile","DBSCAN_Clusters"], 
                    prefix=["fm", "profile","dbscan_cluster"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)

# regressor = Ridge(alpha=0.01)
# regressor.fit(X_train, y_train)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_prediction = regressor.predict(X_test)

RMSE = sqrt(mean_squared_error(y_true = y_test, y_pred = y_prediction))
print(RMSE)