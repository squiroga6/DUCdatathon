# import libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# explore clean dataset (from data_clean.py)
well_header_clean.sample(10)


well_header_validation = well_header_clean[pd.isna(well_header_clean.TVD)]
# get unique values of formation, clusters, and well profile 
# from validation set to filter the training set
validation_clusters = well_header_validation.DBSCAN_Clusters.unique()
validation_formation = well_header_validation.Formation.unique()
validation_profile = well_header_validation.WellProfile.unique()

# filter training set
well_header_train = well_header_clean[pd.notna(well_header_clean.TVD)][
        well_header_clean.DBSCAN_Clusters.isin(validation_clusters)][
        well_header_clean.Formation.isin(validation_formation)][
        well_header_clean.WellProfile.isin(validation_profile)]

# select features and target
features = [
            'TotalDepth',
            'Formation',
            'WellProfile',
            'KBElevation',
            'DBSCAN_Clusters']

target = ['TVD']

X = well_header_train[features]

y = well_header_train[target]

# hot encode
X = pd.get_dummies(X, 
                    columns=[
                            "Formation", 
                            "WellProfile",
                            "DBSCAN_Clusters"], 
                    prefix=["fm", "profile","dbscan_cluster"])

# X.drop('profile_Directional',axis=1,inplace=True)

# split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)

# fit regressor
regr = RandomForestRegressor(
                        # max_depth=10, 
                        # random_state=0
                        # max_samples=5970
                        )
regr.fit(X, y)

# get prediction
y_prediction = regr.predict(X_test)

# get RMSE
RMSE = sqrt(mean_squared_error(y_true = y_test, y_pred = y_prediction))
print(RMSE)

X_validation = well_header_validation[features]
# hot encode
X_validation = pd.get_dummies(X_validation, 
                    columns=[
                            "Formation", 
                            "WellProfile",
                            "DBSCAN_Clusters"], 
                    prefix=["fm", "profile","dbscan_cluster"])

# X_validation.drop('profile_Directional',axis=1,inplace=True)

# get prediction
y_validation = regr.predict(X_validation)

well_header_submission = well_header_validation['EPAssetsId'].to_frame()

well_header_submission['Predicted_TVD'] = y_validation

well_header_submission.to_csv("../predictions/predicted_tvd.csv")