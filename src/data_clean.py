# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import well header data
well_header = pd.read_csv("../data/WellHeader_Datathon.csv")

#select colummns of interest
well_header = well_header[[
                        'EPAssetsId',
                        'TVD','TotalDepth','Formation',
                        'Surf_Longitude','Surf_Latitude',
                        # 'BH_Longitude','BH_Latitude',
                        'WellProfile','Field','KBElevation'
                        ]]

# get submission ids for testing
submission_sample = pd.read_csv("../data/Submission_Sample.csv")
EPAssetId_submission = submission_sample['EPAssetsId']

# filter well header to only include training/cv (non-na TVD) plus final test
well_header = well_header[
      (well_header['EPAssetsId'].isin(EPAssetId_submission)) | (pd.notna(well_header.TVD))
      ]

# check if missing ids are equal to submission test: want TRUE
missingID = well_header[pd.isna(well_header.TVD)].EPAssetsId.reset_index(drop=True)
missingID.sort_values(ignore_index=True).equals(EPAssetId_submission.sort_values(ignore_index=True))


# label field column (for cluster metrics comparisons)
well_header['Field'] = well_header['Field'].astype('category')
well_header['Field_code'] = well_header['Field'].cat.codes

# quick plot
plt.figure(figsize = (15,8))
sns.scatterplot(well_header['Surf_Longitude'], 
                well_header['Surf_Latitude'],alpha=0.2,
                hue=well_header['Field'],legend=False)

# Compute DBSCAN clustering
from sklearn.cluster import DBSCAN
from sklearn import metrics

# cluster lat long
lat_longs = well_header[['Surf_Longitude','Surf_Latitude']]
db = DBSCAN(eps=0.1, min_samples=10).fit(lat_longs)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
labels_true = np.array(well_header.Field_code)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# print out metrics
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(lat_longs, labels))

# re-run with prediction values
db = DBSCAN(eps=0.1, min_samples = 10)
clusters = db.fit_predict(lat_longs)

# quick plot with clusters
plt.figure(figsize = (15,8))
sns.scatterplot(well_header['Surf_Longitude'], 
                well_header['Surf_Latitude'],
                alpha=0.2,hue=clusters,palette="plasma")

# check length of clusters
len(np.unique(np.array(clusters)))

# add clusters back to df
well_header['DBSCAN_Clusters']=clusters

#select colummns of interest
well_header_clean = well_header[[
                              'EPAssetsId',
                              'TVD','TotalDepth','Formation',
                              'WellProfile','DBSCAN_Clusters',
                              'KBElevation'
                              ]]
# remove "noise" clusters
# well_header_clean = well_header[well_header['DBSCAN_Clusters']>-1]
