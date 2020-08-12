# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# import well header data
well_header = pd.read_csv("../data/WellHeader_Datathon.csv")

#select colummns of interest
well_header_clean = well_header[['TVD','TotalDepth','Formation',
                                 'Surf_Longitude','Surf_Latitude',
                                 'BH_Longitude','BH_Latitude',
                                 'WellProfile','Field','KBElevation']]
# drop na
well_header_clean = well_header_clean.dropna()
# remove vertical wells
# well_header_clean = well_header_clean[well_header_clean.WellProfile != "Vertical"]
# label field column
well_header_clean['Field'] = well_header_clean['Field'].astype('category')
# well_header_clean.info()
well_header_clean['Field_code'] = well_header_clean['Field'].cat.codes

# quick plot
plt.figure(figsize = (15,8))
sns.scatterplot(well_header_clean['Surf_Longitude'], 
                well_header_clean['Surf_Latitude'],alpha=0.2,
                hue=well_header_clean['Field'],legend=False)

# Compute DBSCAN

# cluster lat long
from sklearn.cluster import DBSCAN
lat_longs = well_header_clean[['Surf_Longitude','Surf_Latitude']]
db = DBSCAN(eps=0.1, min_samples=10).fit(lat_longs)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
labels_true = np.array(well_header_clean.Field_code)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

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


db = DBSCAN(eps=0.1, min_samples = 10)
clusters = db.fit_predict(lat_longs)

# quick plot with clusters
plt.figure(figsize = (15,8))
sns.scatterplot(well_header_clean['Surf_Longitude'], 
                well_header_clean['Surf_Latitude'],
                alpha=0.2,hue=clusters,palette="plasma")

len(np.unique(np.array(clusters)))

# add clusters back to df
well_header_clean['DBSCAN_Clusters']=clusters

#select colummns of interest
well_header_clean = well_header_clean[['TVD','TotalDepth','Formation',
                                       'WellProfile','DBSCAN_Clusters',
                                       'KBElevation']]

well_header_clean = well_header_clean[well_header_clean['DBSCAN_Clusters']>1]