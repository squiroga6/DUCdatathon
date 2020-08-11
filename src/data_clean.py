# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import well header data
well_header = pd.read_csv("../data/WellHeader_Datathon.csv")

#select colummns of interest
well_header_clean = well_header[['TVD','TotalDepth','Formation',
                                 'Surf_Longitude','Surf_Latitude',
                                 'BH_Longitude','BH_Latitude',
                                 'WellProfile','Field']]
# drop na
well_header_clean = well_header_clean.dropna()
# remove vertical wells
well_header_clean = well_header_clean[well_header_clean.WellProfile != "Vertical"]

# quick plot
plt.figure(figsize = (15,8))
sns.scatterplot(well_header_clean['Surf_Longitude'], 
                well_header_clean['Surf_Latitude'],alpha=0.2,
                hue=well_header_clean['Field'],legend=False)

# cluster lat long
from sklearn.cluster import DBSCAN
lat_longs = well_header_clean[['Surf_Longitude','Surf_Latitude']]
dbscan = DBSCAN(eps=0.1, min_samples = 10)
clusters = dbscan.fit_predict(lat_longs)

# quick plot with clusters
plt.figure(figsize = (15,8))
sns.scatterplot(well_header_clean['Surf_Longitude'], 
                well_header_clean['Surf_Latitude'],
                alpha=0.2,hue=clusters,palette="plasma")

# add clusters back to df
well_header_clean['DBSCAN_Clusters']=clusters

#select colummns of interest
well_header_clean = well_header_clean[['TVD','TotalDepth','Formation',
                                 'WellProfile','DBSCAN_Clusters']]