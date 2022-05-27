from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from keras.datasets import mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(123)

raw_data = datasets.load_wine()
dataframe = pd.DataFrame(raw_data["data"], columns=raw_data["feature_names"])
plt.figure(figsize=(15, 6))
sns.heatmap(dataframe.corr(), annot=True)
plt.show()

scaler = StandardScaler()
scaled_array = scaler.fit_transform(dataframe)
scaled_dataframe = pd.DataFrame(scaled_array, columns=dataframe.columns)
kmeans_model = KMeans(n_clusters=4)
kmeans_model.fit(dataframe)

centroids = kmeans_model.cluster_centers_
dataframe["cluster"] = kmeans_model.labels_
print(dataframe.head())

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

sns.scatterplot(x='proanthocyanins', y="proline", data=scaled_dataframe, hue_order="cluster", palette="Accent", ax=ax1,
                legend=False)
sns.scatterplot(x="color_intensity", y="flavanoids", data=scaled_dataframe, hue_order="cluster", palette="Accent", ax=ax2,
                legend=False)
sns.scatterplot(x="flavanoids", y="ash", data=scaled_dataframe, hue_order="cluster", palette="Accent", ax=ax3, legend=False)
sns.scatterplot(x="total_phenols", y="flavanoids", data=scaled_dataframe, hue_order="cluster", palette="Accent", ax=ax4,
                legend=False)

plt.show()
