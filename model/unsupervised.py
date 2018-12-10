import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score as calinski_harabasz

class UnsupervisedClassifier(object):
    def __init__(self, k_min=2, k_max=10):
        self.__range = range(k_min, k_max + 1)

    def __most_common(self, lst):
        return max(set(lst), key=lst.count)

    def __make_clusters(self, dataset):
        self.__clusters = {}
        for label, sample in zip(self.__instance.labels_, dataset.iterrows()):
            if str(label) not in self.__clusters:
                centroid = self.__instance.cluster_centers_[label]
                content = {'samples': [], 'centroid': centroid}
                self.__clusters[str(label)] =  content

            self.__clusters[str(label)]['samples'].append(sample)
    
        for cluster_id in self.__clusters:
            samples = self.__clusters[cluster_id]['samples']
            classes = map(lambda x: x[1][-1], samples)
            self.__clusters[cluster_id]['label'] = self.__most_common(classes)

    def fit(self, dataset):
        best = None
        x = dataset.iloc[:,:-1]
        self.__dimensions = dataset.iloc[0,:-1].count()

        for k in self.__range:
            instance = KMeans(n_clusters=k)
            instance.fit(x)

            v = calinski_harabasz(x, instance.labels_)
            if best is None or v > best[1]:
                best = (k, v, instance)
       
        self.__instance = best[2]
        self.__make_clusters(dataset)

    def predict(self, sample):
        if self.__instance is not None:
            assignment = self.__instance.predict([sample])[0]
            cluster = self.__clusters[str(assignment)]
            return cluster['label']

        return None

    def centroids(self):
        def mapper(cluster_id):
            cluster = self.__clusters[cluster_id]
            return cluster['centroid']

        return map(mapper, self.__clusters)

    def centroid_by_label(self, label):
        def reducer(amount, cluster):
            return amount + cluster['centroid']
        
        clusters_keys = filter(lambda x: self.__clusters[x]['label'] == label, self.__clusters)
        clusters = map(lambda x: self.__clusters[x], clusters_keys)

        d = len(clusters)
        if d == 0:
            return np.zeros(self.__dimensions)

        return reduce(reducer, clusters, np.zeros(self.__dimensions)) / d

    def centroids_by_label(self, label):
        clusters_keys = filter(lambda x: self.__clusters[x]['label'] == label, self.__clusters)

        if len(clusters_keys) == 0:
            return [np.zeros(self.__dimensions)]
        return map(lambda x: self.__clusters[x]['centroid'], clusters_keys)

    def evaluate(self, dataframe):
        true_predictions = 0
        num_rows = dataframe.iloc[:,0].count()

        for i in range(num_rows):
            sample = dataframe.iloc[i,:]
            unclassified = dataframe.iloc[i,:-1]

            prediction = self.predict(unclassified)
            if prediction == sample.iloc[-1]:
                true_predictions += 1

        return float(true_predictions), float(num_rows)
                
        

