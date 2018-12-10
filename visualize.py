import uuid
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.source import SourceFactory
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier

from data.processing.probability import Gaussian
from data.processing.scaling import Standard
from model.bayes import NaiveBayesClassifier
from model.unsupervised import UnsupervisedClassifier

parser = argparse.ArgumentParser()

parser.add_argument('dataset', help='Determines the source to read data from',
        choices=SourceFactory.values())
parser.add_argument('-n', '--dimensions', help='Number of dimensions to plot',
        type=int, default=3, choices=[2,3])

def to_numeric_id(names):
    id_mapping = {}
    current_id = [1]

    def mapper(name):
        if str(name) not in id_mapping:
            id_mapping[str(name)] = current_id[0]
            current_id[0] = current_id[0] + 1

        return id_mapping[str(name)]
   
    return map(mapper, names)

def plot(dataset, assignments):
    cmap = plt.cm.get_cmap('plasma')

    if dataset.shape[1] == 2:
        x = dataset[:,0]
        y = dataset[:,1]
        plt.scatter(x, y, c=assignments, cmap=cmap)
    else:
        x = dataset[:,0]
        y = dataset[:,1]
        z = dataset[:,2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=assignments, cmap=cmap)

    plt.show()

def plot_predictions(source, sup, unsup, feature_importances):
    num_rows = source.dataset.iloc[:,0].count()
    
    x = {}
    for class_id in sup.classes():
        c = {}
        prototypes = [sup.abduct(class_id).values]
        prototypes = prototypes + unsup.centroids_by_label(class_id)

        for prototype in prototypes:
            c[str(uuid.uuid4())] = prototype

        x[str(class_id)] = c

    def knn(k, sample, dataset):
        distances = []
        def sorter(v):
            return v[1]

        for class_id in dataset:
            for t in dataset[class_id]:
                prototype = dataset[class_id][t]
                a = np.multiply(sample.values, feature_importances)
                b = np.multiply(prototype, feature_importances)

                distance = euclidean(a, b)
                measurement = (class_id, distance)
                distances.append(measurement)


        distances = sorted(distances, key=sorter)[:k]
        distances = map(lambda x: x[0], distances)
        return max(set(distances), key=distances.count)

    for i in range(num_rows):
        class_name = source.dataset.iloc[i,-1]
        unclassified = source.dataset.iloc[i,:-1]
        predicted = knn(1, unclassified, x)

        source.dataset.iloc[i,-1] = predicted

    pca = PCA(n_components=args.dimensions)
    data = pca.fit_transform(source.dataset.iloc[:,:-1].values)
    labels = np.array(to_numeric_id(source.dataset.iloc[:,-1]))
    plot(data, labels)

if __name__ == '__main__':
    args = parser.parse_args()
    source = SourceFactory.build(args.dataset, scaler=Standard())
    train, test = source.split()

    classifier = UnsupervisedClassifier(k_min=3, k_max=20)
    classifier.fit(train)

    bayes = NaiveBayesClassifier(Gaussian())
    bayes.fit(train)

    def mapper(centroid):
        cls = np.array([5])
        data = np.append(centroid, cls)
        return pd.Series(data=data, index=source.dataset.columns)

    # centroids = map(mapper, classifier.centroids())
    # dataset = source.dataset.append(centroids)
    dataset = source.dataset
    
    # print(centroids)
    # print('centroids: ', len(centroids))
    # print('*****')
    # for class_id in bayes.classes():
    #     v = bayes.abduct(class_id)
    #     data = np.append(v.values, np.array([7]))
    #     print(data)
    #     x = pd.Series(data=data, index=source.dataset.columns)
    #     print(x)

    #     dataset = dataset.append(x, ignore_index=True)
    
    labels = dataset.iloc[:,-1]
    unlabeled_dataset = dataset.iloc[:,:-1]

    pca = PCA(n_components=args.dimensions)
    data = pca.fit_transform(unlabeled_dataset)
    
    c = RandomForestClassifier(n_estimators=100, max_depth=2)
    c.fit(train.iloc[:,:-1], train.iloc[:,-1])

    plot(data, to_numeric_id(labels.values)) 
    plot_predictions(source, bayes, classifier, c.feature_importances_)


