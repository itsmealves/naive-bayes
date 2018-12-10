import os
import math
import uuid
import requests
import pandas as pd

class Source(object):
    def __init__(self, path=None, delimiter=None, features=None, scaler=None):
        method = self.__download if path is None else self.__import_dataset

        self.__scaler = scaler
        self.__features = features
        path = self.url if path is None else path
        delimiter = self.delimiter if delimiter is None else delimiter

        data = method(path, delimiter)
        self.dataset = self.__process_features(data)
        self.unlabeled_dataset = self.dataset.iloc[:,:-1]
    
    def split(self, proportion=0.7):
        data = self.dataset.copy()

        randomized = data.sample(frac=1)
        dataset_size = data.iloc[:,0].count()
        split_size = int(math.floor(dataset_size * proportion))

        train = randomized.iloc[0:split_size,:]
        test = randomized.iloc[split_size:,:]
    
        return train, test

    def inverse_scale_sample(self, sample):
        if self.__scaler is not None:
            values = self.__scaler.inverse_transform(sample)
            return pd.Series(data=values, index=sample.index)

        return sample

    def __process_features(self, data):
        pipeline = [ self.__select_features,
                     self.__scale_features ]
        return reduce(lambda s, fn: fn(s), pipeline, data)

    def __select_features(self, data):
        if self.__features is not None:
            features = self.__features[:]
            features.append(-1)

            return data.iloc[:,features]

        return data

    def __scale_features(self, data):
        if self.__scaler is not None:
            data.iloc[:,:-1] = self.__scaler.fit_transform(data.iloc[:,:-1])
            return data
        return data

    def __download(self, url, delimiter=','):
        response = requests.get(url)
        return self.__string_to_dataset(response.text, delimiter)

    def __import_dataset(self, path, delimiter=','):
        return pd.read_csv(path, sep=delimiter)

    def __string_to_dataset(self, text, delimiter):
        path = '/tmp/{}.csv'.format(uuid.uuid4())

        with open(path, 'w+') as f:
            f.write(text)

        data = self.__import_dataset(path, delimiter)
        os.remove(path)

        return data

class Jain(Source):
    delimiter = '\t'
    url = 'https://www.dropbox.com/s/lh1eqxg5ntl75w5/jain.txt?dl=1'

class MislabeledJain(Source):
    delimiter = '\t'    
    url = 'https://www.dropbox.com/s/sqaacjc27ehfssl/mislabeled-jain.txt?dl=1'

class Iris(Source):
    delimiter = ','
    url = 'https://www.dropbox.com/s/zzrqn68bm97ecys/iris.data?dl=1'

class MislabeledIris(Source):
    delimiter = ','
    url = 'https://www.dropbox.com/s/whezliw9a4efrx2/mislabeled-iris.data?dl=1'

class Wine(Source):
    delimiter = ','
    url = 'https://www.dropbox.com/s/4liqqz9fo3ih2fn/wine.data?dl=1'

class MislabeledWine(Source):
    delimiter = ','
    url = 'https://www.dropbox.com/s/pnmrgr8r69p006n/mislabeled-wine.data?dl=1'

class SourceFactory(object):
    @staticmethod
    def build(name, *args, **kwargs):
        if name is None:
            return None

        all_sources = Source.__subclasses__()
        class_object = filter(lambda x: x.__name__ == name, all_sources)[0]
        
        return class_object(*args, **kwargs)
    
    @staticmethod
    def values():
        return map(lambda x: x.__name__, Source.__subclasses__())
