import os
import math
import uuid
import requests
import pandas as pd

class Source(object):
    def __init__(self, path=None, delimiter=None, features=None):
        method = self.__download if path is None else self.__import_dataset

        self.__features = features
        path = self.url if path is None else path
        delimiter = self.delimiter if delimiter is None else delimiter

        data = method(path, delimiter)
        self.dataset = self.__process_features(data)
    
    def split(self, proportion=0.7):
        data = self.dataset.copy()

        randomized = data.sample(frac=1)
        dataset_size = data.iloc[:,0].count()
        split_size = int(math.floor(dataset_size * proportion))

        train = randomized.iloc[0:split_size,:]
        test = randomized.iloc[split_size:,:]
    
        return train, test

    def __process_features(self, data):
        pipeline = [self.__select_features]
        return reduce(lambda s, fn: fn(s), pipeline, data)

    def __select_features(self, dataset):
        if self.__features is not None:
            features = self.__features[:]
            features.append(-1)

            return dataset.iloc[:,features]

        return dataset

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
    url = 'https://www.dropbox.com/s/neahy1hq971qa0f/mislabeled-jain.txt?dl=1'

class Iris(Source):
    delimiter = ','
    url = 'https://www.dropbox.com/s/zzrqn68bm97ecys/iris.data?dl=1'

class MislabeledIris(Source):
    delimiter = ','
    url = 'https://www.dropbox.com/s/whezliw9a4efrx2/mislabeled-iris.data?dl=1'

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
