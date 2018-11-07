import math
import pandas as pd

class Dataset(object):
	@staticmethod
	def fetch(path, scaling_method=None, features=None, split_proportion=0.7):
		dataset = pd.read_csv(path)

		if scaling_method is not None:
			df = dataset.copy().iloc[:,0:-1]
			dataset[df.columns] = scaling_method.fit_transform(df[df.columns])

		randomized = dataset.sample(frac=1)
		dataset_size = dataset.iloc[:,0].count()
		split_size = int(math.floor(dataset_size * split_proportion))

		if features is not None:
			features = features[:]
			features.append(-1)

			train = randomized.iloc[0:split_size,features]
			test = randomized.iloc[split_size:,features]
		else:
			train = randomized.iloc[0:split_size,:]
			test = randomized.iloc[split_size:,:]

		return train, test
