import math
import pandas as pd

class Dataset(object):
	@staticmethod
	def fetch(path, split_proportion=0.7):
		dataset = pd.read_csv(path)
		randomized = dataset.sample(frac=1)
		dataset_size = dataset.iloc[:,0].count()
		split_size = int(math.floor(dataset_size * split_proportion))

		train = randomized.iloc[0:split_size,:]
		test = randomized.iloc[split_size:,:]

		return train, test
