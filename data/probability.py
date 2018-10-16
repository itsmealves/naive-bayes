import math
import numpy as np

# All probability types should be defined with:
# 	* A 'get' method, where a parameter will be computed for the given data slice
# 	* A 'calculate' method, where a probability will be computed for a value given a parameter
# The parameter should be any structure that can be used to produce a probability value by itself for a given value

# For sampling probability calculation
# Should be used for discrete-valued features
class ProbabilityMethod(object):
	pass

class Sampling(ProbabilityMethod):
	def get(self, data):
		parameters = {}
		values, frequencies = np.unique(data, return_counts=True)

		for i, value in enumerate(values):
			parameters[value] = float(frequencies[i]) / float(data.shape[0])

		return parameters

	def calculate(self, parameters, value):
		try:
			return parameters[value]
		except KeyError:
			return 0

# Calculates probabilities based on a Gaussian curve
# Should be used for continuous-valued features
class Gaussian(ProbabilityMethod):
	def get(self, data):
		v = data.astype('float')
		return v.mean(), v.std()

	def calculate(self, parameters, value):
		mean, std = parameters

		a = 1.0 / (std * math.sqrt(2 * math.pi))
		b = ((value - mean) ** 2) / (2 * (std ** 2))

		# Gaussian probability density function
		# Reference: https://en.wikipedia.org/wiki/Gaussian_function
		return a * math.exp(-b)

class ProbabilityMethodFactory(object):
	@staticmethod
	def build(name):
		all_methods = ProbabilityMethod.__subclasses__()
		class_object = filter(lambda x: x.__name__ == name, all_methods)[0]
		return class_object()

	@staticmethod
	def values():
		return map(lambda x: x.__name__, ProbabilityMethod.__subclasses__())

