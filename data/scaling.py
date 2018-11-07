import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class ScalingMethod(object):
	pass

class Standard(ScalingMethod, StandardScaler):
	pass

class MinMax(ScalingMethod, MinMaxScaler):
	pass

class ScalingMethodFactory(object):
	@staticmethod
	def build(name):
		if name is None:
			return None

		all_methods = ScalingMethod.__subclasses__()
		class_object = filter(lambda x: x.__name__ == name, all_methods)[0]
		return class_object()

	@staticmethod
	def values():
		return map(lambda x: x.__name__, ScalingMethod.__subclasses__())

