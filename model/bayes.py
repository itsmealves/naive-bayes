import numpy as np

class NaiveBayesClassifier(object):
	def __init__(self, probability_method=None):
		# For class probabilities storage
		self.__class_parameters = {}

		# For X probabilities storage (using a custom probability method)
		self.__feature_parameters = {}

		# For X given class probabilities storage (using a custom probability method)
		self.__conditional_parameters = {}

		# Custom probability method
		self.__probability_method = probability_method

	def __set_class_parameter(self, class_id, dataframe):
		# Get sample probability for this class_id
		class_columns = dataframe[dataframe.iloc[:,-1].astype('str')
		rows = class_columns.str.match(str(class_id))]
		rows_count = float(rows.iloc[:,0].count())
		dataframe_count = float(dataframe.iloc[:,0].count())

		self.__class_parameters[class_id] = rows_count / dataframe_count

	def __set_feature_parameter(self, feature, dataframe):
		# Get values for this feature
		frame = dataframe[feature].values
		parameters = self.__probability_method.get(frame)
		self.__feature_parameters[feature] = parameters

	def __set_conditional_parameter(self, class_id, feature, dataframe):
		# Get values for this feature considering this class_id
		frame = dataframe[dataframe.iloc[:,-1] == class_id][feature].values
		parameters = self.__probability_method.get(frame)
		self.__conditional_parameters[class_id][feature] = parameters


	def fit(self, dataframe):
		classes = dataframe.iloc[:,-1].unique()

		for class_id in classes:
			self.__conditional_parameters[class_id] = {}
			self.__set_class_parameter(class_id, dataframe)

			for feature in dataframe.iloc[:,0:-1]:
				self.__set_feature_parameter(feature, dataframe)
				self.__set_conditional_parameter(class_id, feature, dataframe)

	def evaluate(self, dataframe):
		true_predictions = 0
		num_rows = dataframe.iloc[:,0].count()

		for i in range(num_rows):
			sample = dataframe.iloc[i,:]
			unclassified = dataframe.iloc[i,:-1]

			prediction, probabilities = self.predict(unclassified)

			if prediction == sample.iloc[-1]:
				true_predictions += 1

		return float(true_predictions), float(num_rows)



	def predict(self, sample):
		result = {}
		max_probability = 0
		max_probability_class = None

		for class_id in self.__class_parameters:
			r = 1		# Probability of X
			s = 1		# Conditional probability of X given class

			for feature in sample.index:
				p1 = self.__feature_parameters[feature]
				p2 = self.__conditional_parameters[class_id][feature]

				r = r * self.__probability_method.calculate(p1, sample[feature])
				s = s * self.__probability_method.calculate(p2, sample[feature])

			t = self.__class_parameters[class_id]  # Probability of class
			result[class_id] = (t * s) / r   # Bayes theorem, calculating probability of class given X

			if result[class_id] > max_probability:
				max_probability = result[class_id]
				max_probability_class = class_id

		return max_probability_class, result