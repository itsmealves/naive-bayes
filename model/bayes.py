import sys
import uuid
import numpy as np
import pandas as pd
import pyswarms as ps
from scipy.spatial.distance import mahalanobis, euclidean
from model.pso import ParticleSwarmOptimization

class NaiveBayesClassifier(object):
    def __init__(self, probability_method=None):
	np.seterr(all='ignore')

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
	rows = dataframe[dataframe.iloc[:,-1].astype('str').str.match(str(class_id))]
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

    def classes(self):
	return sorted(self.__classes)

    def fit(self, dataframe):
	self.__classes = dataframe.iloc[:,-1].unique()
	self.__labels = dataframe.iloc[:,0:-1].columns
	self.__dimensions = len(dataframe.columns) - 1

	for class_id in self.__classes:
	    self.__conditional_parameters[class_id] = {}
	    self.__set_class_parameter(class_id, dataframe)

	    for feature in dataframe.iloc[:,0:-1]:
		self.__set_feature_parameter(feature, dataframe)
		self.__set_conditional_parameter(class_id, feature, dataframe)

    def evaluate_classification(self, dataframe):
	true_predictions = 0
	num_rows = dataframe.iloc[:,0].count()

	for i in range(num_rows):
	    sample = dataframe.iloc[i,:]
	    unclassified = dataframe.iloc[i,:-1]

	    prediction, probabilities = self.predict(unclassified)

	    if prediction == sample.iloc[-1]:
		true_predictions += 1

	return float(true_predictions), float(num_rows)
    
    def verify(self, test, unsupervised, source, feature_importances):
        true_predictions = 0
        num_rows = test.iloc[:,0].count()
        
        x = {}
	for class_id in self.__classes:
            c = {}
            prototypes = [self.abduct(class_id).values]
            prototypes = prototypes + unsupervised.centroids_by_label(class_id) 
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
            class_name = test.iloc[i,-1]
            unclassified = test.iloc[i,:-1]
            predicted = knn(3, unclassified, x)

            if predicted == str(class_name):
               true_predictions += 1
        
        print('{} {} ***************'.format(true_predictions, num_rows))
        return float(true_predictions), float(num_rows)


    def __verify(self, test, unsupervised, source, feature_importances):
        true_predictions = 0
        num_rows = test.iloc[:,0].count()

        x = {}
	for class_id in self.__classes:
            # c = { 'a': self.abduct(class_id),
            #       'b': unsupervised.centroid_by_label(class_id) }

	    x[str(class_id)] = self.abduct(class_id)

        for i in range(num_rows):
            min_dist = None
            sample = test.iloc[i,:]
            unclassified = test.iloc[i,:-1]
            class_name = test.iloc[i,-1]

            for class_id in x:
                prototype = x[class_id]
                # a = np.multiply(unclassified.values, feature_importances)
                # b = np.multiply(prototype.values, feature_importances)
                a = unclassified.values
                b = prototype.values
                distance = euclidean(a, b)
                
                if min_dist is None or distance < min_dist[1]:
                    min_dist = (class_id, distance)

            # up = unsupervised.predict(unclassified)
            # if not min_dist[0] == up:
            #     if self.predict(unclassified)[0] == class_name:
            #         true_predictions += 1
            if min_dist[0] == str(class_name):
               true_predictions += 1
        
        print('{} {} ***************'.format(true_predictions, num_rows))
        return float(true_predictions), float(num_rows)

    def evaluate_abduction(self, test):
	true_predictions = 0
	num_rows = test.iloc[:,0].count()
	# cov = np.linalg.inv(train.iloc[:,:-1].cov())

	x = {}
	for class_id in self.__classes:
	    x[str(class_id)] = self.abduct(class_id)

	for i in range(num_rows):
	    min_dist = (0, sys.maxsize)
	    sample = test.iloc[i,:]
            true_class = test.iloc[i,-1]
    	    unclassified = test.iloc[i,:-1]

	    for class_id in x:
	        # distance = mahalanobis(unclassified, x[class_id], cov)
		distance = euclidean(unclassified, x[class_id])
		if distance < min_dist[1]:
		    min_dist = (class_id, distance)
            
	    if str(true_class) == min_dist[0]:
		true_predictions += 1
        
	return float(true_predictions), float(num_rows)

    def abduct_multiple(self, class_id, k):
        prototypes = []
       
        def optimization_function(x):
            def reducer(a, b):
                return a + euclidean(x, b)

            p = len(prototypes) + 1.0
            dataframe = pd.Series(data=x, index=self.__labels)
            target = self.__inverse_run(class_id, dataframe)
            distances = reduce(reducer, prototypes, 1)

            return target * distances

        for i in range(k):
            pso = ParticleSwarmOptimization(dimensions=self.__dimensions) 
            sample, value = pso.optimize(objective_function=optimization_function)
            print(value)
            prototypes.append(sample)
            
        return map(lambda x: pd.Series(data=x, index=self.__labels), prototypes)
        
    def abduct(self, class_id):
        def optimization_function(x):
	    dataframe = pd.Series(data=x, index=self.__labels)
			
	    other_classes = 0
	    target = self.__inverse_run(class_id, dataframe)
	    other_classes_id = filter(lambda x: x != class_id, self.__classes)

	    for other_class_id in other_classes_id:
		other_classes += self.__inverse_run(other_class_id, dataframe)
            
	    return target - other_classes

	pso = ParticleSwarmOptimization(dimensions=self.__dimensions)
	sample, value = pso.optimize(objective_function=optimization_function)
        
        print(sample, value)
        return pd.Series(data=sample, index=self.__labels)

    def __inverse_run(self, class_id, sample):
	r = 1		# Probability of X

	for feature in sample.index:
	    p1 = self.__feature_parameters[feature]
	    r = r * self.__probability_method.calculate(p1, sample[feature])

	s = self.__run(class_id, sample)		# Conditional probability of class given X
	t = self.__class_parameters[class_id]  	# Probability of class
	return (r * s) / t   # Bayes theorem, calculating probability of class given X

    def __run(self, class_id, sample):
	r = 1		# Probability of X
	s = 1		# Conditional probability of X given class

	for feature in sample.index:
	    p1 = self.__feature_parameters[feature]
	    p2 = self.__conditional_parameters[class_id][feature]

	    r = r * self.__probability_method.calculate(p1, sample[feature])
	    s = s * self.__probability_method.calculate(p2, sample[feature])

        t = self.__class_parameters[class_id]  # Probability of class
        return (t * s) / r   # Bayes theorem, calculating probability of class given X

    def predict(self, sample):
	result = {}
	max_probability = 0
	max_probability_class = None

	for class_id in self.__classes:
	    result[class_id] = self.__run(class_id, sample)

	    # argmax: checking for which class the input gives a higher output
	    if result[class_id] > max_probability:
		max_probability = result[class_id]
		max_probability_class = class_id

	# return the classification output and the probability for each class
	return max_probability_class, result
    
