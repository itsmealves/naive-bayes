import sys
import math
import argparse
import numpy as np

from data.dataset import Dataset
from model.bayes import NaiveBayesClassifier
from data.probability import ProbabilityMethodFactory
from data.scaling import ScalingMethodFactory

parser = argparse.ArgumentParser()

parser.add_argument('dataset', help='Path for a .csv file containing the dataset')
parser.add_argument('-f', '--features', help='Feature filter for the dataset',
					nargs='+', type=int, default=None)
parser.add_argument('-e', '--exclude-classification', help='Remove columns for the classification experiment',
					nargs='+', type=int, default=None)
parser.add_argument('-a', '--exclude-abduction', help='Remove columns for the abduction experiment',
					nargs='+', type=int, default=None)
parser.add_argument('-b', '--exclude-abduction_test', help='Remove columns for the abduction experiment (test dataset)',
					nargs='+', type=int, default=None)
parser.add_argument('-s', '--scaler', help='Determine which method of scaling do you want to apply',
					choices=ScalingMethodFactory.values(), default=None)
parser.add_argument('-n', '--samples', help='Number of samples to enable statistical analysis',
					type=int, default=30)
parser.add_argument('probability_method',
					choices=ProbabilityMethodFactory.values(),
					help='It tells how the probabilities should be computed')

args = parser.parse_args()

def statistics(fn, samples):
	values = []
	for i in range(samples):
		values.append(fn(i))

	mat = np.array(values)
	return mat.mean(), mat.std()

if __name__ == '__main__':
	scaling_method = ScalingMethodFactory.build(args.scaler)
	probability_method = ProbabilityMethodFactory.build(args.probability_method)

	def classification(index):
		train, test = Dataset.fetch(args.dataset, scaling_method, args.features, args.exclude_classification, args.exclude_classification)

		classifier = NaiveBayesClassifier(probability_method)
		classifier.fit(train)
		true_predictions, total = classifier.evaluate(test)

		return true_predictions / total

	def abduction(index):
		train, test = Dataset.fetch(args.dataset, scaling_method, args.features, args.exclude_abduction, args.exclude_abduction_test)

		classifier = NaiveBayesClassifier(probability_method)
		classifier.fit(train)
		true_predictions, total = classifier.evaluate_abduction(train, test)

		return true_predictions / total

				
	print('Accuracy: {} {}'.format(*statistics(classification, args.samples)))
	print('Accuracy: {} {}'.format(*statistics(abduction, args.samples)))


