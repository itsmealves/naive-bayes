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
parser.add_argument('-s', '--scaler', help='Determine which method of scaling do you want to apply',
					choices=ScalingMethodFactory.values(), default=None)
parser.add_argument('-n', '--samples', help='Number of samples to enable statistical analysis',
					type=int, default=30)
parser.add_argument('probability_method',
					choices=ProbabilityMethodFactory.values(),
					help='It tells how the probabilities should be computed')

args = parser.parse_args()

if __name__ == '__main__':
	abductions = {}
	scaling_method = ScalingMethodFactory.build(args.scaler)
	probability_method = ProbabilityMethodFactory.build(args.probability_method)
	classifier = NaiveBayesClassifier(probability_method)

	accuracy_list = []
	for i in range(args.samples):
		train, test = Dataset.fetch(args.dataset, scaling_method, args.features)

		classifier.fit(train)
		true_predictions, total = classifier.evaluate(test)
		accuracy_list.append(true_predictions / total)

	mat = np.array(accuracy_list)
	mean = mat.mean()
	std_dev = mat.std()

	print('Accuracy: {} {}'.format(mean, std_dev))

	accuracy_list = []
	for i in range(args.samples):
		train, test = Dataset.fetch(args.dataset, scaling_method, args.features)

		classifier.fit(train)
		true_predictions, total = classifier.evaluate_abduction(train, test)
		accuracy_list.append(true_predictions / total)

	mat = np.array(accuracy_list)
	mean = mat.mean()
	std_dev = mat.std()

	print('Accuracy: {} {}'.format(mean, std_dev))

