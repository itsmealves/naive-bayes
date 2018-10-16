import argparse

from data.dataset import Dataset
from model.bayes import NaiveBayesClassifier
from data.probability import ProbabilityMethodFactory

parser = argparse.ArgumentParser()

parser.add_argument('dataset', help='Path for a .csv file containing the dataset')
parser.add_argument('probability_method',
					choices=ProbabilityMethodFactory.values(),
					help='It tells how the probabilities should be computed')

args = parser.parse_args()

if __name__ == '__main__':
	probability_method = ProbabilityMethodFactory.build(args.probability_method)
	classifier = NaiveBayesClassifier(probability_method)

	train, test = Dataset.fetch(args.dataset)
	
	classifier.fit(train)
	classifier.evaluate(test)



