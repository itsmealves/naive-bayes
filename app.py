import sys
import math
import argparse
import numpy as np

from tqdm import tqdm
from data.source import SourceFactory
from model.bayes import NaiveBayesClassifier
from data.processing.scaling import ScalingMethodFactory
from data.processing.probability import ProbabilityMethodFactory

parser = argparse.ArgumentParser()

parser.add_argument('dataset', help='Determines the source to read data from',
        choices=SourceFactory.values())
parser.add_argument('-f', '--features', help='Feature filter for the dataset',
        nargs='+', type=int, default=None)
parser.add_argument('-s', '--scaler', help='Determine which method of scaling do you want to apply',
        choices=ScalingMethodFactory.values(), default=None)
parser.add_argument('-n', '--samples', help='Number of samples to enable statistical analysis',
        type=int, default=30)
parser.add_argument('-a', '--abduct', help='Use a different source for abduction phase',
        choices=SourceFactory.values(), default=None)
parser.add_argument('probability_method', choices=ProbabilityMethodFactory.values(),
        help='It tells how the probabilities should be computed')

def statistics_of(fn, samples):
    values = []
    for i in tqdm(range(samples), total=samples):
        values.append(fn(i))

    mat = np.array(values)
    return mat.mean(), mat.std()

if __name__ == '__main__':
    args = parser.parse_args()
    source = SourceFactory.build(args.dataset)
    abduct_source = SourceFactory.build(args.abduct)
    scaling_method = ScalingMethodFactory.build(args.scaler)
    probability_method = ProbabilityMethodFactory.build(args.probability_method)

    def classify(index):
        train, test = source.split()

        classifier = NaiveBayesClassifier(probability_method)
        classifier.fit(train)
        true_predictions, total = classifier.evaluate_classification(test)

        return true_predictions / total

    def abduct(index):
        train, test = source.split()

        if abduct_source is not None:
            _, test = abduct_source.split()

        classifier = NaiveBayesClassifier(probability_method)
        classifier.fit(train)
        true_predictions, total = classifier.evaluate_abduction(test)

        return true_predictions / total


    print('Loaded {} source'.format(source.__class__.__name__))
    if abduct_source is not None:
        print('Also loaded {} source for abduction test phase'.format(abduct_source.__class__.__name__))

    print('Accuracy: {} {}'.format(*statistics_of(classify, args.samples)))
    print('Accuracy: {} {}'.format(*statistics_of(abduct, args.samples)))


