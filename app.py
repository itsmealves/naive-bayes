import argparse
import numpy as np

from tqdm import tqdm
from data.source import SourceFactory
from model.bayes import NaiveBayesClassifier
from sklearn.ensemble import RandomForestClassifier
from model.unsupervised import UnsupervisedClassifier
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
parser.add_argument('-t', '--abduct-source', help='Use a different source for abduction test phase',
        choices=SourceFactory.values(), default=None)
parser.add_argument('-c', '--class-source', help='Use a different source for classification test phase',
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
    scaling_method = ScalingMethodFactory.build(args.scaler)
    probability_method = ProbabilityMethodFactory.build(args.probability_method)
    source = SourceFactory.build(args.dataset, 
            features=args.features, scaler=scaling_method)
    class_source = SourceFactory.build(args.class_source,
            features=args.features, scaler=scaling_method)
    abduct_source = SourceFactory.build(args.abduct_source,
            features=args.features, scaler=scaling_method)

    def classify(index):
        train, test = source.split()

        if class_source is not None:
            _, test = class_source.split()

        classifier = NaiveBayesClassifier(probability_method)
        classifier.fit(train)
        true_predictions, total = classifier.evaluate_classification(test)

        return true_predictions / total

    def verify(index):
        train, test = source.split()

        if class_source is not None:
            _, test = class_source.split()

        classifier = NaiveBayesClassifier(probability_method)
        c = RandomForestClassifier(n_estimators=100, max_depth=2)
        c.fit(train.iloc[:,:-1], train.iloc[:,-1])

        classifier.fit(train)
        unsupervised = UnsupervisedClassifier(k_min=len(classifier.classes()), k_max=25)
        unsupervised.fit(train)

        if class_source is None:
            true_predictions, total = classifier.verify(test, unsupervised, source, c.feature_importances_)
        else:
            true_predictions, total = classifier.verify(test, unsupervised, class_source, c.feature_importances_)

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
    if class_source is not None:
        print('\t* Also loaded {} source for classification test phase'.format(class_source.__class__.__name__))
    if abduct_source is not None:
        print('\t* Also loaded {} source for abduction test phase'.format(abduct_source.__class__.__name__))

    print('')
    print('Accuracy: {} {}'.format(*statistics_of(classify, args.samples)))
    print('Accuracy: {} {}'.format(*statistics_of(verify, args.samples)))
    # print('Accuracy: {} {}'.format(*statistics_of(abduct, args.samples)))

    # train, test = source.split()
    # classifier = NaiveBayesClassifier(probability_method)
    # classifier.fit(train)

    # for class_id in classifier.classes():
    #     print(class_id)
    #     for i in range(3):
    #         for prototype in classifier.abduct_multiple(class_id, 3):
    #             print(source.inverse_scale_sample(prototype))


