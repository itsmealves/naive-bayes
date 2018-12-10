import numpy as np
from tqdm import tqdm
from data.source import SourceFactory
from data.processing.scaling import Standard
from model.unsupervised import UnsupervisedClassifier


def statistics_of(fn, samples):
    values = []
    for i in tqdm(range(samples), total=samples):
        values.append(fn(i))
    
    mat = np.array(values)
    return mat.mean(), mat.std()
    
if __name__ == '__main__':
    source = SourceFactory.build('Jain', scaler=Standard()) 

    def classify(i):
        train, test = source.split()
        
        classifier = UnsupervisedClassifier(k_min=3, k_max=20) 
        classifier.fit(train)
        true_predictions, total = classifier.evaluate(test)
        return true_predictions / total

    print('Accuracy: {} {}'.format(*statistics_of(classify, 30)))

