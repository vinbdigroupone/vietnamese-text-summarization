import torch 
import rouge
import model
from evaluate import evaluate_vi

def prepare_results(p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


if __name__ == '__main__':
    path = '../../../vietnews/data/test_tokenized'
    aggregator = 'Avg'
    summarizer = model.CentroidBOWSummarizer(language='english')

    evaluate_vi(path, aggregator, summarizer)