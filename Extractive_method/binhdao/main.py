import torch 
import rouge
import model
from evaluate import evaluate_vi


if __name__ == '__main__':
    path = '../../../vietnews/data/test_tokenized'
    aggregator = 'Avg'
    summarizer = model.CentroidBOWSummarizer(language='english')

    evaluate_vi(path, aggregator, summarizer)