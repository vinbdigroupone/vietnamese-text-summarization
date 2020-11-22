import torch 
import rouge
import os
from sklearn.feature_extraction.text import TfidfVectorizer

from evaluate import evaluate_vi
import model


def compute_tfidf(data_path):
    article_list = []
    for file in os.listdir(data_path):
        #print(file)
        with open(f'{data_path}/{file}', 'r') as f:
            article = f.read().splitlines()
            article_list.append(article[4:-1])   # leave out titles, abstracts
            f.close()

    for i in range(len(article_list)):
        article_list[i] = list(filter(None, article_list[i]))   # Remove blank spaces
        article_list[i] = ' '.join(article_list[i])

    # Fit Tfidf
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(article_list)
    return tfidf, tfidf_matrix
                    

if __name__ == '__main__':
    path = '../../../vietnews/data/test_tokenized'
    aggregator = 'Avg'
    tfidf, tfidf_matrix = compute_tfidf(path)

    # summarizer = model.CentroidBOWSummarizer(language='english')
    # summarizer = model.tfidfSummarizer(tfidf, tfidf_matrix, kmeans=False)
    summarizer = model.phoBertSummarizer()

    evaluate_vi(path, aggregator, summarizer)