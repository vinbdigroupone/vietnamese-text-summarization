import nltk
nltk.download('punkt')

import torch
import numpy as np
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.models.roberta import RobertaModel
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


phoBERT = RobertaModel.from_pretrained('PhoBERT_base_fairseq', checkpoint_file='model.pt')

class BPE():
    bpe_codes = 'PhoBERT_base_fairseq/bpe.codes'

class phoBertExtractiveModel():
    def __init__(self):
        self.model = phoBERT
        self.model.bpe = fastBPE(BPE())

    def get_representation(self, tokenized_sentences):
        """
            Input a list of tokenized sentences
            Output an list of encoded sentences
        """
        tokenized_ids = [self.model.encode(sent) for sent in tokenized_sentences]
        tokenized_tensor = [torch.tensor(item) for item in tokenized_ids]
        encoded_list = []
        for i in range(len(tokenized_tensor)):
            with torch.no_grad():
                encoded_layers = self.model.extract_features(tokenized_tensor[i])
            encoded_list.append(encoded_layers)
        sentence_list = [torch.mean(vec, dim=1).reshape((768)).numpy() for vec in encoded_list]
        return sentence_list

    def get_index_from_kmeans(self, sentence_embedding_list):
        """
            Input a list of embeded sentence vectors
            Output an list of indices of sentence in the paragraph, represent the clustering of key sentences
            Note: Kmeans is used here for clustering
        """
        n_clusters = np.ceil(len(sentence_embedding_list) ** 0.5)
        kmeans = KMeans(n_clusters=int(n_clusters))
        kmeans = kmeans.fit(sentence_embedding_list)
        sum_index, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, sentence_embedding_list,
                                                     metric='euclidean')
        sum_index = sorted(sum_index)
        return sum_index

    def get_summarization(self, text):
        """
            Input a text
            Output a summary
        """
        tokenized_sentences = sent_tokenize(text)
        sentences_representation = self.get_representation(tokenized_sentences)
        sum_index = self.get_index_from_kmeans(sentences_representation)
        summary = ' '.join([tokenized_sentences[ind] for ind in sum_index])
        return summary

