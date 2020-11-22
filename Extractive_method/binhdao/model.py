import base
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.models.roberta import RobertaModel
from nltk.tokenize import sent_tokenize
import torch 
from pytorch_pretrained_bert import BertTokenizer, BertModel


# ==================================== Kmeans with Bert ====================================
class phoBertSummarizer():
    def __init__(self):
        self.phoBERT = RobertaModel.from_pretrained('PhoBERT_base_fairseq', checkpoint_file='model.pt')
        args = BPE()
        self.phoBERT.bpe = fastBPE(args) #Incorporate the BPE encoder into PhoBERT
    
    def summarize(self, text):
        tokenized_sentences = sent_tokenize(text)
        sentences_representation = get_representation(self.phoBERT, tokenized_sentences)
        sum_index = get_index_from_kmeans(sentences_representation)
        summary = ' '.join([tokenized_sentences[ind] for ind in sum_index])
        return summary

class BPE():
    bpe_codes = 'PhoBERT_base_fairseq/bpe.codes'

def get_representation(model, tokenized_sentences):
    tokenized_ids = [model.encode(sent) for sent in tokenized_sentences]
    tokenized_tensor = [torch.tensor(item) for item in tokenized_ids]
    encoded_list = []
    for i in range(len(tokenized_tensor)):
        with torch.no_grad():
            encoded_layers = model.extract_features(tokenized_tensor[i])
            encoded_list.append(encoded_layers)
    sentence_list = [torch.mean(vec, dim=1).reshape((768)).numpy() for vec in encoded_list]
    return sentence_list

def get_index_from_kmeans(sentence_embedding_list):
    """
    Input a list of embeded sentence vectors
    Output an list of indices of sentence in the paragraph, represent the clustering of key sentences
    Note: Kmeans is used here for clustering
    """
    n_clusters = np.ceil(len(sentence_embedding_list)**0.5)
    kmeans = KMeans(n_clusters=int(n_clusters))
    kmeans = kmeans.fit(sentence_embedding_list)
    sum_index,_ = pairwise_distances_argmin_min(kmeans.cluster_centers_, sentence_embedding_list,metric='euclidean')
    sum_index = sorted(sum_index)
    return sum_index


# ==================================== TFIDF Model =====================================

class tfidfSummarizer():
    def __init__(self, tfidf, tfidf_matrix, kmeans=True):
        self.tfidf_matrix = tfidf_matrix
        self.tfidf = tfidf
        self.kmeans = kmeans

    def summarize(self, text):
        sentence_score_list, sentence_list = score_article(text, self.tfidf, self.tfidf_matrix, 0)
        if self.kmeans:
            summary = generate_summary_kmeans(sentence_score_list, sentence_list)
        else:
            summary = generate_summary_max(sentence_score_list, sentence_list, alpha=1.15)
        return summary


def score_sentence(sentence, tfidf, tfidf_matrix, a_index_):
  # Calculate the sum tfidf score of words as sentence score
  # Take in a sentence (string), and the article's index in the corpus (for word lookup)

  sentence_score = 0
  sentence = sentence.replace(' ,', '').strip()

  # For each word in a sentence, look up its score from tfidf matrix, return 0 if lookup fails
  for i_word in sentence.split(' '):
    #print(i_word)
    sentence_score += tfidf_matrix[a_index_, tfidf.vocabulary_.get(i_word.lower(), 10)]
  
  return sentence_score


def score_article(article, tfidf, tfidf_matrix, a_index_):
  # Generate a list of scores of each sentence of an article
  # Take in an article (list of sentences), and the article's index in the corpus (for word lookup)

  sentence_score_list = []
  sentence_list = article.split('.')

  # For each sentence in an article, call score_sentence and append output to sentence_score_list
  for i_sentence in sentence_list:
    sentence_score = score_sentence(i_sentence, tfidf, tfidf_matrix, a_index_)
    sentence_score_list.append(sentence_score)

  return sentence_score_list, sentence_list


def generate_summary_max(sentence_score_list, sentence_list, alpha=1.15):
  # Generate a summary from sentences that have score alpha times larger than the article' sentence mean score
  # Take in a list of sentence scores, list of sentences, thresholding multiplier alpha

  sentence_score_array = np.asarray(sentence_score_list)
  mean_score = np.mean(sentence_score_array)
  sentence_list_arr = np.asarray(sentence_list, dtype=object)
  summary = sentence_list_arr[sentence_score_array > alpha*mean_score]
  summary = ' '.join(summary)

  return summary

def generate_summary_kmeans(sentence_score_list, sentence_list, k=4):
    sentence_score_list = np.expand_dims(np.array(sentence_score_list), axis=1)
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(np.array(sentence_score_list))
    label_max_tfidf = kmeans.labels_[np.argmax(sentence_score_list)]
    idxes = np.where(kmeans.labels_ == label_max_tfidf)[0]
    
    summary = ''
    for idx in idxes:
        summary += sentence_list[idx]
        
    return summary

# ==================================== BOW Model =====================================
class CentroidBOWSummarizer(base.BaseSummarizer):

    def __init__(self,
                 language='english',
                 preprocess_type='nltk',
                 stopwords_remove=True,
                 length_limit=10,
                 debug=False,
                 topic_threshold=0.3,
                 sim_threshold=0.95):
        super().__init__(language, preprocess_type, stopwords_remove, length_limit, debug)
        self.topic_threshold = topic_threshold
        self.sim_threshold = sim_threshold
        return

    def summarize(self, text, limit_type='word', limit=100):
        raw_sentences = self.sent_tokenize(text)
        clean_sentences = self.preprocess_text(text)

        vectorizer = CountVectorizer()
        sent_word_matrix = vectorizer.fit_transform(clean_sentences)

        transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
        tfidf = transformer.fit_transform(sent_word_matrix)
        tfidf = tfidf.toarray()

        centroid_vector = tfidf.sum(0)
        centroid_vector = np.divide(centroid_vector, centroid_vector.max())
        for i in range(centroid_vector.shape[0]):
            if centroid_vector[i] <= self.topic_threshold:
                centroid_vector[i] = 0

        sentences_scores = []
        for i in range(tfidf.shape[0]):
            score = base.similarity(tfidf[i, :], centroid_vector)
            sentences_scores.append((i, raw_sentences[i], score, tfidf[i, :]))

        sentence_scores_sort = sorted(sentences_scores, key=lambda el: el[2], reverse=True)

        count = 0
        sentences_summary = []
        for s in sentence_scores_sort:
            if count > limit:
                break
            include_flag = True
            for ps in sentences_summary:
                sim = base.similarity(s[3], ps[3])
                # print(s[0], ps[0], sim)
                if sim > self.sim_threshold:
                    include_flag = False
            if include_flag:
                # print(s[0], s[1])
                sentences_summary.append(s)
                if limit_type == 'word':
                    count += len(s[1].split())
                else:
                    count += len(s[1])

        summary = "\n".join([s[1] for s in sentences_summary])
        return summary