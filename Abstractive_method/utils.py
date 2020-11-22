import glob
import os
import pickle

import torch

SOS_token = 0
EOS_token = 1

def flatten_list(lists):
  flat_list = [item for sublist in lists for item in sublist]
  return flat_list

def build_dict(lists):
  word2index = {}
  word2count = {}
  index2word = {0: "SOS", 1: "EOS"}
  n_words = 2  # Count SOS and EOS

  flatten = ''.join(flatten_list(lists))
  for word in flatten.split(' '):
    if word not in word2index:
      word2index[word] = n_words
      word2count[word] = 1
      index2word[n_words] = word
      n_words += 1
    else:
      word2count[word] += 1
  
  return word2index, word2count, index2word, n_words


def readLangs(directory):
    files_path = glob.glob(f'{directory}/*')
    print("Reading lines...")
    sample = []
    text = []
    target = []
    for file in os.listdir(directory):
      with open(os.path.join(directory,file), 'r') as f:
          file_content = f.readlines()
          abstract = file_content[2]
          body = ' '.join(file_content[3:]).replace('\n', '').replace('.\n', '').rstrip("\n")
          pairs = [abstract, body]
          sample.append(pairs)
          text.append(body)
          target.append(abstract)

    return text, target, sample

def prepareData(directory):
    input_lang, output_lang, pairs = readLangs(directory)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    word2index, word2count, index2word, n_words = build_dict(input_lang)
    print("Counted words:")
    print(n_words)

    return input_lang, output_lang, pairs, word2index, word2count, index2word, n_words


def load_pickle(path):
    with open(f'{path}/input_lang.pkl', 'rb') as fp:
        input_lang = pickle.load(fp)

    with open(f'{path}/output_lang.pkl', 'rb') as fp:
        output_lang = pickle.load(fp)

    with open(f'{path}/pairs.pkl', 'rb') as fp:
        pairs = pickle.load(fp)

    with open(f'{path}/word2index.pkl', 'rb') as fp:
        word2index = pickle.load(fp)

    with open(f'{path}/word2count.pkl', 'rb') as fp:
        word2count = pickle.load(fp)

    with open(f'{path}/index2word.pkl', 'rb') as fp:
        index2word = pickle.load(fp)

    with open(f'{path}/n_words.pkl', 'rb') as fp:
        n_words = pickle.load(fp)

    return input_lang, output_lang, pairs, word2index, word2count, index2word, n_words

def indexesFromSentence(lang, sentence, word2index):
    return [word2index.get(word,10) for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, word2index):
    indexes = indexesFromSentence(lang, sentence, word2index)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)


def tensorsFromPair(pair, input_lang, output_lang, word2index):
    input_tensor = tensorFromSentence(input_lang, pair[0], word2index)
    target_tensor = tensorFromSentence(output_lang, pair[1], word2index)
    return (input_tensor, target_tensor)