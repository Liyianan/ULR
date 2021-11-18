from nltk.tokenize import sent_tokenize
import time
import sys
import numpy as np
import re
MAX_LEN = 6
def compute_pmi(sentences):
  # count ngram frequency
  vocab = []
  s = time.time()
  for L in range(1, MAX_LEN + 1):
    vocab.append({})
    for sent in sentences:
      sent = re.sub("[().,?;\'!:]+", '', sent)
      sent = sent.split()
      for i in range(len(sent) - L+1):
        cand_ngram = ' '.join(sent[i: i+L])
        if '[' in cand_ngram or ']' in cand_ngram:
          continue
        vocab[-1][cand_ngram] = vocab[-1].get(cand_ngram, 0) + 1
  
  # compute pmi score for all ngrams (n>=2)
  for word in vocab[0]:
    vocab[0][word] = np.log(vocab[0][word])
  for i in range(1, MAX_LEN):
    for ngram in vocab[i]:
      ngram_split = ngram.split()
      vocab[i][ngram] = np.log(vocab[i][ngram])
      for tok in ngram_split:
        vocab[i][ngram] -= vocab[0][tok]
      #vocab[i][ngram] /= i+1
  ngrams_all = {}
  for i in range(1, MAX_LEN):
    for item in vocab[i]:
      ngrams_all[item] = vocab[i][item]
  
  # select 3000 ngrams with highest pmi scores (n>=2)
  d_order = sorted(ngrams_all.items(), key=lambda x:x[1], reverse=True) 
  pruned_ngrams =set()
  for i in range(min(len(d_order), 3000)):
    pruned_ngrams.add(d_order[i][0])
  len_all = {}

  return pruned_ngrams

def ngram_match(sentences, ngrams):
  out_sent = []
  for sent in sentences:
    sent = re.sub("[().,?;\'!:]+", '', sent).split()
    i = 0
    out = []
    while i < len(sent):
      flag = False
      for j in range(MAX_LEN-1):# 0,1,2,3
        L = MAX_LEN - j  # 5,4,3,2
        if i + L > len(sent):
          continue
        cand = ' '.join(sent[i:i+L])
        if '[' in cand or ']' in cand:
          continue
        if cand in ngrams:
          flag = True
          out.append('(' + cand + ')')
          i += L
          break
      if not flag:
        out.append(sent[i])
        i += 1
    out = ' '.join(out)
    out_sent.append(out)
  return out_sent

def check(sentences):
  len_all = {}
  tokens_all = 0
  tokens_ngram = 0
  for sent in sentences:
    tokens_all += len(sent.split())
    ngrams = re.findall('\([^\(\)]*\)', sent)
    for phra in ngrams:
      len_all[str(len(phra.split()))] = len_all.get(str(len(phra.split())), 0) + 1
      tokens_ngram += len(phra.split())


i = 0
counter = 0
sentences = []
fo = open('wiki_clean_ngrams_sent_per_line.txt', 'w')
num_articles = 0
with open('wiki_clean_articles.txt') as f:
  for i, line in enumerate(f):
    if line[0] == '【':
      if i == 0:
        continue
      counter += 1
      if counter == 1:
        num_articles += 1
        if num_articles % 100 == 0:
          print('processing article: ', num_articles)
        start = time.time()
        ngrams = compute_pmi(sentences)
        out_sent = ngram_match(sentences, ngrams)
        #check(out_sent)
        for sent in out_sent:
          fo.write(sent + '\n')
        sentences = []
        counter = 0
    else:
      for _ in sent_tokenize(line.strip()):  
        sentences.append(_)



     
