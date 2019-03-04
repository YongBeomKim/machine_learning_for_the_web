
# coding: utf-8

# In[1]:

#import files
import os
import numpy as np
#get titles
from BeautifulSoup import BeautifulSoup
moviehtmldir = './movie/'
moviedict = {}
for filename in [f for f in os.listdir(moviehtmldir) if f[0]!='.']:
    id = filename.split('.')[0]
    f = open(moviehtmldir+'/'+filename)
    parsed_html = BeautifulSoup(f.read())
    try:
       title = parsed_html.body.h1.text
       
    except:
       title = 'none'
    moviedict[id] = title


# In[2]:

def ListDocs(dirname):
    docs = []
    titles = []
    for filename in [f for f in os.listdir(dirname) if str(f)[0]!='.']:
        f = open(dirname+'/'+filename,'r')
        id = filename.split('.')[0].split('_')[1]
        titles.append(moviedict[id])
        docs.append(f.read())
    return docs,titles

dir = './review_polarity/txt_sentoken/'
pos_textreviews,pos_titles = ListDocs(dir+'pos/')
neg_textreviews,neg_titles = ListDocs(dir+'neg/')
tot_textreviews = pos_textreviews+neg_textreviews
tot_titles = pos_titles+neg_titles


# In[6]:

#LDA
import gensim.models
from gensim import models

from nltk.tokenize import RegexpTokenizer
tknzr = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
class GenSimCorpus(object):
           def __init__(self, texts, stoplist=[],bestwords=[],stem=False):
               self.texts = texts
               self.stoplist = stoplist
               self.stem = stem
               self.bestwords = bestwords
               self.dictionary = gensim.corpora.Dictionary(self.iter_docs(texts, stoplist))
            
           def __len__(self):
               return len(self.texts)
           def __iter__(self):
               for tokens in self.iter_docs(self.texts, self.stoplist):
                   yield self.dictionary.doc2bow(tokens)
           def iter_docs(self,texts, stoplist):
               for text in texts:
                   if self.stem:
                      yield (stemmer.stem(w) for w in [x for x in tknzr.tokenize(text) if x not in stoplist])
                   else:
                      if len(self.bestwords)>0:
                         yield (x for x in tknzr.tokenize(text) if x in self.bestwords)
                      else:
                         yield (x for x in tknzr.tokenize(text) if x not in stoplist)            
            
num_topics = 10
corpus = GenSimCorpus(tot_textreviews, stoplist,[],False)
dict_lda = corpus.dictionary
lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dict_lda,passes=10, iterations=50)
print lda.show_topics(num_topics=num_topics)


# In[7]:

import copy
#filter out very common words like mobie and film or very unfrequent terms
out_ids = [tokenid for tokenid, docfreq in dict_lda.dfs.iteritems() if docfreq > 1000 or docfreq < 3 ]
dict_lfq = copy.deepcopy(dict_lda)
dict_lfq.filter_tokens(out_ids)
dict_lfq.compactify()
corpus = [dict_lfq.doc2bow(tknzr.tokenize(text)) for text in tot_textreviews]


# In[8]:

lda_lfq = models.LdaModel(corpus, num_topics=num_topics, id2word=dict_lfq,passes=10, iterations=50,alpha=0.01,eta=0.01)
for t in range(num_topics):
    print 'topic ',t,'  words: ',lda_lfq.print_topic(t,topn=10)
    print


# In[9]:

#topics for each doc
def GenerateDistrArrays(corpus):
         for i,dist in enumerate(corpus[:10]):
             dist_array = np.zeros(num_topics)
             for d in dist:
                 dist_array[d[0]] =d[1]
             if dist_array.argmax() == 6 :
                print tot_titles[i]
corpus_lda = lda_lfq[corpus]
GenerateDistrArrays(corpus_lda)


# In[3]:

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
tknzr = WordPunctTokenizer()

from nltk.tokenize import RegexpTokenizer
tknzr = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)

nltk.download('stopwords')
stoplist = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


from collections import namedtuple

def PreprocessReviews(text,stop=[],stem=False):
    #print profile
    words = tknzr.tokenize(text)
    if stem:
       words_clean = [stemmer.stem(w) for w in [i.lower() for i in words if i not in stop]]
    else:
       words_clean = [i.lower() for i in words if i not in stop]
    return words_clean

Review = namedtuple('Review','words title tags')
dir = './review_polarity/txt_sentoken/'
do2vecstem = True
reviews_pos = []
cnt = 0
for filename in [f for f in os.listdir(dir+'pos/') if str(f)[0]!='.']:
    f = open(dir+'pos/'+filename,'r')
    id = filename.split('.')[0].split('_')[1]
    reviews_pos.append(Review(PreprocessReviews(f.read(),stoplist,do2vecstem),moviedict[id],['pos_'+str(cnt)]))
    cnt+=1
    
reviews_neg = []
cnt= 0
for filename in [f for f in os.listdir(dir+'neg/') if str(f)[0]!='.']:
    f = open(dir+'neg/'+filename,'r')
    id = filename.split('.')[0].split('_')[1]
    reviews_neg.append(Review(PreprocessReviews(f.read(),stoplist,do2vecstem),moviedict[id],['neg_'+str(cnt)]))
    cnt+=1

tot_reviews = reviews_pos + reviews_neg


# In[11]:

#split in test training sets
def word_features(words):
    return dict([(word, True) for word in words])
negfeatures = [(word_features(r.words), 'neg') for r in reviews_neg]
posfeatures = [(word_features(r.words), 'pos') for r in reviews_pos]
portionpos = int(len(posfeatures)*0.8)
portionneg = int(len(negfeatures)*0.8)
print portionpos,'-',portionneg
trainfeatures = negfeatures[:portionneg] + posfeatures[:portionpos]
print len(trainfeatures)
testfeatures = negfeatures[portionneg:] + posfeatures[portionpos:]
#shuffle(testfeatures)


# In[12]:

from nltk.classify import NaiveBayesClassifier
#training naive bayes 
classifier = NaiveBayesClassifier.train(trainfeatures)
##testing
err = 0
print 'test on: ',len(testfeatures)
for r in testfeatures:
    sent = classifier.classify(r[0])
    if sent != r[1]:
       err +=1.
print 'error rate: ',err/float(len(testfeatures))


# In[117]:




# In[16]:

import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from random import shuffle

#train bigram:
def bigrams_words_features(words, nbigrams=200,measure=BigramAssocMeasures.chi_sq):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(measure, nbigrams)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])

negfeatures = [(bigrams_words_features(r.words,500), 'neg') for r in reviews_neg]
posfeatures = [(bigrams_words_features(r.words,500), 'pos') for r in reviews_pos]
portionpos = int(len(posfeatures)*0.8)
portionneg = int(len(negfeatures)*0.8)
print portionpos,'-',portionneg
trainfeatures = negfeatures[:portionpos] + posfeatures[:portionneg]
print len(trainfeatures)
classifier = NaiveBayesClassifier.train(trainfeatures)
##test bigram
testfeatures = negfeatures[portionneg:] + posfeatures[portionpos:]
shuffle(testfeatures)
err = 0
print 'test on: ',len(testfeatures)
for r in testfeatures:
    sent = classifier.classify(r[0])
    #print r[1],'-pred: ',sent
    if sent != r[1]:
       err +=1.
print 'error rate: ',err/float(len(testfeatures))


# In[21]:

import nltk.classify.util, nltk.metrics
tot_poswords = [val for l in [r.words for r in reviews_pos] for val in l]
tot_negwords = [val for l in [r.words for r in reviews_neg] for val in l]
from nltk.probability import FreqDist, ConditionalFreqDist
word_fd = FreqDist()
label_word_fd = ConditionalFreqDist()
 
for word in tot_poswords:
    word_fd[word.lower()] +=1
    label_word_fd['pos'][word.lower()] +=1
 
for word in tot_negwords:
    word_fd[word.lower()] +=1
    label_word_fd['neg'][word.lower()] +=1
pos_words = len(tot_poswords)
neg_words = len(tot_negwords)

tot_words = pos_words + neg_words
#select the best words in terms of information contained in the two classes pos and neg
word_scores = {}
 
for word, freq in word_fd.iteritems():
    pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],
                (freq, pos_words), tot_words)
    neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],
                (freq, neg_words), tot_words)
    word_scores[word] = pos_score + neg_score
print 'total: ',len(word_scores)
best = sorted(word_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:10000]
bestwords = set([w for w, s in best])


# In[22]:

#training naive bayes with chi square feature selection of best words
def best_words_features(words):
    return dict([(word, True) for word in words if word in bestwords])

negfeatures = [(best_words_features(r.words), 'neg') for r in reviews_neg]
posfeatures = [(best_words_features(r.words), 'pos') for r in reviews_pos]
portionpos = int(len(posfeatures)*0.8)
portionneg = int(len(negfeatures)*0.8)
print portionpos,'-',portionneg
trainfeatures = negfeatures[:portionpos] + posfeatures[:portionneg]
print len(trainfeatures)
classifier = NaiveBayesClassifier.train(trainfeatures)
##test with feature chi square selection
testfeatures = negfeatures[portionneg:] + posfeatures[portionpos:]
shuffle(testfeatures)
err = 0
print 'test on: ',len(testfeatures)
for r in testfeatures:
    sent = classifier.classify(r[0])
    #print r[1],'-pred: ',sent
    if sent != r[1]:
       err +=1.
print 'error rate: ',err/float(len(testfeatures))


# In[23]:

from gensim.models import Doc2Vec

import multiprocessing

shuffle(tot_reviews)
cores = multiprocessing.cpu_count()
vec_size = 500
model_d2v = Doc2Vec(dm=1, dm_concat=0, size=vec_size, window=5, negative=0, hs=0, min_count=1, workers=cores)

#build vocab
model_d2v.build_vocab(tot_reviews)
#train
numepochs= 20
for epoch in range(numepochs):
    try:
        print 'epoch %d' % (epoch)
        model_d2v.train(tot_reviews)
        model_d2v.alpha *= 0.99
        model_d2v.min_alpha = model_d2v.alpha
    except (KeyboardInterrupt, SystemExit):
        break


# In[24]:

#split train,test sets
trainingsize = 2*int(len(reviews_pos)*0.8)

train_d2v = np.zeros((trainingsize, vec_size))
train_labels = np.zeros(trainingsize)
test_size = len(tot_reviews)-trainingsize
test_d2v = np.zeros((test_size, vec_size))
test_labels = np.zeros(test_size)

cnt_train = 0
cnt_test = 0
for r in reviews_pos:
    name_pos = r.tags[0]
    if int(name_pos.split('_')[1])>= int(trainingsize/2.):
        test_d2v[cnt_test] = model_d2v.docvecs[name_pos]
        test_labels[cnt_test] = 1
        cnt_test +=1
    else:
        train_d2v[cnt_train] = model_d2v.docvecs[name_pos]
        train_labels[cnt_train] = 1
        cnt_train +=1

for r in reviews_neg:
    name_neg = r.tags[0]
    if int(name_neg.split('_')[1])>= int(trainingsize/2.):
        test_d2v[cnt_test] = model_d2v.docvecs[name_neg]
        test_labels[cnt_test] = 0
        cnt_test +=1
    else:
        train_d2v[cnt_train] = model_d2v.docvecs[name_neg]       
        train_labels[cnt_train] = 0
        cnt_train +=1


# In[27]:

#train log regre
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(train_d2v, train_labels)
print 'accuracy:',classifier.score(test_d2v,test_labels)

from sklearn.svm import SVC
clf = SVC()
clf.fit(train_d2v, train_labels)
print 'accuracy:',clf.score(test_d2v,test_labels)


# In[108]:

#svm linear
clf = SVC(kernel='linear')
clf.fit(train_d2v, train_labels)
print clf.score(test_d2v,test_labels)

