
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

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
tknzr = WordPunctTokenizer()
nltk.download('stopwords')
stoplist = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
stemmer = PorterSt`emmer()
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


# In[4]:

#test tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer

def PreprocessTfidf(texts,stoplist=[],stem=False):
    newtexts = []
    for text in texts:
        if stem:
           tmp = [w for w in tknzr.tokenize(text) if w not in stoplist]
        else:
           tmp = [stemmer.stem(w) for w in [w for w in tknzr.tokenize(text) if w not in stoplist]]
        newtexts.append(' '.join(tmp))
    return newtexts
vectorizer = TfidfVectorizer(min_df=1)
processed_reviews = PreprocessTfidf(tot_textreviews,stoplist,True)
mod_tfidf = vectorizer.fit(processed_reviews)
vec_tfidf = mod_tfidf.transform(processed_reviews)
tfidf = dict(zip(vectorizer.get_feature_names(),vectorizer.idf_))


# In[110]:

#dump tf-idf into file
import cPickle as pickle
#print mod_tfidf.get_feature_names()
print len(processed_reviews),'--',len(mod_tfidf.get_feature_names())
v= mod_tfidf.transform(processed_reviews)
#print v
with open('vectorizer.pk', 'wb') as fin:
      pickle.dump(mod_tfidf, fin)
file = open("vectorizer.pk",'r')
load_tfidf =  pickle.load(file)
        
print load_tfidf.transform(PreprocessTfidf([' '.join(['drama'])],stoplist,True))


# In[5]:

#test LSA
import gensim
from gensim import models
class GenSimCorpus(object):
           def __init__(self, texts, stoplist=[],stem=False):
               self.texts = texts
               self.stoplist = stoplist
               self.stem = stem
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
                      yield (x for x in tknzr.tokenize(text) if x not in stoplist)

corpus = GenSimCorpus(tot_textreviews,stoplist,True)
dict_corpus = corpus.dictionary
ntopics = 10
lsi =  models.LsiModel(corpus, num_topics=ntopics, id2word=dict_corpus)


# In[6]:

U = lsi.projection.u
Sigma = np.eye(ntopics)*lsi.projection.s
#calculate V
V = gensim.matutils.corpus2dense(lsi[corpus], len(lsi.projection.s)).T / lsi.projection.s
dict_words = {}
for i in range(len(dict_corpus)):
    dict_words[dict_corpus[i]] = i


# In[7]:

from collections import namedtuple

def PreprocessDoc2Vec(text,stop=[],stem=False):
    words = tknzr.tokenize(text)
    if stem:
       words_clean = [stemmer.stem(w) for w in [i.lower() for i in words if i not in stop]]
    else:
       words_clean = [i.lower() for i in words if i not in stop]
    return words_clean

Review = namedtuple('Review','words tags')
dir = './review_polarity/txt_sentoken/'
do2vecstem = False
reviews_pos = []
cnt = 0
for filename in [f for f in os.listdir(dir+'pos/') if str(f)[0]!='.']:
    f = open(dir+'pos/'+filename,'r')
    reviews_pos.append(Review(PreprocessDoc2Vec(f.read(),stoplist,do2vecstem),['pos_'+str(cnt)]))
    cnt+=1
    
reviews_neg = []
cnt= 0
for filename in [f for f in os.listdir(dir+'neg/') if str(f)[0]!='.']:
    f = open(dir+'neg/'+filename,'r')
    reviews_neg.append(Review(PreprocessDoc2Vec(f.read(),stoplist,do2vecstem),['neg_'+str(cnt)]))
    cnt+=1

tot_reviews = reviews_pos + reviews_neg


# In[ ]:




# In[8]:

#define doc2vec
from gensim.models import Doc2Vec
import multiprocessing

cores = multiprocessing.cpu_count()
vec_size = 500
model_d2v = Doc2Vec(dm=1, dm_concat=0, size=vec_size, window=10, negative=0, hs=0, min_count=1, workers=cores)

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


# In[9]:

#query
query = ['science','future','action']


# In[10]:

#similar tfidf
#sparse matrix so the metrics transform into regular vectors before computing cosine
from sklearn.metrics.pairwise import cosine_similarity
query_vec = mod_tfidf.transform(PreprocessTfidf([' '.join(query)],stoplist,True))
sims= cosine_similarity(query_vec,vec_tfidf)[0]
indxs_sims = sims.argsort()[::-1]
for d in list(indxs_sims)[:5]:
    print 'sim:',sims[d],' title:',tot_titles[d]


# In[11]:

#LSA query
def TransformWordsListtoQueryVec(wordslist,dict_words,stem=False):
    q = np.zeros(len(dict_words.keys()))
    for w in wordslist:
        if stem:
            q[dict_words[stemmer.stem(w)]]=1.
        else:
            q[dict_words[w]] = 1.
    return q

q = TransformWordsListtoQueryVec(query,dict_words,True)

qk =   np.dot(np.dot(q,U),Sigma)

sims = np.zeros(len(tot_textreviews))
for d in range(len(V)):
    sims[d]=np.dot(qk,V[d])
indxs_sims = np.argsort(sims)[::-1]  
for d in list(indxs_sims)[:5]:
    print 'sim:',sims[d],' doc:',tot_titles[d]


# In[12]:

#doc2vec query
#force inference to get the same result
model_d2v.random = np.random.RandomState(1)
query_docvec = model_d2v.infer_vector(PreprocessDoc2Vec(' '.join(query),stoplist,do2vecstem))

reviews_related = model_d2v.docvecs.most_similar([query_docvec], topn=5)#model_d2v.docvecs.most_similar([query_docvec], topn=3)
for review in reviews_related:
    print 'relevance:',review[1],'  title:',tot_titles[review[0]]


# In[ ]:



