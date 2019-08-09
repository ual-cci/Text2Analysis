import numpy as np
import gensim
from gensim.parsing.preprocessing import remove_stopwords, stem_text, strip_punctuation
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

documents = np.load("data/documents.npz")['a']
titles = np.load("data/titles.npz")['a']

# How would we continue?
# Gensim loading:
from collections import defaultdict
stoplist = set(', . : / ( ) [ ] - _ ; * & ? ! – a b c d e t i p an us on 000 if it ll to as are then '
               'they our the you we s in if a m I x re to this at ref do and'.split()) # additional stopwords
texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in documents
]

# Task specific
# - remove generic words from ads such as "work", "strong" ...
stoplist = set('experience job ensure able working join key apply strong recruitment work team successful '
               'paid contact email role skills company day good high time required want right success'
               'ideal needs feel send yes no arisen arise title true'.split()) # additional stopwords
texts = [
    [word for word in document if word not in stoplist]
    for document in texts
]

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
top_k = 5
top = sorted(frequency.items(), key=lambda x:-x[1])[:top_k]
for x in top:
    print("{0}: {1}".format(*x))

texts = [
    [token for token in text if frequency[token] > 1]
    for text in texts
]

# After removing stop words in their original form, we can convert all words into tokenized representations
# Optional step! This might work better without this step
ALLOW_STEMMED_REPR = False

if ALLOW_STEMMED_REPR:
    for i in range(len(texts)):
        for j in range(len(texts[i])):
            word = texts[i][j]
            word = stem_text(word) # do we want "porter-stemmed version" ?
            texts[i][j] = word

"""
DBG_freq_between = (10,40) # <a,b>
for term, freq in frequency.items():
    if freq >= DBG_freq_between[0] and freq <= DBG_freq_between[1]:
        print(freq, term)
"""

print(len(texts), "documents")
print("EXAMPLES: ")

for i in range(3):
    print(len(texts[i]), texts[i])

# =====================================

from gensim import corpora
dictionary = corpora.Dictionary(texts)
print(dictionary)

corpus = [dictionary.doc2bow(text) for text in texts]

print("We then have", len(corpus), "len of corpus.")
for i in range(3):
    print(len(corpus[i]), corpus[i])

tfidf = gensim.models.TfidfModel(corpus)  # step 1 -- initialize a model

for i in range(3):                        # step 2 -- use the model to transform vectors
    transformed = tfidf[corpus[i]]
    print(len(transformed), transformed)

corpus_tfidf = tfidf[corpus]
# LSI
print("LSI ---")
lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300)  # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
for i in range(3):
    print(len(corpus_lsi[i]), corpus_lsi[i])
lsi.print_topics(2)

# LDA
print("LDA ---")
## Actually seems to be failing !
## There might need to be more cleaning up done ...
## I also found another tutorial to try specifically for LDA ... https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

# using 1 pass and updating once every 1 chunk (10,000 documents)
lda_topics = 100
lda = gensim.models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=lda_topics, update_every=1, passes=1)
# using 20 full passes, no online updates
#lda = gensim.models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=lda_topics, update_every=0, passes=20)
corpus_lda = lda[corpus_tfidf]
for i in range(3):
    print(len(corpus_lda[i]), corpus_lda[i])
lda.print_topics(2)


xs = []
ys = []
x_dim = 0
y_dim = 1
for i in range(len(corpus_lsi)):
    xs.append(corpus_lsi[i][x_dim][1])
    ys.append(corpus_lsi[i][y_dim][1])

#plt.scatter(xs, ys)
#plt.title('Whole dataset projected into 2 dimensions using bow->tfidf->fold-in-lsi')
#plt.show()

index = gensim.similarities.MatrixSimilarity(lsi[corpus_tfidf])
index_lda = gensim.similarities.MatrixSimilarity(lda[corpus_tfidf])

# SAVE then LOAD
dictionary.save('data/dict.dict')
corpora.MmCorpus.serialize('data/corpus.mm', corpus)
corpora.MmCorpus.serialize('data/corpus_tfidf.mm', corpus_tfidf)
corpora.MmCorpus.serialize('data/corpus_lsi.mm', corpus_lsi)
corpora.MmCorpus.serialize('data/corpus_lda.mm', corpus_lda)
lsi.save('data/model.lsi')
lda.save('data/model.lda')
tfidf.save('data/model.tfidf')
index.save('data/index.index')
index_lda.save('data/index_lda.index')

documents_represented = texts
#np.save("data/documents_represented.npy", documents_represented)
np.savez_compressed("data/documents_represented.npz", a=documents_represented)

