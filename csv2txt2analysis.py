# Load data from CSV files:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gensim
import json


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# This scraped dataset specific functions:
csv_files = [
    "../JobAdsData/Source1-Monster/creative_jobs_clean_1363.csv",
    "../JobAdsData/Source1-Monster/IT_jobs_clean_1511.csv",
    "../JobAdsData/Source2-Reed/creative2_pages_clean_1359.csv",
    "../JobAdsData/Source2-Reed/it2_pages_clean_1556.csv",
    "../JobAdsData/Source3-Indeed/creative2_pages_clean_4164.csv",
    "../JobAdsData/Source3-Indeed/it2_pages_clean_986.csv"
]

# Structure:
# ﻿Page_URL,JobTitle,JobDescription,Location,Type,Posted,Industries,Salary

total_lines = 0
total_ignored_short = 0
full_text_dataset = []
full_text_titles = []
lengths_descriptions = []

THR_min_desc_length_allowed = 400
THR_exclude_long_errs = 26000
THR_supposed_to_have_values = 8

for csv_file in csv_files:
    df = pd.read_csv(csv_file, delimiter=',')

    for line in df.values:
        title = line[1]
        description = line[2]
        category_sort_of_like = line[6]

        # one line contains incorrectly scraped data (np.nan):
        if description is not np.nan and title is not np.nan:
            # those broken by splitting (only 3 records)
            if len(line) > THR_supposed_to_have_values and line[-1] is not np.nan:
                #print(len(line))
                #print(csv_file)
                #print(line)
                continue


            if len(description) < THR_min_desc_length_allowed:
                #print(csv_file)
                #print(line)
                total_ignored_short += 1
                continue

            if len(description) > THR_exclude_long_errs:
                total_ignored_short += 1
                continue

            # print(title, "|||", len(description), description)
            full_text_dataset.append(description)
            full_text_titles.append(title)
            lengths_descriptions.append(len(description))
            total_lines += 1


lengths_descriptions = np.asarray(lengths_descriptions)

print("--=============================--")
print("Read",total_lines,"lines in total.")
print("Skipped",total_ignored_short,"too short lines (using thr", THR_min_desc_length_allowed,")")
print("Statistics of lenghts (min, max, avg):",np.min(lengths_descriptions), np.max(lengths_descriptions), np.average(lengths_descriptions))


#plt.plot(lengths_descriptions)
#plt.ylabel('lenghts of descriptions in scraped data')
#plt.show()

# Check the longest entry ? Seems alright.
longest_i = np.argmax(lengths_descriptions)
#print(lengths_descriptions[longest_i])
#print(full_text_dataset[longest_i])

from gensim.parsing.preprocessing import remove_stopwords, stem_text, strip_punctuation
# some hacks - words were joined together just by "." separate them
for i in range(len(full_text_dataset)):
    full_text_dataset[i] = full_text_dataset[i].replace('!', ' ! ')
    full_text_dataset[i] = full_text_dataset[i].replace('?', ' ? ')
    full_text_dataset[i] = full_text_dataset[i].replace('.', ' . ')
    full_text_dataset[i] = full_text_dataset[i].replace(',', ' , ')
    full_text_dataset[i] = full_text_dataset[i].replace(':', ' : ')
    full_text_dataset[i] = full_text_dataset[i].replace(';', ' ; ')
    full_text_dataset[i] = full_text_dataset[i].replace('(', ' ( ')
    full_text_dataset[i] = full_text_dataset[i].replace(')', ' ) ')
    full_text_dataset[i] = full_text_dataset[i].replace('/', ' / ')
    full_text_dataset[i] = full_text_dataset[i].replace('*', ' * ')
    full_text_dataset[i] = full_text_dataset[i].replace(']', ' ] ')
    full_text_dataset[i] = full_text_dataset[i].replace('[', ' [ ')
    full_text_dataset[i] = full_text_dataset[i].replace('&', ' & ')
    full_text_dataset[i] = full_text_dataset[i].replace('_', ' _ ')

    full_text_dataset[i] = remove_stopwords(full_text_dataset[i])

    # slower processing from:
    #full_text_dataset[i] = strip_punctuation(full_text_dataset[i]) # our default seems to be better
    #full_text_dataset[i] = stem_text(full_text_dataset[i]) # do we want "porter-stemmed version" ?

    full_text_dataset[i] = " ".join(list(gensim.utils.tokenize(full_text_dataset[i])))

documents = full_text_dataset

# Save into a txt?

# How would we continue?
# Gensim loading:
from collections import defaultdict
stoplist = set(', . : / ( ) [ ] - _ ; * & ? ! – a b c d e t i p an us on 000 if it ll to as are then '
               'they our the you we s in if a re to this at ref do and'.split()) # additional stopwords
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
lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300)  # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

for i in range(3):
    print(len(corpus_lsi[i]), corpus_lsi[i])

lsi.print_topics(2)

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

# Similarity
check_doc_id = 23

vec_lsi = corpus_lsi[check_doc_id]
index = gensim.similarities.MatrixSimilarity(lsi[corpus_tfidf])

# SAVE then LOAD
#>>> dictionary.save('/tmp/deerwester.dict')
#>>> corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)  # store to disk, for later use
#>>> lsi.save('/tmp/model.lsi')  # same for tfidf, lda, ...
#>>> index.save('/tmp/deerwester.index')

#>>> dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
#>>> corpus = corpora.MmCorpus('/tmp/deerwester.mm')
#>>> lsi = models.LsiModel.load('/tmp/model.lsi')
#>>> index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')


sims = index[vec_lsi]  # perform a similarity query against the corpus
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims)  # print (document_number, document_similarity) 2-tuples

print("Queried document:")
print(full_text_titles[check_doc_id])
print(texts[check_doc_id])
print("Closest documents:")
for i in range(5):
    closest_doc_id = sims[1+i][0]
    print(i, " - id=",closest_doc_id, ", distance=", sims[1+i][1])
    print(full_text_titles[closest_doc_id])
    print(texts[closest_doc_id])

### SEEMS TO BE WORKING: (showing titles for human debugging/insight,
#                         however the distance was calculated purely from descriptions)
# Queried document:
# Graduate Sales Executive - Creative Media - BMS

# Closest documents:
# 0  - id= 1003 , distance= 0.92888165 # another run => 0.9279927 so it's not deterministic, just like for example tSNE
# Graduate Sales Executive - Media and Advertising - BMS
# 1  - id= 1258 , distance= 0.9147828
# Graduate Sales Trainee- Branded Business Gifts - BMS
# 2  - id= 941 , distance= 0.9144758
# Graduate Sales Trainee - Branded Business Gifts - BMS
# 3  - id= 2535 , distance= 0.8929088
# Graduate Sales Executive - Cyber Security - BMS
# 4  - id= 2275 , distance= 0.8842137
# Graduate Sales Consultant - Cyber Security Software - BMS
