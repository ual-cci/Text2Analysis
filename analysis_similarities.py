from gensim import corpora
import gensim
import numpy as np

documents = np.load("data/documents.npy")
titles = np.load("data/titles.npy")
documents_represented = np.load("data/documents_represented.npy")

dictionary = corpora.Dictionary.load('data/dict.dict')
corpus = corpora.MmCorpus('data/corpus.mm')
corpus_tfidf = corpora.MmCorpus('data/corpus_tfidf.mm')
corpus_lsi = corpora.MmCorpus('data/corpus_lsi.mm')
corpus_lda = corpora.MmCorpus('data/corpus_lda.mm')
#lsi = gensim.models.LsiModel.load('data/model.lsi')
#lda = gensim.models.LsiModel.load('data/model.lda')
tfidf = gensim.models.LsiModel.load('data/model.tfidf')
index = gensim.similarities.MatrixSimilarity.load('data/index.index')
index_lda = gensim.similarities.MatrixSimilarity.load('data/index_lda.index')

# Similarity
check_doc_id = 234
#check_doc_id = 102
vec_lsi = corpus_lsi[check_doc_id]
vec_lda = corpus_lda[check_doc_id]

# LSI
print("LSI ----")

sims = index[vec_lsi]  # perform a similarity query against the corpus
sims = sorted(enumerate(sims), key=lambda item: -item[1])
#print(sims)  # print (document_number, document_similarity) 2-tuples

print("Queried document:")
print(titles[check_doc_id])
print(documents[check_doc_id])
print(documents_represented[check_doc_id])
print()
print("Closest documents:")
for i in range(5):
    closest_doc_id = sims[1+i][0]
    print(i, " - id=",closest_doc_id, ", distance=", sims[1+i][1])
    print(titles[closest_doc_id])
    print(documents[closest_doc_id])
    print(documents_represented[closest_doc_id])

print()
# LDA
print("LDA ----")

sims_lda = index_lda[vec_lda]  # perform a similarity query against the corpus
sims_lda = sorted(enumerate(sims_lda), key=lambda item: -item[1])
#print(sims_lda)  # print (document_number, document_similarity) 2-tuples

print("Queried document:")
print(titles[check_doc_id])
print(documents[check_doc_id])
print(documents_represented[check_doc_id])
print()
print("Closest documents:")
for i in range(5):
    closest_doc_id = sims_lda[1+i][0]
    print(i, " - id=",closest_doc_id, ", distance=", sims_lda[1+i][1])
    print(titles[closest_doc_id])
    print(documents[closest_doc_id])
    print(documents_represented[closest_doc_id])


##########################################################################################
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


##########################################################################################
# Queried document:
# Embedded Software Engineer - Embedded C - Creative Personnel London

# Closest documents:
# 0  - id= 344 , distance= 1.0
# Embedded Software Engineer - Embedded C - Creative Personnel London
# 1  - id= 2229 , distance= 1.0  # <<< Detected a duality in the dataset
# Embedded Software Engineer - Embedded C - Creative Personnel London
# 2  - id= 1991 , distance= 0.869358
# Embedded Software Engineer - Spectrum IT
# 3  - id= 2360 , distance= 0.86588764
# Embedded Software Engineer - Bedfordshire - £30,000 - £45,000 - Premier Group
# 4  - id= 5253 , distance= 0.8601873
# Embedded Software Engineer - Sunbury-On-Thames
