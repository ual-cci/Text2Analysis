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
    #"../JobAdsData/Source2-Reed/creative2_pages_clean_1359.csv",
    #"../JobAdsData/Source2-Reed/it2_pages_clean_1556.csv",
    #"../JobAdsData/Source3-Indeed/creative2_pages_clean_4164.csv",
    #"../JobAdsData/Source3-Indeed/it2_pages_clean_986.csv"
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
titles = full_text_titles

np.save("data/documents.npy", documents)
np.save("data/titles.npy", titles)
