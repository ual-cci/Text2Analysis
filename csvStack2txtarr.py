# Load data from CSV files:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gensim
import json
from tqdm import tqdm

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# This scraped dataset specific functions:
csv_files = [
    "../QuestionData/graphicdesign_all_q.csv",
    #"../QuestionData/graphicdesign_all_a_50k.csv",
    #"../QuestionData/gamedev_all_q.csv", # << even larger file

]

# Structure:
# ﻿0 Id	1 PostTypeId	2 AcceptedAnswerId	3 ParentId	4 CreationDate	5 DeletionDate	6 Score	7 ViewCount	8 Body
#  9 OwnerUserId	10 OwnerDisplayName	11 LastEditorUserId	12 LastEditorDisplayName	13 LastEditDate	14 LastActivityDate
#  15 Title	16 Tags	17 AnswerCount	18 CommentCount	19 FavoriteCount	20 ClosedDate	21 CommunityOwnedDate
# Use: 8=Body and 15=Title, body can contain HTML code ...
#      plus maybe 16=Tags? (as "<adobe-photoshop><perspective>")

total_lines = 0
total_ignored_short = 0
full_text_dataset = []
full_text_titles = []
full_text_ids = [] # question ids to look up answers
lengths_questions = []

THR_min_desc_length_allowed = 200 #400
THR_exclude_long_errs = 26000
THR_supposed_to_have_values = 22

for csv_file in csv_files:
    df = pd.read_csv(csv_file, delimiter=',')

    for line in df.values:
        body = line[8]
        title = line[15]
        id = line[0]
        #print(id)
        #print(title)
        #print(body)
        #print(line)


        if body is not np.nan and title is not np.nan:
            # those broken by splitting (only 3 records)
            if len(line) > THR_supposed_to_have_values and line[-1] is not np.nan:
                print(len(line))
                #print(csv_file)
                #print(line)
                continue

            if len(body) < THR_min_desc_length_allowed:
                #print(csv_file)
                #print(line)
                total_ignored_short += 1
                continue

            if len(body) > THR_exclude_long_errs:
                total_ignored_short += 1
                continue

            # print(title, "|||", len(description), description)
            full_text_dataset.append(body)
            full_text_titles.append(title)
            full_text_ids.append(id)
            lengths_questions.append(len(body))
            total_lines += 1

lengths_questions = np.asarray(lengths_questions)

print("--=============================--")
print("Read",total_lines,"lines in total.")
print("Skipped",total_ignored_short,"too short lines (using thr", THR_min_desc_length_allowed,")")
print("Statistics of lenghts (min, max, avg):", np.min(lengths_questions), np.max(lengths_questions), np.average(lengths_questions))

#plt.plot(lengths_descriptions)
#plt.ylabel('lenghts of descriptions in scraped data')
#plt.show()

# Check the longest entry ? Seems alright.
longest_i = np.argmax(lengths_questions)
#print(lengths_questions[longest_i])
#print(full_text_dataset[longest_i])

# Remove HTML from bodies

from bs4 import BeautifulSoup
from gensim.parsing.preprocessing import remove_stopwords

i = 0
for question in tqdm(full_text_dataset):
    soup = BeautifulSoup(question, 'html.parser')
    text = soup.find_all(text=True)

    output = ''
    blacklist = [
        '[document]',
        'noscript',
        'header',
        'html',
        'meta',
        'head',
        'input',
        'script',
        # there may be more elements you don't want, such as "style", etc.
    ]

    for t in text:
        if t.parent.name not in blacklist:
            output += '{} '.format(t)

    # Replaces:
    #print(output)

    output = output.replace('\n', ' ')
    output = output.replace('!', ' ! ')
    output = output.replace('?', ' ? ')
    output = output.replace('.', ' . ')
    output = output.replace(',', ' , ')
    output = output.replace(':', ' : ')
    output = output.replace(';', ' ; ')
    output = output.replace('(', ' ( ')
    output = output.replace(')', ' ) ')
    output = output.replace('/', ' / ')
    output = output.replace('*', ' * ')
    output = output.replace(']', ' ] ')
    output = output.replace('[', ' [ ')
    output = output.replace('&', ' & ')
    output = output.replace('_', ' _ ')

    output = remove_stopwords(output)
    output = " ".join(list(gensim.utils.tokenize(output)))

    full_text_dataset[i] = output
    i += 1

    #print(question)
    #print(output)



documents = full_text_dataset
titles = full_text_titles

# on purpose the same files - just run it in sequence probably ...
plus = ""
plus = "Stack"
#np.save("data/documents"+plus+".npy", documents)
#np.save("data/titles"+plus+".npy", titles)
np.savez_compressed("data/documents"+plus+".npz", a=documents)
np.savez_compressed("data/titles"+plus+".npz", a=titles)
