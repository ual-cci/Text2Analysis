from nlp_functions import *
import numpy as np
import os
import re
import shutil

from pprint import pprint
import pandas as pd

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import matplotlib.colors as mcolors

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class NLPTools(object):
    """
    Main class for NLP related functions ...
    """

    def __init__(self, settings):
        self.settings = settings
        self.verbose = 1 # 1 normal, 2 print everything including examples

        self.colors_topics = [color for name, color in mcolors.TABLEAU_COLORS.items()] + [color for name, color in
                                                                                     mcolors.XKCD_COLORS.items()]


    def restart_workspace(self):
        # restart / file cleanup!:
        files = ["save.zip", "templates/plots/wordclouds_00.png", "templates/plots/wordclouds_01.png", "templates/plots/wordclouds_02.png", "templates/plots/wordclouds_03.png"]
        for file in files:
            if os.path.exists(file):
                os.remove(file)
                print("deleted", file)

    def prepare_workspace(self, plot_dir):
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        if not os.path.exists("data"):
            os.makedirs("data")

    def load_stopwords(self):
        # NLTK Stop words
        stop_words = stopwords.words('english')
        stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

        # CUSTOM STOP WORDS?
        stoplist = set(', . : / ( ) [ ] - _ ; * & ? ! – a b c d e t i p an us on 000 if it ll to as are then '
                       'they our the you we s in if a m I x re to this at ref do and'.split())
        stop_words.extend(stoplist)
        stoplist = set('experience job ensure able working join key apply strong recruitment work team successful '
                       'paid contact email role skills company day good high time required want right success'
                       'ideal needs feel send yes no arisen arise title true work role application process contract '
                       'interested touch'.split())
        stop_words.extend(stoplist)

        return stop_words

    def preprocessing(self, data, stop_words):

        if self.verbose > 1:
            print("loaded text")
            print(len(data[:1][0]))
            pprint(data[:1])

        # Remove Emails
        data = [re.sub('\S*@\S*\s?', '', doc) for doc in data]
        # Remove new line characters
        data = [re.sub('\s+', ' ', doc) for doc in data]
        # Remove distracting single quotes
        data = [re.sub("\'", "", doc) for doc in data]

        if self.verbose > 1:
            print("removed special chars text")
            print(len(data[:1][0]))
            pprint(data[:1][0])

        data_words = list(sentences_to_words(data))

        if self.verbose > 1:
            print(data_words[:1])

        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        # Remove Stop Words
        data_words_nostops = remove_stopwords(data_words, stop_words)
        # Form Bigrams
        data_words_bigrams = make_bigrams(bigram_mod, data_words_nostops)

        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        # python3 -m spacy download en
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

        # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = lemmatization(data_words_bigrams, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        if self.verbose > 1:
            print(data_lemmatized[:1])

        return data, data_lemmatized

    def analyze(self, text):
        #return "Yes, that's very nice text!"

        self.restart_workspace()

        data = [text]
        stop_words = self.load_stopwords()
        data, data_lemmatized = self.preprocessing(data, stop_words)

        ### Data prepared, save
        texts = data_lemmatized
        data = [text]
        titles = ["foobar"]

        DATASET = ""
        plot_dir = "templates/plots" + DATASET + "/"
        self.prepare_workspace(plot_dir)

        ### Settings:
        METAOPT_num_of_topics = False
        NAME_METAOPT_plot = plot_dir + "LDA_best_number_of_topics_"

        LOAD_lda_model = False

        #LDA_number_of_topics = 13  # Wait for metaopt!
        LDA_number_of_topics = 4  # Wait for metaopt!
        CALC_coherence = True

        VIZ_wordclouds = True
        VIZ_html_interactive = True  # SLOW and last
        NAME_wordclouds = plot_dir + "wordclouds_"  # +i+.png
        NAME_html_interactive = plot_dir + "LDA_Visualization.html"  # "vis.html"

        DEBUG_print_docs = False
        DEBUG_print_topics = False

        # GENSIM analysis
        id2word = corpora.Dictionary(texts)
        corpus = [id2word.doc2bow(text) for text in texts]
        """
        if DEBUG_print_docs:
            print("document in vector format:", corpus[:1])
            print("readable form:", [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])
            print()

        if METAOPT_num_of_topics:
            # Can take a long time to run.
            topics_start = 5
            topics_end = 15  # non exclusive
            topics_step = 1
            plot_name = NAME_METAOPT_plot + str(topics_start) + "TO" + str(topics_end) + "BY" + str(
                topics_step) + ".png"
            LDA_best_number_of_topics(id2word, corpus, texts, topics_start, topics_end, topics_step,
                                      mallet_lda=LDA_tryMallet, plot_name=plot_name)
            print("Analyze the results of the meta-optimalization ^^^")
            assert False
        """

        # Build LDA model
        print("Building/Loading LDA model (takes time)")
        if LOAD_lda_model:
            lda_model = gensim.models.LdaModel.load('data/model_LDAEXAMPLE.lda')
        else:
            lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=LDA_number_of_topics,
                                                        random_state=100, update_every=1, chunksize=100, passes=10,
                                                        alpha='auto', per_word_topics=True)
            lda_model.save('data/model_LDAEXAMPLE.lda')

        if DEBUG_print_topics:
            print("Topics:")
            pprint(lda_model.print_topics(num_topics=LDA_number_of_topics, num_words=5))

        doc_lda = lda_model[corpus]

        # Evaluation - Perplexity, Coherence Score
        print('\nPerplexity: ',
              lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
        if CALC_coherence:
            coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            print('\nCoherence Score: ', coherence_lda)

        ######################################################################################################

        # Dominant topics per document
        """
        if SAVE_doc2topic or SAVE_topic2docs or SAVE_topic2num or VIZ_hist:
            df_topic_sents_keywords, dominant_topics_as_arr = format_topics_sentences(ldamodel=lda_model, corpus=corpus,
                                                                                      texts=data)
            df_dominant_topic = df_topic_sents_keywords.reset_index()
            df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
        """

        # Wordclouds of Top N words in each topic
        if VIZ_wordclouds:
            print("Saving wordclouds into >", NAME_wordclouds, "...")

            # topics = lda_model.show_topics(formatted=False)
            topics = lda_model.show_topics(num_topics=LDA_number_of_topics, formatted=False)
            for i_t, topic in enumerate(topics):
                topic_i = topic[0]
                topic_words = topic[1]
                print("topic", topic_i, "===", topic_words)

                fig = plt.figure()
                topic_words = dict(topic_words)
                cloud = WordCloud(stopwords=stop_words, background_color='white', width=2500, height=1800,
                                  max_words=10, colormap='tab10',
                                  color_func=lambda *args, **kwargs: self.colors_topics[topic_i],
                                  prefer_horizontal=1.0)

                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                plt.gca().set_title('Topic ' + str(topic_i), fontdict=dict(size=16))
                plt.axis('off')
                plt.margins(x=0, y=0)
                plt.tight_layout()
                plt.savefig(NAME_wordclouds + str(topic_i).zfill(2) + ".png")
                plt.close()

        if VIZ_html_interactive:
            # Takes forever!
            print("creating visualization...")
            vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
            print("saving it...")
            pyLDAvis.save_html(vis, NAME_html_interactive)
            print("done")


        output_filename = "save"
        dir_name = "templates/plots"
        shutil.make_archive(output_filename, 'zip', dir_name)

        return NAME_html_interactive









    # Specific visualizations:

    def viz_tsne(self, titles, plot_dir, lda_model, corpus, LDA_number_of_topics):
        NAME_tsne = plot_dir + "tsne.html"

        print("Saving TSNE visualization into >", NAME_tsne)

        from sklearn.manifold import TSNE
        from bokeh.plotting import figure, output_file, show, save, ColumnDataSource

        # Get topic weights
        doc_features = []
        doc_titles = []
        doc_dominanttopics = []
        for i, row_list in enumerate(lda_model[corpus]):
            # What we have in the encoding:
            # row_list[0] = Document topics: [(0, 0.87507219282484316), (1, 0.12492780717515681)]
            # row_list[1] = Word topics: [(0, [0, 1]), (3, [0, 1]), (4, [0, 1]), (7, [0, 1])]
            # row_list[2] = Phi values: [(0, [(0, 0.9783234200583657), (1, 0.021676579941634355)]), (3, [(0, 0.93272653621872503), (1, 0.067273463781275009)]), (4, [(0, 0.98919912227661466), (1, 0.010800877723385368)]), (7, [(0, 0.97541896333079636), (1, 0.024581036669203641)])]

            # row_list[0] has the weights to topics
            # This means that one document was encoded into the LDA_number_of_topics topics we chose

            tmp = np.zeros(LDA_number_of_topics)
            max_w = -1
            max_w_idx = -1
            for j, w in row_list[0]:
                tmp[j] = w
                if max_w < w:
                    max_w_idx = j
                    max_w = w
            doc_features.append(tmp)

            doc_dominanttopics.append(max_w_idx)
            doc_titles.append(titles[i])

        arr = pd.DataFrame(doc_features).fillna(0).values
        # tSNE Dimension Reduction
        tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
        tsne_lda = tsne_model.fit_transform(arr)

        TOOLTIPS = [
            ("index", "$index"),
            ("(x,y)", "($x, $y)"),
            ("desc", "@desc"),
        ]

        mycolors = np.array(self.colors_topics)

        from bokeh.models import HoverTool, WheelZoomTool, PanTool, BoxZoomTool, ResetTool, SaveTool
        hover = HoverTool(tooltips=TOOLTIPS)
        tools = [hover, WheelZoomTool(), PanTool(), BoxZoomTool(), ResetTool(), SaveTool()]

        # plot = figure(title="t-SNE Clustering of {} LDA Topics".format(LDA_number_of_topics),
        #              tooltips=TOOLTIPS, plot_width=900, plot_height=700)
        plot = figure(title="t-SNE Clustering of {} LDA Topics".format(LDA_number_of_topics),
                      tools=tools, plot_width=900, plot_height=700)
        source = ColumnDataSource(data=dict(
            x=tsne_lda[:, 0],
            y=tsne_lda[:, 1],
            desc=doc_titles,
            color=mycolors[doc_dominanttopics],
        ))

        plot.scatter(x='x', y='y', source=source, color='color')
        output_file(NAME_tsne)
        save(plot)
        # show(plot)

    def viz_hist(self, plot_dir, dominant_topics_as_arr, LDA_number_of_topics, lda_model):
        NAME_hist = plot_dir + "hist_topics.png"

        print("Saving histogram into >", NAME_hist)
        hist_tmp = {}
        for val in dominant_topics_as_arr:
            if val not in hist_tmp:
                hist_tmp[val] = 0
            else:
                hist_tmp[val] += 1
        print("My own hist:", hist_tmp)
        xs = list(range(LDA_number_of_topics))
        ys = [0] * LDA_number_of_topics
        for topic_num, val in hist_tmp.items():
            wp = lda_model.show_topic(topic_num)
            topic_keywords = ", ".join([word for word, prop in wp])  # removed probabilities
            print("topic", topic_num, ":", val, "time(s) , keywords = ", topic_keywords)
            ys[topic_num] = val

        plt.bar(xs, ys)
        plt.xticks(np.arange(LDA_number_of_topics), np.arange(LDA_number_of_topics))
        plt.savefig(NAME_hist)
        plt.close()

    def save_csv_documents(self, plot_dir, df_dominant_topic, df_topic_sents_keywords):
        SAVE_doc2topic = False  # don't really need
        NAME_doc2topic = plot_dir + "doc2topic.csv"
        SAVE_topic2docs = True
        NAME_topic2docs = plot_dir + "topic2docs.csv"
        SAVE_topic2num = True
        NAME_topic2num = plot_dir + "topic2num.csv"

        if SAVE_doc2topic:
            df_dominant_topic.to_csv(NAME_doc2topic, index=True)

        # Topic 2 representative documents
        if SAVE_topic2docs:
            sent_topics_sorteddf_mallet = pd.DataFrame()
            sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
            for i, grp in sent_topics_outdf_grpd:
                sent_topics_sorteddf_mallet = pd.concat(
                    [sent_topics_sorteddf_mallet, grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)],
                    axis=0)
            sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
            sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

            sent_topics_sorteddf_mallet.to_csv(NAME_topic2docs, index=True)

        # Topic 2 number of docs
        if SAVE_topic2num:
            topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()
            topic_contribution = round(topic_counts / topic_counts.sum(), 4)
            topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]
            df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)
            df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

            df_dominant_topics.to_csv(NAME_topic2num, index=True)
