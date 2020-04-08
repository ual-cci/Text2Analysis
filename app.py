from flask import Flask, redirect, url_for, request, render_template, send_file, Response
from analysis_handler import AnalysisHandler
import os, random
from os import listdir
from os.path import isfile, join
from timeit import default_timer as timer
#from multiprocessing import Pool
#from multiprocessing.dummy import Pool
import threading, time
import time
import pandas as pd
from werkzeug.utils import secure_filename

from stopwords_hardcoded_collection import adjectives_500

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True # nice, this one works to update /pyldaviz when the analysis changes it

# uploading texts:
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# GLOBALS:
LAST_ANALYSIS_N_TOPICS = 0
async_ready_flag = False
async_answer = None
tsne_on = False

@app.route('/')
def index():
    return redirect(url_for('enter'))

@app.route('/enter', methods=['GET', 'POST'])
def enter():
    sample_text = "Lorem Ipsum"
    try:
        file = open("static/examples/sample_input.txt", "r")
        sample_text = file.read()
        file.close()
    except:
        print("Failed to load sample_input.txt, loading default text string.")
    return render_template('enter.html', sample_text = sample_text)

def processing_function_extracted(input_type, user_input, number_of_topics):
    # input_type ... 0=RawText, 1=ListOfTexts, 2=ListOfTextsWithCaptions
    print("-processing_function_extracted called!")

    global async_answer, async_ready_flag, tsne_on
    async_answer = None
    async_ready_flag = False
    tsne_on = False

    start_analysis = timer()

    # Start processing
    folder_name = unique_random_name_gen()
    settings = None
    analysis_handler = AnalysisHandler(settings, folder_name)

    if input_type == 0:
        analysis_handler.load_raw_text(user_input)

    else:
        texts = None
        captions = None
        if input_type == 1:
            texts = user_input
        elif input_type == 2:
            texts, captions = user_input
            tsne_on = True
        analysis_handler.load_list_of_text(texts, captions)

    # todo: split raw vs list
    analysis_reply, n_topics, n_chars, n_documents = analysis_handler.call_analysis_raw_text(number_of_topics)

    analysis_handler.cleanup() # cleanup so that we regain the memory on server ...
    del analysis_handler

    end_analysis = timer()
    n_seconds_analysis = (end_analysis - start_analysis)
    print("This analysis took " + str(n_seconds_analysis) + "s (" + str(n_seconds_analysis / 60.0) + "min)")
    n_seconds_analysis = float(int(n_seconds_analysis * 100.0)) / 100.0

    """
    # test long processing times ...
    # refer to https://librenepal.com/article/flask-and-heroku-timeout/
    t_to_wait = 140
    #t_to_wait = 0
    t_rem = int(max(t_to_wait - n_seconds_analysis, 0))
    print("waiting for extra", t_rem, "sec!")
    import time
    time.sleep(t_rem)
    """

    global LAST_ANALYSIS_N_TOPICS
    LAST_ANALYSIS_N_TOPICS = n_topics

    async_answer = analysis_reply, n_topics, n_chars, n_documents, n_seconds_analysis, folder_name
    async_ready_flag = True
    #return analysis_reply, n_topics, n_chars, n_documents, n_seconds_analysis

@app.route('/processRenderTemplateVersion', methods=['GET', 'POST'])
def process(user_input=None):
    if request.method == 'POST':
        user_input = request.form['user_input']
        number_of_topics = int(request.form['number_of_topics'])
        input_type = 0  # 0=RawText, 1=ListOfTexts, 2=ListOfTextsWithCaptions

        processing_function_extracted(input_type, user_input, number_of_topics)
        global async_answer
        analysis_reply, n_topics, n_chars, n_documents, n_seconds_analysis, folder_name = async_answer

        preview = user_input[0:20]+"..."
    else:
        print("Wasn't able to load the text data!")

        preview = None
        analysis_reply = None
        n_topics = 0
        n_chars = 0
        n_documents = 0
        n_seconds_analysis = 0

    # Show results ...
    return render_template('process.html', user_input=preview, analysis_reply=analysis_reply, n_topics=n_topics,
                           n_chars = n_chars, n_documents = n_documents, n_seconds_analysis = n_seconds_analysis,
                           rand_i = random.randint(1,9999))


@app.route('/process', methods=['GET', 'POST'])
def check():
    input_type = 0 # 0=RawText, 1=ListOfTexts, 2=ListOfTextsWithCaptions
    user_input = None
    file_contents = None
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file:
                txt_len = 0
                allowed, ext = allowed_file(file.filename)
                if allowed:
                    if ext == 'txt':
                        file_contents = file.read()
                        #file_contents = str(file_contents) # keeps the b'' tags etc
                        file_contents = file_contents.decode('utf-8')
                        txt_len = len(file_contents)

                    elif ext == 'csv':
                        # load as csv

                        contains_captions = False
                        texts = []
                        captions = []
                        df = pd.read_csv(file, delimiter=',')
                        for line in df.values:
                            text = line[0]
                            caption = ""

                            cols = len(line)
                            if cols > 1:
                                contains_captions = True
                                caption = line[1]

                            texts.append(text)
                            captions.append(caption)
                            txt_len += len(text)

                        if contains_captions:
                            input_type = 2
                            file_contents = texts, captions
                        else:
                            input_type = 1
                            file_contents = texts

                file.close()
                print("loaded from file, number of chars:", txt_len)

        user_input = request.form['user_input']
        number_of_topics = int(request.form['number_of_topics'])

        if file_contents is not None:
            user_input = file_contents


    def generate(input_type, user_input, number_of_topics):
        yield "Started ... please wait ... <br><br>"  # notice that we are yielding something as soon as possible

        global async_ready_flag
        async_ready_flag = False

        start = timer()

        #pool = Pool(processes=1)
        #result = pool.apply_async(processing_function_extracted, [user_input, number_of_topics], callback)  # Evaluate "processing_function_extracted" asynchronously calling callback when finished.
        threading.Thread(target=processing_function_extracted,args=[input_type, user_input, number_of_topics]).start()

        while not async_ready_flag:
            current = timer()
            so_far_waited = (current - start)
            print("So far took " + str(so_far_waited) + "s (" + str(so_far_waited / 60.0) + "min)")
            so_far_waited = float(int(so_far_waited * 100.0)) / 100.0

            sec_wait = 5
            yield "Still running (so far "+str(so_far_waited)+" seconds) ... please wait ... (will check again in "+str(sec_wait)+" seconds)<br>"
            time.sleep(sec_wait)

        analysis_reply, n_topics, n_chars, n_documents, n_seconds_analysis, folder_name = async_answer
        answer = "<br><br><h2>Finished! Please look at the <a href='saved/"+folder_name+"'>results</a></h2>" # html like returned to Response
        yield answer

    return Response(generate(input_type, user_input, number_of_topics), mimetype='text/html')

    #return render_template('process.html', user_input=preview, analysis_reply=analysis_reply, n_topics=n_topics,
    #                       n_chars = n_chars, n_documents = n_documents, n_seconds_analysis = n_seconds_analysis,
    #                       rand_i = random.randint(1,9999))


@app.route('/saved/<analysis_name>', methods=['GET', 'POST'])
def last(analysis_name):
    analysis_name = secure_filename(analysis_name)
    folder_wordclouds = "static/"+analysis_name+"/"
    wordclouds_names = [f for f in listdir(folder_wordclouds) if isfile(join(folder_wordclouds, f)) and ".png" in f]
    print(wordclouds_names)
    LAST_ANALYSIS_N_TOPICS = len(wordclouds_names)

    folder_analysis = "templates/plots/"+analysis_name+"/"
    analysis_names = [f for f in listdir(folder_analysis) if isfile(join(folder_analysis, f))]
    print(analysis_names)

    tsne_on = ("tsne.html" in analysis_names)
    print("LAST_ANALYSIS_N_TOPICS=",LAST_ANALYSIS_N_TOPICS)
    print("tsne_on=",tsne_on)

    return render_template('last.html', n_topics=LAST_ANALYSIS_N_TOPICS, tsne_on=tsne_on, rand_i=random.randint(1, 9999), analysis_name=analysis_name)

@app.route('/pyldaviz/<analysis_name>', methods=['GET', 'POST'])
def pyldaviz(analysis_name):
    analysis_name = secure_filename(analysis_name)
    folder_analysis = "plots/"+analysis_name+"/LDA_Visualization.html"
    return render_template(folder_analysis)

@app.route('/tsne/<analysis_name>', methods=['GET', 'POST'])
def tsne(analysis_name):
    analysis_name = secure_filename(analysis_name)
    folder_analysis = "plots/"+analysis_name+"/tsne.html"
    return render_template(folder_analysis)

@app.route('/example_csv', methods=['GET', 'POST'])
def example_csv():
    return send_file("static/examples/example_list_data_bbc-science-20-3-2020.csv", as_attachment=True)

@app.route('/download/<analysis_name>', methods=['GET', 'POST'])
def download(analysis_name):
    analysis_name = secure_filename(analysis_name)
    folder_analysis = "templates/plots/"+analysis_name+"/"
    path = folder_analysis+"analysis.zip"
    return send_file(path, as_attachment=True)

@app.route('/debug_list', methods=['GET', 'POST'])
def debug_list():
    folder_analysis = "templates/plots/"
    analysis_names = [f for f in listdir(folder_analysis) if not isfile(join(folder_analysis, f))]
    print("folders:", analysis_names)
    for folder in analysis_names:
        print("/saved/"+folder)

    return Response("Foo!", mimetype='text/html')


def allowed_file(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    return ('.' in filename and ext in ALLOWED_EXTENSIONS), ext

def unique_random_name_gen():
    """
    from nltk.corpus import wordnet as wn
    adjectives = []
    for synset in list(wn.all_synsets(wn.ADJ)):
        for adj in synset.lemma_names():
            adjectives.append(str(adj))
    adjectives = random.sample(adjectives,500)
    print(len(adjectives), adjectives)
    """
    adjectives = adjectives_500
    random_name = random.sample(adjectives, 1)[0] + "_" + str(random.randint(0,9999))
    return random_name

#if __name__ == '__main__':
#    with app.app_context():
#        app.run(debug=True)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    #app.run(debug=True, port=port)
    app.run(debug=False, port=port)

