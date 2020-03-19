from flask import Flask, session, redirect, url_for, request, render_template, send_file, Response
from analysis_handler import AnalysisHandler
import os, random
from timeit import default_timer as timer
#from multiprocessing import Pool
#from multiprocessing.dummy import Pool
import threading, time
import time

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True # nice, this one works to update /pyldaviz when the analysis changes it

# GLOBALS:
LAST_ANALYSIS_N_TOPICS = 0
async_ready_flag = False
async_answer = None

@app.route('/')
def index():
    return redirect(url_for('enter'))

@app.route('/enter', methods=['GET', 'POST'])
def enter():
    sample_text = "Lorem Ipsum"
    try:
        file = open("sample_input.txt", "r")
        sample_text = file.read()
        file.close()
    except:
        print("Failed to load sample_input.txt, loading default text string.")
    return render_template('enter.html', sample_text = sample_text)

def processing_function_extracted(user_input, number_of_topics):
    print("-processing_function_extracted called!")

    global async_answer, async_ready_flag
    async_answer = None
    async_ready_flag = False

    start_analysis = timer()

    # Start processing
    settings = None
    analysis_handler = AnalysisHandler(settings)

    analysis_handler.load_text(user_input)
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

    async_answer = analysis_reply, n_topics, n_chars, n_documents, n_seconds_analysis
    async_ready_flag = True
    #return analysis_reply, n_topics, n_chars, n_documents, n_seconds_analysis

@app.route('/processRenderTemplateVersion', methods=['GET', 'POST'])
def process(user_input=None):
    if request.method == 'POST':
        user_input = request.form['user_input']
        number_of_topics = int(request.form['number_of_topics'])

        processing_function_extracted(user_input, number_of_topics)
        global async_answer
        analysis_reply, n_topics, n_chars, n_documents, n_seconds_analysis = async_answer

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
    if request.method == 'POST':
        user_input = request.form['user_input']
        number_of_topics = int(request.form['number_of_topics'])

    def generate(user_input, number_of_topics):
        yield "Started ... please wait ... <br><br>"  # notice that we are yielding something as soon as possible

        global async_ready_flag
        async_ready_flag = False

        start = timer()

        #pool = Pool(processes=1)
        #result = pool.apply_async(processing_function_extracted, [user_input, number_of_topics], callback)  # Evaluate "processing_function_extracted" asynchronously calling callback when finished.
        threading.Thread(target=processing_function_extracted,args=[user_input, number_of_topics]).start()

        while not async_ready_flag:
            current = timer()
            so_far_waited = (current - start)
            print("So far took " + str(so_far_waited) + "s (" + str(so_far_waited / 60.0) + "min)")
            so_far_waited = float(int(so_far_waited * 100.0)) / 100.0

            sec_wait = 5
            yield "Still running (so far "+str(so_far_waited)+" seconds) ... please wait ... (will check again in "+str(sec_wait)+" seconds)<br>"
            time.sleep(sec_wait)

        #analysis_reply, n_topics, n_chars, n_documents, n_seconds_analysis = async_answer
        answer = "<br><br><h2>Finished! Please look at the <a href='last'>results</a></h2>" # html like returned to Response
        yield answer

    return Response(generate(user_input, number_of_topics), mimetype='text/html')

    #return render_template('process.html', user_input=preview, analysis_reply=analysis_reply, n_topics=n_topics,
    #                       n_chars = n_chars, n_documents = n_documents, n_seconds_analysis = n_seconds_analysis,
    #                       rand_i = random.randint(1,9999))



@app.route('/last', methods=['GET', 'POST'])
def last():
    global LAST_ANALYSIS_N_TOPICS

    return render_template('last.html', n_topics=LAST_ANALYSIS_N_TOPICS, rand_i = random.randint(1,9999))
    #return render_template('plots/LDA_Visualization.html')

@app.route('/pyldaviz', methods=['GET', 'POST'])
def pyldaviz():
    return render_template('plots/LDA_Visualization.html')

@app.route('/download')
def download():
    path = "save.zip"
    return send_file(path, as_attachment=True)

@app.route('/forget')
def forget():
    # remove the text data from the session if it's there
    session.pop('user_input', None)
    return redirect(url_for('enter'))

#if __name__ == '__main__':
#    with app.app_context():
#        app.run(debug=True)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    #app.run(debug=True, port=port)
    app.run(debug=False, port=port)

