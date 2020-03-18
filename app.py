from flask import Flask, session, redirect, url_for, request, render_template, send_file
from analysis_handler import AnalysisHandler
import os, random
from timeit import default_timer as timer

app = Flask(__name__)

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

@app.route('/process', methods=['GET', 'POST'])
def process(user_input=None):
    if request.method == 'POST':
        user_input = request.form['user_input']
        number_of_topics = int(request.form['number_of_topics'])

        # TODO: Process data here ...
        settings = None
        analysis_handler = AnalysisHandler(settings)

        start_analysis = timer()

        analysis_handler.load_text(user_input)
        analysis_reply, n_topics, n_chars, n_documents = analysis_handler.call_analysis_raw_text(number_of_topics)

        end_analysis = timer()
        n_seconds_analysis = (end_analysis - start_analysis)
        print("This analysis took " + str(n_seconds_analysis) + "s (" + str(n_seconds_analysis / 60.0) + "min)")

        n_seconds_analysis = float(int(n_seconds_analysis * 100.0))/100.0

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


@app.route('/last', methods=['GET', 'POST'])
def last():
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

