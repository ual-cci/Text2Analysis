from flask import Flask, session, redirect, url_for, request, render_template
from analysis_handler import AnalysisHandler

app = Flask(__name__)
# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = b'F\xc4\xea]\xab$E$X\xd6H?\xbc\xd9oszblunk'

@app.route('/')
def index():
    return redirect(url_for('enter'))

@app.route('/enter', methods=['GET', 'POST'])
def enter():
    return render_template('enter.html')

@app.route('/process', methods=['GET', 'POST'])
def process(user_input=None):
    if request.method == 'POST':
        user_input = request.form['user_input']

        # TODO: Process data here ...
        settings = None
        analysis_handler = AnalysisHandler(settings)

        analysis_handler.load_text(user_input)
        analysis_reply = analysis_handler.call_analysis()

        preview = user_input[0:20]+"..."
    else:
        print("Wasn't able to load the text data!")

        preview = None
        analysis_reply = None

    # Show results ...
    return render_template('process.html', user_input=preview, analysis_reply=analysis_reply)


@app.route('/forget')
def forget():
    # remove the text data from the session if it's there
    session.pop('user_input', None)
    return redirect(url_for('enter'))

#if __name__ == '__main__':
#    with app.app_context():
#        app.run(debug=True)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True, port=33507)
