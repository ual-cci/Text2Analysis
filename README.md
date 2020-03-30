# Text2Analysis

## Install 

Prerequisite libraries:

- `pip install gensim pandas spacy pyLDAvis nltk bokeh wordcloud`
- run python script with `import nltk` and `nltk.download('stopwords')`
- `python3 -m spacy download en`

### Fast setup:

```
git clone https://github.com/previtus/Text2Analysis.git
#(optionally make an environment)
pip3 install -r requirements.txt
python3 -m nltk.downloader stopwords
python3 -m spacy download en
python3 app.py
#(starts it on http://localhost:5000/)
```

## Demo

- Run app.py file: `python3 app.py`

Now it's also working online on Heroku (with some setup pains...), see: https://previtus-nlp-api-heroku.herokuapp.com/

[![Demo on Heroku](https://raw.githubusercontent.com/previtus/Text2Analysis/master/illustration_screen.png)](https://previtus-nlp-api-heroku.herokuapp.com/ "Demo on Heroku")
