import nlp_tools

class AnalysisHandler(object):
    """
    Main class for the analysis ...
    """

    def __init__(self, settings):
        self.settings = settings
        self.input_text = None

        self.nlp_tools = nlp_tools.NLPTools(self.settings)

    def load_text(self, text):

        self.input_text = text
        print("Loaded text with", len(text), "characters!")

    def call_analysis(self):
        return self.nlp_tools.analyze(self.input_text)


"""
settings = None
testAnalysis = AnalysisHandler(settings)

text = 'lorem ipsum'
testAnalysis.load_text(text)
reply = testAnalysis.call_analysis()

print(reply)
"""