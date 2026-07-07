from preprocessing.clean_review import TextPreprocessor

class TransformerPreprocessor:

    def __init__(self, text_preprocessor):
        self.text_preprocessor = text_preprocessor

    def cleanReview(self, text: str) -> list:

        text = self.text_preprocessor.remove_html(text)
        text = self.text_preprocessor.remove_urls(text)

        words = self.text_preprocessor.tokenize(text)

        return words