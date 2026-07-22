from preprocessing.clean_review import TextPreprocessor

class TraditionalReviewPreprocessor:

    def __init__(self, text_preprocessor):
        self.text_preprocessor = text_preprocessor

    def cleanReview(self, text: str) -> str:
        
        text = self.text_preprocessor.lowercase(text)
        text = self.text_preprocessor.remove_html(text)
        text = self.text_preprocessor.remove_urls(text)
        text = self.text_preprocessor.convert_emojis(text)
        text = self.text_preprocessor.remove_special_characters(text)
        text = self.text_preprocessor.remove_extra_spaces(text)
        
        words = self.text_preprocessor.tokenize(text)
        words = self.text_preprocessor.remove_stopwords(words)
        words = self.text_preprocessor.lemmatize(words)

        return " ".join(words)