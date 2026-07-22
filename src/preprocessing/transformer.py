class TransformerReviewPreprocessor:

    def __init__(self, text_preprocessor):
        self.text_preprocessor = text_preprocessor

    def cleanReview(self, text: str) -> str:

        text = self.text_preprocessor.remove_html(text)
        text = self.text_preprocessor.remove_urls(text)
        text = self.text_preprocessor.convert_emojis(text)
        text = self.text_preprocessor.remove_extra_spaces(text)

        return text
       