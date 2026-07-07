import re
import emoji
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:

    def __init__(self):
        pass

    def lowercase(self, text: str) -> str:
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        return re.sub(r"[^\w\s]", "", text)

    def remove_html(self, text: str) -> str:
        return BeautifulSoup(text, "html.parser").get_text()

    def remove_urls(self, text: str) -> str:
        return re.sub(r"http\S+|www\S+", "", text)

    def convert_emojis(self, text: str) -> str:
        return emoji.demojize(text)

    def remove_special_characters(self, text: str) -> str:
        return re.sub(r"[^a-zA-Z\s]", " ", text)

    def remove_extra_spaces(self, text: str) -> str:
        return " ".join(text.split())

    def tokenize(self, text: str) -> list:
        return word_tokenize(text)

    def remove_stopwords(self, words: list) -> list:
        stop_words = set(stopwords.words("english"))
        newlist = []

        for word in words:
            if word not in stop_words or word in ["not", "no", "never"]:
                newlist.append(word)

        return newlist

    def lemmatize(self, words: list) -> list:
        lemmatizer = WordNetLemmatizer()

        for word in words:
            lemmatizer.lemmatize(word)

        return words