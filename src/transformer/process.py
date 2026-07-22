import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocessing.clean_review import TextPreprocessor
from preprocessing.transformer import TransformerReviewPreprocessor
from preprocessing.prepare_dataset import DataPreprocessor
from evaluation.evaluate import Evaluator
from utils.file_csv import FileHandler
from utils.sentiment import LabelSentiment
from utils.logger import logger

class TransformerReviewProcessor():

    def __init__(self, raw_data_dir, raw_data_filename, processed_data_dir, clean_data_filename, pretrained_model_details):

        self.processed_df = None

        self.raw_data_dir = raw_data_dir
        self.raw_data_filename = raw_data_filename
        self.processed_data_dir = processed_data_dir
        self.clean_data_filename = clean_data_filename

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_details)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_details)

    def getProcessedDF(self):
        
        file_handler = FileHandler()
        logger.info("Created 'FileHandler' object.")
        text_preprocessor = TextPreprocessor()
        logger.info("Created 'TextPreprocessor' object.")
        self.review_preprocessor = TransformerReviewPreprocessor(text_preprocessor)
        logger.info("Created 'TransformerReviewPreprocessor' object.")
        dataPreprocessor = DataPreprocessor(self.review_preprocessor, file_handler)
        logger.info("Created 'DataPreprocessor' object.")
        
        self.processed_df = dataPreprocessor.processDataFrame(self.raw_data_dir, self.raw_data_filename, self.processed_data_dir, self.clean_data_filename)

    def calculateSentiment(self, review):

        tokens = self.tokenizer.encode(review, truncation=True, max_length=512, return_tensors='pt')
        result = self.model(tokens)

        rating = int(torch.argmax(result.logits))+1
        sentiment = LabelSentiment(rating)

        return sentiment

    def predictSentimentForAmazonReviews(self, df_rows_range):

        self.getProcessedDF()

        predicted_sentiments = []
        reviews = self.processed_df["Review Text"].iloc[df_rows_range["first_row"]:df_rows_range["last_row"]]

        for review in reviews:
            sentiment = self.calculateSentiment(review)
            predicted_sentiments.append(sentiment)

        return predicted_sentiments

    def evaluatePredictionsForAmazonReviews(self, predictions, df_rows_range, metrics_dir, imp_metrics_filename, c_matrix_filename, c_report_filename):

        actual_sentiments = self.processed_df["Sentiment"].iloc[df_rows_range["first_row"]:df_rows_range["last_row"]]

        evaluator = Evaluator(actual_sentiments, predictions)
        logger.info("Created 'Evaluator' object.")

        metrics = evaluator.getImportantMetrics()
        confusion_mat = evaluator.getConfusionMatrix().tolist()
        classi_report = evaluator.getClassificationReport()
        
        evaluator.saveMetrics(metrics, metrics_dir, imp_metrics_filename)
        evaluator.saveMetrics(confusion_mat, metrics_dir, c_matrix_filename)
        evaluator.saveMetrics(classi_report, metrics_dir, c_report_filename)