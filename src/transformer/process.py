import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocessing.clean_review import TextPreprocessor
from preprocessing.transformer import TransformerReviewPreprocessor
from preprocessing.prepare_dataset import DataPreprocessor
from evaluation.evaluate import Evaluator
from utils.file_csv import FileHandler
from utils.logger import logger

class TransformerReviewProcessor():

    def __init__(self, pretrained_model_details):

        self.processed_df = None
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_details)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_details)
        
    def calculateSentiment(self, review):

        tokens = self.tokenizer.encode(review, return_tensors='pt')
        result = self.model(tokens)

        rating = int(torch.argmax(result.logits))+1

        sentiment = self.extractSentiment(rating)

        return sentiment

    def extractSentiment(self, rating):
            if (rating <= 2):
                return "Negative"
            elif (rating == 3):
                return "Neutral"
            else:
                return "Positive"

    def createProcessedDF(self, raw_data_dir, raw_data_filename, processed_data_dir, processed_data_filename):
    
        file_handler = FileHandler()
        logger.info("Created 'FileHandler' object.")
        text_preprocessor = TextPreprocessor()
        logger.info("Created 'TextPreprocessor' object.")
        self.review_preprocessor = TransformerReviewPreprocessor(text_preprocessor)
        logger.info("Created 'TransformerReviewPreprocessor' object.")
        dataPreprocessor = DataPreprocessor(self.review_preprocessor, file_handler, raw_data_dir, raw_data_filename)
        logger.info("Created 'DataPreprocessor' object.")
        
        self.processed_df = dataPreprocessor.processDataFrame(processed_data_dir, processed_data_filename)

    def predictSentimentForAmazonReviews(self, raw_data_dir, raw_data_filename, processed_data_dir, processed_data_filename, X_col_name):

        self.createProcessedDF(raw_data_dir, raw_data_filename, processed_data_dir, processed_data_filename, X_col_name)

        predictions = self.processed_df[X_col_name].astype(str).apply(
                lambda x: self.calculateSentiment(x[:512]))

        return predictions

    def evaluatePredictionsForAmazonReviews(self, predictions, metrics_dir, imp_metrics_filename, c_matrix_filename, c_report_filename):

        evaluator = Evaluator(self.test_data["y"], predictions)
        logger.info("Created 'Evaluator' object.")

        metrics = evaluator.getImportantMetrics()
        confusion_mat = evaluator.getConfusionMatrix().tolist()
        classi_report = evaluator.getClassificationReport()
        
        evaluator.saveMetrics(metrics, metrics_dir, imp_metrics_filename)
        evaluator.saveMetrics(confusion_mat, metrics_dir, c_matrix_filename)
        evaluator.saveMetrics(classi_report, metrics_dir, c_report_filename)