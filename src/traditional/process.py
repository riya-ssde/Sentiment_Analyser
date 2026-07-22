from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils.file_csv import FileHandler
from preprocessing.clean_review import TextPreprocessor
from preprocessing.traditional import TraditionalReviewPreprocessor
from preprocessing.prepare_dataset import DataPreprocessor
from traditional.train import Trainer
from traditional.splitter import DataSplitter
from traditional.predict import Predictor
from traditional.evaluate import Evaluator
from utils.logger import logger
from utils.configuration import *

class Process_Traditional():

    def __init__(self):

        text_preprocessor = TextPreprocessor()
        logger.info("Created 'TextPreprocessor' object.")
        self.review_preprocessor = TraditionalReviewPreprocessor(text_preprocessor)
        logger.info("Created 'TraditionalPreprocessor' object.")

        self.processed_df = None
        self.test_data = None
        self.vectorizerPath = f"{vectorizers_dir}/{vectorizer_filename}{vectorizer_file_ext}"
        self.modelPath = f"{models_dir}/{model_filename}{model_file_ext}"

    def createProcessedDF(self):

        file_handler = FileHandler()
        logger.info("Created 'FileHandler' object.")
        dataPreprocessor = DataPreprocessor(self.review_preprocessor, file_handler, raw_data_dir, raw_data_filename)
        logger.info("Created 'DataPreprocessor' object.")
        
        self.processed_df = dataPreprocessor.processDataFrame(processed_data_dir, processed_data_filename)

    def fitModel(self):

        X = self.processed_df[X_col_name]
        y = self.processed_df[y_col_name]

        if (X.isnull().sum() > 0 or y.isnull().sum() > 0):
            self.processed_df = self.processed_df.dropna(subset=[X_col_name, y_col_name])
        
        data_splitter = DataSplitter(self.processed_df, X_col_name, y_col_name, test_size_split, random_state_split)
        logger.info("Created 'DataSplitter' object.")
        tfidf_vectorizer = TfidfVectorizer(max_features = tfidf_max_features)
        logger.info("Created 'TfidfVectorizer' object.")
        model_log_reg = LogisticRegression(max_iter = log_reg_max_iter)
        logger.info("Created 'LogisticRegression' object.")

        trainer = Trainer(data_splitter, tfidf_vectorizer, model_log_reg)
        logger.info("Created 'Trainer' object.")
        trainer.train(vectorizers_dir, models_dir, vectorizer_filename, model_filename)

        self.populateTestData(trainer)

    def populateTestData(self, trainer):

        X_test_vec = trainer.getXTestVectorized()
        y_test = trainer.getYTest()

        self.test_data = {
            "X_vec": X_test_vec,
            "y": y_test
        }

    def trainModel(self):
        self.createProcessedDF()
        self.fitModel()

    def predictTestDataSentiment(self):

        predictor = Predictor(self.vectorizerPath, self.modelPath)
        logger.info("Created 'Predictor' object.")

        predictions = predictor.predictForVectorizedData(self.test_data["X_vec"])

        return predictions

    def evaluateTestDataPredictions(self, predictions):

        evaluator = Evaluator(self.test_data["y"], predictions)
        logger.info("Created 'Evaluator' object.")

        metrics = evaluator.getImportantMetrics()
        confusion_mat = evaluator.getConfusionMatrix().tolist()
        classi_report = evaluator.getClassificationReport()
        
        evaluator.saveMetrics(metrics, metrics_dir, imp_metrics_filename)
        evaluator.saveMetrics(confusion_mat, metrics_dir, c_matrix_filename)
        evaluator.saveMetrics(classi_report, metrics_dir, c_report_filename)

    def predictReviewSentiment(self, review):

        predictor = Predictor(self.vectorizerPath, self.modelPath, self.review_preprocessor)
        logger.info("Created 'Predictor' object.")

        sentiment = predictor.predictForReview(review)

        return sentiment