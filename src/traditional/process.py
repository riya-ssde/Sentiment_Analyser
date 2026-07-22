import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils.file_csv import FileHandler
from preprocessing.clean_review import TextPreprocessor
from preprocessing.traditional import TraditionalReviewPreprocessor
from preprocessing.prepare_dataset import DataPreprocessor
from traditional.split import DataSplitter
from traditional.train import Trainer
from traditional.predict import Predictor
from evaluation.evaluate import Evaluator
from utils.configuration import *
from utils.logger import logger

class TraditionalReviewProcessor():

    def __init__(self):

        self.processed_df = None
        self.test_data = None
        self.data_trainer = None

        text_preprocessor = TextPreprocessor()
        logger.info("Created 'TextPreprocessor' object.")
        self.review_preprocessor = TraditionalReviewPreprocessor(text_preprocessor)
        logger.info("Created 'TraditionalReviewPreprocessor' object.")

        self.vectorizerPath = f"{vectorizers_dir}/{vectorizer_filename}{vectorizer_file_ext}"
        self.modelPath = f"{models_dir}/{model_filename}{model_file_ext}"

    def getProcessedDF(self):

        file_handler = FileHandler()
        logger.info("Created 'FileHandler' object.")
        dataPreprocessor = DataPreprocessor(self.review_preprocessor, file_handler)
        logger.info("Created 'DataPreprocessor' object.")
        
        self.processed_df = dataPreprocessor.processDataFrame(raw_data_dir, raw_data_filename, processed_data_dir, clean_data_filename)

    def getSplitData(self):

        X = self.processed_df[X_col_name]
        y = self.processed_df[y_col_name]

        if (X.isnull().sum() > 0 or y.isnull().sum() > 0):
            self.processed_df = self.processed_df.dropna(subset=[X_col_name, y_col_name])

        data_splitter = DataSplitter(self.processed_df, X_col_name, y_col_name, test_size_split, random_state_split)
        logger.info("Created 'DataSplitter' object.")
        split_data = data_splitter.splitDataSet()

        return split_data

    def createTrainerObject(self):

        tfidf_vectorizer = TfidfVectorizer(max_features = tfidf_max_features)
        logger.info("Created 'TfidfVectorizer' object.")
        model_log_reg = LogisticRegression(max_iter = log_reg_max_iter)
        logger.info("Created 'LogisticRegression' object.")

        self.data_trainer = Trainer(tfidf_vectorizer, model_log_reg, vectorizers_dir, models_dir, vectorizer_filename, model_filename)
        logger.info("Created 'Trainer' object.")

    def trainModel(self):

        if Path(self.vectorizerPath).exists() and Path(self.modelPath).exists():
            logger.info("The model has already been trained.")
        else:
            self.getProcessedDF()
            split_data = self.getSplitData()
            self.createTrainerObject()
            self.data_trainer.train(split_data["X_train"], split_data["y_train"])

    def populateTestData(self):

        if (self.processed_df is None):
            self.getProcessedDF()

        split_data = self.getSplitData()

        if self.data_trainer is None:
            vectorizer = joblib.load(self.vectorizerPath)
            X_test_vectorized = vectorizer.transform(split_data["X_test"])
        else:
            X_test_vectorized = self.data_trainer.getXTestVectorized(split_data["X_test"])

        self.test_data = {
            "X_vectorized": X_test_vectorized,
            "y": split_data["y_test"]
        }

    def predictTestDataSentiment(self):
        
        self.populateTestData()

        predictor = Predictor(self.vectorizerPath, self.modelPath)
        logger.info("Created 'Predictor' object.")

        predictions = predictor.predictForVectorizedData(self.test_data["X_vectorized"])

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