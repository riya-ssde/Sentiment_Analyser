from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils.file_csv import FileHandler
from preprocessing.clean_review import TextPreprocessor
from preprocessing.traditional import TraditionalPreprocessor
from preprocessing.prepare_dataset import DataPreprocessor
from training.train import Trainer
from training.splitter import DataSplitter
from prediction.predict import Predictor
from evaluation.evaluate import Evaluator
from utils.logger import logger
from utils.configuration import config

def main():

    try:

        # Clean the log file.
        with open("logs/app.log", "w"):
            pass

        logger.info("Welcome to the Sentiment Analyser.")

        raw_data_dir = config["RAW_DATA"]["directory"]
        raw_data_filename = config["RAW_DATA"]["filename"]
        
        processed_data_dir = config["PROCESSED_DATA"]["directory"]
        processed_data_filename = config["PROCESSED_DATA"]["filename"]

        X_col_name = config["COLUMN_NAMES"]["X"]
        y_col_name = config["COLUMN_NAMES"]["y"]

        test_size_split = float(config["TRAIN_TEST_SPLIT"]["test_size"])
        random_state_split = int(config["TRAIN_TEST_SPLIT"]["random_state"])

        tfidf_max_features = int(config["TFIDF"]["max_features"])
        log_reg_max_iter = int(config["LOGISTIC_REGRESSION"]["max_iter"])

        vectorizers_dir = config["VECTORIZERS"]["directory"]
        vectorizer_filename = config["VECTORIZERS"]["filename"]
        vectorizer_file_ext = config["VECTORIZERS"]["file_extension"]

        models_dir = config["MODELS"]["directory"]
        model_filename = config["MODELS"]["filename"]
        model_file_ext = config["MODELS"]["file_extension"]

        metrics_dir = config["METRICS"]["directory"]
        imp_metrics_filename = config["METRICS"]["imp_metrics_filename"]
        c_matrix_filename = config["METRICS"]["c_matrix_filename"]
        c_report_filename = config["METRICS"]["c_report_filename"]

        text_preprocessor = TextPreprocessor()
        logger.info("Created 'TextPreprocessor' object.")
        traditional_preprocessor = TraditionalPreprocessor(text_preprocessor)
        logger.info("Created 'TraditionalPreprocessor' object.")
        file_handler = FileHandler()
        logger.info("Created 'FileHandler' object.")
        dataPreprocessor = DataPreprocessor(traditional_preprocessor, file_handler, raw_data_dir, raw_data_filename)
        logger.info("Created 'DataPreprocessor' object.")
        
        processed_df = dataPreprocessor.processDataFrame(processed_data_dir, processed_data_filename)

        X = processed_df[X_col_name]
        y = processed_df[y_col_name]

        if (X.isnull().sum() > 0 or y.isnull().sum() > 0):
            processed_df = processed_df.dropna(subset=[X_col_name, y_col_name])
        
        data_splitter = DataSplitter(processed_df, X_col_name, y_col_name, test_size_split, random_state_split)
        logger.info("Created 'DataSplitter' object.")
        tfidf_vectorizer = TfidfVectorizer(max_features = tfidf_max_features)
        logger.info("Created 'TfidfVectorizer' object.")
        model_log_reg = LogisticRegression(max_iter = log_reg_max_iter)
        logger.info("Created 'LogisticRegression' object.")

        trainer = Trainer(data_splitter, tfidf_vectorizer, model_log_reg)
        logger.info("Created 'Trainer' object.")
        trainer.train(vectorizers_dir, models_dir, vectorizer_filename, model_filename)

        X_test_vec = trainer.getXTestVectorized()
        y_test = trainer.getYTest()

        vectorizerPath = f"{vectorizers_dir}/{vectorizer_filename}{vectorizer_file_ext}"
        modelPath = f"{models_dir}/{model_filename}{model_file_ext}"

        predictor = Predictor(vectorizerPath, modelPath, traditional_preprocessor)
        logger.info("Created 'Predictor' object.")

        review1 = "neither liked it nor hated it"
        predictedSentiment = predictor.predictForReview(review1)
        print(predictedSentiment)

        predictions = predictor.predictForVectorizedData(X_test_vec)

        evaluator = Evaluator(y_test, predictions)
        logger.info("Created 'Evaluator' object.")

        metrics = evaluator.getImportantMetrics()
        confusion_mat = evaluator.getConfusionMatrix().tolist()
        classi_report = evaluator.getClassificationReport()
        
        evaluator.saveMetrics(metrics, metrics_dir, imp_metrics_filename)
        evaluator.saveMetrics(confusion_mat, metrics_dir, c_matrix_filename)
        evaluator.saveMetrics(classi_report, metrics_dir, c_report_filename)

        logger.info("We have reached the end of the project!")

    except Exception as e:
        logger.exception(e)

if __name__ == "__main__":
    main()