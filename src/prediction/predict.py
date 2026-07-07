import joblib
from preprocessing.traditional import TraditionalPreprocessor
from utils.logger import logger

class Predictor:

    def __init__(self, vectorizerPath, modelPath, text_preprocessor):
        self.text_preprocessor = text_preprocessor
        self.vectorizer = joblib.load(vectorizerPath)
        self.model = joblib.load(modelPath)

    def predictForVectorizedData(self, X_test_tfidf):

        logger.info("Started prediction for some vectorized data...")
        predictions = self.model.predict(X_test_tfidf)
        logger.info("The prediction is complete.")

        return predictions

    def predictForReview(self, review):

        logger.info("Started prediction for a review...")

        processedReview = self.text_preprocessor.cleanReview(review)

        vectorizedReview = self.vectorizer.transform([processedReview])
        predictedSentiment = self.model.predict(vectorizedReview)
        logger.info("The prediction is complete.")

        return predictedSentiment
