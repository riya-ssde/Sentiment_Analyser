import joblib
from pathlib import Path
from utils.logger import logger

class Trainer:

    def __init__(self, vectorizer, model, vectorizers_dir, models_dir, vectorizer_filename, model_filename, extension = ".pkl"):
        self.vectorizer = vectorizer
        self.model = model
        self.vectorizers_dir = vectorizers_dir
        self.models_dir = models_dir
        self.vectorizer_file_path = Path(f"{vectorizers_dir}/{vectorizer_filename}{extension}")
        self.model_file_path = Path(f"{models_dir}/{model_filename}{extension}")
        
    def vectorizeTrainingData(self, X_train):
        logger.info("Started vectorizing the training data...")
        return self.vectorizer.fit_transform(X_train)

    def vectorizeTestData(self, X_test):
        logger.info("Started vectorizing the test data...")
        return self.vectorizer.transform(X_test)

    def fitModel(self, X_train_vec, y_train):
        self.model.fit(X_train_vec, y_train)
        logger.info("Fitted the model.")

    def saveVectorizer(self):
        Path(self.vectorizers_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, self.vectorizer_file_path)
        logger.info(f"Saved the vectorizer. Path: {self.vectorizer_file_path}.")


    def saveModel(self):
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_file_path)
        logger.info(f"Saved the model. Path: {self.model_file_path}.")

    def train(self, X_train, y_train):

        if self.vectorizer_file_path.exists() and self.model_file_path.exists():
            logger.info("The model has already been trained.")
        else:

            logger.info("Started training...")

            X_train_vec = self.vectorizeTrainingData(X_train)
            self.fitModel(X_train_vec, y_train)

            self.saveVectorizer()
            self.saveModel()

            logger.info("The training is complete.")

    def getXTestVectorized(self, X_test):        
        return self.vectorizeTestData(X_test)