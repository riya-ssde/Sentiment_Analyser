from pathlib import Path
import joblib
from utils.logger import logger

class Trainer:

    def __init__(self, data_splitter, vectorizer, model):
        self.data_splitter = data_splitter
        self.vectorizer = vectorizer
        self.model = model
        
    def vectorizeTrainingData(self, X_train):
        logger.info("Started vectorizing the training data...")
        return self.vectorizer.fit_transform(X_train)

    def vectorizeTestData(self, X_test):
        logger.info("Started vectorizing the test data...")
        return self.vectorizer.transform(X_test)

    def fitModel(self, X_train_vec, y_train):
        self.model.fit(X_train_vec, y_train)
        logger.info("Fitted the model.")

    def saveVectorizer(self, directory, filename, file_extension):
        Path(directory).mkdir(parents=True, exist_ok=True)
        file_path = f"{directory}/{filename}{file_extension}"
        joblib.dump(self.vectorizer, file_path)
        logger.info(f"Saved the vectorizer. Path: {file_path}.")


    def saveModel(self, directory, filename, file_extension):
        Path(directory).mkdir(parents=True, exist_ok=True)
        file_path = f"{directory}/{filename}{file_extension}"
        joblib.dump(self.model, file_path)
        logger.info(f"Saved the model. Path: {file_path}.")

    def train(self, vectorizers_dir, models_dir, vectorizer_filename, model_filename, extension = ".pkl"):

        logger.info("Started training...")

        self.data_splitter.splitDataSet()
        X_train = self.data_splitter.getXTrain()
        y_train = self.data_splitter.getYTrain()

        X_train_vec = self.vectorizeTrainingData(X_train)
        self.fitModel(X_train_vec, y_train)

        self.saveVectorizer(vectorizers_dir, vectorizer_filename, extension)
        self.saveModel(models_dir, model_filename, extension)

        logger.info("The training is complete.")

    def getXTrain(self):
        return self.data_splitter.getXTrain()

    def getYTrain(self):
        return self.data_splitter.getYTrain()
    
    def getXTest(self):
        return self.data_splitter.getXTest()

    def getYTest(self):
        return self.data_splitter.getYTest()

    def getXTestVectorized(self):
        X_test = self.data_splitter.getXTest()
        return self.vectorizeTestData(X_test)