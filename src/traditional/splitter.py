from sklearn.model_selection import train_test_split
from utils.logger import logger

class DataSplitter:

    def __init__(self, df, X_col_name, y_col_name, test_data_size = 0.2, random_state_shuffling = 42):
        self.df = df
        self.X_col_name = X_col_name
        self.y_col_name = y_col_name
        self.test_data_size = test_data_size
        self.random_state_shuffling = random_state_shuffling

    def splitDataSet(self):

        logger.info("Started splitting the data set...")

        X = self.df[self.X_col_name]
        y = self.df[self.y_col_name]

        logger.info("Dataset Size: ", len(X))

        X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = self.test_data_size,
        random_state = self.random_state_shuffling,
        stratify=y
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        logger.info(f"Size of training data: {len(self.X_train)}")
        logger.info(f"Size of testing data: {len(self.X_test)}")

    def getXTrain(self):
        return self.X_train

    def getYTrain(self):
        return self.y_train

    def getXTest(self):
        return self.X_test

    def getYTest(self):
        return self.y_test