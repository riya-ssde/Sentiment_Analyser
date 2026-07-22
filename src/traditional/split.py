from sklearn.model_selection import train_test_split
from utils.logger import logger

class DataSplitter:

    def __init__(self, df, X_col_name, y_col_name, test_data_size = 0.2, random_state_shuffling = 42):
        self.df = df
        self.X_col_name = X_col_name
        self.y_col_name = y_col_name
        self.test_data_size = test_data_size
        self.random_state_shuffling = random_state_shuffling
        self.split_data = None

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

        self.split_data = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }

        logger.info(f"Size of training data: {len(self.split_data["X_train"])}")
        logger.info(f"Size of testing data: {len(self.split_data["X_test"])}")

        return self.split_data

    def getXTrain(self):
        return self.split_data["X_train"]

    def getYTrain(self):
        return self.split_data["y_train"]

    def getXTest(self):
        return self.split_data["X_test"]

    def getYTest(self):
        return self.split_data["y_test"]