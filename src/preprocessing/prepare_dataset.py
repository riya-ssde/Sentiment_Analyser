import pandas as pd
from utils.logger import logger

class DataPreprocessor:

    def __init__(self, text_preprocessor, file_handler, data_directory, data_filename):
        self.text_preprocessor = text_preprocessor
        self.file_handler = file_handler
        self.data_directory = data_directory
        self.data_filename = data_filename

    def loadFileIntoDF(self):
        self.df = self.file_handler.load_csv(self.data_directory, self.data_filename)
        logger.info("Loaded raw data to a data frame.")

    def dropUnnecessaryColumns(self):
        self.df = self.df.drop(["Profile Link", "Country", "Date of Experience"], axis=1)
        logger.info("Dropped Columns: 'Profile Link', 'Country', and 'Data of Experience'")
    
    def updateDateFormat(self):
        self.df['Review Date'] = pd.to_datetime(self.df['Review Date']).dt.date
        self.df['Review Date'] = pd.to_datetime(self.df['Review Date'], format = '%Y-%m-%d')
        logger.info("Updated 'Review Date' column format.")
    
    def dropRowsMissingReviewText(self):
        self.df = self.df.dropna(subset=["Review Text"])
        logger.info("Dropped rows with no review text.")

    def dropRowsMissingRating(self):
        self.df = self.df.dropna(subset=["Rating"])
        logger.info("Dropped rows with missing rating data.")
    
    def extractRating(self):
        self.df['Rating'] = (
            self.df['Rating']
            .str.extract(r'(\d)')
            .astype(int)
        )
        logger.info("Extracted rating and updated 'Rating' column.")

    def labelSentiment(self, rating):
        if (rating <= 2):
            return "Negative"
        elif (rating == 3):
            return "Neutral"
        else:
            return "Positive"

    def addSentimentColumn(self):
        self.df["Sentiment"] = self.df["Rating"].apply(self.labelSentiment)
        logger.info("Added 'Sentiment' column.")

    def cleanReviewText(self):
        
        self.df["Review Text"] = self.df["Review Text"].astype(str).apply(
        self.text_preprocessor.cleanReview)

        logger.info("Cleaned reviews using the traditioanl preprocessor.")

    def saveDFToFile(self, directory, filename):
        self.file_handler.save_df_to_csv(self.df, directory, filename)
        logger.info("Saved the processed data to a csv file.")

    def processDataFrame(self, processed_data_dir = "", processed_data_filename = ""):

        logger.info("Started processing the raw data...")

        self.loadFileIntoDF()

        self.dropUnnecessaryColumns()
        self.updateDateFormat()
        self.dropRowsMissingReviewText()
        self.dropRowsMissingRating()
        self.extractRating()
        self.cleanReviewText()
        self.addSentimentColumn()
        
        if (self.df["Review Text"].isnull().sum() > 0):
            self.dropRowsMissingReviewText()

        if (len(processed_data_dir) > 0 and len(processed_data_filename) > 0):
            self.saveDFToFile(processed_data_dir, processed_data_filename)

        logger.info("The raw data has been processed.")
        
        return self.df