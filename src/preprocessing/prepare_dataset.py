import pandas as pd
from pathlib import Path
from utils.sentiment import LabelSentiment 
from utils.logger import logger

class DataPreprocessor:

    def __init__(self, text_preprocessor, file_handler):
        self.df = None
        self.text_preprocessor = text_preprocessor
        self.file_handler = file_handler

    def loadFileIntoDF(self, data_directory, data_filename):
        self.df = self.file_handler.load_csv(data_directory, data_filename)
        logger.info("Loaded CSV file data to a data frame.")

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

    def addSentimentColumn(self):
        self.df["Sentiment"] = self.df["Rating"].apply(LabelSentiment)
        logger.info("Added 'Sentiment' column.")

    def cleanReviewText(self):
        
        self.df["Review Text"] = self.df["Review Text"].astype(str).apply(
        self.text_preprocessor.cleanReview)

        logger.info("Cleaned reviews.")

    def saveDFToFile(self, directory, filename):
        self.file_handler.save_df_to_csv(self.df, directory, filename)
        logger.info("Saved the processed data to a csv file.")

    def processDataFrame(self, raw_data_directory, raw_data_filename, processed_data_dir, processed_data_filename):

        file_path = Path(f"{processed_data_dir}/{processed_data_filename}.csv")

        if file_path.exists():
            logger.info("Processed file already exists.")
            self.loadFileIntoDF(processed_data_dir, processed_data_filename)
        else:
        
            logger.info("Started processing the raw data...")

            self.loadFileIntoDF(raw_data_directory, raw_data_filename)
            self.dropUnnecessaryColumns()
            self.updateDateFormat()
            self.dropRowsMissingReviewText()
            self.dropRowsMissingRating()
            self.extractRating()
            self.cleanReviewText()
            self.addSentimentColumn()
            
            if (self.df["Review Text"].isnull().sum() > 0):
                self.dropRowsMissingReviewText()

            self.saveDFToFile(processed_data_dir, processed_data_filename)

            logger.info("The raw data has been processed.")

        return self.df