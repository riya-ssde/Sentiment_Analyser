from utils.logger import logger
from utils.configuration import *
from traditional.process import Process_Traditional

def main():

    try:

        # Clean the log file.
        with open("logs/app.log", "w"):
            pass

        logger.info("Welcome to the Sentiment Analyser.")

        # Baseline
        process_traditional = Process_Traditional()
        process_traditional.trainModel()

        predictions = process_traditional.predictTestDataSentiment()
        process_traditional.evaluateTestDataPredictions(predictions)

        review = "neither liked it nor hated it"
        logger.info(f"Review: {review}")
        predicted_sentiment = process_traditional.predictReviewSentiment(review)
        logger.info(f"Predicted Sentiment:{predicted_sentiment}")

        logger.info("We have reached the end of the project!")

    except Exception as e:
        logger.exception(e)

if __name__ == "__main__":
    main()