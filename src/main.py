from utils.logger import logger
from traditional.process import TraditionalReviewProcessor
from transformer.process import TransformerReviewProcessor

def main():

    try:

        # Clean the log file.
        with open("logs/app.log", "w"):
            pass

        logger.info("Welcome to the Sentiment Analyser.")

        # Baseline

        traditional_processor = TraditionalReviewProcessor()
        traditional_processor.trainModel()

        predictions = traditional_processor.predictTestDataSentiment()
        traditional_processor.evaluateTestDataPredictions(predictions)

        review = "neither liked it nor hated it"
        sentiment = traditional_processor.predictReviewSentiment(review)

        logger.info(f"Review: {review}")
        logger.info(f"Predicted Sentiment: {sentiment}")

        # Transformer

        transformer_processor = TransformerReviewProcessor()

        review = "neither liked it nor hated it"
        sentiment = transformer_processor.calculateSentiment(review)

        logger.info(f"Review: {review}")
        logger.info(f"Predicted Sentiment: {sentiment}")

        df_rows_range = {
            "first_row": 0,
            "last_row": 10
        }

        predicted_sentiments = transformer_processor.predictSentimentForAmazonReviews(df_rows_range)
        transformer_processor.evaluatePredictionsForAmazonReviews(predicted_sentiments, df_rows_range)

        logger.info("We have reached the end of the project!")

    except Exception as e:
        logger.exception(e)

if __name__ == "__main__":
    main()