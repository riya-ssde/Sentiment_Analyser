def LabelSentiment(rating):
    if (rating <= 2):
        return "Negative"
    elif (rating == 3):
        return "Neutral"
    else:
        return "Positive"