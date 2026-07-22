# Sentiment Analyser
It's a python-based sentiment analysis application that processes Amazon reviews using NLP and machine learning techniques to classify customer sentiment into three categories - positive, negative, and neutral - and generate insights.

## Project Overview
Sentiment Analyser is designed to:
- Ingest and preprocess Amazon reviews
- Train one or more models (e.g., Logistic Regression, Transformer Model, etc)
- Predict the right sentiment for reviews
- Evaluate the model

## Folder Structure
```
Sentiment_Analyser/
|
|__ artifacts/
|   |__ metrics/
|   |__ models/
|   |__ vectorizers/
|   
|__ config/
|
|__ data/
|   |__ processed/
|   |__ raw/
|
|__ deployment/
|
|__ logs/
|
|__ src/
|   |__ evaluation/
|   |__ prediction/
|   |__ preprocessing/
|   |__ training/
|   |__ utils/
|   |__ __init__.py
|   |__ main.py
|
|__ tests/
|
|__ pyproject.toml
|
|__ README.md
|
|__ requirements.txt
```
## Datasets
- Amazon Reviews (Kaggle - https://www.kaggle.com/datasets/dongrelaxman/amazon-reviews-dataset?resource=download)

## Features
### Baseline
- TF-IDF Transformer
- Logistic Regression Classifier
### Advanced Model
- BERT/ RoBERT

## Installation
```
pip install -r requirements.txt
```
