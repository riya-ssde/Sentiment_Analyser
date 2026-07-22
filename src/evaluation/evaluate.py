import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from utils.logger import logger

class Evaluator:

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def evaluateAccuracyScore(self):
        accuracy = accuracy_score(self.y_true, self.y_pred)
        logger.info("Accuracy: %.4f", accuracy)
        return accuracy

    def evaluatePrecisionScore(self, avg = "weighted"):
        precision = precision_score(self.y_true, self.y_pred, average = avg)
        logger.info("Precision: %.4f", precision)
        return precision

    def evaluateRecallScore(self, avg = "weighted"):
        recall = recall_score(self.y_true, self.y_pred, average = avg)
        logger.info("Recall Score: %.4f", recall)
        return recall

    def evaluateF1Score(self, avg = "weighted"):
        f1 = f1_score(self.y_true, self.y_pred, average = avg)
        logger.info("F1 Score: %.4f", f1)
        return f1

    def getConfusionMatrix(self):
        confusion_mat = confusion_matrix(self.y_true, self.y_pred)
        logger.info(f"Confusion Matrix:\n{confusion_mat}")
        return confusion_mat
    
    def getClassificationReport(self):
        classi_report = classification_report(self.y_true, self.y_pred)
        logger.info(f"Classification Report:\n{classi_report}")
        return classi_report

    def getImportantMetrics(self):
        accuracy = self.evaluateAccuracyScore()
        precision = self.evaluatePrecisionScore()
        recall = self.evaluateRecallScore()
        f1 = self.evaluateF1Score()
        c_matrix_list = self.getConfusionMatrix().tolist()

        metrics = {
            "Accuracy": float(accuracy),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1-Score": float(f1),
            "Confusion Matrix": c_matrix_list
        }

        logger.info("Created a dictionary of important metrices.")

        return metrics

    def saveMetrics(self, metrics, directory, filename):
        Path(directory).mkdir(parents=True, exist_ok=True)
        filepath = f"{directory}/{filename}.json"
        
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Saved metrices to a file. Path: {filepath}.")