from configparser import ConfigParser

config = ConfigParser()

config_path = "config/config.ini"
config.read(config_path)

raw_data_dir = config["RAW_DATA"]["directory"]
raw_data_filename = config["RAW_DATA"]["filename"]

processed_data_dir = config["PROCESSED_DATA"]["directory"]
processed_data_filename = config["PROCESSED_DATA"]["filename"]

X_col_name = config["COLUMN_NAMES"]["X"]
y_col_name = config["COLUMN_NAMES"]["y"]

test_size_split = float(config["TRAIN_TEST_SPLIT"]["test_size"])
random_state_split = int(config["TRAIN_TEST_SPLIT"]["random_state"])

tfidf_max_features = int(config["TFIDF"]["max_features"])
log_reg_max_iter = int(config["LOGISTIC_REGRESSION"]["max_iter"])

vectorizers_dir = config["VECTORIZERS"]["directory"]
vectorizer_filename = config["VECTORIZERS"]["filename"]
vectorizer_file_ext = config["VECTORIZERS"]["file_extension"]

models_dir = config["MODELS"]["directory"]
model_filename = config["MODELS"]["filename"]
model_file_ext = config["MODELS"]["file_extension"]

metrics_dir = config["METRICS"]["directory"]
imp_metrics_filename = config["METRICS"]["imp_metrics_filename"]
c_matrix_filename = config["METRICS"]["c_matrix_filename"]
c_report_filename = config["METRICS"]["c_report_filename"]