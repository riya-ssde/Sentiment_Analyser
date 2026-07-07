from pathlib import Path
import pandas as pd
from utils.logger import logger

class FileHandler:

    def load_csv(self, directory, filename):
        file_path = f"{directory}/{filename}.csv"
        df = pd.read_csv(file_path, engine="python", on_bad_lines="skip")

        logger.info("File (%s) loaded into a data frame.", file_path)
        logger.info("The data frame has %d rows.", len(df))

        return df

    def save_df_to_csv(self, df, directory, filename):
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        file_path = f"{directory}/{filename}.csv"
        df.to_csv(file_path, index=False)

        logger.info(f"Saved a data frame to a file. Path: {file_path}.")