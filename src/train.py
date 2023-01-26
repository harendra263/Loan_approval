import pandas as pd
import numpy as np
import os


TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")



FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    df = pd.read_parquet(TRAINING_DATA)

    df_train = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
    df_test = df[df.kfold == FOLD].reset_index(drop=True)
