import os
import math
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
from sklearn.neural_network import MLPRegressor
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from process import TimeSeriesPreProcessor
from pathlib import Path
PROJECT_ROOT_DIR = str(Path(__file__).parent.parent)


class Data:
    def __init__(self):
        self.df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "data/oil.csv"))
        self.labels = ["dcoilwtico"]
        self.date_column = "date"


class TimeSeriesTrainer:
    def __init__(self):
        self.preprocessor= TimeSeriesPreProcessor()

    def train_model(self, df, labels, date_column):
        label_count = len(labels)
        freq = 'D'  # transfer to payload config
        training_range = len(df)  # transfer to payload config
        train_df, features_requests = self.preprocessor.preprocess_series(df, labels, date_column, freq, training_range)
        train_df_for_training = train_df.copy()
        train_df_for_training.drop(columns=labels, inplace=True)
        scaler_features, scaler_label = self.preprocessor.scale_lagged_df(train_df_for_training, label_count)
        X_train, y_train, X_test, y_test = self.preprocessor.split_lagged_df(train_df_for_training, label_count)

        feature_count = len(np.squeeze(np.array([train_df.columns.values[:-label_count]])))
        h_cells = tuple([feature_count] * label_count)
        model = self.mlpregressor(X_train, y_train, X_test, y_test, h_cells)

        model.features_requests = features_requests
        model.train_df = train_df
        model.scaler_label = scaler_label
        model.labels = labels
        model.date_column = date_column
        model.freq = freq

        return model

    @staticmethod
    def mlpregressor(X_train, y_train, X_test, y_test, h_cells):
        mlp = MLPRegressor(
            hidden_layer_sizes=h_cells,
            max_iter=100,
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            validation_fraction=0.2,
            shuffle=False,
            random_state=10,
            batch_size=math.ceil(len(X_train) / 10))
        params = {
            # "n_iter_no_change": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "alpha": Categorical(categories=[0.001, 0.05]),
        }
        n_iter = 10
        # callback = TqdmCallback(total=n_iter)
        simplefilter("ignore", category=ConvergenceWarning)
        model = BayesSearchCV(mlp, params, n_iter=n_iter, n_jobs=3)
        # model.fit(X_train, y_train.ravel(), callback=callback)
        model.fit(X_train, y_train.ravel())

        return model

    def __call__(self, *args, **kwargs):
        data = Data()
        model_dir = os.path.join(PROJECT_ROOT_DIR, "model_storage")
        local_model_storage_path = os.path.join(model_dir, "model-new.sav")
        model = self.train_model(data.df, data.labels, data.date_column)
        joblib.dump(model, local_model_storage_path)

#
# # Create a custom callback to update tqdm
# class TqdmCallback:
#     def __init__(self, total):
#         self.pbar = tqdm(total=total, desc='BayesSearchCV Progress')
#
#     def __call__(self, res):
#         self.pbar.update(1)
#
#     def close(self):
#         self.pbar.close()