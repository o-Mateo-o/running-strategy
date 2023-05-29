import re
from typing import Union

import numpy as np
import pandas as pd

from src.transformers import DistanceBounder, QualityAssessor, RecordChooser


class ProcessingError(Exception):
    ...


class DataHandler:
    def __init__(self) -> None:
        self.df_raw = None
        self.df_working = None
        self.data_quality = [0, 0, 0]
        self.model = dict()

    @property
    def cols(self) -> np.ndarray:
        return self.df_raw.columns.values

    def _read_file(self, path: str) -> None:
        try:
            self.df_raw = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            raise ProcessingError("Zestaw danych jest pusty")
        except UnicodeDecodeError:
            raise ProcessingError("Napotkano problem podczas dekodowania pliku")

    def process_file(self, path: str) -> None:
        self.reset_data()
        if bool(re.search(".csv$", path)):
            self._read_file(path)
        else:
            raise ProcessingError("Akceptowanym formatem jest tylko CSV")

    def preprocess_data(self, col_d: str, col_t: str) -> None:
        data_preprocessor = DataPreprocessor(self.df_raw, col_d, col_t)
        self.df_working, self.data_quality = data_preprocessor.process()

    def estim_model_params(self) -> None:
        self.model = KellerFitter(self.df_working).fit()

    def predict(self, distance, weight_change) -> Union[float, None]:
        est_time, quality_warning_level = Predictor(self.model).predict(
            distance, weight_change
        )
        quality_warning = ""
        return est_time, quality_warning

    def reset_data(self) -> None:
        self.df_raw = None
        self.df_working = None
        self.data_quality = [0, 0, 0]
        self.model = dict()


class DataPreprocessor:
    # ! DATA UNITS MUST BE [KM] AND [SEC]
    def __init__(self, df: pd.DataFrame, col_d: str, col_t: str) -> None:
        self.df = df
        self.col_d = col_d
        self.col_t = col_t
        self.data_quality = [3, 3, 3]

    def _validate_format(self) -> None:
        if not isinstance(self.df, pd.DataFrame):
            raise ProcessingError("Nie wybrano zestawu danych")
        if self.df.empty:
            raise ProcessingError("Zestaw danych jest pusty")

    def _subset(self) -> None:
        try:
            self.df = self.df[[self.col_d, self.col_t]]
        except KeyError:
            raise ProcessingError("Niepoprawnie wybrane kolumny")
        self.df.columns = ["D", "T"]

    def _cleanse(self) -> None:
        self.df = self.df.dropna()

    def _floatify(self) -> None:
        data_types = self.df.applymap(lambda x: isinstance(x, (int, float, str)))
        if not data_types.all().all():
            raise ProcessingError("Wybrane dane mają nieprawidłowy format")
        try:
            self.df = self.df.applymap(float)
        except (TypeError, ValueError):
            raise ProcessingError("Wybrane dane nie mogą być traktowane jako liczby")

    def _bound(self) -> None:
        self.df = DistanceBounder(self.df).bound()

    def _keep_records(self) -> None:
        self.df = RecordChooser(self.df).cleanse()

    def _assess_quality(self) -> None:
        penalties = QualityAssessor(self.df).assess()
        self.data_quality = [
            max(0, dq - p) for (dq, p) in zip(self.data_quality, penalties)
        ]

    def _approve_quality(self) -> None:
        if not any([q > 0 for q in self.data_quality]):
            raise ProcessingError(
                "Niezidentyfikowane błędy w danych uniemożliwiają kalkulacje"
            )

    def process(self) -> tuple:
        self._validate_format()
        self._subset()
        self._cleanse()
        self._floatify()
        self._bound()
        self._keep_records()
        self._assess_quality()
        self._approve_quality()
        return self.df, self.data_quality


class KellerFitter:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def fit(self) -> dict:
        # 1. divide data into groups (with extensions)
        # - might be done by an external transformer
        # 2. for each sector if quality > 0:
        # - estimate params by fitting and save them
        # - if needed use them
        # - if the model wants to use the args estimated before but cannot find them
        # raise an error
        return {
            "E0": 0,
            "sigma": 0,
            "tau": 0,
            "F": 0,
            "gamma": 0,
        }


class Predictor:
    def __init__(self, model: dict) -> None:
        self.model = model
        self.altered_model = self.model.copy()

    def get_time(self, distance: float) -> float:
        # TODO: WRITE CONDITIONS AND MODEL
        # 1. determine an estimation sector with the optimal range bounds
        # 2. using the model and params get the time (range switch)
        # 3. return the quality warning level and the time
        return None

    def predict(self, distance: float, weight_change: float) -> float:
        weight_factor = 1 / (1 + weight_change)
        # TODO: TRY and in case of key error handle somehow
        self.altered_model["sigma"] = self.altered_model["sigma"] * weight_factor
        self.altered_model["F"] = self.altered_model["F"] * weight_factor
        return self.get_time(distance), 0
