import re
from typing import Union

import numpy as np
import pandas as pd

from src.transformers import DistanceBounder, Predictor, QualityAssessor, RecordChooser


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
        self.df_working, self.data_quality = data_preprocessor.run()

    def estim_model_params(self) -> None:
        # self.model = ...
        pass

    def predict(self, distance, weight_change) -> Union[float, None]:
        return Predictor(self.model).predict(distance, weight_change)

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
        self.df = DistanceBounder(self.df).run()

    def _keep_records(self) -> None:
        self.df = RecordChooser(self.df).run()

    def _assess_quality(self) -> None:
        penalties = QualityAssessor(self.df).run()
        self.data_quality = [
            max(0, dq - p) for (dq, p) in zip(self.data_quality, penalties)
        ]

    def _approve_quality(self) -> None:
        if not any([q > 0 for q in self.data_quality]):
            raise ProcessingError(
                "Niezidentyfikowane błędy w danych uniemożliwiają kalkulacje"
            )

    def run(self) -> tuple:
        self._validate_format()
        self._subset()
        self._cleanse()
        self._floatify()
        self._bound()
        self._keep_records()
        self._assess_quality()
        self._approve_quality()
        return self.df, self.data_quality
