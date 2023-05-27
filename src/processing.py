import pandas as pd
import re
import numpy as np
import yaml
from pathlib import Path

class ProcessingError(Exception):
    ...


class DataHandler:
    def __init__(self) -> None:
        self.df_raw = None
        self.df_working = None
        self.data_quality = [0, 0, 0]

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

    def reset_data(self) -> None:
        self.df_raw = None
        self.df_working = None
        self.data_quality = [0, 0, 0]


class DataPreprocessor:
    def __init__(self, df: pd.DataFrame, col_d: str, col_t: str) -> None:
        self.df = df
        self.col_d = col_d
        self.col_t = col_t
        self.data_quality = [3, 3, 3]

    def _validate_format(self) -> None:
        if not isinstance(self.df, pd.DataFrame):
            raise ProcessingError("Nie wybrano zestawu danych")
        if self.df.empty:
            return ProcessingError("Zestaw danych jest pusty")

    def _subset(self) -> None:
        try:
            self.df = self.df[[self.col_d, self.col_t]]
        except KeyError:
            return ProcessingError("Nie znaleziono wybranej kolumny w zestawie danych")

    def _cleanse(self) -> None:
        self.df.dropna(inplace=True)

    def _floatify(self) -> None:
        # TODO: WRITE IT
        pass

    def _bound(self) -> None:
        with open(Path("config", "ranges.yml"), 'r') as f:
            model_const = yaml.safe_load(f)

        # TODO: WRITE IT

    def _keep_records(self) -> None:
        # TODO: WRITE IT
        pass

    def _assess_quality(self) -> None:
        # TODO: WRITE IT
        pass

    def _approve_quality(self) -> None:
        if not any([q > 0 for q in self.data_quality]):
            raise ProcessingError(
                "Niezidentyfikowane błędy w danych uniemożliwiają kalkulacje"
            )

    def run(self) -> tuple:
        self._validate_format()
        self.df_working = self._subset()
        self.df_working = self._cleanse()
        self._bound()
        self._keep_records()
        self._assess_quality()
        self._approve_quality()
        return self.df, self.data_quality
