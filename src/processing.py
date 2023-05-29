import re
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import yaml

from src.model import ExtendedKellerApproxModel as formulas
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

    @staticmethod
    def describe_quality(quality_warning_level: int) -> str:
        if quality_warning_level >= 3:
            return ""
        elif quality_warning_level == 2:
            return "Z powodu niewielkiej ilości danych, wyniki mogą być obciążone delikatnymi niedokładnościami"
        elif quality_warning_level == 1:
            return "Z powodu małych ilości danych na pewnych przedziałach, wynik może być mało wiarygodny"
        else:
            return (
                "Nie można przewidywać wyniku dla tego dystansu z powodu braku danych"
            )

    def predict(self, distance, weight_change) -> Union[float, None]:
        est_time = None
        try:
            est_time, quality_warning_level = Predictor(
                self.model, self.data_quality
            ).predict(distance, weight_change)
            quality_warning = self.describe_quality(quality_warning_level)
        except ProcessingError as msg:
            quality_warning = msg
        except KeyError:
            quality_warning = "Z powodu niekompletnego modelu nie można dokonać predykcji" 
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
    def __init__(self, model: dict, data_quality: list) -> None:
        self.model = model
        self.sector_qualites = {
            "short": data_quality[0],
            "mid": data_quality[1],
            "long": data_quality[2],
        }
        self.altered_model = self.model.copy()
        self.limits = self.get_config_limits()

    @staticmethod
    def get_config_limits() -> list:
        with open(Path("config", "model_u_bounds.yml"), "r") as f:
            model_u_bounds = yaml.safe_load(f)
        with open(Path("config", "model_const.yml"), "r") as f:
            model_const = yaml.safe_load(f)
        bound_limits = [model_u_bounds[kind]["optimal"] for kind in ["short", "mid"]]
        return (
            [model_const["min_distance"]] + bound_limits + [model_const["max_distance"]]
        )

    def evaluate_sector(self, distance: float) -> int:
        lower_checks = [distance >= lim for lim in self.limits]
        upper_checks = [distance < lim for lim in self.limits]
        in_sector_check = [l and u for l, u in zip(lower_checks[:-1], upper_checks[1:])]
        try:
            ix = in_sector_check.index(True)
        except ValueError:
            ix = None
        sector_names = {
            0: "short",
            1: "mid",
            2: "long",
            None: "none",
        }
        return sector_names[ix]

    def predict_simple(self, distance: float) -> float:
        sector = self.evaluate_sector(distance)
        if sector == "none":
            raise ProcessingError("Wartość dystansu jest spoza dozwolonego zakresu")
        else:
            quality = self.sector_qualites[sector]

        if sector == "short" and quality > 0:
            time = formulas.short(distance, self.model["tau"], self.model["F"])
        elif sector == "mid" and quality > 0:
            time = formulas.mid(
                distance,
                self.model["E0"],
                self.model["sigma"],
                self.model["tau"],
                self.model["F"],
            )
        elif sector == "long" and quality > 0:
            time = formulas.long(
                distance,
                self.model["gamma"],
                self.model["E0"],
                self.model["sigma"],
                self.model["tau"],
                self.model["F"],
            )
        else:
            time = None
            quality = 0 # ^ just in case
            
        return time, quality

    def predict(self, distance: float, weight_change: float) -> float:
        weight_factor = 1 / (1 + weight_change)
        self.altered_model["sigma"] = self.altered_model["sigma"] * weight_factor
        self.altered_model["F"] = self.altered_model["F"] * weight_factor
        return self.predict_simple(distance)
