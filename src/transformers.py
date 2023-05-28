import yaml
from pathlib import Path
import pandas as pd


class DistanceBounder:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        with open(Path("config", "model_const.yml"), "r") as f:
            self.model_const = yaml.safe_load(f)

    def run(self) -> pd.DataFrame:
        min_dist = self.model_const["min_distance"]
        max_dist = self.model_const["max_distance"]
        return self.df[(self.df["D"] >= min_dist) & (self.df["D"] <= max_dist)]


class RecordChooser:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def run(self) -> pd.DataFrame:
        # TODO: WRITE IT
        return self.df


class QualityAssessor:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def run(self) -> list:
        # TODO: WRITE IT
        return [0, 0, 0]


class Predictor:
    def __init__(self, model: dict) -> None:
        self.model = model
        self.altered_model = self.model.copy()

    def get_time(self, distance: float) -> float:
        # TODO: WRITE CONDITIONS AND MODEL
        return None

    def predict(self, distance: float, weight_change: float) -> float:
        weight_factor = 1 / (1 + weight_change)
        # TODO: TRY and in case of key error handle somehow
        # self.altered_model["sg"] = self.altered_model["sg"] * weight_factor
        # self.altered_model["F"] = self.altered_model["F"] * weight_factor
        # return self.get_time(distance)
