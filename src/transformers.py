"""Helper transformers that support the processing classes."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class DistanceBounder:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        with open(Path("config", "model_const.yml"), "r") as f:
            self.model_const = yaml.safe_load(f)

    def bound(self, bounds: tuple = tuple()) -> pd.DataFrame:
        if bounds:
            min_dist = bounds[0]
            max_dist = bounds[1]
        else:
            min_dist = self.model_const["min_distance"]
            max_dist = self.model_const["max_distance"]
        return self.df[(self.df["D"] >= min_dist) & (self.df["D"] <= max_dist)]


class RecordChooser:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    @staticmethod
    def categorizer(distance: float) -> float:
        if distance < 0.1:
            return np.round(distance, 3)
        elif distance < 0.5:
            return np.round(distance, 2)
        elif distance < 2:
            return np.round(distance, 1)
        else:
            return np.round(distance)

    def cleanse(self) -> pd.DataFrame:
        self.df["D_categ"] = self.df["D"].apply(self.categorizer)
        self.df["Pace"] = self.df["T"] / self.df["D"]
        self.df = self.df.sort_values("Pace")
        return self.df.drop_duplicates("D_categ", keep="first")


class QualityAssessor:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        with open(Path("config", "model_u_bounds.yml"), "r") as f:
            self.model_u_bounds = yaml.safe_load(f)
        with open(Path("config", "model_const.yml"), "r") as f:
            self.model_const = yaml.safe_load(f)
        self.model_u_bounds["zero"] = {"optimal": 0}
        self.sectors = ["zero", "short", "mid", "long"]

    def _range_value(self, sector: str, kind: str) -> int:
        bound = self.model_u_bounds[sector][kind]
        return bound if bound != None else np.Inf

    @staticmethod
    def _extension_decision(ix: int, sector_ix: int) -> int:
        if ix > sector_ix:
            return 1
        elif ix == sector_ix:
            return 2
        else:
            return 0

    def test_sector(self, sector_ix: int) -> list:
        # ! extend only upper boundaries
        curr_sector = self.sectors[sector_ix]
        before_sector = self.sectors[sector_ix - 1]

        lower_opt = self._range_value(before_sector, "optimal")
        upper_opt = self._range_value(curr_sector, "optimal")
        upper_ext = self._range_value(curr_sector, "extended")

        if (
            DistanceBounder(self.df).bound((lower_opt, upper_opt)).shape[0]
            >= self.model_const["required_points"][curr_sector]["reliable"]
        ):
            return [0, 0, 0]
        elif (
            DistanceBounder(self.df).bound((lower_opt, upper_opt)).shape[0]
            >= self.model_const["required_points"][curr_sector]["min"]
        ):
            return [1 if ix == sector_ix else 0 for ix in range(1, 4)]
        elif (
            DistanceBounder(self.df).bound((lower_opt, upper_ext)).shape[0]
            >= self.model_const["required_points"][curr_sector]["min"]
        ):
            return [self._extension_decision(ix, sector_ix) for ix in range(1, 4)]
        else:
            return [3 if ix >= sector_ix else 0 for ix in range(1, 4)]

    def assess(self) -> list:
        sector_penalties = {
            self.sectors[sector_ix]: self.test_sector(sector_ix)
            for sector_ix in range(1, 4)
        }
        # List of penalties based on the partial penalties evaluated on each section
        penalties = [
            sector_penalties["short"][i]
            + sector_penalties["mid"][i]
            + sector_penalties["long"][i]
            for i in range(3)
        ]
        return penalties


class ConfigLimits:
    def __init__(self) -> None:
        with open(Path("config", "model_u_bounds.yml"), "r") as f:
            self.model_u_bounds = yaml.safe_load(f)
        with open(Path("config", "model_const.yml"), "r") as f:
            self.model_const = yaml.safe_load(f)

    def get_optimal(self) -> list:
        bound_limits = [
            self.model_u_bounds[kind]["optimal"] for kind in ["short", "mid"]
        ]
        return (
            [self.model_const["min_distance"]]
            + bound_limits
            + [self.model_const["max_distance"]]
        )

    def get_extended(self) -> list:
        bound_limits = [
            self.model_u_bounds[kind]["extended"] for kind in ["short", "mid"]
        ]
        return (
            [self.model_const["min_distance"]]
            + bound_limits
            + [self.model_const["max_distance"]]
        )
