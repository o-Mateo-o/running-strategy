"""Helper transformers that support the processing classes."""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import yaml


class DistanceBounder:
    """Bounder for the data using the distance and the config limits.

    Attributes:
        df: Data frame of distances and times.
        model_const: Model constants.

    Args:
        df: Data frame of (unbounded) distances and times.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        with open(Path("config", "model_const.yml"), "r") as f:
            self.model_const = yaml.safe_load(f)

    def bound(self, bounds: tuple = tuple()) -> pd.DataFrame:
        """Bound the data.
        If the bounds are given, use them. Otherwise use the maximal possible bounds
        from the config file.

        Args:
            bounds (tuple, optional): . Defaults to the empty tuple.

        Returns:
            pd.DataFrame: Data frame with the bounded data.
        """
        if bounds:
            min_dist = bounds[0]
            max_dist = bounds[1]
        else:
            min_dist = self.model_const["min_distance"]
            max_dist = self.model_const["max_distance"]
        return self.df[(self.df["D"] >= min_dist) & (self.df["D"] <= max_dist)]


class RecordChooser:
    """Transformer that chooses only records from the data set.

    Attributes:
        df: Data frame with all the observations.

    Args:
        df: Data frame with all the observations.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    @staticmethod
    def categorizer(distance: float) -> float:
        """Round a given distance, with precision respectively to the scale.

        Args:
            distance (float): Original distance.

        Returns:
            float: Distance category (rounded value).
        """
        if distance < 0.1:
            return np.round(distance, 3)
        elif distance < 0.5:
            return np.round(distance, 2)
        elif distance < 2:
            return np.round(distance, 1)
        else:
            return np.round(distance)

    def cleanse(self) -> pd.DataFrame:
        """Choose only the "liminf" of the times.
        Categorize the times, and having sorted it by peace values, remove duplicates.

        Returns:
            pd.DataFrame: Frame of a cleansed data.
        """
        self.df["D_categ"] = self.df["D"].apply(self.categorizer)
        self.df["Pace"] = self.df["T"] / self.df["D"]
        self.df = self.df.sort_values("Pace")
        return self.df.drop_duplicates("D_categ", keep="first")


class QualityAssessor:
    """Assess quality of the data on all three sectors.

    Attributes:
        df: Data frame of distances and times.
        model_const: Model constants.
        model_u_bounds: Upper bounds of the sectors.

    Args:
        df: Data frame of distances and times.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        with open(Path("config", "model_u_bounds.yml"), "r") as f:
            self.model_u_bounds = yaml.safe_load(f)
        with open(Path("config", "model_const.yml"), "r") as f:
            self.model_const = yaml.safe_load(f)
        self.model_u_bounds["zero"] = {"optimal": 0}
        self.sectors = ["zero", "short", "mid", "long"]

    def _range_value(self, sector: str, kind: str) -> int:
        """Easily find a range bound value.

        Args:
            sector (str): Distance sector name.
            kind (str): Kind of the range ("optimal" or "extended").

        Returns:
            int: A bound value.
        """
        bound = self.model_u_bounds[sector][kind]
        return bound if bound != None else np.Inf

    @staticmethod
    def _extension_decision(ix: int, sector_ix: int) -> int:
        """Find the entailed penalty for an affected sector index
        and the origin sector index.
        The origin distance has the worst penalty, and the succeeding ones
        have some smaler. The distances before are not affected at all.

        Args:
            ix (int): An index currently analized (affected)
            sector_ix (int): A reference sector index (origin).

        Returns:
            int: Entailed penalty.
        """
        if ix > sector_ix:
            return 1
        elif ix == sector_ix:
            return 2
        else:
            return 0

    def test_sector(self, sector_ix: int) -> list[int]:
        """Find the penalties for all the sectors, related to the data in a given sector.
        Analyze the quality and make a quality decision for each case, based on the data
        counts and placement.

        .. note::
            Only the upper boundaries can be extedned.

        Args:
            sector_ix (int): Index of a specific sector.

        Returns:
            list: List of sector penalties caused by the data in the current sector.
        """
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

    def assess(self) -> list[int]:
        """Find the combined penalties by assessing it for each sector and summarizing
        the partial results.

        Returns:
            list: Combined penalties for each sector.
        """
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
    """Sectror limits server. Uses the settings from the config files.
    On init, load the config.

    Attributes:
        model_const: Model constants.
        model_u_bounds: Upper bounds of the sectors.
    """

    def __init__(self) -> None:
        with open(Path("config", "model_u_bounds.yml"), "r") as f:
            self.model_u_bounds = yaml.safe_load(f)
        with open(Path("config", "model_const.yml"), "r") as f:
            self.model_const = yaml.safe_load(f)

    def get_optimal(self) -> list[Union[int, float]]:
        """Get the optimal limits of all the data sectors, so those four separators
        in the |short|mid|long| representation.

        Returns:
            list: Sector limiters.
        """
        bound_limits = [
            self.model_u_bounds[kind]["optimal"] for kind in ["short", "mid"]
        ]
        return (
            [self.model_const["min_distance"]]
            + bound_limits
            + [self.model_const["max_distance"]]
        )

    def get_extended(self) -> list[Union[int, float]]:
        """Get the extended (maximally right-shifted) limits of all the data sectors,
        so those four separators in the |short|mid|long| representation.

        Returns:
            list: Sector limiters.
        """
        bound_limits = [
            self.model_u_bounds[kind]["extended"] for kind in ["short", "mid"]
        ]
        return (
            [self.model_const["min_distance"]]
            + bound_limits
            + [self.model_const["max_distance"]]
        )
