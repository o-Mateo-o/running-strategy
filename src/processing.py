"""The main data processing classes. Core data keeper,
along with the ones regarding preprocessing, fitting and predicting."""

import re
from pathlib import Path

import numpy as np
from typing import Union
import pandas as pd
import yaml
from scipy.optimize import curve_fit

from src.model import ExtendedKellerApproxModel as formulas
from src.transformers import (
    ConfigLimits,
    DistanceBounder,
    QualityAssessor,
    RecordChooser,
)


class ProcessingError(Exception):
    """Error in processing, displayed then by the app."""

    ...


class DataHandler:
    """A class that keeps the data and handles the processing operations.

    Attributes:
        df_raw: Data frame fetched from file.
        df_working: Data frame processed by the app.
        data_quality: A list of quality levels for all the three distance sectors.
        model: A dictionary of model parameter values.
    """

    def __init__(self) -> None:
        self.df_raw = None
        self.df_working = None
        self.data_quality = [0, 0, 0]
        self.model = dict()

    @property
    def cols(self) -> np.ndarray:
        """Property of column names of a pandas DataFrame as a NumPy array.

        Returns:
            np.ndarray: Columns of a frame.
        """
        return self.df_raw.columns.values

    def _read_file(self, path: str) -> None:
        """Read data from a csv.

        Args:
            path (str): Path to the input data file.

        Raises:
            ProcessingError: If no data or a file cannot be decoded.
        """
        try:
            self.df_raw = pd.read_csv(path)
            # TODO: ISSUE #1 from TODO file
        except pd.errors.EmptyDataError:
            raise ProcessingError("Zestaw danych jest pusty")
        except UnicodeDecodeError:
            raise ProcessingError("Napotkano problem podczas dekodowania pliku")

    def process_file(self, path: str) -> None:
        """Read the file unless it is of the wrong format.

        Args:
            path (str): Path to the input data file.

        Raises:
            ProcessingError: When the file is not csv.
        """
        self.reset_data()
        if bool(re.search(".csv$", path)):
            self._read_file(path)
        else:
            raise ProcessingError("Akceptowanym formatem jest tylko CSV")

    def preprocess_data(self, col_d: str, col_t: str) -> None:
        """Preprocess the data to preare it for modeling and find the qualities.

        Args:
            col_d (str): Name of the distance column.
            col_t (str): Name of the time column.
        """
        data_preprocessor = DataPreprocessor(self.df_raw, col_d, col_t)
        self.df_working, self.data_quality = data_preprocessor.process()

    def estim_model_params(self) -> None:
        """Estimate the extended Keller model parameters.

        Raises:
            ProcessingError: If somethong is wron with the computation.
        """
        try:
            self.model = KellerFitter(self.df_working, self.data_quality).fit()
        except (RuntimeError, RuntimeWarning):
            raise ProcessingError(
                "Nie można dopasowac modelu do podanych danych.\nSprawdź wybrane kolumny"
            )

    @staticmethod
    def describe_quality(quality_warning_level: int) -> str:
        """Translate the quality levels to the correspondong warning messages.

        Args:
            quality_warning_level (int): Number describing the single quality level.

        Returns:
            str: A quality warning message.
        """
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

    def predict(
        self, distance: float, weight_change: float
    ) -> tuple[Union[float, str]]:
        """Perform time predictions for the given distance and weight change.
        Find also the quality warning for the distance sector.

        Args:
            distance (float): Distace to esimate time from.
            weight_change (float): Percent weight change of a runner.

        Returns:
            tuple: Etimated time and the quality warning message.
        """
        est_time = None
        try:
            est_time, quality_warning_level = Predictor(
                self.model, self.data_quality
            ).predict(distance, weight_change)
            quality_warning = self.describe_quality(quality_warning_level)
        except ProcessingError as msg:
            quality_warning = str(msg)
        except KeyError:
            quality_warning = (
                "Z powodu niekompletnego modelu nie można dokonać predykcji"
            )
        return est_time, quality_warning

    def reset_data(self) -> None:
        """Reset all the attributes. Data frames, model and the quality levels."""
        self.df_raw = None
        self.df_working = None
        self.data_quality = [0, 0, 0]
        self.model = dict()


class DataPreprocessor:
    """Class containing operations related to data preprocessing.
    This is the entire pipeline of operations to prepare the raw data set for
    model fitting.

    .. warning:
        Input data units must be [km] for distance and [sec] for time.

    Attributes:
        df: Data frame that is transformed.
        col_d: Name of the distance column.
        col_t: Name of the time column.
        data_quality: Quality levels for data sectors.

    Args:
        df (pd.DataFrame): Raw data frame.
        col_d (str): Name of the distance column.
        col_t (str): Name of the time column.

    """

    def __init__(self, df: pd.DataFrame, col_d: str, col_t: str) -> None:
        self.df = df
        self.col_d = col_d
        self.col_t = col_t
        self.data_quality = [3, 3, 3]

    def _validate_format(self) -> None:
        """Validate the raw data set and raise an error if it is not appropriate.

        Raises:
            ProcessingError: Raise it when the further calculations cannot be preformed.
                Namely, if the data set is empty or is not chosen.
        """
        if not isinstance(self.df, pd.DataFrame):
            raise ProcessingError("Nie wybrano zestawu danych")
        if self.df.empty:
            raise ProcessingError("Zestaw danych jest pusty")

    def _subset(self) -> None:
        """Select only the distance and time columns from the raw data frame.
        Use the given column names.

        Raises:
            ProcessingError: If the chosen columns does not exist.
        """
        try:
            self.df = self.df[[self.col_d, self.col_t]]
        except KeyError:
            raise ProcessingError("Niepoprawnie wybrane kolumny")
        self.df.columns = ["D", "T"]

    def _cleanse(self) -> None:
        """Remove the rows with blank data."""
        self.df = self.df.dropna()

    def _floatify(self) -> None:
        """Change the type of all data to float.

        Raises:
            ProcessingError: If the data cannot be converted to float.
        """
        data_types = self.df.applymap(lambda x: isinstance(x, (int, float, str)))
        if not data_types.all().all():
            raise ProcessingError("Wybrane dane mają nieprawidłowy format")
        try:
            self.df = self.df.applymap(float)
        except (TypeError, ValueError):
            raise ProcessingError("Wybrane dane nie mogą być traktowane jako liczby")

    def _bound(self) -> None:
        """Remove the data with distances smaller than the default minimal limit
        and the records larger than the maximal limit accordingly.
        """
        self.df = DistanceBounder(self.df).bound()

    def _keep_records(self) -> None:
        """Choose only the "liminf" of the times by a specific method of categorization
        to drop the non-record times. Use the `RecordChooser` class.
        """
        self.df = RecordChooser(self.df).cleanse()

    def _assess_quality(self) -> None:
        """Find and save the quality of data on each distance sector.
        Get the penalties using the `QualityAssessor` class and by subtraction
        evaluate the right qualities.
        """
        penalties = QualityAssessor(self.df).assess()
        self.data_quality = [3, 3, 3]
        self.data_quality = [
            max(0, dq - p) for (dq, p) in zip(self.data_quality, penalties)
        ]

    def _approve_quality(self) -> None:
        """Check if the quality is acceptable at least for one distance sector.
        Otherwise, raise an error to block the further processing.

        Raises:
            ProcessingError: If all the qualities are equal to zero.
        """
        if not any([q > 0 for q in self.data_quality]):
            raise ProcessingError(
                "Jakość danych jest nieakceptowalna na wszystkich zakresach danych"
            )

    def process(self) -> tuple:
        """A pipeline of preprocessing. Calls a range of methods to get clean data.

        Returns:
            tuple: Ready-to-fit dataframe and data qualities.
        """
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
    """Fitter for the extended Keller model.


    Attributes:
        df: Frame of preprocessed data.
        data_quality: Quality levels for data sectors.
        required_points: Required point counts for each distancce sector.

    Args:
        df (pd.DataFrame): Frame of preprocessed data.
        data_quality (list): Quality levels for data sectors.
    """

    def __init__(self, df: pd.DataFrame, data_quality: list[int]) -> None:
        self.df = df
        self.sector_qualites = data_quality
        with open(Path("config", "model_const.yml"), "r") as f:
            self.required_points = yaml.safe_load(f)["required_points"]

    def slice_data(self, sector_ix: int) -> pd.DataFrame:
        """Slice the data and get the division of a given index.
        Evaluate the sector bounds and find the best set of points
        for the sector.

        Args:
            sector_ix (int): Index of the distance sector (0 - short, 1 - mid, 2 - long).

        Returns:
            pd.DataFrame: Division of the data set.
        """
        config_limits = ConfigLimits()
        limits_optimal = config_limits.get_optimal()
        limits_extended = config_limits.get_extended()

        extended_slice = self.df[
            (limits_optimal[sector_ix] <= self.df["D"])
            & (self.df["D"] < limits_extended[sector_ix + 1])
        ]
        optimal_slice = extended_slice[
            extended_slice["D"] < limits_optimal[sector_ix + 1]
        ]

        translator = ["short", "mid", "long"]

        min_point_count = self.required_points[translator[sector_ix]]["min"]
        if optimal_slice.shape[0] < min_point_count:
            extended_slice = extended_slice.sort_values("D")
            return extended_slice.iloc[: min(extended_slice.shape[0], min_point_count)]
        else:
            return optimal_slice

    def fit(self) -> dict[str]:
        """Fit the 5 parameters of the model. With the divided set of data
        fit the model on three sectors getting the right parameters on each.
        Take the data quality into account and do not fit if the quality is too low.

        Returns:
            dict: Model parameters. If the parameter is not found, it defaults to NaN.
        """
        data_div = {
            "short": self.slice_data(0),
            "mid": self.slice_data(1),
            "long": self.slice_data(2),
        }

        tau, F, E0, sigma, gamma = [np.NAN] * 5
        if self.sector_qualites[0] > 0:
            tau, F = curve_fit(
                formulas.short, data_div["short"]["D"], data_div["short"]["T"]
            )[0]
        if self.sector_qualites[1] > 0:
            E0, sigma = curve_fit(
                lambda D, E0, sigma: formulas.mid(D, E0, sigma, tau, F),
                data_div["mid"]["D"],
                data_div["mid"]["T"],
            )[0]
        if self.sector_qualites[2] > 0:
            (gamma,) = curve_fit(
                lambda D, gamma: formulas.long(D, gamma, E0, sigma, tau, F),
                data_div["long"]["D"],
                data_div["long"]["T"],
            )[0]

        return {
            "gamma": gamma,
            "E0": E0,
            "sigma": sigma,
            "tau": tau,
            "F": F,
        }


class Predictor:
    """Time predictor for the extended Keller model for a given time
    and a percent weight change.

    Attributes:
        model_pure: Pure model parameters fitted by the `KellerFitter` class.
        model: Model altered by the weight changes. Initially it is an exact copy
            of a pure model.
        sector_qualities: Dictionary of qualities in each distance sector.
        limits: Optimal sector limits set in the config files.

    Args:
        model (dict): Pure model parameters fitted by the `KellerFitter` class.
        data_quality (list): Quality levels for data sectors.

    """

    def __init__(self, model: dict[float], data_quality: list[int]) -> None:
        self.model_pure = model
        self.sector_qualites = {
            "short": data_quality[0],
            "mid": data_quality[1],
            "long": data_quality[2],
        }
        self.model = self.model_pure.copy()
        self.limits = ConfigLimits().get_optimal()

    def evaluate_sector(self, distance: float) -> str:
        """Find the sector which a given distance belongs to.

        Args:
            distance (float): Distance to categorize.

        Returns:
            str: Distance sector name.
        """
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

    def predict_simple(self, distance: float) -> tuple[Union[float, None], int]:
        """Predict a time from a current model.
        Depending of the quality and a distance sector use different formulas.

        Args:
            distance (float): Some distnace.

        Raises:
            ProcessingError: If some error occurs in the computation process.

        Returns:
            float: Estimated time and quality of the result. If they cannot be found,
                the tuple defaults to `(None, 0)`
        """
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
            quality = 0  # ^ just in case

        return time, quality

    def predict(
        self, distance: float, weight_change: float
    ) -> tuple[Union[float, None], int]:
        """Perform a simple prediction, but having altered the model by the weight factor.

        Args:
            distance (float): Some distance.
            weight_change (float): Some small percent time change.

        Returns:
            float: Estimated time and quality of the result. If they cannot be found,
                the tuple defaults to `(None, 0)`
        """
        weight_factor = 1 / (1 + weight_change)
        self.model["sigma"] = self.model_pure["sigma"] * weight_factor
        self.model["F"] = self.model_pure["F"] * weight_factor
        self.model["E0"] = self.model_pure["E0"] * weight_factor
        try:
            return self.predict_simple(distance)
        except RuntimeWarning:
            return None, 0
