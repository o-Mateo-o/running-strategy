"""Custom Kivy widgets."""

from pathlib import Path

import yaml
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.spinner import Spinner


class FileInfo(Label):
    """Label informing about the events and errors regardning the file selection.

    As default, print the basic prompt.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.print_prompt()

    def print_prompt(self) -> None:
        """Print the prompt message (no color)."""
        self.color = (1, 1, 1)
        self.text = "Wybierz plik z danymi z listy."

    def print_success(self, filename: str) -> None:
        """Print the file read success message with the filename (no color).

        Args:
            filename (str): Name of the selected file.
        """
        self.color = (1, 1, 1)
        self.text = f'Załadowano plik "{filename}".\nPoniżej wybierz kolumny z czasem i dystansem.'

    def print_error(self, msg: str, file_hint: bool = False) -> None:
        """Print the error either related to the file processing or the data processing (color red).
        If this is only a data processing error, do not display an additional prompt.

        Args:
            msg (str): Error message.
            file_hint (bool, optional): If error is related to the file itself. Defaults to False.
        """
        self.color = (1, 0, 0)
        file_hint_txt = "\nSpróbuj wybrać plik jeszcze raz." if file_hint else ""
        self.text = f"BŁĄD: {msg}.{file_hint_txt}"


class MySpinner(Spinner):
    """File selector trigger with a custom prompt name.
    On default, reset the values and display the prompt."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.reset()

    def reset(self) -> None:
        """Remove all the possible values and set the current value to prompt."""
        self.text = "- rozwiń -"
        self.values = []


class SmartSlider(Slider):
    """A scaled slider with given range and scale type that
    allows to get the transformed value.

    Attributes:
        min: Artificial left bound of a slider represenred in % (equals 0).
        max: Artificial right bound of a slider represenred in % (equals 100).
        min_real_value: Minimal real value on the slider.
        max_real_val: Maximal real value on the slider.
        log: Bool info about the scale. `True` means the logarithmic out scale. Defaults to `False`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.min = 0
        self.max = 100
        self.min_real_val = 0
        self.max_real_val = 100
        self.log = False

    @staticmethod
    def _logify(x: float) -> float:
        """Transform a value from 0-1 range by an exponential function
        to get the beter "granularity" on the smaller values.

        .. note::
            The profile can be changed by altering the `PROFILE` constant.
            Larger profile would mean a "stronger" curve.

        Args:
            x (float): Value to be transformed.

        Returns:
            float: Curved value.
        """
        PROFILE = 300
        return ((1 + PROFILE) ** x - 1) / PROFILE

    @property
    def real_value(self) -> float:
        """Based on the percent covered by a slider,
        evaluate a real value represented by the widget.

        Returns:
            float: Real value from a slider.
        """
        prop = self.value / 100
        if self.log:
            prop = self._logify(prop)
        return self.min_real_val + prop * (self.max_real_val - self.min_real_val)


class DistanceSlider(SmartSlider):
    """Custom smart slider for distances.
    Automatically assign the ranges from the config file.
    Log scale set.

    Attributes:
        min_real_value: Minimal real value on the slider (from config file).
        max_real_val: Maximal real value on the slider (from config file - 1).
        log: `True` flag for a log scale.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.log = True
        with open(Path("config", "model_const.yml"), "r") as f:
            model_const = yaml.safe_load(f)
        self.min_real_val = model_const["min_distance"]
        self.max_real_val = model_const["max_distance"] - 1
        self.log = True


class WeightSlider(SmartSlider):
    """Custom smart slider for weights.

    Attributes:
        min_real_value: Minimal real value on the slider (-5 [%]).
        max_real_val: Maximal real value on the slider (5 [%]).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.min_real_val = -5
        self.max_real_val = 5
