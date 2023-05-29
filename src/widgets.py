from pathlib import Path

import yaml
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.spinner import Spinner


class FileInfo(Label):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.print_prompt()

    def print_prompt(self) -> None:
        self.color = (1, 1, 1)
        self.text = "Wybierz plik z danymi z listy."

    def print_success(self, filename: str) -> None:
        self.color = (1, 1, 1)
        self.text = f'Załadowano plik "{filename}".\nPoniżej wybierz kolumny z czasem i dystansem.'

    def print_error(self, msg: str, file_hint: bool = False) -> None:
        self.color = (1, 0, 0)
        file_hint_txt = "\nSpróbuj wybrać plik jeszcze raz." if file_hint else ""
        self.text = f"BŁĄD: {msg}.{file_hint_txt}"


class MySpinner(Spinner):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.reset()

    def reset(self) -> None:
        self.text = "- rozwiń -"
        self.values = []

class SmartSlider(Slider):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.min = 0
        self.max = 100
        self.min_real_val = 0
        self.max_real_val = 100
        self.log = False

    @staticmethod
    def _logify(x: float) -> float:
        profile = 300
        return ((1 + profile) ** x - 1) / profile

    @property
    def real_value(self) -> float:
        prop = self.value / 100
        if self.log:
            prop = self._logify(prop)
        return self.min_real_val + prop * (self.max_real_val - self.min_real_val)


class DistanceSlider(SmartSlider):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.log = True
        with open(Path("config", "model_const.yml"), "r") as f:
            model_const = yaml.safe_load(f)
        self.min_real_val = model_const["min_distance"]
        self.max_real_val = model_const["max_distance"]
        self.log = True


class WeightSlider(SmartSlider):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.min_real_val = -5
        self.max_real_val = 5
