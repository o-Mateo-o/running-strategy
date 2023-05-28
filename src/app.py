from pathlib import Path
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

from typing import Any, Union
from src.processing import DataHandler, ProcessingError
from src.assets import AssetPaths
from src.widgets import FileInfo, MySpinner, DistanceSlider, WeightSlider


class MainScreen(Screen):
    def __init__(self, **kw) -> None:
        self.data_handler = None
        super().__init__(**kw)

    def _reset_spinners(self) -> None:
        self.ids.spinner_time.reset()
        self.ids.spinner_distance.reset()

    def display_error(self, msg: str, file_hint: bool = False) -> None:
        self.ids.file_info.print_error(msg, file_hint)
        if file_hint:
            self._reset_spinners()

    def go_results_action(self) -> None:
        try:
            self.data_handler.preprocess_data(
                self.ids.spinner_distance.text, self.ids.spinner_time.text
            )
            self.data_handler.estim_model_params()
            self.manager.current = "results_screen"
        except ProcessingError as msg:
            self.display_error(msg)

    def choose_file(self, paths: list) -> None:
        try:
            # read and digest the file
            self.data_handler.process_file(paths[0])
            # show the filename info and add its columns to the spinners
            self.ids.file_info.print_success(Path(paths[0]).name)
            self.ids.spinner_distance.values = self.data_handler.cols
            self.ids.spinner_time.values = self.data_handler.cols
        except ProcessingError as msg:
            self.display_error(msg, file_hint=True)

    def click_file(self, paths: list) -> None:
        if paths:
            self.choose_file(paths)


class ResultsScreen(Screen):
    def __init__(self, **kw) -> None:
        self.data_handler = None
        super().__init__(**kw)

    def display_warning(self, msg: str) -> None:
        self.ids.result_warnings.text = msg

    def update_input(self) -> None:
        distance_raw = self.ids.distance_slider.real_value
        if distance_raw < 1000:
            self.ids.distance_display.text = f"{distance_raw:.0f} m"
        else:
            self.ids.distance_display.text = f"{(distance_raw / 1000):.1f} km"
        self.ids.weight_display.text = f"{self.ids.weight_slider.real_value:.1f} %"

    def show_predictions(self) -> None:
        if self.manager.current == "results_screen":
            distance = self.ids.distance_slider.real_value
            weight_change_v = self.ids.weight_slider.real_value / 100
            prediction = self.data_handler.predict(distance, weight_change_v)
            self.ids.est_time.text = "---" if prediction == None else str(prediction)
            self.display_warning("")


class InfoScreen(Screen):
    ...


class WindowManager(ScreenManager):
    def __init__(self, **kwargs) -> None:
        self.data_handler = DataHandler()
        super().__init__(**kwargs)

    def add_widget(self, widget: Screen, *args, **kwargs) -> None:
        super().add_widget(widget, *args, **kwargs)
        widget.data_handler = self.data_handler


class KarczRunApp(App):
    assets = AssetPaths()

    def build(self) -> Union[Any, None]:
        return Builder.load_file(str(Path("src", "app.kv")))
