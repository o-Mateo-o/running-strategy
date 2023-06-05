"""Main Kivy app clases. App, screens and their managers."""

import math
from pathlib import Path
from typing import Any, Union

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager
from numpy import NAN

from src.assets import AssetPaths
from src.processing import DataHandler, ProcessingError
from src.widgets import DistanceSlider, FileInfo, MySpinner, WeightSlider


class MainScreen(Screen):
    """The MainScreen class contains methods for handling user input and displaying information
    related to data processing and file selection within a main screen of GUI application.

    Attributes:
        data_handler: A class to keep the data and handle the processing operations
    """

    def __init__(self, **kw) -> None:
        self.data_handler = None
        super().__init__(**kw)

    def _reset_spinners(self) -> None:
        """Reset the time and distance spinners."""
        self.ids.spinner_time.reset()
        self.ids.spinner_distance.reset()

    def display_error(self, msg: str, file_hint: bool = False) -> None:
        """Show an error message for either a file related error or not.

        Args:
            msg (str): Error message.
            file_hint (bool, optional): Whether the error type is related to the files.
                Defaults to False.
        """
        self.ids.file_info.print_error(msg, file_hint)
        if file_hint:
            self._reset_spinners()

    def go_results_action(self) -> None:
        """Change the screen to results.
        Preprocess the data and fitt the model. In case any errors were catched
        do not change the screen and print the message.
        """
        try:
            self.data_handler.preprocess_data(
                self.ids.spinner_distance.text, self.ids.spinner_time.text
            )
            self.data_handler.estim_model_params()
            self.manager.current = "results_screen"
        except ProcessingError as msg:
            self.display_error(msg)

    def choose_file(self, paths: list[str]) -> None:
        """Do the actions after file choosing event.
        Read and digest the file, then show the file name in the
        info tab and update the spinner values.
        In case any errors were catched do not change the screen and print the message.

        Args:
            paths (list): Path list. Usualy a list of one file path.
                The method always uses its first element.
        """
        try:
            # read and digest the file
            self.data_handler.process_file(paths[0])
            # show the filename info and add its columns to the spinners
            self.ids.file_info.print_success(Path(paths[0]).name)
            self._reset_spinners()
            self.ids.spinner_distance.values = sorted(self.data_handler.cols, key=lambda x: 'distance' not in x.lower() and 'dystans' not in x.lower())
            self.ids.spinner_time.values = sorted(self.data_handler.cols, key=lambda x: 'time' not in x.lower() and 'czas' not in x.lower())
        except ProcessingError as msg:
            self.display_error(msg, file_hint=True)

    def click_file(self, paths: list[str]) -> None:
        """Handle an event of file/dir clicking. Only if an object is a file
        perform the procedures related to file choice.

        Args:
            paths (list): Path list. Usualy a list of one file path.
                The method always uses its first element.
        """
        if paths:
            self.choose_file(paths)


class ResultsScreen(Screen):
    """The ResultsScreen class contains methods to set the input for predictions
    and display its results along with the warning messages in the second screen of GUI application.

    Attributes:
        data_handler: A class to keep the data and handle the processing operations.
    """

    def __init__(self, **kw) -> None:
        self.data_handler = None
        super().__init__(**kw)

    def display_warning(self, msg: str) -> None:
        self.ids.result_warnings.text = f"UWAGA: {msg}." if msg else ""
        self.ids.result_warnings.color = (0,0,0)

    def update_input(self) -> None:
        distance_raw = self.ids.distance_slider.real_value
        if distance_raw < 1:
            self.ids.distance_display.text = f"{(distance_raw * 1000):.0f} m"
        else:
            self.ids.distance_display.text = f"{distance_raw:.1f} km"
        self.ids.weight_display.text = f"{self.ids.weight_slider.real_value:.1f} %"

    @staticmethod
    def format_time(total_seconds: Union[int, float, None]) -> str:
        if total_seconds == None or total_seconds is NAN or math.isnan(total_seconds):
            return "---"
        m, s = divmod(total_seconds, 60)
        h, m = divmod(m, 60)
        h, m, s, d = int(h), int(m), int(s // 1), int((s % 1) * 10)
        if h > 0:
            return f"{h} h {m} min"
        elif m > 0:
            return f"{m} min {s} s"
        else:
            return f"{s},{d} s"

    def show_predictions(self) -> None:
        if self.manager.current == "results_screen":
            distance = self.ids.distance_slider.real_value
            weight_change_v = self.ids.weight_slider.real_value / 100
            prediction, warning = self.data_handler.predict(distance, weight_change_v)
            self.ids.est_time.text = self.format_time(prediction)
            self.display_warning(warning)


class InfoScreen(Screen):
    """The InfoScreen class contains displays the user manual in a third screen of GUI application.

    Attributes:
        data_handler: A class to keep the data and handle the processing operations.
        manual: User manual string content.
    """
    def __init__(self, **kw):
        super().__init__(**kw)
        with open(Path("config", "user_manual.txt"), "r", encoding="utf-8") as f:
            self.manual = f.read()


class WindowManager(ScreenManager):
    """Screen manager for the main, result and info areas.
    Automatically connect the screens and data handler both ways.

    Attributes:
        data_handler: A class to keep the data and handle the processing operations.
    """
    def __init__(self, **kwargs) -> None:
        self.data_handler = DataHandler()
        super().__init__(**kwargs)

    def add_widget(self, widget: Screen, *args, **kwargs) -> None:
        super().add_widget(widget, *args, **kwargs)
        widget.data_handler = self.data_handler


class KarczRunApp(App):
    """Main app class.

    Attributes:
        assets: Asset path keeper.
    """
    assets = AssetPaths()

    def build(self) -> Union[Any, None]:
        """Build a kivy app structure based on the .kv file.

        Returns:
            Union[Any, None]: Kivy app structure.
        """
        return Builder.load_file(str(Path("src", "app.kv")))
