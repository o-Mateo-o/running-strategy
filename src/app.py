from pathlib import Path
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen


from src.processing import DataHandler, ProcessingError
from src.assets import AssetPaths
from src.widgets import FileInfo, MySpinner


class MainScreen(Screen):
    def __init__(self, **kw):
        self.data_handler = None
        super().__init__(**kw)

    def _reset_spinners(self):
        self.ids.spinner_time.reset()
        self.ids.spinner_distance.reset()

    def display_error(self, msg, file_hint=False):
        self.ids.file_info.print_error(msg, file_hint)
        self._reset_spinners()

    def go_results_action(self):
        try:
            self.data_handler.preprocess_data(
                self.ids.spinner_distance.text, self.ids.spinner_time.text
            )
            self.manager.current = "results_screen"
        except ProcessingError as msg:
            self.display_error(msg)

    def choose_file(self, paths):
        try:
            # read and digest the file
            self.data_handler.process_file(paths[0])
            # show the filename info and add its columns to the spinners
            self.ids.file_info.print_success(Path(paths[0]).name)
            self.ids.spinner_distance.values = self.data_handler.cols
            self.ids.spinner_time.values = self.data_handler.cols
        except ProcessingError as msg:
            self.display_error(msg, file_hint=True)

    def click_file(self, paths):
        if paths:
            self.choose_file(paths)


class ResultsScreen(Screen):
    pass


class InfoScreen(Screen):
    pass


class WindowManager(ScreenManager):
    def __init__(self, **kwargs):
        self.data_handler = DataHandler()
        super().__init__(**kwargs)

    def add_widget(self, widget, *args, **kwargs):
        super().add_widget(widget, *args, **kwargs)
        widget.data_handler = self.data_handler


class KarczRunApp(App):
    assets = AssetPaths()

    def build(self):
        return Builder.load_file(str(Path("src", "app.kv")))
