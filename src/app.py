from pathlib import Path
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

from src.processing import DataHandler


class MainScreen(Screen):
    def __init__(self, **kw):
        self.data_handler = None
        super().__init__(**kw)

    def go_results_action(self):
        if True:
            self.manager.current = "results_screen"
        else:
            # show an error
            pass

    def choose_file(self, paths):
        # self.ids.file_info = "Wybierz plik z listy."
        self.data_handler.process_file(paths[0])

        if self.data_handler.digested:
            filename = Path(paths[0]).name
            self.ids.file_info.text = f"""Załadowano plik "{filename}".\nPoniżej wybierz kolumny z czasem i dystansem."""  # TODO:class
            self.ids.spinner_distance.values = self.data_handler.cols
            self.ids.spinner_time.values = self.data_handler.cols
        else:
            pass
            # wyświetlić `self.data_handler.error_messages[0]`

    def click_file(self, paths):
        if paths:
            self.choose_file(paths)
# klasa do file info która na kolorowo wyświetla rzeczhy


class ResultsScreen(Screen):
    # df = None
    # def check(self):
    #     self.df = MyApp.WindowMenager.FirstWindow.df
    #     print(self.df.columns)
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
    def build(self):
        return Builder.load_file(str(Path("src", "app.kv")))
