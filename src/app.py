from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
import pandas as pd


class InfoScreen(Screen):
    pass


class MainScreen(Screen):
    df = None

    def selected(self, filename):
        try:
            self.df = pd.read_csv(filename[0])
            self.ids.file.text = str(self.df.columns.values)
            self.ids.spinner_distance.values = self.df.columns.values
            self.ids.spinner_time.values = self.df.columns.values
        except:
            try:
                self.ids.file.text = str(filename[0])
                self.ids.spinner_distance.values = ""
                self.ids.spinner_time.values = ""
            except:
                pass


class ResultsScreen(Screen):
    # df = None
    # def check(self):
    #     self.df = MyApp.WindowMenager.FirstWindow.df
    #     print(self.df.columns)
    pass


class WindowManager(ScreenManager):
    pass


class MyApp(App):
    def build(self):
        return kv


if __name__ == "__main__":
    # os.chdir()
    kv = Builder.load_file("layout.kv")
    Window.size = (800, 500)
    MyApp().run()
