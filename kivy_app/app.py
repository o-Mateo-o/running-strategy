from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
import os
import json
import numpy as np
import pandas as pd



class InfoWindow(Screen):
    pass



class FirstWindow(Screen):
    df = None
    def selected(self, filename):

        try:
            self.df = pd.read_csv(filename[0])
            # print(self.df)
            self.ids.plik.text  = str(self.df.columns.values)
            self.ids.spinner_dist.values = self.df.columns.values
            self.ids.spinner_time.values = self.df.columns.values
        except:
            try:
                self.ids.plik.text = str(filename[0])
                self.ids.spinner_dist.values = ''
                self.ids.spinner_time.values = ''
            except:
                pass



class SecondWindow(Screen):
    # df = None
    # def check(self):
    #     self.df = MyApp.WindowMenager.FirstWindow.df
    #     print(self.df.columns)
    pass
 


class WindowMenager(ScreenManager):
    pass



class MyApp(App):
    def build(self):
        return kv
    


if __name__ == '__main__':
    # os.chdir(os.path.dirname(__file__))
    print(os.getcwd(), '----------------------------------')
    kv = Builder.load_file('program.kv')
    Window.size = (1000, 800)
    MyApp().run()