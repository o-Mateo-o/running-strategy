"""Main file running the window app."""

import os
from pathlib import Path

from kivy.config import Config

if __name__ == "__main__":
    # change the path to the app dir
    karczrun_path = Path(os.path.abspath(__file__))
    os.chdir(karczrun_path / Path(".."))
    # configure Kivy
    Config.set("graphics", "resizable", True)
    Config.set("graphics", "height", "500")
    Config.set("graphics", "width", "700")
    # Window import can be done only after configuration is completed
    from src.app import *

    # run the app
    KarczRunApp().run()
