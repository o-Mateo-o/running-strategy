import os
from pathlib import Path
import kivy

kivy.require("1.0.6")
from kivy.config import Config


if __name__ == "__main__":
    karczrun_path = Path(os.path.abspath(__file__))
    
    os.chdir(karczrun_path / Path(".."))
    Config.set("graphics", "resizable", True)
    Config.set("graphics", "height", "500")
    Config.set("graphics", "width", "700")
    from src.app import *

    KarczRunApp().run()
