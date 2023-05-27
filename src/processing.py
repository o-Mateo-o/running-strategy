import pandas as pd
import re

class DataHandler:
    def __init__(self) -> None:
        self.df = None
        self.digested = False
        self.cols = []
        self.error_messages = []

    def _read_file(self, path):
        try:
            self.df = pd.read_csv(path)
            self.digested = True
        except pd.errors.EmptyDataError:
            self.error_messages.append("Empty data")
        except UnicodeDecodeError:
            self.error_messages.append("Unicode decode")

    def process_file(self, path):
        self.reset_data()
        if bool(re.search(".csv$", path)):
            self._read_file(path)
        else:
            self.error_messages.append("Only csv files are accepted")
        if self.digested:
            self.cols = self.df.columns.values


    def reset_data(self):
        self.df = None
        self.digested = False
        self.cols = []
        self.error_messages = []