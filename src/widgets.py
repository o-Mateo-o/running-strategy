from kivy.uix.label import Label
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
        self.text = "<rozwiń>"
        self.values = []
