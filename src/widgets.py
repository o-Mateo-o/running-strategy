from kivy.uix.label import Label

class FileInfo(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.print_prompt()

    def print_prompt(self):
        self.color = (1, 1, 1)
        self.text = "Wybierz plik z danymi z listy."

    def print_success(self, filename):
        self.color = (1, 1, 1)
        self.text = f'Załadowano plik "{filename}".\nPoniżej wybierz kolumny z czasem i dystansem.'

    def print_error(self, message):
        self.color = (1, 0, 0)
        self.text = f"BŁĄD: {message}.\nSpróbuj wybrać plik jeszcze raz."
