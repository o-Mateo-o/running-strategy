# Kalkulator rekordów biegowych
<img src="assets/karcz_run.png" height="100">

[![version](https://img.shields.io/badge/Version-development-red)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)

## Opis projektu

__KarczRun__ to aplikacja przewidująca rekordy biegów na podstawie dostarczonych historycznych danych. Powstała ona w ramach projektu dotyczącego __strategii biegania__, podjętego na kursie Matematyka dla Przemysłu, W13 PWr.

Użyta w aplikacji estymacja, opiera się na modelu Kellera, a właściwie jego aproksymacji na konkretnych przedziałach dystansów. Na podstawie danych, zawierających czasy i dystansy biegów określonej osoby, jest ona w stanie określić jego możliwe rekordy dla żądanego dystansu i ewentualnej procentowej zmiany masy ciała zawodnika.

Przygotowanie projektu wymagało szeregu analiz i implementacji odpowiednich metod, a ostatecznie z połączenia tych części powstał przyjazny użytkownikowi program.

## Jak używać?

Aby użyć aplikacji w wersji deweloperskiej, należy w wirtualnym środowisku zainstalować wszystkie zależności (`pip install -r requirements.txt`).

Następnie, z wersją Pythona 3.9.7 uruchamiamy w terminalu główny plik komendą

    $ python karcz-run.py
    
Powinna wtedy otworzyć się okienkowa aplikacja z całą dostępną funkcjonalnością.

Do przechowywania lokalnie danych przeznaczony jest folder `./data`, jednak można wybierać także pliki spoza niego.

## Technologie
[![Python](https://img.shields.io/badge/Python-3.9.7-blue?logo=python&logoColor=white)](https://python.org "Go to Python homepage")
[![Kivy](https://img.shields.io/badge/Kivy-2.2.0-blue)](https://kivy.org/ "Go to Kivy homepage")

## Autorzy
 - [Natalia Iwańska](https://github.com/natalia185),
 - [Klaudia Janicka](https://github.com/klaudynka245),
 - [Maciej Karczewski](https://github.com/maciejkar),
 - [Adam Kawałko](https://github.com/Adasiek01),
 - [Mateusz Machaj](https://github.com/o-Mateo-o).
