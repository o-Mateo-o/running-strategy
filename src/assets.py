from pathlib import Path
from typing import Any, Union


class AssetPaths:
    """Easy usage in the kivy file."""

    files = {"runner_icon": "runner.png"}

    def __getitem__(self, key: Any) -> Union[Any, None]:
        try:
            filename = self.files[key]
            return str(Path("assets", filename))
        except KeyError:
            return None
