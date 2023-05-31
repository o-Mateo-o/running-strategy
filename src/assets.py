"""Kivy asset read support."""

from pathlib import Path
from typing import Any, Union


class AssetPaths:
    """Asset paths keeper providing easy usage in the kivy file.
    
    Attributes:
        files: Dictionary of the filenames and their keys.
    """

    files = {
        "runner_icon": "runner.png",
        "weight_icon": "weight.png",
        "logo": "karcz_run.png",
    }

    def __getitem__(self, key: Any) -> Union[Any, None]:
        """Get an assset file path by its key name.
        
        Args:
            key (Any): Key name of an asset.

        Return:
            (Union[Any, None]): Asset full path.
        """
        try:
            filename = self.files[key]
            return str(Path("assets", filename))
        except KeyError:
            return None
