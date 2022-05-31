"""Parse the corpus.

"""
__author__ = 'Paul Landes'

from typing import Tuple
from dataclasses import dataclass, field
import logging
import pandas as pd
from zensols.dataframe import ResourceFeatureDataframeStash

logger = logging.getLogger(__name__)


@dataclass
class SouthParkDataframeStash(ResourceFeatureDataframeStash):
    _CHAR_COL = 'character'

    characters: Tuple[str] = field(default=None)
    """The lower case southpark character names used as labels.

    """

    def _get_dataframe(self) -> pd.DataFrame:
        df = super()._get_dataframe()
        chars = self.characters
        df['character'] = df['character'].apply(str.lower)
        df = df[df[self._CHAR_COL].isin(chars)]
        df['line'] = df['line'].apply(str.strip)
        if logger.isEnabledFor(logging.INFO):
            chars = df['character'].drop_duplicates().to_list()
            logger.info(f"using labels: {', '.join(chars)}")
        return df
