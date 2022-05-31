"""Contains a class to parse the corpus.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple
from dataclasses import dataclass, field
import logging
import pandas as pd
from zensols.config import ConfigFactory
from zensols.dataframe import ResourceFeatureDataframeStash

logger = logging.getLogger(__name__)


@dataclass
class SouthParkDataframeStash(ResourceFeatureDataframeStash):
    _CHAR_COL = 'character'

    characters: Tuple[str] = field(default=None)
    """The lower case southpark character names used as labels.

    """
    rand_sample: int = field(default=None)
    """"""

    def _get_loudest_character_names(self, df: pd.DataFrame) -> List[str]:
        """Return the southpark characters from the most to least verbose per mentioned
        utterances in the dataset.

        """
        col = self._CHAR_COL
        df = df.groupby(col).agg({col: 'count'}).\
            rename(columns=({col: 'count'}))
        df = df.sort_values('count', ascending=False)
        return df.iloc[:self.n_characters].index.to_list()

    def _get_dataframe(self) -> pd.DataFrame:
        df = super()._get_dataframe()
        chars = self.characters
        df['character'] = df['character'].apply(str.lower)
        df = df[df[self._CHAR_COL].isin(chars)]
        df['line'] = df['line'].apply(str.strip)
        if logger.isEnabledFor(logging.INFO):
            chars = df['character'].drop_duplicates().to_list()
            logger.info(f"using labels: {', '.join(chars)}")
        if self.rand_sample is not None:
            df = df.sample(frac=1)
            df = df[:self.rand_sample]
        return df
