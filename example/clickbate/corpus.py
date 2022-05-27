"""Contains a class to parse the corpus.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
import logging
from pathlib import Path
import pandas as pd
from zensols.install import Installer, Resource
from zensols.dataframe import AutoSplitDataframeStash

logger = logging.getLogger(__name__)


@dataclass
class ClickbateDataframeStash(AutoSplitDataframeStash):
    """Create the dataframe by reading the newline delimited set of clickbate
    headlines from the corpus files.

    """
    installer: Installer = field()
    """The installer used to download and uncompress the clickbate headlines.
    """

    cb_data_resource: Resource = field()
    """Use to resolve the file that has the the positive clickbate headlines.

    """
    non_cb_data_resource: Resource = field()
    """Use to resolve the file that has the the negative clickbate headlines.

    """
    def _parse_corpus(self, path: Path, label: bool) -> pd.DataFrame:
        """Parse the corpus file identified and give it a label.

        :param path: the path to the corpus file to read

        :param label: the label to add to the dataframe as a column

        """
        logger.info(f'parsing {path} with label {label}')
        with open(path) as f:
            sents = filter(lambda ln: len(ln) > 0,
                           map(lambda ln: ln.strip(), f.readlines()))
        rows = map(lambda ln: (label, ln), sents)
        return pd.DataFrame(rows, columns='label sent'.split())

    def _get_dataframe(self) -> pd.DataFrame:
        self.installer()
        cb_path: Path = self.installer[self.cb_data_resource]
        non_cb_path: Path = self.installer[self.non_cb_data_resource]
        # the "right" way to do this would be to also stratify across labels,
        # but this a simple example
        return pd.concat([self._parse_corpus(cb_path, 'y'),
                          self._parse_corpus(non_cb_path, 'n')],
                         ignore_index=True)
