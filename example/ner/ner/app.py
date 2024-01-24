"""Simple application to print corpus stats.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
from zensols.config import ConfigFactory
from . import SentenceStatsCalculator


@dataclass
class App(object):
    """Provides additional functionality to the NER example.

    """
    config_factory: ConfigFactory = field()
    stats_calc: SentenceStatsCalculator = field()

    def _inst(self, name: str):
        return self.config_factory(name)

    def _clean(self):
        cleaner = self._inst('cleaner_cli')
        cleaner.clean_level = 1
        cleaner()

    def stats(self):
        """Print corpus statistics."""
        self.stats_calc.write()

    def proto(self):
        "Test"
        import itertools as it
        if 1:
            self._clean()
        o = self._inst('dataset_stash')
        o.write()
        for b in it.islice(o.values(), 2):
            b.write()
