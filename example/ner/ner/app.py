"""
"""

from dataclasses import dataclass
from zensols.config import ConfigFactory
from . import SentenceStatsCalculator


@dataclass
class App(object):
    config_factory: ConfigFactory
    stats_calc: SentenceStatsCalculator

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
        print(type(self._inst('facade')))
        return
        import itertools as it
        if 0:
            self._clean()
        o = self._inst('dataset_stash')
        o.write()
        for b in it.islice(o.values(), 2):
            b.write()
