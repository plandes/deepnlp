"""Application configuration class.

"""
__author__ = 'Paul Landes'

from pathlib import Path
from dataclasses import dataclass
from zensols.config import DictionaryConfig
from zensols.deeplearn.cli import FacadeApplicationFactory


@dataclass
class CliFactory(FacadeApplicationFactory):
    @classmethod
    def instance(cls: type, root_dir: Path, gpu_primary_index: int = 0,
                 **kwargs):
        dconf = DictionaryConfig(
            {'env': {'root_dir': str(root_dir),
                     'gpu_primary_index': str(gpu_primary_index)}})
        return cls(
            package_resource='movie.facade',
            app_config_resource=root_dir / 'resources' / 'app.conf',
            children_configs=(dconf,), **kwargs)
