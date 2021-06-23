"""Application configuration class.

"""
__author__ = 'Paul Landes'

from pathlib import Path
from zensols.config import DictionaryConfig
from zensols.deeplearn.cli import FacadeApplicationFactory


class CliFactory(FacadeApplicationFactory):
    """The application specific factory."""
    def __init__(self, root_dir: Path, temporary_dir: Path = None,
                 **kwargs):
        """Initialize the factory.

        :param root_dir: the path to the root directory where data and resource
                         directories reside

        :param temporary_dir: the path where temporary files are created

        """
        if temporary_dir is None:
            temporary_dir = root_dir / 'target'
        dconf = DictionaryConfig(
            {'env': {'root_dir': str(root_dir),
                     'temporary_dir': str(temporary_dir)}})
        super().__init__(
            package_resource='ner.facade',
            app_config_resource=root_dir / 'resources' / 'app.conf',
            children_configs=(dconf,),
            **kwargs)
