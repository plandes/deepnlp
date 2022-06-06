"""Environemnt configuration and set up: add this (deepnlp) library to the
Python path and framework entry point.

"""
__author__ = 'Paul Landes'

from pathlib import Path


class JupyterManagerFactory(object):
    """Bootstrap and import libraries to automate notebook testing.

    """
    def __init__(self, app_root_dir: Path = Path('..')):
        """Set up the interpreter environment so we can import local packages.

        :param app_root_dir: the application root directory

        """
        from zensols import deepnlp
        deepnlp.init()
        from zensols.cli import ConfigurationImporterCliHarness
        self._harness = ConfigurationImporterCliHarness(
            package_resource='ner',
            root_dir=app_root_dir.absolute())

    def __call__(self):
        """Create a new ``JupyterManager`` instance and return it."""
        from zensols.deeplearn.cli import JupyterManager

        def map_args(config_name: str = 'wordvec', embedding: str = 'glove_50',
                     model_id: str = 'bert-base-cased'):
            cfile: Path = self._harness.root_dir / f'models/{config_name}.yml'
            overrides = {'name': embedding,
                         'trans_model_id': model_id}
            oarg = ','.join(map(lambda x: f'ner_default.{x[0]}={x[1]}',
                                overrides.items()))
            return ('--config', str(cfile), '--override', oarg)

        return JupyterManager(self._harness, cli_args_fn=map_args,
                              reduce_logging=True)
