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
        self._harness = ConfigurationImporterCliHarness(root_dir=app_root_dir)

    def __call__(self):
        """Create a new ``JupyterManager`` instance and return it."""
        from zensols.deeplearn.cli import JupyterManager

        def map_args(embedding: str = None):
            args = []
            if embedding is not None:
                args.extend(['--override', f'cb_default.name={embedding}'])
            return args

        return JupyterManager(self._harness, cli_args_fn=map_args,
                              reduce_logging=True)
