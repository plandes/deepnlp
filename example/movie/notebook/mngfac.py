"""Environemnt configuration and set up: add this (deepnlp) library to the
Python path and framework entry point.

"""
__author__ = 'Paul Landes'

from pathlib import Path


class JupyterManagerFactory(object):
    """Bootstrap and import libraries to automate notebook testing.

    """
    def __init__(self, app_root_dir: Path = Path('..'),
                 deepnlp_path: Path = Path('../../../src/python')):
        """Set up the interpreter environment so we can import local packages.

        :param app_root_dir: the application root directory

        """
        from zensols.cli import ConfigurationImporterCliHarness
        ConfigurationImporterCliHarness.add_sys_path(deepnlp_path)
        from zensols import deepnlp
        deepnlp.init()
        from zensols.cli import ConfigurationImporterCliHarness
        self._harness = ConfigurationImporterCliHarness(
            package_resource='mr',
            root_dir=app_root_dir.absolute())

    def __call__(self):
        """Create a new ``JupyterManager`` instance and return it."""
        from zensols.deeplearn.cli import JupyterManager

        def map_args(config_name: str = 'wordvec', embedding: str = None):
            args = []
            if embedding is not None:
                args.extend(['--override', f'mr_default.name={embedding}'])
            if config_name is not None:
                config_file = self._harness.root_dir / f'models/{config_name}.conf'
                args.extend(['--config', str(config_file)])
            return args

        return JupyterManager(self._harness, cli_args_fn=map_args,
                              reduce_logging=True)


f = JupyterManagerFactory()
f()
