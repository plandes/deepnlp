"""Environemnt configuration and set up: add this (deepnlp) library to the
Python path and framework entry point."""
__author__ = 'Paul Landes'

from pathlib import Path


class JupyterManagerFactory(object):
    def __init__(self, app_root_dir: Path = Path('..')):
        """Set up the interpreter environment so we can import local packages.

        :param app_root_dir: the application root directory

        """
        from zensols import deepnlp
        deepnlp.init()
        from zensols.cli import ConfigurationImporterCliHarness
        self._harness = ConfigurationImporterCliHarness(root_dir=app_root_dir)

    def __call__(self, cuda_device_index: int = None) -> 'JupyterManager':
        from zensols.deeplearn.cli import JupyterManager

        def map_args(model: str = None, embedding: str = None):
            model = 'wordvec' if model is None else model
            path = str(self._harness.root_dir / 'models' / f'{model}.conf')
            args = ['-c', path]
            if embedding is not None:
                args.extend(['--override', f'cb_default.name={embedding}'])
            return args

        mng = JupyterManager(self._harness, cli_args_fn=map_args)
        mng.reduce_logging()
        if cuda_device_index is not None:
            # tell which GPU to use
            mng.config('gpu_torch_config', cuda_device_index=cuda_device_index)
        return mng
