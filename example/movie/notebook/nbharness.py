"""Environemnt configuration and set up: add this (deepnlp) library to the
Python path and framework entry point."""
__author__ = 'Paul Landes'

from pathlib import Path


class AppNotebookHarness(object):
    def __init__(self, app_root_dir: Path = Path('..'), src_dir: str = 'mr'):
        """Set up the interpreter environment so we can import local packages.

        :param app_root_dir: the application root directory

        :param src_dir: the relative source directory from the project root

        """
        import sys
        # add the root path
        sys.path.append(str(app_root_dir))
        # add the example to the Python library path
        sys.path.append(str(app_root_dir / src_dir))
        # initialize the deep learning libraries and random state
        import harness
        harness.init()
        self._harness = harness.create_harness(root_dir=app_root_dir)

    def __call__(self, cuda_device_index: int = None):
        from zensols.deeplearn.cli import JupyterManager

        def map_args(model: str = None, embedding: str = None):
            model = 'wordvec' if model is None else model
            path = str(self._harness.root_dir / 'models' / f'{model}.conf')
            args = ['-c', path]
            if embedding is not None:
                args.extend(['--override', f'mr_default.name={embedding}'])
            return args

        class AppJupyterManager(JupyterManager):
            def _init_jupyter(self):
                import logging
                # turn off more logging so only the progress bar shows
                logging.getLogger('zensols.deeplearn.model.executor.status').\
                    setLevel(logging.WARNING)

        mng = AppJupyterManager(self._harness, cli_args_fn=map_args)
        if cuda_device_index is not None:
            # tell which GPU to use
            mng.config('gpu_torch_config', cuda_device_index=cuda_device_index)
        return mng
