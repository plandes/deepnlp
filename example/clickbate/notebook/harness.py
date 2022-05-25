# environemnt configuration and set up: add this (deepnlp) library to the
# Python path and framework entry point
class NotebookHarness(object):
    """Configure the Jupyter notebook environment and create model resources.

    """
    def __init__(self, app_root_dir: str = '..', deepnlp_path: str = '..'):
        """Set up the interpreter environment so we can import local packages.

        :param app_root_dir: the application root directory
        :param deepnlp_path: the path to the DeepNLP source code
        """
        import sys
        from pathlib import Path
        self.app_root_dir = Path(app_root_dir)
        # add the example to the Python library path
        sys.path.append(str(self.app_root_dir / 'cb'))
        # add the deepnlp path
        sys.path.append(deepnlp_path)
        # reset random state for consistency before any other packages are
        # imported
        from zensols.deeplearn import TorchConfig
        TorchConfig.init()
        # initialize the NLP system
        from zensols.deepnlp import init
        init()

    def __call__(self, cuda_device_index: int = None,
                 temporary_dir_name: str = None):
        """Create and return an instance a :class:`.JupyterManager`.

        :param cuda_device_index: the CUDA (GPU) device to use or ``None`` to
               use the default

        :param temporary_dir_name: the temporary directory to use for temporary
                                   space and results

        """
        from pathlib import Path
        from zensols.config import DictionaryConfig
        from zensols.deeplearn.cli import (
            JupyterManager, FacadeApplicationFactory
        )

        # we use the CliHarness in the entry point ``run.py`` script, so define
        # the glue code here since there isn't a module with this example
        class CliFactory(FacadeApplicationFactory):
            def __init__(self, root_dir: Path, temporary_dir: Path = None,
                         **kwargs):
                dconf = DictionaryConfig({'appenv': {'root_dir': str(root_dir)}})
                super().__init__(
                    package_resource='app',
                    app_config_resource=root_dir / 'resources/app.conf',
                    children_configs=(dconf,), **kwargs)

        factory_args = {'root_dir': self.app_root_dir}
        if temporary_dir_name is not None:
            factory_args['temporary_dir'] = self.app_root_dir / temporary_dir_name

        class NBJupyterManager(JupyterManager):
            def _init_jupyter(self):
                import logging
                # turn off more logging so only the progress bar shows
                logging.getLogger('zensols.deeplearn.model.executor.status').\
                    setLevel(logging.WARNING)

        mng = NBJupyterManager(
            allocation_tracking='counts',
            cli_class=CliFactory,
            factory_args=factory_args,
            cli_args_fn=lambda cnf, name: [
                '-c', str(self.app_root_dir / 'models' / f'{cnf}.conf'),
                '--override', f'clickbate_default.name={name}'
            ])
        if cuda_device_index is not None:
            # tell which GPU to use
            mng.config('gpu_torch_config', cuda_device_index=cuda_device_index)
        return mng
