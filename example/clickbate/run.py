#!/usr/bin/env python

from zensols.cli import ConfigurationImporterCliHarness


def init():
    # reset random state for consistency before any other packages are
    # imported
    from zensols.deeplearn import TorchConfig
    TorchConfig.init()
    # initialize the NLP system
    from zensols import deepnlp
    deepnlp.init()


if (__name__ == '__main__'):
    init()
    ConfigurationImporterCliHarness(
        src_dir_name='cb',
        config_path='models/glove50.conf',
        proto_args={
            0: ['batch',
                '--clear',
                '--override=batch_stash.workers=1,batch_stash.batch_limit=1,batch_stash.batch_size=2'],
            1: 'batch',
            2: 'debug',
            3: 'traintest',
        }[1],
        proto_factory_kwargs={'reload_pattern': '^cb'},
        app_factory_class='zensols.deeplearn.cli.FacadeApplicationFactory',
    ).run()
