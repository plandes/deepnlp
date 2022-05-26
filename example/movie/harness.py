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
        src_dir_name='src',
        proto_args='debug',
        proto_factory_kwargs={'reload_pattern': '^(?:app|dataset|domain)'},
    ).run()
