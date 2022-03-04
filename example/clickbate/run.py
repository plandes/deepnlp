#!/usr/bin/env python

from zensols.cli import CliHarness


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
    CliHarness(
        app_config_resource='resources/app.conf',
        proto_args='-c models/transformer.conf traintest',
        proto_factory_kwargs={'reload_pattern': '^cb'},
        app_factory_class='zensols.deeplearn.cli.FacadeApplicationFactory',
    ).run()
