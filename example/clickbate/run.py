#!/usr/bin/env python

from zensols.cli import CliHarness
from zensols.deeplearn import TorchConfig
from zensols import deepnlp


# reset random state for consistency before any other packages are
# imported
TorchConfig.init()
# initialize the NLP system
deepnlp.init()


CliHarness(
    app_config_resource='resources/app.conf',
    proto_args='-c models/transformer.conf traintest',
    proto_factory_kwargs={'reload_pattern': '^cb'},
    app_factory_class='zensols.deeplearn.cli.FacadeApplicationFactory',
).run()
