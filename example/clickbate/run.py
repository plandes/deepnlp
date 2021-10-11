#!/usr/bin/env python

from zensols.cli import CliHarness
from zensols.deeplearn.cli import FacadeApplicationFactory


if (__name__ == '__main__'):
    from zensols import deepnlp
    deepnlp.init()


CliHarness(
    app_config_resource='resources/app.conf',
    proto_args='-c models/transformer.conf traintest',
    proto_factory_kwargs={'reload_pattern': '^cb'},
    app_factory_class=FacadeApplicationFactory,
).run()
