#!/usr/bin/env python

from pathlib import Path
from zensols.cli import CliHarness


if (__name__ == '__main__'):
    CliHarness(
        app_config_resource=Path('app.conf'),
        proto_args='traintest',
        proto_factory_kwargs={'reload_pattern': '^app'},
    ).run()
