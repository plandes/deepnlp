#!/usr/bin/env python

from zensols import deepnlp

# initialize the NLP system
deepnlp.init()


if (__name__ == '__main__'):
    from zensols.cli import CliHarness
    CliHarness(
        package_resource='cb',
        proto_args='debug',
        proto_factory_kwargs={'reload_pattern': '^cb'},
    ).run()
