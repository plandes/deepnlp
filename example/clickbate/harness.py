#!/usr/bin/env python

from zensols.cli import ConfigurationImporterCliHarness


if (__name__ == '__main__'):
    # initialize the NLP system
    from zensols import deepnlp
    deepnlp.init()
    ConfigurationImporterCliHarness(
        proto_args='debug',
        proto_factory_kwargs={'reload_pattern': '^corpus'},
    ).run()
