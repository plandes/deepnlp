#!/usr/bin/env python


if (__name__ == '__main__'):
    # initialize the NLP system
    from zensols import deepnlp
    deepnlp.init()
    from zensols.cli import CliHarness
    CliHarness(
        proto_args='debug',
        proto_factory_kwargs={'reload_pattern': '^corpus'},
    ).run()
