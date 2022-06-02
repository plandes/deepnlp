#!/usr/bin/env python

from zensols import deepnlp

# initialize the NLP system
deepnlp.init()


if (__name__ == '__main__'):
    from zensols.cli import CliHarness
    CliHarness(
        package_resource='sp',
        proto_args='debug --override sp_default.name=transformer_trainable',
        proto_factory_kwargs={'reload_pattern': '^sp'},
    ).run()
