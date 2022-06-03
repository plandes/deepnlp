#!/usr/bin/env python

from zensols import deepnlp

# initialize the NLP system
deepnlp.init()


if (__name__ == '__main__'):
    from zensols.cli import CliHarness
    CliHarness(
        package_resource='ner',
        #proto_args='proto',
        #proto_args='info -i config',
        proto_args={0: 'proto',
                    1: 'info -i config',
                    2: 'debug'
                    }[2],
        proto_factory_kwargs={'reload_pattern': '^ner'},
    ).run()
