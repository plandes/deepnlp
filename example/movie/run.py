#!/usr/bin/env python

from typing import List
import sys
from pathlib import Path


def main(args: List[str], **factory_kwargs):
    entry_path = Path(args[0])
    src_path = entry_path.parent / 'src'
    sys.path.append(str(src_path))
    from movie import CliFactory
    cli = CliFactory.instance(entry_path.parent, **factory_kwargs)
    cli.invoke(args[1:])


def tmp(args: List[str], **factory_kwargs):
    entry_path = Path(args[0])
    src_path = entry_path.parent / 'src'
    sys.path.append(str(src_path))
    from movie import CliFactory
    cli = CliFactory.instance(entry_path.parent, **factory_kwargs)
    facade = cli.create_facade()
    import itertools as it
    vec = facade.language_vectorizer_manager.vectorizers['transformer']
    for b in it.islice(facade.feature_stash.values(), 1):
        print(type(b))
        print(b)
        print(vec.encode(b))
    for b in it.islice(facade.batch_stash.values(), 2):
        print(b['label'].size(0), b['transformer_embedding'].size(0))


def proto():
    print('-->proto')
    try:
        args = './run.py debug --override models/transformer.conf'.split()
        #args = './run.py tmp'.split()
        #main(args, reload_factory=True)
        tmp(args, reload_factory=True)
    except SystemExit as e:
        print(f'exit: {e}')


if (__name__ == '__main__'):
    if 0:
        import logging
        fmt = '%(asctime)-15s [%(name)s] %(message)s'
        logging.basicConfig(format=fmt, level=logging.INFO)

    from zensols.deeplearn.batch import TorchMultiProcessStash
    TorchMultiProcessStash.init()

    # when running from a shell, run the CLI entry point
    import __main__ as mmod
    if hasattr(mmod, '__file__'):
        main(sys.argv)
    # otherwise, assume a Python REPL and run the prototyping method
    else:
        proto()
