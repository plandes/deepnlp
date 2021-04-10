#!/usr/bin/env python

from typing import List
import sys
from pathlib import Path


def main(args: List[str], **factory_kwargs):
    entry_path = Path(args[0])
    src_path = entry_path.parent / 'src'
    sys.path.append(str(src_path))
    from ner import CliFactory
    cli = CliFactory.instance(entry_path.parent, **factory_kwargs)
    cli.invoke(args[1:])


def proto():
    print('-->proto')
    try:
        run = 0
        reload = True
        {0: lambda: main('./run.py tmp'.split(), reload_factory=reload),
         1: lambda: main('./run.py debug'.split(), reload_factory=reload),
         2: lambda: main('./run.py batch'.split(), reload_factory=reload),
         3: lambda: main('./run.py train'.split(), reload_factory=reload),
         }[run]()
    except SystemExit as e:
        print(f'exit: {e}')


if (__name__ == '__main__'):
    from zensols.deeplearn import TorchConfig
    TorchConfig.init()
    if 0:
        import logging
        fmt = '%(asctime)-15s [%(name)s] %(message)s'
        logging.basicConfig(format=fmt, level=logging.INFO)
    # when running from a shell, run the CLI entry point
    import __main__ as mmod
    if hasattr(mmod, '__file__'):
        main(sys.argv)
    # otherwise, assume a Python REPL and run the prototyping method
    else:
        proto()
