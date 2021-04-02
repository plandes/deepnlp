#!/usr/bin/env python

from typing import List
import sys
from pathlib import Path


def main(args: List[str], **factory_kwargs):
    entry_path = Path(args[0])
    src_path = entry_path.parent / 'src'
    sys.path.append(str(src_path))
    from bner import CliFactory
    cli = CliFactory.instance(entry_path.parent, **factory_kwargs)
    cli.invoke(args[1:])


def proto():
    print('-->proto')
    try:
        run = 0
        {0: lambda: main('./run.py tmp'.split(), reload_factory=True),
         1: lambda: main('./run.py debug'.split(), reload_factory=True),
         2: lambda: main('./run.py train'.split(), reload_factory=True),
         }[run]()
    except SystemExit as e:
        print(f'exit: {e}')


if (__name__ == '__main__'):
    if 0:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    # when running from a shell, run the CLI entry point
    import __main__ as mmod
    if hasattr(mmod, '__file__'):
        main(sys.argv)
    # otherwise, assume a Python REPL and run the prototyping method
    else:
        proto()
