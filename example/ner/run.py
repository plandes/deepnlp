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
        model = {0: 'glove50',
                 1: 'transformer-trainable',
                 }[0]
        args = f'-c models/{model}.conf'
        #args += ' --execlevel 2'
        # other reload patterns read from app.conf
        rl_mods = 'ner.app'.split()
        action = {0: 'proto',
                  1: 'debug',
                  2: 'all',
                  3: 'batch',
                  4: 'train',
                  }[0]
        main(f'./run.py {action} {args}'.split(),
             reload_pattern=f'^(?:{"|".join(rl_mods)})'),
    except SystemExit as e:
        print(f'exit: {e}')


if (__name__ == '__main__'):
    from zensols.deepnlp import init
    init()
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
