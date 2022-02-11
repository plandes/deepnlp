#!/usr/bin/env python

from typing import List
import sys
from pathlib import Path
import logging
from zensols.deeplearn import TorchConfig
from zensols import deepnlp


# reset random state for consistency before any other packages are
# imported
TorchConfig.init()
# initialize the NLP system
deepnlp.init()


class CliHarness(object):
    """A utility class to automate the creation of execution of the model from
    either the command line or a Python REPL.

    """
    def __init__(self, args: List[str] = sys.argv, src_dir_name: str = 'src'):
        """Configure the Python interpreter and this run class.

        :param args: the command line arguments

        :param src_dir_name: the directory add the Python path containing the
                             source for the application

        """
        self.args = args[1:]
        self.entry_path = Path(args[0])
        src_path = self.entry_path.parent / src_dir_name
        sys.path.append(str(src_path))

    @classmethod
    def run(cls):
        """The command line script entry point."""
        self = cls()
        if (__name__ == '__main__'):
            # when running from a shell, run the CLI entry point
            import __main__ as mmod
            if hasattr(mmod, '__file__'):
                self.invoke()
            # otherwise, assume a Python REPL and run the prototyping method
            else:
                self.proto()

    def configure_logging(self):
        """Configure the logging package system."""
        fmt = '%(asctime)-15s [%(name)s] %(message)s'
        logging.basicConfig(format=fmt, level=logging.INFO)

    def invoke(self, args: List[str] = None, **factory_kwargs):
        """Invoke the application.

        :param args: the command line arguments without the first argument (the
                     program name)

        :param factory_kwargs: arguments given to the command line factory

        """
        from movie import CliFactory
        cli = CliFactory(self.entry_path.parent, **factory_kwargs)
        return cli.invoke(self.args if args is None else args)

    def proto(self):
        """Called from the Python REPL to evaluate."""
        print('-->proto')
        #self.configure_logging()
        #logging.getLogger('zensols.config.importini').setLevel(logging.INFO)
        #logging.getLogger('zensols.config.iniconfig').setLevel(logging.INFO)
        try:
            model = {0: 'glove50',
                     1: 'transformer-fixed',
                     2: 'transformer-trainable'
                     }[0]
            action = {0: 'proto',
                      1: 'debug',
                      2: 'batch',
                      3: 'predictions',
                      4: 'info -i conf',
                      }[4]
            args = f'-c models/{model}.conf'
            # other reload patterns read from app.conf
            rl_mods = 'movie.app'.split()
            self.invoke(f'{action} {args}'.split(),
                        reload_pattern=f'^(?:{"|".join(rl_mods)})'),
        except SystemExit as e:
            print(f'Prevented exit: {e}')


CliHarness.run()
