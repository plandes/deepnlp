#!/usr/bin/env python

from zensols.cli import ApplicationFactory
from io import StringIO


CONFIG = """
[cli]
class_name = zensols.cli.ActionCliManager
apps = list: app

[import]
files = parser.conf

[app]
class_name = app.Application
vec_mng = instance: language_feature_manager
"""


def silencewarn():
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.simplefilter("ignore", ResourceWarning)


def main():
    print()
    silencewarn()
    rl_mods = 'app zensols.deepnlp.transformer zensols.deepnlp.vectorize.layer'.split()
    cli = ApplicationFactory(
        'nlparse', StringIO(CONFIG),
        reload_pattern=f'^(?:{"|".join(rl_mods)})')
    import __main__ as mmod
    if hasattr(mmod, '__file__'):
        cli.invoke()
    else:
        cli.invoke_protect('go'.split())


if __name__ == '__main__':
    main()


main()
