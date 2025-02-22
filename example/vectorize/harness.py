#!/usr/bin/env python

from io import StringIO
from zensols.cli import CliHarness


CONFIG = """
[cli]
class_name = zensols.cli.ActionCliManager
apps = list: cleaner_cli, app

[import]
sections = list: imp_conf, imp_obj, imp_conf

[imp_conf]
config_files = list: parser.conf

[imp_obj]
type = importini
config_files = list:
  resource(zensols.util): resources/escape.conf,
  resource(zensols.deeplearn): resources/default.conf,
  resource(zensols.deepnlp): resources/default.conf,
  resource(zensols.deeplearn): resources/obj.conf,
  resource(zensols.nlp): resources/obj.conf,
  resource(zensols.deepnlp): resources/default.conf,
  resource(zensols.deepnlp): resources/obj.conf,
  resource(zensols.deepnlp): resources/bigbird.conf

[cleaner_cli]
class_name = zensols.cli.Cleaner
paths = eval: (('*.log', '**/__pycache__'),)

[app]
class_name = app.Application
vec_mng = instance: language_vectorizer_manager
"""


def init():
    # reset random state for consistency before any other packages are
    # imported
    from zensols.deeplearn import TorchConfig
    TorchConfig.init()
    # initialize the NLP system
    from zensols import deepnlp
    deepnlp.init()


if (__name__ == '__main__'):
    init()
    rl_mods = ['zensols.deepnlp.transformer',
               'zensols.deepnlp.vectorize.layer',
               'app']
    reload_pattern = f'^(?:{"|".join(rl_mods)})'
    CliHarness(
        app_config_resource=StringIO(CONFIG),
        proto_args='proto',
        proto_factory_kwargs={'reload_pattern': reload_pattern},
        app_factory_class='zensols.deeplearn.cli.FacadeApplicationFactory',
    ).run()
