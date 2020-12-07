"""Command line entrance point to the application.

"""
__author__ = 'plandes'

from typing import Type
from dataclasses import dataclass
from zensols.deeplearn.model import ModelFacade
from zensols.deeplearn.cli import (
    FacadeCli,
    EnvironmentVariable,
    EnvironmentFormatter,
    FacadeCommandLine,
)
from . import AppConfig, NERModelFacade


@dataclass
class NERFacadeCli(FacadeCli):
    def _create_environment_formatter(self) -> EnvironmentFormatter:
        return EnvironmentFormatter(
            self.config,
            (EnvironmentVariable('corpus_dir'),
             EnvironmentVariable('data_dir'),
             EnvironmentVariable('path', 'word2vec_300_embedding', 'w2v_path'),
             EnvironmentVariable('source_path', 'sent_factory_stash', 'connl_dir'),
             EnvironmentVariable('path', 'glove_50_embedding', 'glove_dir')))

    def _get_facade_class(self) -> Type[ModelFacade]:
        return NERModelFacade


class ConfAppCommandLine(FacadeCommandLine):
    def __init__(self):
        super().__init__(cli_class=NERFacadeCli, pkg_dist='ner',
                         config_type=AppConfig)
