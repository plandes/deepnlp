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
from . import AppConfig, ReviewModelFacade


@dataclass
class ReviewFacadeCli(FacadeCli):
    def _create_environment_formatter(self) -> EnvironmentFormatter:
        return EnvironmentFormatter(
            self.config,
            (EnvironmentVariable('corpus_dir'),
             EnvironmentVariable('data_dir'),
             EnvironmentVariable('path', 'glove_50_embedding', 'glove_dir'),
             EnvironmentVariable('stanford_path', 'dataset_factory', 'stanford_dir'),
             EnvironmentVariable('path', 'word2vec_300_embedding', 'w2v_path'),
             EnvironmentVariable('rt_pol_path', 'dataset_factory', 'cornell_dir')))

    def _get_facade_class(self) -> Type[ModelFacade]:
        return ReviewModelFacade


class ConfAppCommandLine(FacadeCommandLine):
    def __init__(self):
        super().__init__(cli_class=ReviewFacadeCli, config_type=AppConfig)


def main():
    cl = ConfAppCommandLine()
    cl.invoke()
