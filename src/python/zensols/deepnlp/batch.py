"""Domain objects to support natural language processing applications.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Any
from dataclasses import dataclass, field
import sys
from io import TextIOBase
from zensols.deeplearn import DeepLearnError
from zensols.persist import persisted
from zensols.nlp import (
    TokenContainer, FeatureDocument,
    TokenAnnotatedFeatureSentence, TokenAnnotatedFeatureDocument,
)
from zensols.deeplearn.batch import DataPoint


@dataclass
class TokenContainerDataPoint(DataPoint):
    """A convenience class that uses data, such as tokens, a sentence or a
    document (:class:`~zensols.nlp.container.TokenContainer`) as a data point.

    """
    container: TokenContainer = field()
    """The token cotainer used for this data point."""

    @property
    @persisted('_doc')
    def doc(self) -> FeatureDocument:
        """The container as a document.  If it is a sentence, it will create a
        document with the single sentence.

        """
        return self.container.to_document()

    def _get_token_labels(self) -> Tuple[Any, ...]:
        if isinstance(
                self.container,
                (TokenAnnotatedFeatureDocument, TokenAnnotatedFeatureSentence)):
            return self.container.annotations
        else:
            raise DeepLearnError(
                'Need instance of TokenAnnotatedFeature{Sentence,Document} ' +
                f'(got {type(self.sent)}) or override _get_token_labels')

    @property
    def token_labels(self) -> Tuple[Any, ...]:
        """The label that corresponds to each normalized token."""
        return self._get_token_labels()

    def __len__(self) -> int:
        """The number or normalized tokens in the container."""
        return self.container.token_len

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        super().write(depth, writer)
        self._write_line('container:', depth, writer)
        self.sent.write(depth + 1, writer)

    def __str__(self):
        return self.container.__str__()

    def __repr__(self):
        return self.conatiner.__repr__()
