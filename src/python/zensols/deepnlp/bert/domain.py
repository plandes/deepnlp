from __future__ import annotations
"""Container classes for Bert models

"""
__author__ = 'Paul Landes'

from typing import Tuple, Dict, Any, List
from dataclasses import dataclass, field
import logging
import sys
from io import TextIOBase
from itertools import chain
import itertools as it
from torch import Tensor
from zensols.config import Dictable
from zensols.persist import PersistableContainer
from zensols.deepnlp import FeatureToken
#from zensols.deeplearn.vectorize import TensorFeatureContext

logger = logging.getLogger(__name__)


@dataclass
class WordPiece(PersistableContainer, Dictable):
    """The output of a word piece tokenization algorithm for whole token parsed
    (i.e. by spaCy's tokenizer).  A word/token can be broken up in to several
    word pieces.  For this reason, the number for tokens is never greater than
    word pieces.

    """
    WRITABLE__DESCENDANTS = True

    tokens: List[str] = field()
    """The word piece tokens for this word piece."""

    feature: FeatureToken = field()
    """The spaCy parsed token features."""

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'tokens: {self.tokens}', depth, writer)
        if self.feature is not None:
            self._write_line('feature:', depth, writer)
            self._write_object(self.feature, depth + 1, writer)

    def __len__(self) -> int:
        return len(self.tokens)

    def __str__(self) -> str:
        return f'{self.tokens}: {self.feature}'


@dataclass
class WordPieceSentence(PersistableContainer, Dictable):
    """A sentence made up of word piece tokens.

    """
    pieces: Tuple[WordPiece] = field()
    """The tokenized data."""

    def __post_init__(self):
        # Whether or not the [CLS] and [SEP] tokens exist.
        self.has_cls_sep = self.pieces[0].tokens[0] == 'CLS'

    def truncate(self, limit: int) -> WordPieceSentence:
        """Return a truncated version of this sentence.

        :param limit: the max number of word pieces (not to be confused with
                      non-word piece tokens) to keep

        :return: an instance of a sentence with no more than ``limit`` word
                 pieces

        """
        trunced = WordPieceSentence
        if limit >= len(self.pieces):
            trunced = self
        else:
            if self.has_cls_sep:
                limit = limit - 1
            toks = list(it.islice(self.pieces, limit))
            if self.has_cls_sep:
                toks.append(self.pieces[-1])
            trunced = self.__class__(tuple(toks))
        return trunced

    @property
    def word_piece_tokens(self) -> Tuple[str]:
        """Return word piece token strings."""
        return tuple(chain.from_iterable(map(lambda p: p.tokens, self.pieces)))

    def __len__(self) -> int:
        """The length of this sentence in word pieces."""
        return sum(map(len, self.pieces))

    def __str__(self):
        return '|'.join(chain.from_iterable(
            map(lambda w: w.tokens, self.pieces)))


@dataclass
class Tokenization(PersistableContainer, Dictable):
    """The output of the model tokenization.

    """
    WRITABLE__DESCENDANTS = True

    piece_list: WordPieceSentence = field()
    """The transformer tokens paired with features."""

    tensor: Tensor = field()
    """The vectorized tokenized data."""

    # def __post_init__(self):
    #     if self.piece_list is not None:
    #         if logger.isEnabledFor(logging.DEBUG):
    #             logger.debug(f'tokens: {len(self.piece_list)}, ' +
    #                          f'shape: {self.input_ids.shape}')
    #         assert len(self.piece_list) == self.input_ids.size(1)

    @property
    def input_ids(self) -> Tensor:
        """The token IDs as the output from the tokenizer."""
        return self.tensor[0]

    @property
    def attention_mask(self) -> Tensor:
        """The attention mask (0/1s)."""
        return self.tensor[1]

    @property
    def position_ids(self) -> Tensor:
        """The position IDs (only given for Bert currently for huggingface bug.

        :see: `HF Issue <https://github.com/huggingface/transformers/issues/2952>`_

        """
        return self.tensor[2]

    def params(self) -> Dict[str, Any]:
        dct = {}
        atts = 'input_ids attention_mask'
        if self.tensor.size(0) >= 3:
            atts += ' position_ids'
        for att in atts.split():
            dct[att] = getattr(self, att).unsqueeze(1)
        return dct

    def __str__(self) -> str:
        return self.piece_list.__str__()
