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
from torch import Tensor
from zensols.config import Dictable
from zensols.persist import PersistableContainer
from zensols.deepnlp import FeatureToken

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
        if len(self) <= limit:
            trunced = self
        else:
            wp_toks = []
            tot = 1 if self.has_cls_sep else 0
            for p in self.pieces:
                tot += len(p.tokens)
                if tot > limit:
                    break
                wp_toks.append(p)
            if self.has_cls_sep:
                wp_toks.append(self.pieces[-1])
            trunced = self.__class__(tuple(wp_toks))
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

    def _format(self, obj: Any) -> str:
        if isinstance(obj, Tensor):
            return str(obj.shape)
        else:
            return super()._format(obj)

    @classmethod
    def get_input_ids(cls, tensor: Tensor) -> Tensor:
        return tensor[0]

    @classmethod
    def get_attention_mask(cls, tensor: Tensor) -> Tensor:
        return tensor[1]

    @property
    def input_ids(self) -> Tensor:
        """The token IDs as the output from the tokenizer."""
        return self.get_input_ids(self.tensor)

    @property
    def attention_mask(self) -> Tensor:
        """The attention mask (0/1s)."""
        return self.get_attention_mask(self.tensor)

    def params(self) -> Dict[str, Any]:
        dct = {}
        atts = 'input_ids attention_mask'
        for att in atts.split():
            dct[att] = getattr(self, att).unsqueeze(0)
        return dct

    def __str__(self) -> str:
        return self.piece_list.__str__()
