from __future__ import annotations
"""Container classes for Bert models

"""
__author__ = 'Paul Landes'

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field
import sys
import logging
from io import TextIOBase
import torch
from torch import Tensor
from zensols.deepnlp import FeatureDocument
from zensols.persist import PersistableContainer
from zensols.config import Writable
from zensols.deepnlp import FeatureToken

logger = logging.getLogger(__name__)


@dataclass
class TokenizedDocument(PersistableContainer):
    tensor: Tensor = field()
    """Encodes the input IDs, attention mask, and word piece offset map."""

    def __post_init__(self):
        super().__init__()

    @property
    def input_ids(self) -> Tensor:
        """The token IDs as the output from the tokenizer."""
        return self.tensor[0]

    @property
    def attention_mask(self) -> Tensor:
        """The attention mask (0/1s)."""
        return self.tensor[1]

    @property
    def offsets(self) -> Tensor:
        """The offsets from word piece (transformer's tokenizer) to feature document
        index mapping.

        """
        return self.tensor[2]

    @property
    def shape(self) -> torch.Size:
        """Return the shape of the vectorized document."""
        return self.tensor.shape

    def __len__(self) -> int:
        return self.tensor.size(-1)

    def detach(self) -> TokenizedDocument:
        """Return a version of the document that is pickleable.
        """
        return self

    def params(self) -> Dict[str, Any]:
        dct = {}
        atts = 'input_ids attention_mask'
        for att in atts.split():
            dct[att] = getattr(self, att)
        return dct

    def map_word_pieces(self, token_offsets: List[int]) -> \
            List[Tuple[FeatureToken, List[int]]]:
        ftoks = []
        n_ftok = -1
        for wix, tix in enumerate(token_offsets):
            if tix >= 0:
                if tix > n_ftok:
                    wptoks = []
                    ftoks.append((tix, wptoks))
                    n_ftok += 1
                wptoks.append(wix)
        return ftoks

    def deallocate(self):
        super().deallocate()
        del self.tensor

    def __str__(self) -> str:
        return f'doc: {self.tensor.shape}'


@dataclass
class TokenizedFeatureDocument(TokenizedDocument, Writable):
    """This is the tokenized document output of
    :class:`.TransformerDocumentTokenizer`.

    """
    feature: FeatureDocument = field()
    """The document to tokenize."""

    id2tok: Dict[int, str] = field()
    """If provided, a mapping of indexes to transformer tokens.  This attribute is
    always nulled out after being persisted.

    """

    def detach(self) -> TokenizedDocument:
        return TokenizedDocument(self.tensor)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        if self.id2tok is not None:
            def id2tok(x):
                return self.id2tok[x]
        else:
            def id2tok(x):
                return str(x)
        input_ids = self.input_ids.cpu().numpy()
        sent_offsets = self.offsets
        doc = self.feature
        for six, (sent, tok_offsets) in enumerate(zip(doc, sent_offsets)):
            input_sent = input_ids[six]
            wps = self.map_word_pieces(sent_offsets[six])
            self._write_line(f'sentence: {sent}', depth, writer)
            for tix, ixs in wps:
                tok = sent[tix]
                ttoks = '|'.join(map(lambda i: id2tok(input_sent[i]), ixs))
                self._write_line(f'{tok.text} -> {ttoks}', depth + 1, writer)
