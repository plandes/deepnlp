from __future__ import annotations
"""Container classes for Bert models

"""
__author__ = 'Paul Landes'

from typing import List, Tuple, Dict, Any, Union, Iterable, Callable
from dataclasses import dataclass, field
import sys
import logging
import itertools as it
from io import TextIOBase
import torch
from torch import Tensor
from zensols.nlp import FeatureDocument
from zensols.persist import PersistableContainer
from zensols.config import Writable
from zensols.nlp import FeatureToken, FeatureSentence

logger = logging.getLogger(__name__)


@dataclass
class TokenizedDocument(PersistableContainer):
    """This is the tokenized document output of
    :class:`.TransformerDocumentTokenizer`.  Instances of this class are
    pickelable, in a feature context.  Then give to the in the decoding phase
    to create a tensor with a transformer model such as
    :class:`.TransformerEmbedding`.

    """

    tensor: Tensor = field()
    """Encodes the input IDs, attention mask, and word piece offset map."""

    boundary_tokens: bool = field()
    """If the token document has sentence boundary tokens, such as ``[CLS]`` for
    Bert.

    """

    def __post_init__(self):
        super().__init__()

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> TokenizedDocument:
        """Create an instance of the class using a tensor.  This is useful for
        re-creating documents for mapping with :meth:`.map_word_pieces` after
        unpickling from a document created with
        :class:`.TransformerDocumentTokenizer.tokenize`.

        :param tensor: the tensor to set in :obj:`.tensor`

        """
        return cls(tensor, None)

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
        """Return the size of the document in number of word pieces."""
        return self.tensor.size(-1)

    def detach(self) -> TokenizedDocument:
        """Return a version of the document that is pickleable."""
        return self

    def params(self) -> Dict[str, Any]:
        dct = {}
        atts = 'input_ids attention_mask'
        for att in atts.split():
            dct[att] = getattr(self, att)
        return dct

    @staticmethod
    def map_word_pieces(token_offsets: List[int]) -> \
            List[Tuple[FeatureToken, List[int]]]:
        """Map word piece tokens to linguistic tokens.

        :return:

            a list of tuples in the form:

            ``(<linguistic token|token index>, <list of word piece indexes>)``

            if detatched, the linguistic token is an index as a tensor scalar

        """
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

    def map_to_word_pieces(self, sentences: Iterable[List[Any]] = None,
                           map_wp: Union[Callable, Dict[int, str]] = None) -> \
            List[Dict[str, Union[List[Any], Tuple[FeatureToken, Tuple[str]]]]]:
        """Map word piece tokens to linguistic tokens.

        :return: a list sentence maps, each with:

                   * ``sent`` -> the ``i``th list in ``sentences``

                   * ``map`` -> list of ``(sentence 'token', word pieces)``

        """
        def map_wp_by_id(x: int, six: int, input_ids: List[int]):
            tix = input_ids[x]
            tok = id2tok[tix]
            return tok

        id2tok = None
        input_ids = self.input_ids.cpu().numpy()
        sent_offsets = self.offsets
        sents_map = []
        if map_wp is None:
            def map_str(x, *args, **kwargs):
                return str(x)

            map_wp = map_str
        elif isinstance(map_wp, dict):
            id2tok = map_wp
            map_wp = map_wp_by_id
        if sentences is None:
            sentences = self.input_ids.cpu().numpy()
        sents = enumerate(zip(sentences, sent_offsets))
        for six, (sent, tok_offsets) in sents:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'sent idx: {six}, sent: {sent}, ' +
                             f'offsets: {tok_offsets}')
            input_sent = input_ids[six]
            wps = self.map_word_pieces(tok_offsets)
            sent_map = []
            sents_map.append({'sent': sent, 'map': sent_map})
            for tix, ixs in wps:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'{ixs} -> {tix}')
                tok = sent[tix]
                ttoks = tuple(map(lambda i: map_wp(i[0], six, input_sent),
                                  zip(ixs, it.count())))
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'{tok} -> {ttoks}')
                sent_map.append((tok, ttoks))
        return sents_map

    def deallocate(self):
        super().deallocate()
        del self.tensor

    def __str__(self) -> str:
        return f'doc: {self.tensor.shape}'

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class TokenizedFeatureDocument(TokenizedDocument, Writable):
    """Instance of this class are created, then a picklable version returned with
    :meth:`detach` as an instance of the super class.

    """
    feature: FeatureDocument = field()
    """The document to tokenize."""

    id2tok: Dict[int, str] = field()
    """If provided, a mapping of indexes to transformer tokens.  This attribute is
    always nulled out after being persisted.

    """

    char_offsets: Tuple[Tuple[int, int]] = field()
    """The valid character offsets for each word piece token."""

    def detach(self) -> TokenizedDocument:
        return TokenizedDocument(self.tensor, self.boundary_tokens)

    def map_word_pieces_to_tokens(self) -> \
            List[Dict[str, Union[FeatureSentence,
                                 Tuple[FeatureToken, Tuple[str]]]]]:
        """Map word piece tokens to linguistic tokens.

        :return: a list sentence maps, each with:

                   * ``sent`` -> :class:`.FeatureSentence`

                   * ``map`` -> list of ``(token, word pieces)``

        """
        def id2tok(x: int, six: int, input_ids: List[int]):
            tix = input_ids[x]
            tok = self.id2tok[tix]
            off = self.char_offsets[six][x]
            olen = off[1] - off[0]
            if len(tok) > olen:
                # bert
                if tok.startswith('##'):
                    start = 2
                # roberta
                else:
                    start = 1
            else:
                start = 0
            end = (off[1] - off[0]) + (start * 2)
            tok = tok[start:end]
            return tok

        return super().map_to_word_pieces(self.feature, id2tok)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        sent_map: Dict[str, Union[FeatureSentence,
                                  Tuple[FeatureToken, Tuple[str]]]]
        for sent_map in self.map_word_pieces_to_tokens():
            sent: FeatureSentence = sent_map['sent']
            tmap: Tuple[FeatureToken, Tuple[str]] = sent_map['map']
            self._write_line(f'sentence: {sent}', depth, writer)
            tok: FeatureToken
            ttoks: Tuple[str]
            for tok, ttoks in tmap:
                self._write_line(f'{tok.text} -> {ttoks}', depth + 1, writer)
