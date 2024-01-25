"""Container classes for Bert models

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import List, Tuple, Dict, Any, Set, Union, Iterable, Callable
from dataclasses import dataclass, field
import sys
import logging
from io import TextIOBase
import numpy as np
import torch
from torch import Tensor
from zensols.nlp import FeatureDocument
from zensols.persist import PersistableContainer
from zensols.config import Writable
from zensols.nlp import FeatureToken, FeatureSentence
from zensols.deeplearn import DeepLearnError

logger = logging.getLogger(__name__)


@dataclass
class TokenizedDocument(PersistableContainer, Writable):
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
        """The offsets from word piece (transformer's tokenizer) to feature
        document index mapping.

        """
        return self.tensor[2]

    @property
    def token_type_ids(self) -> Tensor:
        """The token type IDs (0/1s)."""
        if self.tensor.size(0) > 3:
            return self.tensor[3]

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

    def truncate(self, size: int) -> TokenizedDocument:
        """Truncate the the last (token) dimension to ``size``.

        :return: a new instance of this class truncated to size

        """
        cls = self.__class__
        return cls(self.tensor[:, :, 0:size], self.boundary_tokens)

    def params(self) -> Dict[str, Any]:
        dct = {}
        atts = 'input_ids attention_mask token_type_ids'
        for att in atts.split():
            val = getattr(self, att)
            if val is not None:
                dct[att] = val
        return dct

    @staticmethod
    def map_word_pieces(token_offsets: List[int]) -> \
            List[Tuple[FeatureToken, List[int]]]:
        """Map word piece tokens to linguistic tokens.

        :return:

            a list of tuples in the form:

            ``(<token index>, <list of word piece indexes>)``

        """
        ftoks = []
        n_ftok = -1
        wix: int
        tix: int
        for wix, tix in enumerate(token_offsets):
            if tix >= 0:
                if tix > n_ftok:
                    wptoks = []
                    ftoks.append((tix, wptoks))
                    n_ftok += 1
                wptoks.append(wix)
        return ftoks

    def map_to_word_pieces(self, sentences: Iterable[List[Any]] = None,
                           map_wp: Union[Callable, Dict[int, str]] = None,
                           add_indices: bool = False,
                           index_tokens: bool = True) -> \
            List[Dict[str, Dict[str, Any]]]:
        """Map word piece tokens to linguistic tokens.

        :param sentences: an iteration of sentences, which is returned in the
                          output (i.e. :class:`~zensols.nlp.FeatureSentence`),
                          or :obj:`input_ids` if ``None``

        :param map_wp: either a function that takes the token index, sentence ID
                       and input IDs, or the mapping from word piece ID to
                       string token; return output is the string token (or
                       numerical output if no mapping is provided)

        :param add_indices: whether to add the token ID and index after the
                            token string when ``id2tok`` is provided for
                            ``map_wp``

        :return:

            a list sentence maps, each with:

                * ``sent_ix`` -> the ``i``th list in ``sentences``

                * ``sent`` -> a :class:`~zensols.nlp.container.FeatureSentence`
                  or a tensor of vocab indexes if ``map_wp`` is ``None``

                * ``map`` -> list of ``(sentence 'token', word pieces)``

        """
        def map_str(x: int, six: int, input_ids: List[int]):
            return str(x)

        def map_wp_by_id(x: int, six: int, input_ids: List[int]):
            tix = input_ids[x]
            tok = id2tok[tix]
            if tok.startswith('##'):
                tok = tok[2:]
            return (tok, tix, x) if add_indices else tok

        id2tok: Dict[int, str] = None
        input_ids: np.ndarray = self.input_ids.cpu().numpy()
        sent_offsets: List[int] = self.offsets.numpy().tolist()
        sents_map: List[Dict[str, Union[List[Any], Tuple]]] = []
        if map_wp is None:
            map_wp = map_str
        elif isinstance(map_wp, dict):
            id2tok = map_wp
            map_wp = map_wp_by_id
        if sentences is None:
            sentences = self.input_ids.cpu().numpy()
        assert len(sentences) == len(sent_offsets)
        sents = enumerate(zip(sentences, sent_offsets))
        six: int
        sent: Union[FeatureSentence, Tensor]
        for six, (sent, tok_offsets) in sents:
            input_sent: np.ndarray = input_ids[six]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'sent idx: {six}, sent: {sent}')
                logger.debug(f'offsets: {tok_offsets}')
                logger.debug(f'ids: {input_sent}')
            sent_map: Dict[str, Dict[str, Any]] = []
            if index_tokens:
                wps: List[Tuple[Tensor, List[int]]] = \
                    self.map_word_pieces(tok_offsets)
                tix: int
                for tix, ixs in wps:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'idx: {ixs} -> {tix}')
                    tok: str = sent[tix]
                    toks = tuple(map(lambda i: map_wp(i, six, input_sent), ixs))
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'tok: {tok} -> {toks}')
                    sent_map.append((tok, toks))
            else:
                stoks: List[FeatureToken] = []
                toks: List[List[str]] = []
                tix: int = 0
                wps: List[str]
                for i, (ix, off) in enumerate(zip(input_sent, tok_offsets)):
                    if off == -1:
                        toks.append([map_wp(i, six, input_sent)])
                        tix += 1
                    elif off >= 0:
                        lix = off + tix
                        if lix > len(toks) - 1:
                            wps = []
                            toks.append(wps)
                        wps = toks[lix]
                        wps.append(map_wp(i, six, input_sent))
                    else:
                        raise DeepLearnError(f'Unknown offset index: {off}')
                for i, wps in enumerate(toks):
                    tix: int = len(stoks)
                    tok: str = ''.join(wps)
                    sent_map.append((tok, tuple(wps)))
                    stoks.append(FeatureToken(i, i, six, tok))
                sent = FeatureSentence(tuple(stoks))
            sents_map.append({'sent_ix': six, 'sent': sent, 'map': sent_map})
        return sents_map

    def deallocate(self):
        super().deallocate()
        del self.tensor

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_tokens: bool = True, id2tok: Dict[int, str] = None):
        for sent_map in self.map_to_word_pieces(
                map_wp=id2tok, index_tokens=id2tok is None):
            self._write_line(f"sentence: {sent_map['sent']}", depth, writer)
            if include_tokens:
                self._write_line('tokens:', depth, writer)
                tok: str
                wps: Tuple[str]
                for tok, wps in sent_map['map']:
                    self._write_line(f'{tok} -> {wps}', depth + 1, writer)

    def __str__(self) -> str:
        return f'doc: {self.tensor.shape}'

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class TokenizedFeatureDocument(TokenizedDocument):
    """Instance of this class are created, then a picklable version returned
    with :meth:`detach` as an instance of the super class.

    """
    feature: FeatureDocument = field()
    """The document to tokenize."""

    id2tok: Dict[int, str] = field()
    """If provided, a mapping of indexes to transformer tokens.  This attribute
    is always nulled out after being persisted.

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
                if tok.startswith('##'):
                    # bert
                    start = 2
                else:
                    # roberta
                    start = 1
            else:
                start = 0
            end = (off[1] - off[0]) + (start * 2)
            tok = tok[start:end]
            return tok

        return super().map_to_word_pieces(self.feature, id2tok)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_tokens: bool = True, id2tok: Dict[int, str] = None):
        id2tok = self.id2tok if id2tok is None else id2tok
        sent_map: Dict[str, Union[FeatureSentence,
                                  Tuple[FeatureToken, Tuple[str]]]]
        for sent_map in self.map_word_pieces_to_tokens():
            sent: FeatureSentence = sent_map['sent']
            tmap: Tuple[FeatureToken, Tuple[str]] = sent_map['map']
            self._write_line(f'sentence: {sent}', depth, writer)
            if include_tokens:
                self._write_line('tokens:', depth, writer)
                tok: FeatureToken
                ttoks: Tuple[str]
                for tok, ttoks in tmap:
                    stext = tok.text.replace('\n', '\\n')
                    stext = f'<{stext}>'
                    self._write_line(f'{stext} -> {ttoks}', depth + 1, writer)
