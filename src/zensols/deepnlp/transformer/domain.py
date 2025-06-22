"""Container classes for Bert models

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import (
    List, Tuple, Dict, Any, Set, Union, Iterable, Callable, ClassVar
)
from dataclasses import dataclass, field
import sys
import logging
import itertools as it
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
    _INPUT_ID_IX: ClassVar[int] = 0
    _ATTENTION_MASK_IX: ClassVar[int] = 1
    _OFFSETS_IX: ClassVar[int] = 2
    _TOKEN_TYPE_ID_IX: ClassVar[int] = 3

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
    def is_empty(self) -> bool:
        """Whether the document in this instance has no sentences."""
        return self.tensor.size(-1) == 0

    @property
    def input_ids(self) -> Tensor:
        """The token IDs as the output from the tokenizer."""
        return self.tensor[self._INPUT_ID_IX]

    @property
    def attention_mask(self) -> Tensor:
        """The attention mask (0/1s)."""
        return self.tensor[self._ATTENTION_MASK_IX]

    @property
    def offsets(self) -> Tensor:
        """The offsets from word piece (transformer's tokenizer) to feature
        document index mapping.

        """
        return self.tensor[self._OFFSETS_IX]

    @property
    def token_type_ids(self) -> Tensor:
        """The token type IDs (0/1s)."""
        if self.tensor.size(0) > 3:
            return self.tensor[self._TOKEN_TYPE_ID_IX]

    @property
    def shape(self) -> torch.Size:
        """Return the shape of the vectorized document."""
        return self.tensor.shape

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

    def _map_sent(self, index_tokens: bool, tok_offsets: List[int],
                  map_wp: Callable, sent: Union[FeatureSentence, Tensor],
                  six: int, input_sent: np.ndarray, includes: Set[str],
                  mask_sent: np.ndarray, special_tokens: Set[str]):
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
            stoks: List[FeatureToken] = [] if 'sent' in includes else None
            toks: List[List[str]] = []
            tix: int = 0
            wps: List[str]
            i: int
            for i, (mask, off) in enumerate(zip(mask_sent, tok_offsets)):
                if mask == 0:
                    continue
                wtok: str = map_wp(i, six, input_sent)
                if wtok in special_tokens:
                    continue
                if off == -1:
                    toks.append([wtok])
                    tix += 1
                elif off >= 0:
                    lix = off + tix
                    if lix > len(toks) - 1:
                        wps = []
                        toks.append(wps)
                    wps = toks[lix]
                    wps.append(wtok)
                else:
                    raise DeepLearnError(f'Unknown offset index: {off}')
            for i, wps in enumerate(toks):
                tok: str = ''.join(map(str, wps))
                sent_map.append((tok, tuple(wps)))
                if stoks is not None:
                    stoks.append(FeatureToken(i, i, six, tok))
            if stoks is not None:
                sent = FeatureSentence(tuple(stoks))
        return sent_map, sent

    def map_to_word_pieces(self, sentences: Iterable[List[Any]] = None,
                           map_wp: Union[Callable, Dict[int, str]] = None,
                           add_indices: bool = False,
                           special_tokens: Set[str] = None,
                           index_tokens: bool = True,
                           includes: Set[str] = frozenset({'map'})) -> \
            List[Dict[str, Any]]:
        """Map word piece tokens to linguistic tokens.

        :param sentences: an iteration of sentences, which is returned in the
                          output (i.e. :class:`~zensols.nlp.FeatureSentence`),
                          or :obj:`input_ids` if ``None``

        :param map_wp: either a function that takes the token index, sentence ID
                       and input IDs, or the mapping from word piece ID to
                       string token; return output is the string token (or
                       numerical output if no mapping is provided); if an
                       instance of :class:`.TransformerDocumentTokenizer`, its
                       vocabulary and special tokens are utilized for mapping
                       and special token consideration

        :param add_indices: whether to add the token ID and index after the
                            token string when ``id2tok`` is provided for
                            ``map_wp``

        :param special_tokens: a list of tokens (such BERT's as ``[CLS]`` and
                               ``[SEP]`` tokens) to remove; to keep special
                               tokens when passing in a tokenizer in ``kwargs``,
                               add ``special_tokens={}``.

        :param index_tokens: whether to index tokens positionally, which is used
                             for mapping with feature or tokenized sentences;
                             set this to ``False`` when ``sentences`` are
                             anything but a feature document / sentences

        :param includes: what data to return, which is a set of the keys listed
                         in the ``return`` documentation below

        :return:

            a list sentence maps, each with:

                * ``sent_ix`` -> the ``i``th sentence (always provided)

                * ``map`` -> list of ``(sentence 'token', word pieces)``

                * ``sent`` -> a :class:`~zensols.nlp.container.FeatureSentence`
                  or a tensor of vocab indexes if ``map_wp`` is ``None``

                * ``word_pieces`` -> the word pieces of the sentences

        """
        def map_identity(x: int, six: int, input_ids: List[int]):
            return (x, input_ids[x], x) if add_indices else x

        def map_wp_by_id(x: int, six: int, input_ids: List[int]) -> str:
            tix = input_ids[x]
            tok = id2tok[tix]
            if tok.startswith('##'):
                # bert
                tok = tok[2:]
            elif tok.startswith('Ġ'):
                # roberta
                tok = tok[1:]
            return (tok, tix, x) if add_indices else tok

        id2tok: Dict[int, str] = None
        cdata: np.ndarray = self.tensor.cpu().numpy()
        input_ids: np.ndarray = cdata[self._INPUT_ID_IX]
        mask: np.ndarray = cdata[self._ATTENTION_MASK_IX]
        sent_offsets: List[int] = cdata[self._OFFSETS_IX].tolist()
        sents_map: List[Dict[str, Union[List[Any], Tuple]]] = []
        if special_tokens is None:
            if hasattr(map_wp, 'all_special_tokens'):
                special_tokens = map_wp.all_special_tokens
                if not isinstance(special_tokens, Set):
                    special_tokens = set(special_tokens)
            else:
                special_tokens = {}
        if map_wp is None:
            map_wp = map_identity
        elif hasattr(map_wp, 'id2tok'):
            # .tokenizer.Tokenizer
            id2tok = map_wp.id2tok
            map_wp = map_wp_by_id
        elif isinstance(map_wp, dict):
            id2tok = map_wp
            map_wp = map_wp_by_id
        if sentences is None:
            sentences = input_ids
        assert len(sentences) == len(sent_offsets)
        sents = enumerate(zip(sentences, sent_offsets))
        six: int
        sent: Union[FeatureSentence, Tensor]
        tok_offsets: List[int]
        for six, (sent, tok_offsets) in sents:
            input_sent: np.ndarray = input_ids[six]
            mask_sent: np.ndarray = mask[six]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'sent idx: {six}, sent: {sent}')
                logger.debug(f'offsets: {tok_offsets}')
                logger.debug(f'ids: {input_sent}')
            smap: Dict[str, Any] = {'sent_ix': six}
            if 'map' in includes:
                sent_map, sent = self._map_sent(
                    index_tokens, tok_offsets, map_wp, sent, six, input_sent,
                    includes, mask_sent, special_tokens)
                smap['map'] = sent_map
            if 'sent' in includes:
                smap['sent'] = sent
            if 'word_pieces' in includes:
                smap['word_pieces'] = tuple(filter(
                    lambda t: t not in special_tokens,
                    map(lambda t: map_wp(t[1], six, input_sent),
                        filter(lambda t: t[0], zip(mask_sent, it.count())))))
            sents_map.append(smap)
        return sents_map

    def get_wordpiece_count(self, **kwargs) -> int:
        """The size of the document (sum over sentences) in number of word
        pieces.  To keep special tokens (such BERT's as ``[CLS]`` and ``[SEP]``
        tokens) when passing in a tokenizer in ``kwargs``, add
        ``special_tokens={}``.

        :param kwargs: any keyword arguments passed on to
                       :meth:`.map_to_word_pieces` except (do not add
                       ``index_tokens`` and ``includes``)

        """
        sents: List[Dict[str, Any]] = self.map_to_word_pieces(
            index_tokens=False, includes={'word_pieces'}, **kwargs)
        return sum(map(lambda s: len(s['word_pieces']), sents))

    def deallocate(self):
        super().deallocate()
        del self.tensor

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_tokens: bool = True, id2tok: Dict[int, str] = None):
        for sent_map in self.map_to_word_pieces(
                map_wp=id2tok,
                index_tokens=id2tok is None,
                includes={'map', 'sent'}):
            self._write_line(f"sentence: {sent_map['sent']}", depth, writer)
            if include_tokens:
                self._write_line('tokens:', depth, writer)
                tok: str
                wps: Tuple[str]
                for tok, wps in sent_map['map']:
                    self._write_line(f'{tok} -> {wps}', depth + 1, writer)

    def __len__(self) -> int:
        """Longest sentence in word pieces with special tokens and padding."""
        return self.tensor.size(-1)

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

    def map_to_word_pieces(self, sentences: Iterable[List[Any]] = None,
                           map_wp: Union[Callable, Dict[int, str]] = None,
                           **kwargs) -> List[Dict[str, Any]]:
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

        map_wp = id2tok if map_wp is None else map_wp
        return super().map_to_word_pieces(self.feature, map_wp=map_wp, **kwargs)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_tokens: bool = True, id2tok: Dict[int, str] = None):
        id2tok = self.id2tok if id2tok is None else id2tok
        sent_map: Dict[str, Any]
        for sent_map in self.map_to_word_pieces(includes={'map', 'sent'}):
            sent: FeatureSentence = sent_map['sent']
            tmap: Tuple[FeatureToken, Tuple[str]] = sent_map['map']
            self._write_line(f'sentence: {sent}', depth, writer)
            if include_tokens:
                self._write_line('tokens:', depth, writer)
                tok: FeatureToken
                ttoks: Tuple[str, ...]
                for tok, ttoks in tmap:
                    stext = tok.text.replace('\n', '\\n')
                    stext = f'<{stext}>'
                    self._write_line(f'{stext} -> {ttoks}', depth + 1, writer)
