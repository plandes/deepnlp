from __future__ import annotations
"""Word piece mappings to feature tokens, sentences and documents.

"""
__author__ = 'Paul Landes'


from typing import Tuple, List, Dict, Any
from dataclasses import dataclass, field
import sys
from io import TextIOBase
from torch import Tensor
from zensols.persist import PersistableContainer
from zensols.config import Dictable
from zensols.nlp import FeatureToken, FeatureSentence, FeatureDocument
from . import (
    TokenizedFeatureDocument, TransformerDocumentTokenizer, TransformerEmbedding
)


@dataclass(repr=False)
class WordPieceBase(Dictable):
    def __repr__(self) -> str:
        return self.__str__()


@dataclass(repr=False)
class WordPiece(PersistableContainer, Dictable):
    """The word piece data.

    """
    name: str = field()
    """The string representation of the word piece."""

    vocab_index: int = field()
    """The vocabulary index."""

    index: int = field()
    """The index of the word piece subword in the tokenization tensor, which
    will have the same index in the output embeddings for
    :obj:`.TransformerEmbedding.output` = ``last_hidden_state``.

    """
    def __str__(self):
        s: str = self.name
        if s.startswith('##'):
            s = s[2:]
        return s


@dataclass(repr=False)
class WordPieceToken(WordPieceBase):
    """The token and the word pieces that repesent it.

    """
    feature: FeatureToken = field()
    """The token from the initial :class:`~zensols.nlp.FeatureSentence`."""

    words: Tuple[WordPiece] = field()
    """The word pieces that make up this token."""

    embedding: Tensor = field(default=None)
    """The embedding for :obj:`words` after using the transformer.

    :shape: (|words|, <embedding dimension>)

    """
    @property
    def indexes(self) -> Tuple[int]:
        """The indexes of the word piece subwords (see :obj:`.WordPiece.index`).

        """
        return tuple(map(lambda wp: wp.index, self.words))

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'{self.feature.norm}:', depth, writer)
        for w in self.words:
            self._write_line(f'{w}: i={w.index}, v={w.vocab_index}',
                             depth + 1, writer)
        if self.embedding is not None:
            self._write_line(f'embedding: {self.embedding.shape}',
                             self, depth + 1, writer)

    def __str__(self) -> str:
        return ''.join(map(str, self.words))


@dataclass(repr=False)
class WordPieceSentence(WordPieceBase):
    """A sentence made up of word pieces.

    """
    feature: FeatureSentence = field()
    """The initial sentence that was used to create the word piences."""

    tokens: Tuple[WordPieceToken] = field()
    """The word piece tokens that make up the sentence."""

    sentence_index: int = field()
    """The index of the sentence in the document."""

    embedding: Tensor = field(default=None)
    """The sentence embedding level (i.e. ``[CLS]``) embedding from the
    transformer.

    :shape: (|words|, <embedding dimension>)

    """

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(self.feature, depth, writer)
        self._write_line(self, depth + 1, writer)
        if self.embedding is not None:
            self._write_line(f'embedding: {self.embedding.shape}',
                             self, depth + 1, writer)

    def __str__(self) -> str:
        return ' '.join(map(str, self.tokens))


@dataclass(repr=False)
class WordPieceDocument(WordPieceBase):
    """A document made up of word piece sentences.

    """
    feature: FeatureDocument = field()
    """The initial document that was used to create the word piences."""

    tokenized: TokenizedFeatureDocument = field()
    """The tokenized feature document."""

    sents: Tuple[WordPieceSentence] = field()
    """The word piece sentences that make up the document."""

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(self.feature, depth, writer)
        sent: WordPieceSentence
        for sent in self.sents:
            self._write_object(sent, depth + 1, writer)

    def __str__(self) -> str:
        return '. '.join(map(str, self.sents))


@dataclass
class WordPieceDocumentFactory(object):
    """Create word piece resources.

    """
    tokenizer: TransformerDocumentTokenizer = field()
    embed_model: TransformerEmbedding = field()

    def add_token_embeddings(self, doc: WordPieceDocument, arr: Tensor):
        six: int
        sent: WordPieceSentence
        for six, sent in enumerate(doc.sents):
            tok: WordPieceToken
            for tok in sent.tokens:
                tok.embedding = arr[six, tok.indexes]

    def add_sent_embeddings(self, doc: WordPieceDocument, arr: Tensor):
        print('A', arr.shape)
        six: int
        sent: WordPieceSentence
        for six, sent in enumerate(doc.sents):
            pass

    def __call__(self, fdoc: FeatureDocument,
                 tdoc: TokenizedFeatureDocument = None,
                 add_token_embeddings: bool = False,
                 add_sent_embeddings: bool = False) -> WordPieceDocument:
        """Return an object graph that relates word pieces to feature tokens.

        :param fdoc: the feature document used to create `tdoc`

        :param tdoc: a tokenized feature document generated by :meth:`tokenize`

        :param add_token_embeddings: whether to add
                                     :class:`.WordPieceToken.embeddings`

        :param add_sent_embeddings: whether to add
                                    class:`.WordPieceSentence.embeddings`

        :return: a data structure with the word piece information

        """
        tdoc = self.tokenizer.tokenize(fdoc) if tdoc is None else tdoc
        sents: List[WordPieceSentence] = []
        wps: List[Dict[str, Any]] = tdoc.map_to_word_pieces(
            sentences=fdoc,
            map_wp=self.tokenizer.id2tok,
            add_indices=True)
        wp: Dict[str, Any]
        for six, wp in enumerate(wps):
            tokens: Tuple[WordPieceToken] = tuple(
                map(lambda x: WordPieceToken(
                    x[0], tuple(map(lambda wt: WordPiece(*wt), x[1]))),
                    wp['map']))
            sents.append(WordPieceSentence(
                feature=wp['sent'],
                tokens=tokens,
                sentence_index=six))
        doc = WordPieceDocument(fdoc, tdoc, sents)
        if add_token_embeddings:
            arr: Tensor = self.embed_model.transform(
                tdoc, output='last_hidden_state')
            self.add_token_embeddings(doc, arr)
        if add_sent_embeddings:
            arr: Tensor = self.embed_model.transform(
                tdoc, output='pooler')
            self.add_sent_embeddings(doc, arr)
        return doc
