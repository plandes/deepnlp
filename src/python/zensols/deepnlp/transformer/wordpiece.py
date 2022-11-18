from __future__ import annotations
"""Word piece mappings to feature tokens, sentences and documents.

There are often edges cases and tricky situations with certain model's usage of
special tokens (i.e. ``[CLS]``) and where they are used.  With this in mind,
this module attempts to:

  * Assist in debugging in cases where token level embeddings are directly
    accessed, and

  * Map corresponding both token and sentence level embeddings to respective
    origin natural langauge feature set data structures.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Dict, Any, Union
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
class WordPiece(PersistableContainer, Dictable):
    """The word piece data.

    """
    word: str = field()
    """The string representation of the word piece."""

    vocab_index: int = field()
    """The vocabulary index."""

    index: int = field()
    """The index of the word piece subword in the tokenization tensor, which
    will have the same index in the output embeddings for
    :obj:`.TransformerEmbedding.output` = ``last_hidden_state``.

    """
    def __str__(self):
        s: str = self.word
        if s.startswith('##'):
            s = s[2:]
        return s


@dataclass(repr=False)
class WordPieceFeatureToken(FeatureToken):
    """The token and the word pieces that repesent it.

    """
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

    def copy_embedding(self, target: FeatureToken):
        """Copy embedding (and children) from this instance to ``target``."""
        target.embedding = self.embedding

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'{self.norm}:', depth, writer)
        for w in self.words:
            self._write_line(f'{w}: i={w.index}, v={w.vocab_index}',
                             depth + 1, writer)
        if self.embedding is not None:
            self._write_line(f'embedding: {tuple(self.embedding.size())}',
                             depth + 1, writer)

    def __str__(self) -> str:
        return ''.join(map(str, self.words))


@dataclass(repr=False)
class WordPieceFeatureSentence(FeatureSentence):
    """A sentence made up of word pieces.

    """
    embedding: Tensor = field(default=None)
    """The sentence embedding level (i.e. ``[CLS]``) embedding from the
    transformer.

    :shape: (|words|, <embedding dimension>)

    """
    def copy_embedding(self, target: FeatureSentence):
        """Copy embeddings (and children) from this instance to ``target``."""
        target.embedding = self.embedding
        targ_tok: FeatureToken
        org_tok: FeatureToken
        for org_tok, targ_tok in zip(self.token_iter(), target.token_iter()):
            org_tok.copy_embedding(targ_tok)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(super().__str__(), depth, writer)
        self._write_line('word pieces:', depth, writer)
        self._write_line(self, depth + 1, writer)
        if self.embedding is not None:
            self._write_line(f'embedding: {tuple(self.embedding.size())}',
                             depth + 1, writer)

    def __str__(self) -> str:
        return ' '.join(map(str, self.tokens))


@dataclass(repr=False)
class WordPieceFeatureDocument(FeatureDocument):
    """A document made up of word piece sentences.

    """
    tokenized: TokenizedFeatureDocument = field(default=None)
    """The tokenized feature document."""

    def copy_embedding(self, target: FeatureDocument):
        """Copy embeddings (and children) from this instance to ``target``."""
        targ_sent: FeatureSentence
        org_sent: WordPieceFeatureSentence
        for org_sent, targ_sent in zip(self.sents, target.sents):
            org_sent.copy_embedding(targ_sent)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(self, depth, writer)
        sent: WordPieceFeatureSentence
        for sent in self.sents:
            self._write_object(sent, depth + 1, writer)

    def __str__(self) -> str:
        return '. '.join(map(str, self.sents))


@dataclass
class WordPieceFeatureDocumentFactory(object):
    """Create instances of :class:`.WordPieceFeatureDocument` from
    :class:`~zensols.nlp.container.FeatureDocument` instances.  It does this by
    iterating through a feature document data structure and adding
    ``WordPiece*`` object data and optionally adding the corresponding sentence
    and/or token level embeddings.

    The embeddings can also be added with :meth:`add_token_embeddings` and
    :meth:`add_sent_embeddings` individually.  If all you want are the sentence
    level embeddings, you can use :meth:`add_sent_embeddings` on a
    :class:`~zensols.nlp.container.FeatureSentence` instance.

    """
    tokenizer: TransformerDocumentTokenizer = field()
    """Used to tokenize documents that aren't already in :meth:`__call__`."""

    embed_model: TransformerEmbedding = field()
    """Used to populate the embeddings in ``WordPiece*`` classes."""

    token_embeddings: bool = field(default=True)
    """Whether to add :class:`.WordPieceFeatureToken.embeddings`.

    """
    sent_embeddings: bool = field(default=True)
    """Whether to add class:`.WordPieceFeatureSentence.embeddings`.

    """
    def add_token_embeddings(self, doc: WordPieceFeatureDocument, arr: Tensor):
        """Add token embeddings to the sentences of ``doc``.  This assumes
        tokens are of type :class:`.WordPieceFeatureToken` since the token
        indices are needed.

        :param doc: sentences of this doc have ``embeddings`` set to the
                    correpsonding sentence tensor with shape (1, <embedding
                    dimension>).

        """
        six: int
        sent: FeatureSentence
        for six, sent in enumerate(doc.sents):
            tok: WordPieceFeatureToken
            for tok in sent.tokens:
                tok.embedding = arr[six, tok.indexes]

    def add_sent_embeddings(
            self, doc: Union[WordPieceFeatureDocument, FeatureDocument],
            arr: Tensor):
        """Add sentence embeddings to the sentences of ``doc``.

        :param doc: sentences of this doc have ``embeddings`` set to the
                    correpsonding sentence tensor with shape ``(1, <embedding
                    dimension>)``.

        """
        six: int
        sent: FeatureSentence
        for six, sent in enumerate(doc.sents):
            sent.embedding = arr[six]

    def __call__(self, fdoc: FeatureDocument,
                 tdoc: TokenizedFeatureDocument = None) -> \
            WordPieceFeatureDocument:
        """Return an object graph that relates word pieces to feature tokens.

        :param fdoc: the feature document used to create `tdoc`

        :param tdoc: a tokenized feature document generated by :meth:`tokenize`

        :return: a data structure with the word piece information

        """
        def map_tok(ftok: FeatureToken, wps: Tuple[str, int, int]) -> \
                WordPieceFeatureToken:
            words = tuple(map(lambda t: WordPiece(*t), wps))
            return ftok.clone(cls=WordPieceFeatureToken, words=words)

        tdoc = self.tokenizer.tokenize(fdoc) if tdoc is None else tdoc
        sents: List[WordPieceFeatureSentence] = []
        wps: List[Dict[str, Any]] = tdoc.map_to_word_pieces(
            sentences=fdoc,
            map_wp=self.tokenizer.id2tok,
            add_indices=True)
        wp: Dict[str, Any]
        for six, wp in enumerate(wps):
            fsent: FeatureSentence = wp['sent']
            tokens: Tuple[WordPieceFeatureToken] = tuple(
                map(lambda t: map_tok(*t), wp['map']))
            sents.append(fsent.clone(
                cls=WordPieceFeatureSentence,
                tokens=tokens))
        doc = fdoc.clone(
            cls=WordPieceFeatureDocument,
            sents=tuple(sents),
            tokenized=tdoc)
        if self.add_token_embeddings:
            arr: Tensor = self.embed_model.transform(
                tdoc, output='last_hidden_state')
            self.add_token_embeddings(doc, arr)
        if self.add_sent_embeddings:
            arr: Tensor = self.embed_model.transform(
                tdoc, output='pooler_output')
            self.add_sent_embeddings(doc, arr)
        return doc
