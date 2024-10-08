"""Word piece mappings to feature tokens, sentences and documents.

There are often edges cases and tricky situations with certain model's usage of
special tokens (i.e. ``[CLS]``) and where they are used.  With this in mind,
this module attempts to:

  * Assist in debugging (works with detached :class:`.TokenizedDocument`) in
    cases where token level embeddings are directly accessed, and

  * Map corresponding both token and sentence level embeddings to respective
    origin natural langauge feature set data structures.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Tuple, List, Dict, Any, Union, Iterable, ClassVar, Type, Set
from dataclasses import dataclass, field
from abc import ABCMeta
import logging
import sys
from itertools import chain
from io import TextIOBase
import torch
from torch import Tensor
from zensols.util import Hasher
from zensols.persist import PersistableContainer, Stash
from zensols.config import Dictable
from zensols.nlp import (
    TokenContainer, FeatureToken, FeatureSentence, FeatureDocument,
    FeatureDocumentDecorator,
)
from zensols.deeplearn import TorchConfig
from zensols.deeplearn.vectorize import FeatureContext, TensorFeatureContext
from ..vectorize import TextFeatureType, EmbeddingFeatureVectorizer
from . import (
    TransformerError, TokenizedFeatureDocument, TransformerDocumentTokenizer,
    TransformerEmbedding,
)

logger = logging.getLogger(__name__)


@dataclass(repr=False)
class WordPiece(PersistableContainer, Dictable):
    """The word piece data.

    """
    UNKNOWN_TOKEN: ClassVar[str] = '[UNK]'
    """The string used for out of vocabulary word piece tokens."""

    word: str = field()
    """The string representation of the word piece."""

    vocab_index: int = field()
    """The vocabulary index."""

    index: int = field()
    """The index of the word piece subword in the tokenization tensor, which
    will have the same index in the output embeddings for
    :obj:`.TransformerEmbedding.output` = ``last_hidden_state``.

    """
    @property
    def is_unknown(self) -> bool:
        """Whether this token is out of vocabulary."""
        return self.word == self.UNKNOWN_TOKEN

    def __str__(self):
        s: str = self.word
        if s.startswith('##'):
            s = s[2:]
        return s


class WordPieceTokenContainer(TokenContainer, metaclass=ABCMeta):
    """Like :class:`~zensols.nlp.container.TokenContainer` but contains word
    pieces.

    """
    def word_iter(self) -> Iterable[WordPiece]:
        """Return an iterable over the word pieces."""
        return chain.from_iterable(
            map(lambda wp: wp.word_iter(), self.token_iter()))

    @property
    def unknown_count(self) -> int:
        """Return the number of out of vocabulary tokens in the container."""
        return sum(map(lambda t: t.is_unknown, self.token_iter()))


@dataclass(eq=False, repr=False)
class WordPieceFeatureToken(FeatureToken):
    """The token and the word pieces that repesent it.

    """
    _PERSITABLE_REMOVE_ATTRIBUTES: ClassVar[Set[str]] = {'embedding'}

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

    @property
    def token_embedding(self) -> Tensor:
        """The embedding of this token, which is the sum of the word piece
        embeddings.

        """
        return self.embedding.sum(dim=0)

    def word_iter(self) -> Iterable[WordPiece]:
        """Return an iterable over the word pieces."""
        return iter(self.words)

    @property
    def is_unknown(self) -> bool:
        """Whether this token is out of vocabulary."""
        return all(map(lambda wp: wp.is_unknown, self.word_iter()))

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

    def clone(self, cls: Type = None, **kwargs) -> FeatureToken:
        kwargs.update(dict(words=self.words, embedding=self.embedding))
        return super().clone(cls, **kwargs)

    def detach(self, *args, **kwargs) -> FeatureToken:
        clone: FeatureToken = super().detach(*args, **kwargs)
        clone.__dict__.pop('embedding', None)
        return clone

    def __str__(self) -> str:
        return ''.join(map(str, self.words))

    def __hash__(self) -> int:
        # even with eq=False, the `dataclasses` library seems to not delegate to
        # the parent's method
        return super().__hash__()

    def __eq__(self, other: WordPieceFeatureToken) -> bool:
        if self is other:
            return True
        return self.i == other.i and \
            self.idx == other.idx and \
            self.norm == other.norm


@dataclass(eq=False, repr=False)
class WordPieceFeatureSpan(FeatureSentence, WordPieceTokenContainer):
    """A sentence made up of word pieces.

    """
    _PERSITABLE_REMOVE_ATTRIBUTES: ClassVar[Set[str]] = {'embedding'}

    embedding: Tensor = field(default=None)
    """The sentence embedding level (i.e. ``[CLS]``) embedding from the
    transformer.

    :shape: (<embedding dimension>,)

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


@dataclass(eq=False, repr=False)
class WordPieceFeatureSentence(WordPieceFeatureSpan, FeatureSentence):
    pass


@dataclass(eq=False, repr=False)
class WordPieceFeatureDocument(FeatureDocument, WordPieceTokenContainer):
    """A document made up of word piece sentences.

    """
    tokenized: TokenizedFeatureDocument = field(default=None)
    """The tokenized feature document."""

    @property
    def embedding(self) -> Tensor:
        """The document embedding (see :obj:`.WordPieceFeatureSpan.embedding`).

        :shape: (|sentences|, <embedding dimension>)

        """
        return torch.stack(tuple(map(lambda s: s.embedding, self.sents)), dim=0)

    def copy_embedding(self, target: FeatureDocument):
        """Copy embeddings (and children) from this instance to ``target``."""
        targ_sent: FeatureSentence
        org_sent: WordPieceFeatureSentence
        if len(self.sents) != len(target.sents):
            raise TransformerError(
                'Can not copy embedding: different length ' +
                f'sentenes: {len(self.sents)} != {len(target.sents)}')
        for org_sent, targ_sent in zip(self.sents, target.sents):
            org_sent.copy_embedding(targ_sent)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(self, depth, writer)
        sent: WordPieceFeatureSentence
        for sent in self.sents:
            self._write_object(sent, depth + 1, writer)

    def __str__(self) -> str:
        return '. '.join(map(str, self.sents))


class _WordPieceDocKey(object):
    """A key class for caching in :class:`.WordPieceFeatureDocumentFactory`
    needed to avoid token level equal compares with embeddings.  These token
    level compares raise a Pytorch error.

    """
    def __init__(self, doc: FeatureDocument, tdoc: TokenizedFeatureDocument):
        self._hash = hash(doc)
        self._doc = doc

    def __eq__(self, other: FeatureDocument) -> bool:
        if self is other:
            return True
        return self._doc.norm == other._doc.norm

    def __hash__(self) -> int:
        return self._hash


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

    def create(self, fdoc: FeatureDocument,
               tdoc: TokenizedFeatureDocument = None) -> \
            WordPieceFeatureDocument:
        """Create a document in to an object graph that relates word pieces to
        feature tokens.  Note that if ``tdoc`` is provided, it must have been
        tokenized from ``fdoc``.

        :param fdoc: the feature document used to create `tdoc`

        :param tdoc: a tokenized feature document generated by :meth:`tokenize`

        :return: a data structure with the word piece information

        """
        def map_tok(ftok: FeatureToken, wps: Tuple[str, int, int]) -> \
                WordPieceFeatureToken:
            words = tuple(map(lambda t: WordPiece(*t), wps))
            return ftok.clone(cls=WordPieceFeatureToken, words=words)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'creating embeddings for: {fdoc} ({len(fdoc.text)})')
        tdoc = self.tokenizer.tokenize(fdoc) if tdoc is None else tdoc
        sents: List[WordPieceFeatureSentence] = []
        wps: List[Dict[str, Any]] = tdoc.map_to_word_pieces(
            sentences=fdoc,
            map_wp=self.tokenizer.id2tok,
            add_indices=True,
            includes={'map', 'sent'})
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
        if self.add_token_embeddings or self.add_sent_embeddings:
            arrs: Dict[str, Tensor] = self.embed_model.transform(
                tdoc, TransformerEmbedding.ALL_OUTPUT)
            if self.token_embeddings:
                logger.debug('adding token embedding')
                arr: Tensor = arrs[
                    TransformerEmbedding.LAST_HIDDEN_STATE_OUTPUT]
                self.add_token_embeddings(doc, arr)
            else:
                logger.debug('skipping token embedding')
                tok: FeatureToken
                for tok in doc.token_iter():
                    tok.embedding = None
            if self.sent_embeddings:
                logger.debug('adding sent embedding')
                arr: Tensor = arrs[TransformerEmbedding.POOLER_OUTPUT]
                self.add_sent_embeddings(doc, arr)
            else:
                logger.debug('skipping sent embedding')
                for sent in doc.sents:
                    sent.embedding = None
        return doc

    def __call__(self, fdoc: FeatureDocument,
                 tdoc: TokenizedFeatureDocument = None) -> \
            WordPieceFeatureDocument:
        return self.create(fdoc, tdoc)

    def populate(self, doc: FeatureDocument, truncate: bool = False):
        """Populate sentence embeddings in a document by first feature parsing a
        new document with :meth:`create` and then copying the embeddings with
        :meth:`.WordPieceFeatureDocument.copy_embeddings`

        :param truncate: if sentence lengths differ (i.e. from using different
                         models to chunk sentences) trim the longer document to
                         match the shorter

        """
        wpdoc: WordPieceFeatureDocument = self.create(doc)
        if truncate:
            slen: int = len(doc)
            wlen: int = len(wpdoc)
            if slen != wlen:
                mx: int = min(slen, wlen)
                doc.sents = doc.sents[:mx]
                wpdoc.sents = wpdoc.sents[:mx]
                logger.warning(
                    'Word piece doc parser chunked unequal sentence ' +
                    f'length (org: {slen}) != (wp: {wlen}), trimming larger')
        wpdoc.copy_embedding(doc)


@dataclass
class CachingWordPieceFeatureDocumentFactory(WordPieceFeatureDocumentFactory):
    """Caches the documents and their embeddings in a
    :class:`~zensols.persist.stash.Stash`.  For those that are cached, the
    embeddings are copied over to the passed document in :meth:`create`.

    """
    stash: Stash = field(default=None)
    """The stash that persists the feature document instances.  If this is not
    provided, no caching will happen.

    """
    hasher: Hasher = field(default_factory=Hasher)
    """Used to hash the natural langauge text in to string keys."""

    def _hash_text(self, text: str) -> str:
        self.hasher.reset()
        self.hasher.update(text)
        return self.hasher()

    def create(self, fdoc: FeatureDocument,
               tdoc: TokenizedFeatureDocument = None) -> \
            WordPieceFeatureDocument:
        key: str = self._hash_text(fdoc.text)
        wdoc: WordPieceFeatureDocument = self.stash.load(key)
        if wdoc is None:
            wdoc = super().create(fdoc, tdoc)
            if self.stash is not None:
                self.stash.dump(key, wdoc)
        else:
            if wdoc.text != fdoc.text:
                raise TransformerError('Document text does not match: ' +
                                       f'<{wdoc.text}> != >{fdoc.text}>')
        return wdoc

    def clear(self):
        """Clear the caching stash."""
        self.stash.clear()


@dataclass
class WordPieceDocumentDecorator(FeatureDocumentDecorator):
    """Populates sentence and token embeddings in the documents.

    :see: :class:`.WordPieceFeatureDocumentFactory`

    """
    word_piece_doc_factory: WordPieceFeatureDocumentFactory = field()
    """The feature document factory that populates embeddings."""

    def decorate(self, doc: FeatureDocument):
        wpdoc: WordPieceFeatureDocument = self.word_piece_doc_factory(doc)
        wpdoc.copy_embedding(doc)


@dataclass
class WordPieceFeatureVectorizer(EmbeddingFeatureVectorizer):
    """Uses the ``embeddings`` attributes added to documents, sentences and
    tokens populated by :class:`.WordPieceFeatureDocumentFactory`.  Currently
    only sentence sequences are supported.  For single sentence or token
    classification, use :mod:`zensols.deepnlp.vectorizers`.

    If aggregated documents are given to the vectorizer, they are flattened into
    sentences and vectorized in the same was a single document's sentences would
    be vectorized.  A batch is created for each document and only one batch is
    created for singleton documents.

    This embedding layer expects the following attribute settings to be left
    with the defaults set: obj:`encode_transformed`, :obj:`fold_method`,
    :obj:`decode_embedding`.

    :shape: (|documents|, |sentences|, |embedding dimension|)

    """
    DESCRIPTION: ClassVar[str] = 'wordpiece'
    FEATURE_TYPE: ClassVar[TextFeatureType] = TextFeatureType.EMBEDDING

    word_piece_doc_factory: WordPieceFeatureDocumentFactory = field(
        default=None)
    """The feature document factory that populates embeddings."""

    embed_model: TransformerEmbedding = field(default=None)
    """This field is not applicable to this vectorizer--keep the default."""

    encode_transformed: bool = field(default=False)
    """This field is not applicable to this vectorizer--keep the default."""

    fold_method: str = field(default='raise')
    """This field is not applicable to this vectorizer--keep the default."""

    decode_embedding: bool = field(default=True)
    """Turn off the :obj:`embed_model` forward pass to use the embeddings we
    vectorized from the ``embedding`` attribute(s).  Keep the default.

    """
    access: str = field(default='raise')
    """What to do when accessing the sentence embedding when encoding.  This is
    one of:

      * ``raise``: raises an error when missing
      * ``add_missing``: create the embedding only if missing
      * ``clobber``: always create a new embedding by replacing (if existed)

    """
    def __post_init__(self):
        super().__post_init__()
        self.embed_model = self.word_piece_doc_factory.embed_model

    def _get_shape(self) -> Tuple[int, int]:
        return -1, self.embed_model.vector_dimension

    def _encode(self, doc: FeatureDocument) -> FeatureContext:
        # parent encode calls this abstract method
        raise ValueError("Method 'encode' is overridden")

    def encode(self, doc: Union[Tuple[FeatureDocument, ...],
                                FeatureDocument]) -> FeatureContext:
        docs: Tuple[FeatureDocument, ...] = \
            doc if isinstance(doc, (list, tuple)) else [doc]
        tc: TorchConfig = self.torch_config
        emb_dim: int = self._get_shape()[1]
        darrs: Tuple[Tensor, ...] = tuple(map(self._encode_doc, docs))
        max_len: int = max(map(lambda t: t.size(0), darrs))
        arr: Tensor = tc.zeros((len(docs), max_len, emb_dim))
        bix: int
        darr: Tensor
        for bix, darr in enumerate(darrs):
            arr[bix, :darr.size(0), :] = darr
        return TensorFeatureContext(self.feature_id, arr)

    def _encode_doc(self, doc: FeatureDocument) -> Tensor:
        def map_sent(sent: FeatureSentence) -> Tensor:
            if not hasattr(sent, 'embedding'):
                raise TransformerError(f'No sentinal embedding in: {sent}')
            return sent.embedding

        if len(doc) == 0:
            raise TransformerError('Empty document')
        has_emb: bool = hasattr(doc[0], 'embedding')
        if not has_emb and self.access == 'raise':
            raise TransformerError(f'No sentinal embedding in: {doc[0]}')
        if not has_emb or self.access == 'clobber':
            self.word_piece_doc_factory.populate(doc)
        tensors: Tuple[Tensor, ...] = tuple(map(map_sent, doc.sents))
        return torch.stack(tensors)


FeatureToken.SKIP_COMPARE_FEATURE_IDS.add('embedding')
