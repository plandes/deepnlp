from __future__ import annotations
"""Domain objects that define features associated with text.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple, Set, Iterable, Any
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import sys
import logging
from io import TextIOBase
from itertools import chain
import itertools as it
from zensols.persist import PersistableContainer, persisted
from zensols.config import Writable
from zensols.nlp import TokenAttributes, TokenFeatures

logger = logging.getLogger(__name__)


class TextContainer(Writable, metaclass=ABCMeta):
    """A class that has a ``text`` property or attribute.  The class also provides
    pretty print utility.

    :attribute text: str

    """
    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'{self.__class__.__name__}: <{self.text}>',
                         depth, writer)


class FeatureToken(TextContainer):
    """A container class for features about a token.  This extracts only a subset
    of features from the heavy object ``TokenFeatures``, which contains Spacy C
    data structures and is hard/expensive to pickle.

    """
    TOKEN_FEATURE_IDS_BY_TYPE = TokenAttributes.FIELD_IDS_BY_TYPE
    TYPES_BY_TOKEN_FEATURE_ID = TokenAttributes.TYPES_BY_FIELD_ID
    TOKEN_FEATURE_IDS = TokenAttributes.FIELD_IDS

    def __init__(self, features: TokenFeatures, feature_ids: Set[str]):
        """Initialize.

        :param features: the features that describes a token

        :param feature_id: a string identifying the type of feature that will

        """
        fd = features.detach(feature_ids).asdict()
        for k in feature_ids:
            if k not in fd:
                fd[k] = None
        self.__dict__.update(fd)

    @property
    def text(self):
        return self.norm

    def to_vector(self, feature_ids: List[str]) -> Iterable[str]:
        """Return an iterable of feature data.

        """
        return map(lambda a: getattr(self, a), feature_ids)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        super().write(depth, writer)
        self._write_line('attributes:', depth, writer)
        for k, v in self.__dict__.items():
            ptype = self.TYPES_BY_TOKEN_FEATURE_ID.get(k)
            ptype = 'missing type' if ptype is None else ptype
            self._write_line(f'{k}={v} ({ptype})', depth + 1, writer)

    def __str__(self):
        attrs = []
        for s in 'norm lemma tag ent'.split():
            v = getattr(self, s) if hasattr(self, s) else None
            if v is not None:
                attrs.append(f'{s}: {v}')
        return ', '.join(attrs)

    def __repr__(self):
        return self.__str__()


class TokensContainer(PersistableContainer, TextContainer, metaclass=ABCMeta):
    """Each instance has the following attributes:

    """
    @abstractmethod
    def token_iter(self, *args) -> Iterable[FeatureToken]:
        """Return an iterator over the token features.

        :param args: the arguments given to :meth:`itertools.islice`

        """
        pass

    def norm_token_iter(self, *args) -> Iterable[str]:
        """Return a list of normalized tokens.

        :param args: the arguments given to :meth:`itertools.islice`

        """
        return map(lambda t: t.norm, self.token_iter(*args))

    @property
    @persisted('_tokens', transient=True)
    def tokens(self) -> Tuple[FeatureToken]:
        """Return the token features as a tuple.

        """
        return tuple(self.token_iter())

    @property
    @persisted('_token_len', transient=True)
    def token_len(self) -> int:
        """Return the number of tokens."""
        return sum(1 for i in self.token_iter())

    @abstractmethod
    def to_sentence(self, limit: int = sys.maxsize) -> FeatureSentence:
        """Coerce this instance to a single sentence.

        :param limit: the limit in the number of chunks to return

        :return: an instance of ``FeatureSentence`` that represents this token
                 sequence

        """
        pass

    @abstractmethod
    def to_document(self, limit: int = sys.maxsize) -> FeatureDocument:
        """Coerce this instance in to a document.

        """
        pass

    @property
    def norms(self) -> Set[str]:
        return set(map(lambda t: t.norm.lower(),
                       filter(lambda t: not t.is_punctuation and not t.is_stop,
                              self.tokens)))

    @property
    def lemmas(self) -> Set[str]:
        return set(map(lambda t: t.lemma.lower(),
                       filter(lambda t: not t.is_punctuation and not t.is_stop,
                              self.tokens)))

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        super().write(depth, writer)
        self._write_line('tokens:', depth, writer)
        for t in self.token_iter():
            t.write(depth + 1, writer)


@dataclass
class FeatureSentence(TokensContainer):
    """A container class of tokens that make a sentence.  Instances of this class
    iterate over :class:`.FeatureToken` instances, and can create documents
    with :meth:`to_document`.

    """
    sent_tokens: Tuple[FeatureToken]
    text: str = field(default=None)

    def __post_init__(self):
        super().__init__()
        if self.text is None:
            self.text = ' '.join(map(lambda t: t.text, self.sent_tokens))

    def token_iter(self, *args) -> Iterable[FeatureToken]:
        if len(args) == 0:
            return iter(self.sent_tokens)
        else:
            return it.islice(self.sent_tokens, *args)

    @property
    def tokens(self) -> Tuple[FeatureToken]:
        return self.sent_tokens

    @property
    def token_len(self) -> int:
        return len(self.sent_tokens)

    def __getitem__(self, key) -> FeatureToken:
        return self.tokens[key]

    def to_sentence(self, limit: int = sys.maxsize) -> FeatureSentence:
        return self

    def to_document(self) -> FeatureDocument:
        return FeatureDocument([self])

    def __len__(self) -> int:
        return self.token_len

    def __iter__(self):
        return self.token_iter()

    def __str__(self):
        return f'<{self.text[:79]}>'

    def __repr__(self):
        return self.__str__()


@dataclass
class FeatureDocument(TokensContainer):
    """A container class of tokens that make a document.  This class contains a one
    to many of sentences.  However, it can be treated like any
    :class:`.TokensContainer` to fetch tokens.  Instances of this class iterate
    over :class:`.FeatureSentence` instances.

    :param sents: the sentences defined for this document

    """
    sents: List[FeatureSentence]

    def token_iter(self, *args) -> Iterable[FeatureToken]:
        sent_toks = chain.from_iterable(map(lambda s: s.tokens, self.sents))
        if len(args) == 0:
            return sent_toks
        else:
            return it.islice(sent_toks, *args)

    def sent_iter(self, *args) -> Iterable[FeatureSentence]:
        if len(args) == 0:
            return iter(self.sents)
        else:
            return it.islice(self.sents, *args)

    def get_text(self, *args):
        return ' '.join(map(lambda s: s.text, self.sent_iter(*args)))

    @property
    def max_sentence_len(self) -> int:
        """Return the length of tokens from the longest sentence in the document.

        """
        return max(map(len, self.sent_iter()))

    def to_sentence(self, *args) -> FeatureSentence:
        sents = self.sent_iter(*args)
        toks = chain.from_iterable(map(lambda s: s.tokens, sents))
        return FeatureSentence(tuple(toks), self.get_text(*args))

    def to_document(self) -> FeatureDocument:
        return self

    @classmethod
    def from_containers(cls, conts: Tuple[TokensContainer]) -> FeatureDocument:
        """Coerce a tuple of token containers (either documents or sentences) in to
        one *synthesized* document.

        """
        return cls(list(map(lambda c: c.to_sentence(), conts)))

    def combine_sentences(self) -> FeatureDocument:
        return self.__class__([self.to_sentence()])

    @property
    @persisted('_text', transient=True)
    def text(self):
        return self.get_text()

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        TextContainer.write(self, depth, writer)
        self._write_line('sentences:', depth, writer)
        for s in self.sents:
            s.write(depth + 1, writer)

    def __getitem__(self, key):
        return self.sents[key]

    def __len__(self):
        return len(self.sents)

    def __iter__(self):
        return self.sent_iter()

    def __str__(self):
        return f'<{self.text[:79]}>'

    def __repr__(self):
        return self.__str__()
