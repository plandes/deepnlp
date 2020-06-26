"""Domain objects that define features associated with text.

"""
__author__ = 'Paul Landes'

import sys
import logging
from abc import ABC, ABCMeta, abstractmethod
from typing import List, Tuple, Set, Iterable, Any
from dataclasses import dataclass
from functools import reduce
from itertools import chain
import itertools as it
from zensols.persist import PersistableContainer, persisted
from zensols.nlp import TokenFeatures

logger = logging.getLogger(__name__)


class TextContainer(ABC):
    """A class that has a ``text`` property or attribute.  The class also provides
    pretty print utility.

    :attribute text: str

    """
    @staticmethod
    def _indspc(depth: int):
        return ' ' * (depth * 2)

    def write(self, depth: int = 0, writer=sys.stdout):
        writer.write(f"{self._indspc(depth)}{self.__class__.__name__}: " +
                     f"<{self.text}>\n")


class FeatureToken(TextContainer):
    """A container class for features about a token.  This extracts only a subset
    of features from the heavy object ``TokenFeatures``, which contains Spacy C
    data structures and is hard/expensive to pickle.

    """
    TOKEN_FEATURE_IDS_BY_TYPE = {
        'bool': frozenset('is_space is_stop is_ent'.split()),
        'int': frozenset('i idx is_punctuation tag ent dep'.split()),
        'str': frozenset('norm lemma tag_ ent_ dep_'.split()),
        'list': frozenset('children'.split())}
    TYPES_BY_TOKEN_FEATURE_ID = dict(chain.from_iterable(
        map(lambda itm: map(lambda f: (f, itm[0]), itm[1]),
            TOKEN_FEATURE_IDS_BY_TYPE.items())))
    TOKEN_FEATURE_IDS = frozenset(
        reduce(lambda res, x: res | x, TOKEN_FEATURE_IDS_BY_TYPE.values()))

    def __init__(self, features: TokenFeatures, feature_ids: Set[str]):
        """Initialize.

        :param features: the features that describes a token
        :param feature_id: a string identifying the type of feature that will

        """
        fd = features.detach(feature_ids).to_dict()
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

    def write(self, depth: int = 0, writer=sys.stdout):
        s2 = self._indspc(depth + 1)
        super().write(depth, writer)
        for k, v in self.__dict__.items():
            ptype = self.TYPES_BY_TOKEN_FEATURE_ID.get(k)
            ptype = 'missing type' if ptype is None else ptype
            writer.write(f'{s2}{k}={v} ({ptype})\n')

    def __str__(self):
        s = self
        return f'{s.norm}: lemma={s.lemma}, tag={s.tag}, ner={s.ent}'

    def __repr__(self):
        return self.__str__()


class TokensContainer(PersistableContainer, TextContainer, metaclass=ABCMeta):
    """Each instance has the following attributes:

    """
    @abstractmethod
    def token_iter(self, *args) -> Iterable[TokenFeatures]:
        pass

    def norm_token_iter(self, *args) -> Iterable[str]:
        return map(lambda t: t.norm, self.token_iter(*args))

    @property
    @persisted('_tokens', transient=True)
    def tokens(self) -> Tuple[TokenFeatures]:
        return tuple(self.token_iter())

    @property
    @persisted('_token_len', transient=True)
    def token_len(self):
        return sum(1 for i in self.token_iter())

    @abstractmethod
    def to_sentence(self, limit=sys.maxsize) -> Any:
        """Coerce this instance to a single sentence.

        :return: an instance of ``FeatureSentence`` that represents this token
                 sequence

        """
        pass

    @property
    def norms(self):
        return set(map(lambda t: t.norm.lower(),
                       filter(lambda t: not t.is_punctuation and not t.is_stop,
                              self.tokens)))

    @property
    def lemmas(self):
        return set(map(lambda t: t.lemma.lower(),
                       filter(lambda t: not t.is_punctuation and not t.is_stop,
                              self.tokens)))

    def write(self, depth: int = 0, writer=sys.stdout):
        super().write(depth, writer)
        for t in self.token_iter():
            t.write(depth + 1, writer)


@dataclass
class FeatureSentence(TokensContainer):
    text: str
    sent_tokens: Tuple[FeatureToken]

    def token_iter(self, *args) -> Iterable[TokenFeatures]:
        if len(args) == 0:
            return iter(self.sent_tokens)
        else:
            return it.islice(self.sent_tokens, *args)

    @property
    def tokens(self) -> Tuple[TokenFeatures]:
        return self.sent_tokens

    @property
    def token_len(self):
        return len(self.sent_tokens)

    def __getitem__(self, key):
        return self.tokens[key]

    def to_sentence(self, limit=sys.maxsize) -> Any:
        return self

    def __len__(self):
        return self.token_len

    def __iter__(self):
        return self.token_iter()

    def __str__(self):
        return f'<{self.text[:79]}>'


@dataclass
class FeatureDocument(TokensContainer):
    sents: List[FeatureSentence]

    def token_iter(self, *args) -> Iterable[TokenFeatures]:
        sent_toks = chain.from_iterable(map(lambda s: s.tokens, self.sents))
        if len(args) == 0:
            return sent_toks
        else:
            return it.islice(sent_toks, *args)

    def sent_iter(self, *args):
        if len(args) == 0:
            return iter(self.sents)
        else:
            return it.islice(self.sents, *args)

    def get_text(self, *args):
        return ' '.join(map(lambda s: s.text, self.sent_iter(*args)))

    def to_sentence(self, *args) -> FeatureSentence:
        sents = self.sent_iter(*args)
        toks = chain.from_iterable(map(lambda s: s.tokens, sents))
        return FeatureSentence(self.get_text(*args), tuple(toks))

    @property
    @persisted('_text', transient=True)
    def text(self):
        return self.get_text()

    def write(self, depth=0, writer=sys.stdout, ):
        TextContainer.write(self, depth, writer)
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
