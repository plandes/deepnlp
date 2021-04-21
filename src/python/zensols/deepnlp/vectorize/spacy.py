"""Feature (ID) normalization.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Any
from dataclasses import dataclass, field
import sys
import math
import itertools as it
from spacy.vocab import Vocab
from torch import Tensor
from zensols.deeplearn import TorchConfig
from zensols.deeplearn.vectorize import FeatureVectorizer


@dataclass
class SpacyFeatureVectorizer(FeatureVectorizer):
    """This normalizes feature IDs of parsed token features in to a number between
    [0, 1].  This is useful for normalized feature vectors as input to neural
    networks.  Input to this would be strings like ``token.ent_`` found on a
    :class:`zensols.nlp.feature.TokenAttributes` instance.

    The class is also designed to create features using indexes, so there are
    methods to resolve to a unique ID from an identifier.

    Instances of this class behave like a ``dict``.

    All symbols are taken from :obj:`spacy.glossary.GLOSSARY`.

    :param vocab: the vocabulary used for ``from_spacy`` to compute the
                  normalized feature from the spacy ID (i.e. ``token.ent_``,
                  ``token.tag_`` etc.)

    :see: :obj:`spacy.glossary.GLOSSARY`

    :see: :class:`zensols.nlp.feature.TokenAttributes`

    """
    torch_config: TorchConfig = field()
    """The torch configuration used to create tensors."""

    vocab: Vocab = field()
    """The spaCy vocabulary used to create IDs from strings.

    :see meth:`id_from_spacy_symbol`

    """

    def __post_init__(self):
        super().__post_init__()
        self.as_list = tuple(self.SYMBOLS.split())
        syms = dict(zip(self.as_list, it.count()))
        self.symbol_to_id = syms
        self.id_to_symbol = dict(map(lambda x: (x[1], x[0]), syms.items()))
        n = len(syms)
        q = n - 1
        arr = self._to_hot_coded_matrix(n)
        rows = zip(syms, map(lambda i: arr[i], range(n)))
        self.symbol_to_vector = dict(rows)
        self.symbol_to_norm = {k: syms[k] / q for k in syms}

    def _is_settable(self, name: str, value: Any) -> bool:
        return False

    def _to_hot_coded_matrix(self, rows: int):
        arr = self.torch_config.zeros((rows, rows))
        for i in range(rows):
            arr[i][i] = 1
        return arr

    def _to_binary_matrix(self, rows: int):
        cols = math.ceil(math.log2(rows))
        arr = self.torch_config.empty((rows, rows))
        for i in range(rows):
            sbin = '{0:b}'.format(i).zfill(cols)
            arr[i] = self.torch_config.from_iterable(map(float, sbin))
        return arr

    def _get_shape(self) -> Tuple[int, int]:
        return 1, len(self.as_list)

    def transform(self, symbol: str) -> Tensor:
        return self.symbol_to_vector[symbol]

    def dist(self, symbol: str) -> float:
        """Return a normalized feature float if ``symbol`` is found.

        :return: a normalized value between [0 - 1] or ``None`` if the symbol
                 isn't found

        """
        return self.symbol_to_norm[symbol]

    def id_from_spacy_symbol(self, id: int, default: int = -1) -> str:
        """Return the Spacy text symbol for it's ID (``token.ent`` -> ``token.ent_``).

        """
        strs = self.vocab.strings
        if id in strs:
            return strs[id]
        else:
            return default

    def from_spacy(self, id: int) -> Tensor:
        """Return a binary feature from a Spacy ID or ``None`` if it doesn't have a
        mapping the ID.

        """
        symbol = self.id_from_spacy_symbol(id)
        return self.symbol_to_vector.get(symbol, None)

    def id_from_spacy(self, id: int, default: int = -1) -> int:
        """Return the ID of this vectorizer for the Spacy ID or -1 if not found.

        """
        symbol = self.id_from_spacy_symbol(id)
        return self.symbol_to_id.get(symbol, default)

    def write(self, writer=sys.stdout):
        """Pretty print a human readable representation of this feature vectorizer.

        """
        syms = self.symbol_to_id
        writer.write(f'{self.description}:\n')
        for k in sorted(syms.keys()):
            writer.write(f'  {k} => {syms[k]} ({self.transform(k)})\n')

    def __str__(self):
        return f'{self.description} ({self.feature_id})'


@dataclass
class NamedEntityRecognitionFeatureVectorizer(SpacyFeatureVectorizer):
    """A feature vectorizor for NER tags.

    :see: :class:`.SpacyFeatureVectorizer`

    """
    DESCRIPTION = 'named entity recognition'
    LANG = 'en'
    FEATURE_ID = 'ent'
    SYMBOLS = """PERSON NORP FACILITY FAC ORG GPE LOC PRODUCT EVENT WORK_OF_ART LAW LANGUAGE
    DATE TIME PERCENT MONEY QUANTITY ORDINAL CARDINAL PER MISC"""


@dataclass
class DependencyFeatureVectorizer(SpacyFeatureVectorizer):
    """A feature vectorizor for dependency head trees.

    :see: :class:`.SpacyFeatureVectorizer`

    """
    DESCRIPTION = 'dependency'
    LANG = 'en'
    FEATURE_ID = 'dep'
    SYMBOLS = """acl acomp advcl advmod agent amod appos attr aux auxpass case cc ccomp clf
complm compound conj cop csubj csubjpass dative dep det discourse dislocated
dobj expl fixed flat goeswith hmod hyph infmod intj iobj list mark meta neg
nmod nn npadvmod nsubj nsubjpass nounmod npmod num number nummod oprd obj obl
orphan parataxis partmod pcomp pobj poss possessive preconj prep prt punct
quantmod rcmod relcl reparandum root vocative xcomp ROOT"""


@dataclass
class PartOfSpeechFeatureVectorizer(SpacyFeatureVectorizer):
    """A feature vectorizor for POS tags.

    :see: :class:`.SpacyFeatureVectorizer`

    """
    DESCRIPTION = 'part of speech'
    LANG = 'en'
    FEATURE_ID = 'tag'
    SYMBOLS = """ADJ ADP ADV AUX CONJ CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM
VERB X EOL SPACE . , -LRB- -RRB- `` " ' $ # AFX CC CD DT EX FW HYPH IN JJ JJR
JJS LS MD NIL NN NNP NNPS NNS PDT POS PRP PRP$ RB RBR RBS RP TO UH VB VBD VBG
VBN VBP VBZ WDT WP WP$ WRB SP ADD NFP GW XX BES HVS NP PP VP ADVP ADJP SBAR PRT
PNP"""


SpacyFeatureVectorizer.VECTORIZERS = \
    {cls.FEATURE_ID: cls for cls in (NamedEntityRecognitionFeatureVectorizer,
                                     DependencyFeatureVectorizer,
                                     PartOfSpeechFeatureVectorizer)}
"""The default set of spaCy feature vectorizers.

"""
