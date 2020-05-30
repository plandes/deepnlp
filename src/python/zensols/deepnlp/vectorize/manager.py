"""An extension of a feature vectorizer manager that parses and vectorized
natural language.

"""
__author__ = 'Paul Landes'

import logging
from typing import List, Union, Set, Dict
from abc import abstractmethod, ABCMeta
from dataclasses import dataclass, field
import collections
from zensols.persist import persisted
from zensols.nlp import LanguageResource
from zensols.deeplearn.vectorize import (
    FeatureContext,
    EncodableFeatureVectorizer,
    FeatureVectorizerManager,
)
from zensols.deepnlp import FeatureDocument, TokensContainer
from zensols.deepnlp import FeatureDocumentParser
from . import SpacyFeatureVectorizer

logger = logging.getLogger(__name__)


@dataclass
class TokenContainerFeatureVectorizer(EncodableFeatureVectorizer,
                                      metaclass=ABCMeta):
    """Creates document or sentence level features using instances of
    ``TokensContainer``.

    """
    @abstractmethod
    def _encode(self, container: TokensContainer) -> FeatureContext:
        pass

    @property
    def token_length(self) -> int:
        return self.manager.token_length


@dataclass
class TokenContainerFeatureVectorizerManager(FeatureVectorizerManager):
    """Creates and manages instances of ``TokenContainerFeatureVectorizer`` and
    parses text in to feature based document.

    This is used to manage the relationship of a given set of parsed features
    keeping in mind that parsing will usually happen as a preprocessing step.
    A second step is the vectorization of those features, which can be any
    proper subset of those features parsed in the previous step.  However,
    these checks, of course, are not necessary if pickling isn't used across
    the parse and vectorization steps.

    :see TokenContainerFeatureVectorizer:
    :see parse:

    """
    doc_parser: FeatureDocumentParser
    token_length: int
    token_feature_types: Set[str] = field(
        default_factory=lambda: FeatureDocumentParser.TOKEN_FEATURE_TYPES)

    def __post_init__(self):
        super().__post_init__()
        feat_diff = self.token_feature_types - self.doc_parser.token_feature_types
        if len(feat_diff) > 0:
            fdiffs = ', '.join(feat_diff)
            s = f'parser token features do not exist in vectorizer: {fdiffs}'
            raise ValueError(s)

    def parse(self, text: Union[str, List[str]], *args, **kwargs) -> FeatureDocument:
        """Parse text or a text as a list of sentences.

        *Important:* Parsing documents through this manager instance is better
         since safe checks are made that features are available from those used
         when documents are parsed before pickling.

        :param text: either a string or a list of strings; if the former a
                     document with one sentence will be created, otherwise a
                     document is returned with a sentence for each string in
                     the list

        """
        return self.doc_parser.parse(text, *args, **kwargs)

    @property
    def langres(self) -> LanguageResource:
        return self.doc_parser.langres

    @property
    @persisted('_spacy_vectorizers')
    def spacy_vectorizers(self) -> Dict[str, SpacyFeatureVectorizer]:
        """Return vectorizers based on the ``token_feature_types`` configured on this
        instance.  Keys are token level feature types found in
        ``SpacyFeatureVectorizer.VECTORIZERS``.

        """
        token_feature_types = set(SpacyFeatureVectorizer.VECTORIZERS.keys())
        token_feature_types = token_feature_types & self.token_feature_types
        token_feature_types = sorted(token_feature_types)
        vectorizers = collections.OrderedDict()
        for feature_type in token_feature_types:
            cls = SpacyFeatureVectorizer.VECTORIZERS[feature_type]
            inst = cls(self.torch_config, self.langres.model.vocab)
            vectorizers[feature_type] = inst
        return vectorizers
