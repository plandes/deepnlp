"""Generate and vectorize language features.

"""
__author__ = 'Paul Landes'

import logging
import sys
from typing import List, Union, Set, Tuple, Any, Dict, Type
from abc import abstractmethod, ABCMeta
from dataclasses import dataclass, field
import collections
import torch
from zensols.persist import persisted
from zensols.nlp import LanguageResource, TokenFeatures
from zensols.deeplearn import (
    TorchConfig,
    FeatureContext,
    EncodableFeatureVectorizer,
    TensorFeatureContext,
    FeatureVectorizerManager,
)
from zensols.deepnlp import (
    FeatureToken,
    FeatureSentence,
    FeatureDocument,
    TokensContainer,
    SpacyFeatureVectorizer,
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureDocumentParser(object):
    """This class parses text in to instances of ``FeatureDocument``.

    *Important:* It is better to use ``TokenContainerFeatureVectorizerManager``
     instead of this class.

    :see TokenContainerFeatureVectorizerManager:

    """
    TOKEN_FEATURE_TYPES = FeatureToken.TOKEN_FEATURE_TYPES

    langres: LanguageResource
    token_feature_types: Set[str] = field(
        default_factory=lambda: FeatureDocumentParser.TOKEN_FEATURE_TYPES)
    doc_class: Type[FeatureDocument] = field(default=FeatureDocument)

    def _create_token(self, feature: TokenFeatures) -> FeatureToken:
        return FeatureToken(feature, self.token_feature_types)

    def from_string(self, text: str) -> List[FeatureSentence]:
        """Parse a document from a string.

        """
        lr = self.langres
        doc = lr.parse(text)
        sent_feats = []
        for sent in doc.sents:
            feats = tuple(map(self._create_token, lr.features(sent)))
            sent_feats.append(FeatureSentence(sent.text, feats))
        return sent_feats

    def from_list(self, text: List[str]) -> List[FeatureSentence]:
        """Parse a document from a list of strings.

        """
        lr = self.langres
        sent_feats = []
        for sent in text:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'sentence: {sent}')
            feats = tuple(map(self._create_token, lr.features(lr.parse(sent))))
            sent_feats.append(FeatureSentence(sent, feats))
        return sent_feats

    def parse(self, text: Union[str, List[str]], *args, **kwargs) -> FeatureDocument:
        """Parse text or a text as a list of sentences.

        :param text: either a string or a list of strings; if the former a
                     document with one sentence will be created, otherwise a
                     document is returned with a sentence for each string in
                     the list

        """
        if isinstance(text, str):
            sents = self.from_string(text)
        else:
            sents = self.from_list(text)
        return self.doc_class(sents, *args, **kwargs)


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
    langres: LanguageResource
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


@dataclass
class TypeTokenContainerFeatureVectorizer(TokenContainerFeatureVectorizer):
    """Creates a stacked binary representation of all configured token level
    features for each token.

    """
    NAME = 'token feature vectorizer'
    FEATURE_TYPE = 'ftvec'

    def _get_shape(self) -> Tuple[int, int]:
        flen = 0
        for fvec in self.manager.spacy_vectorizers.values():
            flen += fvec.shape[1]
        return flen, self.token_length

    def get_feature_vectors(self, container: TokensContainer,
                            fvec: SpacyFeatureVectorizer,
                            arr: torch.Tensor, row_start: int, row_end: int):
        attr_name = fvec.feature_type
        row_end = row_start + fvec.shape[1]
        toks = container.tokens[:arr.shape[1]]
        #print('L', len(toks), arr.shape[1])
        for i, tok in enumerate(toks):
            val = getattr(tok, attr_name)
            vec = fvec.from_spacy(val)
            if vec is not None:
                #print(row_start, row_end, i, tok.norm, val, attr_name, vec.shape)
                arr[row_start:row_end, i] = vec

    def _encode(self, container: TokensContainer) -> FeatureContext:
        row_start = 0
        arr = self.torch_config.zeros(self.shape)
        #print(f'shape: {self.shape}')
        for fvec in self.manager.spacy_vectorizers.values():
            row_end = row_start + fvec.shape[1]
            self.get_feature_vectors(
                container, fvec, arr, row_start, row_end)
            row_start = row_end
        arr = arr.to_sparse()
        return TensorFeatureContext(self.feature_type, arr)

    def _decode(self, context: FeatureContext) -> torch.Tensor:
        return super()._decode(context).to_dense()


TokenContainerFeatureVectorizerManager.register_vectorizer(TypeTokenContainerFeatureVectorizer)


@dataclass
class CountTokenContainerFeatureVectorizer(TokenContainerFeatureVectorizer):
    """Return the count of all tokens as a 1 X M * N tensor where M is the number
    of token feature types and N is the columns of the ``fvec`` vectorizer.
    Each column position's count represents the number of counts for that spacy
    symol for that index position in the ``fvec``.

    """
    NAME = 'token level feature counts'
    FEATURE_TYPE = 'count'

    def _get_shape(self) -> Tuple[int, int]:
        flen = 0
        for fvec in self.manager.spacy_vectorizers.values():
            flen += fvec.shape[1]
        return flen,

    def _encode(self, container: TokensContainer) -> FeatureContext:
        tensors = []
        for fvec in self.manager.spacy_vectorizers.values():
            tensors.append(self.get_feature_counts(container, fvec))
        return TensorFeatureContext(
            self.feature_type, torch.cat(tensors))

    def get_feature_counts(self, container: TokensContainer,
                           fvec: SpacyFeatureVectorizer) -> torch.Tensor:
        """Return the count of all tokens as a 1 X N tensor where N is the columns of
        the ``fvec`` vectorizer.  Each column position's count represents the
        number of counts for that spacy symol for that index position in the
        ``fvec``.

        """
        attr_name = fvec.feature_type
        fcounts = self.torch_config.zeros(fvec.shape[1])
        for tok in container.tokens:
            val = getattr(tok, attr_name)
            fnid = fvec.id_from_spacy(val, -1)
            if fnid > -1:
                fcounts[fnid] += 1
        return fcounts


TokenContainerFeatureVectorizerManager.register_vectorizer(CountTokenContainerFeatureVectorizer)


@dataclass
class DepthTokenContainerFeatureVectorizer(TokenContainerFeatureVectorizer):
    """Return the depths of tokens based on how deep they are in a head dependency
    tree.

    """
    NAME = 'head depth'
    FEATURE_TYPE = 'dep'

    def _get_shape(self) -> Tuple[int, int]:
        return self.token_length,

    def _encode(self, container: TokensContainer) -> FeatureContext:
        arr = self.torch_config.zeros((self.token_length,))
        if isinstance(container, FeatureDocument):
            for sent in container.sents:
                self._transform_sent(sent, arr)
        else:
            self._transform_sent(container, arr)
        return TensorFeatureContext(self.feature_type, arr)

    def _transform_sent(self, container: TokensContainer,  arr: torch.Tensor):
        head_depths = self._get_head_depth(container)
        for root, toks in head_depths:
            if root.i < self.token_length:
                arr[root.i] = 1.
            for ti, t in toks:
                if ti < self.token_length:
                    arr[ti] = 0.5

    def _get_head_depth(self, container: TokensContainer) -> \
            Tuple[FeatureToken, List[Tuple[int, List[FeatureToken]]]]:
        tid_to_idx = {}
        deps = []
        toks = container.tokens
        for i, tok in enumerate(toks):
            tid_to_idx[tok.i] = i
        root = tuple(
            filter(lambda t: t.dep_ == 'ROOT' and not t.is_punctuation, toks))
        if len(root) == 1:
            root = root[0]
            kids = set(root.children)
            ktoks = map(lambda t: (tid_to_idx[t.i], t),
                        filter(lambda t: not t.is_punctuation and t.i in kids,
                               toks))
            deps.append((root, tuple(ktoks)))
        return deps


TokenContainerFeatureVectorizerManager.register_vectorizer(DepthTokenContainerFeatureVectorizer)


@dataclass
class StatisticsTokenContainerFeatureVectorizer(TokenContainerFeatureVectorizer):
    """Return basic statics including: token count, sentence count (for
    FeatureDocuments).

    """
    NAME = 'statistics'
    FEATURE_TYPE = 'stats'

    def _get_shape(self) -> Tuple[int, int]:
        return 9,

    def _encode(self, container: TokensContainer) -> FeatureContext:
        n_toks = len(container.tokens)
        n_sents = 1
        min_tlen = sys.maxsize
        max_tlen = 0
        ave_tlen = 1
        min_slen = sys.maxsize
        max_slen = 0
        ave_slen = 1
        n_char = 0
        for t in container.tokens:
            tlen = len(t.norm)
            n_char += tlen
            min_tlen = min(min_tlen, tlen)
            max_tlen = max(max_tlen, tlen)
        ave_tlen = n_char / n_toks
        if isinstance(container, FeatureDocument):
            n_sents = len(container.sents)
            ave_slen = n_toks / n_sents
            for s in container.sents:
                slen = len(s.tokens)
                min_slen = min(min_slen, slen)
                max_slen = max(max_slen, slen)
        stats = (n_char, n_toks, min_tlen, max_tlen, ave_tlen,
                 n_sents, ave_slen, min_slen, max_slen)
        arr = self.torch_config.from_iterable(stats)
        return TensorFeatureContext(self.feature_type, arr)


TokenContainerFeatureVectorizerManager.register_vectorizer(StatisticsTokenContainerFeatureVectorizer)
