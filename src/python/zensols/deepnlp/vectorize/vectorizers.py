"""Generate and vectorize language features.

"""
__author__ = 'Paul Landes'

import logging
import sys
from typing import List, Tuple, Set
from dataclasses import dataclass, field
import torch
from zensols.deeplearn.vectorize import (
    FeatureContext,
    TensorFeatureContext,
    SparseTensorFeatureContext,
)
from zensols.deepnlp import (
    FeatureToken,
    FeatureDocument,
    TokensContainer,
)
from . import (
    SpacyFeatureVectorizer,
    TokenContainerFeatureVectorizer,
    TokenContainerFeatureVectorizerManager,
)

logger = logging.getLogger(__name__)


@dataclass
class EnumContainerFeatureVectorizer(TokenContainerFeatureVectorizer):
    """Creates a stacked binary representation of all configured token level
    features for each token.  This adds only token vectorizer features
    generated by the SpaCy vectorizers, and not the features themselves (such
    as ``is_stop`` etc).

    """
    NAME = 'spacy feature vectorizer'
    feature_type: str
    decoded_feature_types: Set[str] = field(default=None)

    def _get_shape_with_feature_types(self, feature_types: Set[str]):
        flen = 0
        for fvec in self.manager.spacy_vectorizers.values():
            if feature_types is None or fvec.feature_type in feature_types:
                flen += fvec.shape[1]
        return self.token_length, flen

    def _get_shape_decode(self) -> Tuple[int, int]:
        return self._get_shape_with_feature_types(None)

    def _get_shape(self) -> Tuple[int, int]:
        return self._get_shape_with_feature_types(self.decoded_feature_types)

    def get_feature_vectors(self, container: TokensContainer,
                            fvec: SpacyFeatureVectorizer,
                            arr: torch.Tensor, col_start: int, col_end: int):
        attr_name = fvec.feature_type
        col_end = col_start + fvec.shape[1]
        toks = container.tokens[:arr.shape[0]]
        for i, tok in enumerate(toks):
            val = getattr(tok, attr_name)
            vec = fvec.from_spacy(val)
            if vec is not None:
                arr[i, col_start:col_end] = vec

    def _encode(self, container: TokensContainer) -> FeatureContext:
        col_start = 0
        arr = self.torch_config.zeros(self._get_shape_decode())
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'type array shape: {arr.shape}')
        for fvec in self.manager.spacy_vectorizers.values():
            col_end = col_start + fvec.shape[1]
            self.get_feature_vectors(
                container, fvec, arr, col_start, col_end)
            col_start = col_end
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'array shape: {arr.shape}')
        return SparseTensorFeatureContext.instance(
            self.feature_type, arr, self.torch_config)

    def _slice_by_attributes(self, arr: torch.Tensor) -> torch.Tensor:
        keeps = self.decoded_feature_types
        col_start = 0
        tensors = []
        for fvec in self.manager.spacy_vectorizers.values():
            col_end = col_start + fvec.shape[1]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'type={fvec.feature_type}, to keep={keeps}')
            if fvec.feature_type in keeps:
                tensors.append(arr[:, col_start:col_end])
            col_start = col_end
        return torch.cat(tensors, 1)

    def _decode(self, context: FeatureContext) -> torch.Tensor:
        if isinstance(context, SparseTensorFeatureContext):
            arr = context.to_tensor(self.manager.torch_config)
        else:
            arr = super()._decode(context)
        if self.decoded_feature_types is not None:
            arr = self._slice_by_attributes(arr)
        return arr


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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'array shape: {arr.shape}')
        return TensorFeatureContext(self.feature_type, arr)


TokenContainerFeatureVectorizerManager.register_vectorizer(StatisticsTokenContainerFeatureVectorizer)
