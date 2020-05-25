"""Generate and vectorize language features.

"""
__author__ = 'Paul Landes'

import logging
import sys
from typing import List, Tuple
from dataclasses import dataclass
import torch
from zensols.deeplearn.vectorize import FeatureContext, TensorFeatureContext
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
