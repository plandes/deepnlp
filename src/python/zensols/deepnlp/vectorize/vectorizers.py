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
    TokenContainerFeatureType,
)

logger = logging.getLogger(__name__)


@dataclass
class EnumContainerFeatureVectorizer(TokenContainerFeatureVectorizer):
    """Encode tokens found in the container by aggregating the SpaCy vectorizers
    output.  The results is a concatenated binary representation of all
    configured token level features for each token.  This adds only token
    vectorizer features generated by the SpaCy vectorizers, and not the
    features themselves (such as ``is_stop`` etc).

    Rows are tokens, columns intervals are features.  The encoded matrix is
    sparse, and decoded as a dense matrix.

    """
    NAME = 'spacy feature vectorizer'
    FEATURE_TYPE = TokenContainerFeatureType.TOKEN
    feature_id: str
    decoded_feature_ids: Set[str] = field(default=None)

    def _get_shape_with_feature_ids(self, feature_ids: Set[str]):
        """Compute the shape based on what spacy feature ids are given.

        :param feature_ids: the spacy feature ids used to filter the result

        """
        flen = 0
        for fvec in self.manager.spacy_vectorizers.values():
            if feature_ids is None or fvec.feature_id in feature_ids:
                flen += fvec.shape[1]
        return self.token_length, flen

    def _get_shape_decode(self) -> Tuple[int, int]:
        """Return the shape needed for the tensor when encoding.

        """
        return self._get_shape_with_feature_ids(None)

    def _get_shape(self) -> Tuple[int, int]:
        """Compute the shape based on what spacy feature ids are given.

        """
        return self._get_shape_with_feature_ids(self.decoded_feature_ids)

    def _populate_feature_vectors(self, container: TokensContainer,
                                  fvec: SpacyFeatureVectorizer,
                                  arr: torch.Tensor,
                                  col_start: int, col_end: int):
        """Populate ``arr`` with every feature available from the vectorizer set
        defined in the manager.  This fills in the corresponding vectors from
        the spacy vectorizer ``fvec`` across all tokens for a column range.

        """
        attr_name = fvec.feature_id
        col_end = col_start + fvec.shape[1]
        toks = container.tokens[:arr.shape[0]]
        for i, tok in enumerate(toks):
            val = getattr(tok, attr_name)
            vec = fvec.from_spacy(val)
            if vec is not None:
                arr[i, col_start:col_end] = vec

    def _encode(self, container: TokensContainer) -> FeatureContext:
        """Encode tokens found in the container by aggregating the SpaCy vectorizers
        output.

        """
        col_start = 0
        arr = self.torch_config.zeros(self._get_shape_decode())
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'type array shape: {arr.shape}')
        for fvec in self.manager.spacy_vectorizers.values():
            col_end = col_start + fvec.shape[1]
            self._populate_feature_vectors(
                container, fvec, arr, col_start, col_end)
            col_start = col_end
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'array shape: {arr.shape}')
        return SparseTensorFeatureContext.instance(
            self.feature_id, arr, self.torch_config)

    def _slice_by_attributes(self, arr: torch.Tensor) -> torch.Tensor:
        """Create a new tensor from column based slices of the encoded tensor for each
        specified feature id given in :py:attrib:`~decoded_feature_ids`.

        """
        keeps = set(self.decoded_feature_ids)
        col_start = 0
        tensors = []
        for fvec in self.manager.spacy_vectorizers.values():
            col_end = col_start + fvec.shape[1]
            fid = fvec.feature_id
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'type={fid}, to keep={keeps}')
            if fid in keeps:
                tensors.append(arr[:, col_start:col_end])
                keeps.remove(fid)
            col_start = col_end
        if len(keeps) > 0:
            raise ValueError(f'unknown feature type IDs: {keeps}')
        return torch.cat(tensors, 1)

    def _decode(self, context: FeatureContext) -> torch.Tensor:
        if isinstance(context, SparseTensorFeatureContext):
            arr = context.to_tensor(self.manager.torch_config)
        else:
            arr = super()._decode(context)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'decoded features: {self.decoded_feature_ids}')
        if self.decoded_feature_ids is not None:
            arr = self._slice_by_attributes(arr)
        return arr


@dataclass
class CountTokenContainerFeatureVectorizer(TokenContainerFeatureVectorizer):
    """Return the count of all tokens as a 1 X M * N tensor where M is the number
    of token feature ids and N is the columns of the ``fvec`` vectorizer.  Each
    column position's count represents the number of counts for that spacy
    symol for that index position in the ``fvec``.

    """
    NAME = 'token level feature counts'
    FEATURE_ID = 'count'
    FEATURE_TYPE = TokenContainerFeatureType.DOCUMENT

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
            self.feature_id, torch.cat(tensors))

    def get_feature_counts(self, container: TokensContainer,
                           fvec: SpacyFeatureVectorizer) -> torch.Tensor:
        """Return the count of all tokens as a 1 X N tensor where N is the columns of
        the ``fvec`` vectorizer.  Each column position's count represents the
        number of counts for that spacy symol for that index position in the
        ``fvec``.

        """
        fid = fvec.feature_id
        fcounts = self.torch_config.zeros(fvec.shape[1])
        for tok in container.tokens:
            val = getattr(tok, fid)
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
    FEATURE_ID = 'dep'
    FEATURE_TYPE = TokenContainerFeatureType.DOCUMENT

    def _get_shape(self) -> Tuple[int, int]:
        return self.token_length,

    def _encode(self, container: TokensContainer) -> FeatureContext:
        arr = self.torch_config.zeros((self.token_length,))
        if isinstance(container, FeatureDocument):
            for sent in container.sents:
                self._transform_sent(sent, arr)
        else:
            self._transform_sent(container, arr)
        return TensorFeatureContext(self.feature_id, arr)

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
    FEATURE_ID = 'stats'
    FEATURE_TYPE = TokenContainerFeatureType.DOCUMENT

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
        return TensorFeatureContext(self.feature_id, arr)


TokenContainerFeatureVectorizerManager.register_vectorizer(StatisticsTokenContainerFeatureVectorizer)
