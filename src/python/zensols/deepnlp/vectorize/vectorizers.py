
"""Generate and vectorize language features.

"""
__author__ = 'Paul Landes'

import logging
import sys
from typing import List, Tuple, Set
from dataclasses import dataclass, field
from functools import reduce
import torch
from zensols.deeplearn.vectorize import (
    FeatureContext,
    TensorFeatureContext,
    SparseTensorFeatureContext,
    MultiFeatureContext,
)
from zensols.deepnlp import (
    FeatureToken,
    FeatureSentence,
    FeatureDocument,
    TokensContainer,
)
from . import (
    SpacyFeatureVectorizer,
    TokenContainerFeatureVectorizer,
    TokenContainerFeatureType,
)

logger = logging.getLogger(__name__)


@dataclass
class EnumContainerFeatureVectorizer(TokenContainerFeatureVectorizer):
    """Encode tokens found in the container by aggregating the SpaCy vectorizers
    output.  The result is a concatenated binary representation of all
    configured token level features for each token.  This adds only token
    vectorizer features generated by the spaCy vectorizers, and not the
    features themselves (such as ``is_stop`` etc).

    All spaCy features are encoded given by
    :obj:`~.TokenContainerFeatureVectorizerManager.spacy_vectorizers`.
    However, only those given in :obj:`decoded_feature_ids` are produced in the
    output tensor after decoding.

    The motivation for encoding all, but decoding a subset of features is for
    feature selection during training.  This is because encoding the features
    (in a sparse matrix) takes comparatively less time and space over having to
    re-encode all batches.

    Rows are tokens, columns intervals are features.  The encoded matrix is
    sparse, and decoded as a dense matrix.

    :shape: ``(|sentences|, |token length|, |decoded features|)``

    """
    ATTR_EXP_META = ('decoded_feature_ids',)
    DESCRIPTION = 'spacy feature vectorizer'
    FEATURE_TYPE = TokenContainerFeatureType.TOKEN

    decoded_feature_ids: Set[str] = field(default=None)
    """The spaCy generated features used during *only* decoding (see class docs).
    Examples include ``norm``, ``ent``, ``dep``, ``tag``.  When set to
    ``None``, use all those given in the
    :obj:`~.TokenContainerFeatureVectorizerManager.spacy_vectorizers`.

    """

    def _get_shape_with_feature_ids(self, feature_ids: Set[str]):
        """Compute the shape based on what spacy feature ids are given.

        :param feature_ids: the spacy feature ids used to filter the result

        """
        flen = 0
        for fvec in self.manager.spacy_vectorizers.values():
            if feature_ids is None or fvec.feature_id in feature_ids:
                flen += fvec.shape[1]
        return None, self.token_length, flen

    def _get_shape_decode(self) -> Tuple[int, int]:
        """Return the shape needed for the tensor when encoding."""
        return self._get_shape_with_feature_ids(None)

    def _get_shape_for_document(self, doc: FeatureDocument):
        """Return the shape of the vectorized output for the given document."""
        return (len(doc.sents),
                self.manager.get_token_length(doc),
                self._get_shape_decode()[-1])

    def _get_shape(self) -> Tuple[int, int]:
        """Compute the shape based on what spacy feature ids are given."""
        return self._get_shape_with_feature_ids(self.decoded_feature_ids)

    def _populate_feature_vectors(self, sent: FeatureSentence, six: int,
                                  fvec: SpacyFeatureVectorizer,
                                  arr: torch.Tensor,
                                  col_start: int, col_end: int):
        """Populate ``arr`` with every feature available from the vectorizer set
        defined in the manager.  This fills in the corresponding vectors from
        the spacy vectorizer ``fvec`` across all tokens for a column range.

        """
        attr_name = fvec.feature_id
        col_end = col_start + fvec.shape[1]
        toks = sent.tokens[:arr.shape[1]]
        for tix, tok in enumerate(toks):
            val = getattr(tok, attr_name)
            vec = fvec.from_spacy(val)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'adding vec {fvec} for tok {tok}>: {vec}')
            if vec is not None:
                arr[six, tix, col_start:col_end] = vec

    def _encode(self, doc: FeatureDocument) -> FeatureContext:
        """Encode tokens found in the container by aggregating the SpaCy vectorizers
        output.

        """
        self._assert_doc(doc)
        arr = self.torch_config.zeros(self._get_shape_for_document(doc))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'type array shape: {arr.shape}')
        sent: FeatureSentence
        for six, sent in enumerate(doc.sents):
            col_start = 0
            for fvec in self.manager.spacy_vectorizers.values():
                col_end = col_start + fvec.shape[1]
                self._populate_feature_vectors(
                    sent, six, fvec, arr, col_start, col_end)
                col_start = col_end
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'array shape: {arr.shape}')
        return SparseTensorFeatureContext.instance(
            self.feature_id, arr, self.torch_config)

    def _slice_by_attributes(self, arr: torch.Tensor) -> torch.Tensor:
        """Create a new tensor from column based slices of the encoded tensor for each
        specified feature id given in :obj:`decoded_feature_ids`.

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
                tensors.append(arr[:, :, col_start:col_end])
                keeps.remove(fid)
            col_start = col_end
        if len(keeps) > 0:
            raise ValueError(f'unknown feature type IDs: {keeps}')
        return torch.cat(tensors, 1)

    def _decode(self, context: FeatureContext) -> torch.Tensor:
        arr = super()._decode(context)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'decoded features: {self.decoded_feature_ids}')
        if self.decoded_feature_ids is not None:
            arr = self._slice_by_attributes(arr)
        return arr


@dataclass
class CountEnumContainerFeatureVectorizer(TokenContainerFeatureVectorizer):
    """Return the count of all tokens as a S X M * N tensor where S is the number
    of sentences, M is the number of token feature ids and N is the number of
    columns of the output of the :class:`.SpacyFeatureVectorizer` vectorizer.
    Each column position's count represents the number of counts for that spacy
    symol for that index position in the output of
    :class:`.SpacyFeatureVectorizer`.

    This class uses the same efficiency in decoding features given in
    :class:`.EnumContainerFeatureVectorizer`.

    :shape: ``(|sentences|, |decoded features|,)``

    """
    ATTR_EXP_META = ('decoded_feature_ids',)
    DESCRIPTION = 'token level feature counts'
    FEATURE_TYPE = TokenContainerFeatureType.DOCUMENT

    decoded_feature_ids: Set[str] = field(default=None)

    def _get_shape(self) -> Tuple[int, int]:
        """Compute the shape based on what spacy feature ids are given.

        """
        feature_ids = self.decoded_feature_ids
        flen = 0
        for fvec in self.manager.spacy_vectorizers.values():
            if feature_ids is None or fvec.feature_id in feature_ids:
                flen += fvec.shape[1]
        return -1, flen,

    def get_feature_counts(self, sent: FeatureSentence,
                           fvec: SpacyFeatureVectorizer) -> torch.Tensor:
        """Return the count of all tokens as a S X N tensor where S is the number of
        sentences, N is the columns of the ``fvec`` vectorizer.  Each column
        position's count represents the number of counts for that spacy symol
        for that index position in the ``fvec``.

        """
        fid = fvec.feature_id
        fcounts = self.torch_config.zeros(fvec.shape[1])
        for tok in sent.tokens:
            val = getattr(tok, fid)
            fnid = fvec.id_from_spacy(val, -1)
            if fnid > -1:
                fcounts[fnid] += 1
        return fcounts

    def _encode(self, doc: FeatureDocument) -> FeatureContext:
        sent_arrs = []
        self._assert_doc(doc)
        for sent in doc.sents:
            tok_arrs = []
            for fvec in self.manager.spacy_vectorizers.values():
                tok_arrs.append(self.get_feature_counts(sent, fvec))
            sent_arrs.append(torch.cat(tok_arrs))
        arr = torch.stack(sent_arrs)
        return SparseTensorFeatureContext.instance(
            self.feature_id, arr, self.torch_config)

    def _slice_by_attributes(self, arr: torch.Tensor) -> torch.Tensor:
        """Create a new tensor from column based slices of the encoded tensor for each
        specified feature id given in :obj:`decoded_feature_ids`.

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
                keep_vec = arr[:, col_start:col_end]
                print(f'adding: {fid}: {keep_vec}')
                tensors.append(keep_vec)
                keeps.remove(fid)
            col_start = col_end
        if len(keeps) > 0:
            raise ValueError(f'unknown feature type IDs: {keeps}')
        return torch.cat(tensors, 0)

    def _decode(self, context: FeatureContext) -> torch.Tensor:
        arr = super()._decode(context)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'decoded features: {self.decoded_feature_ids}')
        if self.decoded_feature_ids is not None:
            arr = self._slice_by_attributes(arr)
        return arr


@dataclass
class DepthTokenContainerFeatureVectorizer(TokenContainerFeatureVectorizer):
    """Return the depths of tokens based on how deep they are in a head dependency
    tree.

    Even though this is a document level vectorizer and is usually added in a
    join layer rather than stacked on to the embedded layer, it still assumes
    congruence with the token length, which is used in its shape.

    :shape: ``(token length,)``

    """
    DESCRIPTION = 'head depth'
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


@dataclass
class StatisticsTokenContainerFeatureVectorizer(TokenContainerFeatureVectorizer):
    """Vectorizes basic surface language statics which include:

        * character count
        * token count
        * min token length in characters
        * max token length in characters
        * average token length in characters (|characters| / |tokens|)
        * sentence count (for FeatureDocuments)
        * average sentence length (|tokens| / |sentences|)
        * min sentence length
        * max sentence length

    :shape: ``(9,)``

    """
    DESCRIPTION = 'statistics'
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


@dataclass
class OverlappingTokenContainerFeatureVectorizer(TokenContainerFeatureVectorizer):
    """Vectorize the number of normalized and lemmatized tokens (in this order)
    across multiple documents.

    The input to this feature vectorizer are a tuple N of
    :class:`.TokenContainer` instances.

    :shape: ``(2,)``

    """
    DESCRIPTION = 'overlapping token counts'
    FEATURE_TYPE = TokenContainerFeatureType.DOCUMENT

    def _get_shape(self) -> Tuple[int, int]:
        return 2,

    @staticmethod
    def _norms(ac: TokensContainer, bc: TokensContainer) -> Tuple[int]:
        a = set(map(lambda s: s.norm.lower(), ac.token_iter()))
        b = set(map(lambda s: s.norm.lower(), bc.token_iter()))
        return a & b

    @staticmethod
    def _lemmas(ac: TokensContainer, bc: TokensContainer) -> Tuple[int]:
        a = set(map(lambda s: s.lemma.lower(), ac.token_iter()))
        b = set(map(lambda s: s.lemma.lower(), bc.token_iter()))
        return a & b

    def _encode(self, containers: Tuple[TokensContainer]) -> FeatureContext:
        norms = reduce(self._norms, containers)
        lemmas = reduce(self._lemmas, containers)
        arr = self.torch_config.from_iterable((len(norms), len(lemmas)))
        return TensorFeatureContext(self.feature_id, arr)


@dataclass
class MutualFeaturesContainerFeatureVectorizer(TokenContainerFeatureVectorizer):
    """Vectorize the shared count of all tokens as a S X M * N tensor, where S is
    the number of sentences, M is the number of token feature ids and N is the
    columns of the output of the :class:`.SpacyFeatureVectorizer` vectorizer.

    This uses an instance of :class:`CountEnumContainerFeatureVectorizer` to
    compute across each spacy feature and then sums them up for only those
    features shared.  If at least one shared document has a zero count, the
    features is zeroed.

    The input to this feature vectorizer are a tuple of N
    :class:`.TokenContainer` instances.

    :shape: ``(|sentences|, |decoded features|,)`` from the referenced
            :class:`CountEnumContainerFeatureVectorizer` given by
            :obj:`count_vectorizer_feature_id`

    """
    DESCRIPTION = 'mutual feature counts'
    FEATURE_TYPE = TokenContainerFeatureType.DOCUMENT

    count_vectorizer_feature_id: str = field()
    """The string feature ID configured in the
    :class:`.TokenContainerFeatureVectorizerManager` of the
    :class:`CountEnumContainerFeatureVectorizer` to use for the count features.

    """

    @property
    def count_vectorizer(self) -> CountEnumContainerFeatureVectorizer:
        """Return the count vectorizer used for the count features.

        :see: :obj:`count_vectorizer_feature_id`

        """
        return self.manager[self.count_vectorizer_feature_id]

    @property
    def ones(self) -> torch.Tensor:
        """Return a tensor of ones for the shape of this instance.

        """
        return self.torch_config.ones((1, self.shape[1]))

    def _get_shape(self) -> Tuple[int, int]:
        return self.count_vectorizer.shape

    def _encode(self, containers: Tuple[TokensContainer]) -> FeatureContext:
        ctxs = tuple(map(self.count_vectorizer.encode, containers))
        return MultiFeatureContext(self.feature_id, ctxs)

    def _decode(self, context: MultiFeatureContext) -> torch.Tensor:
        def decode_context(ctx):
            sents = self.count_vectorizer.decode(ctx)
            return torch.sum(sents, axis=0)

        ones = self.ones
        arrs = tuple(map(decode_context, context.contexts))
        if len(arrs) == 1:
            return arrs[0]
        else:
            arrs = torch.stack(arrs, axis=0)
            cnts = self.torch_config.clone(arrs)
            mask = torch.min(torch.cat((cnts.prod(axis=0).unsqueeze(0), ones)),
                             axis=0)[0]
            return (cnts * mask).sum(axis=0)
