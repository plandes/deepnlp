"""Generate and vectorize language features.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple, Set, Union, Dict, Iterable, Callable
from dataclasses import dataclass, field
import logging
import sys
from functools import reduce
import textwrap as tw
from io import TextIOBase
import torch
import numpy as np
from torch import Tensor
from zensols.deeplearn.vectorize import (
    VectorizerError,
    FeatureContext,
    TensorFeatureContext,
    SparseTensorFeatureContext,
    MultiFeatureContext,
    EncodableFeatureVectorizer,
    OneHotEncodedEncodableFeatureVectorizer,
    AggregateEncodableFeatureVectorizer,
    TransformableFeatureVectorizer,
)
from zensols.nlp import (
    FeatureToken, FeatureSentence, FeatureDocument, TokenContainer,
)
from ..embed import WordEmbedModel
from . import (
    SpacyFeatureVectorizer, FeatureDocumentVectorizer,
    TextFeatureType, MultiDocumentVectorizer,
)

logger = logging.getLogger(__name__)


@dataclass
class DecodedContainerFeatureVectorizer(FeatureDocumentVectorizer):
    """A base class that allows for configuring decoded features after batches
    are created at train time.

    """
    decoded_feature_ids: Set[str] = field(default=None)
    """The spaCy generated features used during *only* decoding (see class
    docs).  Examples include ``norm``, ``ent``, ``dep``, ``tag``.  When set to
    ``None``, use all those given in the
    :obj:`~.FeatureDocumentVectorizerManager.spacy_vectorizers`.

    """
    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        dfids: Set[str] = self.decoded_feature_ids
        super().write(depth, writer)
        self._write_line('decoded spacy vectorizers:', depth, writer)
        for fvec in self.manager.spacy_vectorizers.values():
            decoded: bool = dfids is None or fvec.feature_id in dfids
            self._write_line(f'{fvec.feature_id}: {decoded}',
                             depth + 1, writer)


@dataclass
class EnumContainerFeatureVectorizer(DecodedContainerFeatureVectorizer):
    """Encode tokens found in the container by aggregating the spaCy vectorizers
    output.  The result is a concatenated binary representation of all
    configured token level features for each token.  This adds only token
    vectorizer features generated by the spaCy vectorizers (subclasses of
    :class:`.SpacyFeatureVectorizer`), and not the features themselves (such as
    ``is_stop`` etc).

    All spaCy features are encoded given by
    :obj:`~.FeatureDocumentVectorizerManager.spacy_vectorizers`.
    However, only those given in :obj:`decoded_feature_ids` are produced in the
    output tensor after decoding.

    The motivation for encoding all, but decoding a subset of features is for
    feature selection during training.  This is because encoding the features
    (in a sparse matrix) takes comparatively less time and space over having to
    re-encode all batches.

    Rows are tokens, columns intervals of features.  The encoded matrix is
    sparse, and decoded as a dense matrix.

    :shape: (|sentences|, |sentinel tokens|, |decoded features|)

    :see: :class:`.SpacyFeatureVectorizer`

    """
    ATTR_EXP_META = ('decoded_feature_ids',)
    DESCRIPTION = 'spacy feature vectorizer'
    FEATURE_TYPE = TextFeatureType.TOKEN

    string_symbol_feature_ids: Set[str] = field(default=None)
    """Feature IDs of vectorizers that use string symbols rather than their
    integers, which are used to look up the string equivelants in
    :obj:`spacy.vocab.Vocab.strings`.

    """
    def _get_shape_with_feature_ids(self, feature_ids: Set[str]):
        """Compute the shape based on what spacy feature ids are given.

        :param feature_ids: the spacy feature ids used to filter the result

        """
        flen: int = 0
        for fvec in self.manager.spacy_vectorizers.values():
            if feature_ids is None or fvec.feature_id in feature_ids:
                flen += fvec.shape[1]
        return -1, self.token_length, flen

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
                                  fvec: SpacyFeatureVectorizer, arr: Tensor,
                                  col_start: int, col_end: int):
        """Populate ``arr`` with every feature available from the vectorizer set
        defined in the manager.  This fills in the corresponding vectors from
        the spacy vectorizer ``fvec`` across all tokens for a column range.

        """
        fid: str = fvec.feature_id
        col_end: int = col_start + fvec.shape[1]
        toks: List[FeatureToken] = sent.tokens[:arr.shape[1]]
        desc: str = f'in {self.manager.doc_parser}'
        string_fids: Set[str] = self.string_symbol_feature_ids
        map_fn: Callable = fvec.from_spacy
        if string_fids is not None and fid in string_fids:
            map_fn = fvec.symbol_to_vector.get
        tix: int
        tok: FeatureToken
        for tix, tok in enumerate(toks):
            val: Union[str, int] = tok.get_feature(
                feature_id=fid,
                message=desc)
            vec: Tensor = map_fn(val)
            if logger.isEnabledFor(logging.TRACE):
                logger.trace(f'encode value: {val}')
            if vec is not None:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'adding vec {fvec} for {tok}: {vec.shape}')
                arr[six, tix, col_start:col_end] = vec

    def _encode(self, doc: FeatureDocument) -> FeatureContext:
        """Encode tokens found in the container by aggregating the spaCy
        vectorizers output.

        """
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
            logger.debug(f'encoded array shape: {arr.shape}')
        return SparseTensorFeatureContext.instance(
            self.feature_id, arr, self.torch_config)

    def _slice_by_attributes(self, arr: Tensor) -> Tensor:
        """Create a new tensor from column based slices of the encoded tensor
        for each specified feature id given in :obj:`decoded_feature_ids`.

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
            raise VectorizerError(f'Unknown feature type IDs: {keeps}')
        sarr = torch.cat(tensors, dim=2)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'slice dim: {sarr.shape}')
        return sarr

    def to_symbols(self, tensor: Tensor) -> List[List[Dict[str, float]]]:
        """Reverse map the tensor to spaCy features.

        :return: a list of sentences, each with a list of tokens, each having a
                 map of name/count pairs

        """
        sents = []
        for six in range(tensor.size(0)):
            toks = []
            sents.append(toks)
            for tix in range(tensor.size(1)):
                col_start = 0
                by_fid = {}
                toks.append(by_fid)
                for fvec in self.manager.spacy_vectorizers.values():
                    col_end = col_start + fvec.shape[1]
                    fid = fvec.feature_id
                    vec = tensor[six, tix, col_start:col_end]
                    cnts = dict(filter(lambda x: x[1] > 0,
                                       zip(fvec.symbols, vec.tolist())))
                    by_fid[fid] = cnts
                    col_start = col_end
        return sents

    def _decode(self, context: FeatureContext) -> Tensor:
        arr = super()._decode(context)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'decoded features: {self.decoded_feature_ids}, ' +
                         f'shape: {arr.shape}')
        self._assert_decoded_doc_dim(arr, 3)
        if self.decoded_feature_ids is not None:
            arr = self._slice_by_attributes(arr)
        return arr


@dataclass
class CountEnumContainerFeatureVectorizer(DecodedContainerFeatureVectorizer):
    """Vectorize the counts of parsed spaCy features.  This generates the count
    of tokens as a S X M * N tensor where S is the number of sentences, M is the
    number of token feature ids and N is the number of columns of the output of
    the :class:`.SpacyFeatureVectorizer` vectorizer.  Each column position's
    count represents the number of counts for that spacy symol for that index
    position in the output of :class:`.SpacyFeatureVectorizer`.

    This class uses the same efficiency in decoding features given in
    :class:`.EnumContainerFeatureVectorizer`.

    :shape: (|sentences|, |decoded features|)

    """
    ATTR_EXP_META = ('decoded_feature_ids',)
    DESCRIPTION = 'token level feature counts'
    FEATURE_TYPE = TextFeatureType.DOCUMENT

    string_symbol_feature_ids: Set[str] = field(default=None)
    """Feature IDs of vectorizers that use string symbols rather than their
    integers, which are used to look up the string equivelants in
    :obj:`spacy.vocab.Vocab.strings`.

    """
    def _get_spacy_vectorizers(self) -> List[SpacyFeatureVectorizer]:
        feature_ids: Set[str] = self.decoded_feature_ids
        fvecs: List[SpacyFeatureVectorizer] = []
        fvec: SpacyFeatureVectorizer
        for _, fvec in self.manager.ordered_spacy_vectorizers:
            if feature_ids is None or fvec.feature_id in feature_ids:
                fvecs.append(fvec)
        return fvecs

    def _get_shape(self) -> Tuple[int, int]:
        """Compute the shape based on what spacy feature ids are given.

        """
        fln: int = sum(map(lambda v: v.shape[1], self._get_spacy_vectorizers()))
        if logger.isEnabledFor(logging.DEBUG):
            vecs = ', '.join(map(str, self.manager.ordered_spacy_vectorizers))
            logger.debug(f'count enum feature len: {fln}, vecs: {vecs}')
        return -1, fln

    def get_feature_counts(self, sent: FeatureSentence,
                           fvec: SpacyFeatureVectorizer) -> Tensor:
        """Return the count of all tokens as a S X N tensor where S is the
        number of sentences, N is the columns of the ``fvec`` vectorizer.  Each
        column position's count represents the number of counts for that spacy
        symol for that index position in the ``fvec``.

        """
        fid: str = fvec.feature_id
        fcounts: Tensor = self.torch_config.zeros(fvec.shape[1])
        desc: str = f'in vec={self}, parser={self.manager.doc_parser}'
        string_fids: Set[str] = self.string_symbol_feature_ids
        map_fn: Callable = fvec.id_from_spacy
        if string_fids is not None and fid in string_fids:
            map_fn = fvec.symbol_to_id.get
        tok: FeatureToken
        for tok in sent.tokens:
            val: Union[str, int] = tok.get_feature(
                feature_id=fid,
                message=desc)
            fnid: int = map_fn(val, -1)
            if fnid > -1:
                fcounts[fnid] += 1
        return fcounts

    def _encode(self, doc: FeatureDocument) -> FeatureContext:
        if logger.isEnabledFor(logging.DEBUG):
            vec_shape: Tuple[int, int] = self._get_shape()
            text: str = tw.shorten(str(doc), 80)
            logger.debug(f'encoding, shape: {vec_shape} doc: {text}')
        sent_arrs = []
        for sent in doc.sents:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'encoding sentence: {sent}')
            tok_arrs = []
            for _, fvec in self.manager.ordered_spacy_vectorizers:
                cnts: Tensor = self.get_feature_counts(sent, fvec)
                if logger.isEnabledFor(logging.TRACE):
                    logger.trace(f'encoding with {fvec}')
                tok_arrs.append(cnts)
            sent_arrs.append(torch.cat(tok_arrs))
        arr = torch.stack(sent_arrs)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'encoded shape: {arr.shape}')
        return SparseTensorFeatureContext.instance(
            self.feature_id, arr, self.torch_config)

    def _slice_by_attributes(self, arr: Tensor) -> Tensor:
        """Create a new tensor from column based slices of the encoded tensor
        for each specified feature id given in :obj:`decoded_feature_ids`.

        """
        keeps: Set[str] = set(self.decoded_feature_ids)
        col_start: int = 0
        tensors: List[Tensor] = []
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('registered spacy vectorizers: ' +
                         f"{', '.join(self.manager.spacy_vectorizers.keys())}")
            logger.debug(f'keeping: {keeps}')
        fvec: SpacyFeatureVectorizer
        for _, fvec in self.manager.ordered_spacy_vectorizers:
            col_end: int = col_start + fvec.shape[1]
            fid: str = fvec.feature_id
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'type={fid}, to keep={keeps}')
            if fid in keeps:
                keep_vec = arr[:, col_start:col_end]
                tensors.append(keep_vec)
                keeps.remove(fid)
            col_start = col_end
        if len(keeps) > 0:
            raise VectorizerError(f'Unknown feature type IDs: {keeps}')
        sarr = torch.cat(tensors, dim=1)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'slice dim: {sarr.shape}')
        return sarr

    def to_symbols(self, tensor: Tensor) -> List[Dict[str, float]]:
        """Reverse map the tensor to spaCy features.

        :return: a list of sentences, each a map of name/count pairs.

        """
        sents = []
        for six in range(tensor.size(0)):
            col_start = 0
            by_fid = {}
            sents.append(by_fid)
            arr = tensor[six]
            for _, fvec in self.manager.ordered_spacy_vectorizers:
                col_end = col_start + fvec.shape[1]
                fid = fvec.feature_id
                vec = arr[col_start:col_end]
                cnts = dict(filter(lambda x: x[1] > 0,
                                   zip(fvec.symbols, vec.tolist())))
                by_fid[fid] = cnts
                col_start = col_end
        return sents

    def _decode(self, context: FeatureContext) -> Tensor:
        arr = super()._decode(context)
        if logger.isEnabledFor(logging.DEBUG):
            feature_ids: Set[str] = self.decoded_feature_ids
            fvecs: List[SpacyFeatureVectorizer] = self._get_spacy_vectorizers()
            fvstr = ', '.join(map(str, fvecs))
            logger.debug(f'decoding features(configured)={feature_ids}, ' +
                         f'decode shape={tuple(arr.shape)}, ' +
                         f'vec shape={self.shape}, ' +
                         f'vectorizers=[{fvstr}]')
        if self.decoded_feature_ids is not None:
            arr = self._slice_by_attributes(arr)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'decoded shape: {arr.shape}')
        return arr


@dataclass
class DepthFeatureDocumentVectorizer(FeatureDocumentVectorizer):
    """Generate the depths of tokens based on how deep they are in a head
    dependency tree.

    Even though this is a document level vectorizer and is usually added in a
    join layer rather than stacked on to the embedded layer, it still assumes
    congruence with the token length, which is used in its shape.

    **Important**: do not combine sentences in to a single document with
    :meth:`~zensols.nlp.container.FeatureDocument.combine_sentences` since
    features are created as a dependency parse tree at the sentence level.
    Otherwise, the dependency relations are broken and results in a zeored
    tensor.

    :shape: (|sentences|, |sentinel tokens|, 1)

    """
    DESCRIPTION = 'head depth'
    FEATURE_TYPE = TextFeatureType.TOKEN

    def _get_shape(self) -> Tuple[int, int]:
        return -1, self.token_length, 1

    def encode(self, doc: Union[Tuple[FeatureDocument], FeatureDocument]) -> \
            FeatureContext:
        ctx: TensorFeatureContext
        if isinstance(doc, (tuple, list)):
            self._assert_doc(doc)
            docs = doc
            comb_doc = FeatureDocument.combine_documents(docs)
            n_toks = self.manager.get_token_length(comb_doc)
            arrs = tuple(map(lambda d:
                             self._encode_doc(d.combine_sentences(), n_toks),
                             docs))
            arr = torch.cat(arrs, dim=0)
            arr = arr.unsqueeze(-1)
            ctx = SparseTensorFeatureContext.instance(
                self.feature_id, arr, self.torch_config)
        else:
            ctx = super().encode(doc)
        return ctx

    def _encode(self, doc: FeatureDocument) -> FeatureContext:
        n_toks = self.manager.get_token_length(doc)
        arr = self._encode_doc(doc, n_toks)
        arr = arr.unsqueeze(-1)
        return SparseTensorFeatureContext.instance(
            self.feature_id, arr, self.torch_config)

    def _encode_doc(self, doc: FeatureDocument, n_toks: int) -> Tensor:
        n_sents = len(doc.sents)
        arr = self.torch_config.zeros((n_sents, n_toks))
        u_doc = doc.uncombine_sentences()
        if logger.isEnabledFor(logging.DEBUG):
            text: str = tw.shorten(str(doc), 80)
            logger.debug(f'encoding doc: {len(doc)}/{len(u_doc)}: {text}')
        # if the doc is combined as several sentences concatenated in one, un
        # pack and write all features in one row
        if len(doc) != len(u_doc):
            soff = 0
            for sent in u_doc.sents:
                self._transform_sent(sent, arr, 0, soff, n_toks)
                soff += len(sent)
        else:
            # otherwise, each row is a separate sentence
            for six, sent in enumerate(doc.sents):
                self._transform_sent(sent, arr, six, 0, n_toks)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'encoded shape: {arr.shape}')
        return arr

    def _transform_sent(self, sent: FeatureSentence, arr: Tensor,
                        six: int, soff: int, slen: int):
        head_depths = self._get_head_depth(sent)
        for tix, tok, depth in head_depths:
            off = tix + soff
            val = 1. / depth
            in_range = (off < slen)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'setting ({six}, {off}) = {val}: set={in_range}')
            if in_range:
                arr[six, off] = val

    def _dep_branch(self, node: FeatureToken, toks: Tuple[FeatureToken],
                    tid_to_idx: Dict[int, int], depth: int,
                    depths: Dict[int, int]) -> \
            Dict[FeatureToken, List[FeatureToken]]:
        idx = tid_to_idx.get(node.i)
        if idx is not None:
            depths[idx] = depth
        for c in node.children:
            cix = tid_to_idx.get(c)
            if cix is not None:
                child = toks[cix]
                self._dep_branch(child, toks, tid_to_idx, depth + 1, depths)

    def _get_head_depth(self, sent: FeatureSentence) -> \
            Tuple[Tuple[int, FeatureToken, int]]:
        """Calculate the depth of tokens in a sentence.

        :param sent: the sentence that has the tokens to get depts

        :return: a tuple of (sentence token index, token, depth)

        """
        tid_to_idx: Dict[int, int] = {}
        toks = sent.tokens
        for i, tok in enumerate(toks):
            tid_to_idx[tok.i] = i
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('|'.join(
                map(lambda t: f'{tid_to_idx[t.i]}:{t.i}:{t.text}({t.dep_})',
                    sent.token_iter())))
            logger.debug(f'tree: {sent.dependency_tree}')
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'tokens: {toks}')
        root = tuple(
            filter(lambda t: t.dep_ == 'ROOT' and not t.is_punctuation, toks))
        if len(root) == 1:
            root = root[0]
            tree = {tid_to_idx[root.i]: 0}
            try:
                self._dep_branch(root, toks, tid_to_idx, 1, tree)
            except Exception as e:
                dstr: str = 'Could not vectorize depth for'
                try:
                    dstr = f'sentence <{dstr}>, root: {root}, tree: {tree}'
                except Exception as e:
                    dstr = f'{dstr} <error: {e}>'
                raise VectorizerError(
                    f'Could not vectorize depth for : <{dstr}>') from e
            return map(lambda x: (x[0], toks[x[0]], x[1]), tree.items())
        else:
            return ()


@dataclass
class OneHotEncodedFeatureDocumentVectorizer(
        FeatureDocumentVectorizer, OneHotEncodedEncodableFeatureVectorizer):
    """Vectorize nominal enumerated features in to a one-hot encoded vectors.
    The feature is taken from a :class:`~zensols.nlp.FeatureToken`.  If
    :obj:`level` is ``token`` then the features are token attributes identified
    by :obj:`feature_attribute`.  If the :obj:`level` is ``document`` feature is
    taken from the document.

    :shape:

        * level = document: (1, |categories|)

        * level = token: (|<sentences>|, |<sentinel tokens>|, |categories|)

    """
    DESCRIPTION = 'encoded feature document vectorizer'

    feature_attribute: Tuple[str] = field(default=None)
    """The feature attributes to vectorize."""

    level: str = field(default='token')
    """The level at which to take the attribute value, which is ``document``,
    ``sentence`` or ``token``.

    """
    def __post_init__(self):
        super().__post_init__()
        self.optimize_bools = False

    @property
    def feature_type(self) -> TextFeatureType:
        return {'document': TextFeatureType.DOCUMENT,
                'token': TextFeatureType.TOKEN,
                }[self.level]

    def _get_shape(self) -> Tuple[int, int]:
        if self.level == 'document':
            return -1, super()._get_shape()[1]
        else:
            return -1, self.token_length, super()._get_shape()[1]

    def _encode(self, doc: FeatureDocument) -> FeatureContext:
        attr = self.feature_attribute
        if self.level == 'document':
            arr = self.torch_config.zeros((1, self.shape[1]))
            feats = [getattr(doc, attr)]
            self._encode_cats(feats, arr)
        elif self.level == 'token':
            # not tested
            tlen = self.manager.get_token_length(doc)
            arr = self.torch_config.zeros((len(doc), tlen, self.shape[2]))
            for six, sent in enumerate(doc.sents):
                feats = tuple(map(lambda s: getattr(s, attr), sent))
                self._encode_cats(feats, arr[six])
        else:
            raise VectorizerError(f'Unknown doc level: {self.level}')
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'vectorized: {len(doc)} sents in to {arr.shape}')
        return SparseTensorFeatureContext.instance(
            self.feature_id, arr, self.torch_config)


@dataclass
class TokenEmbeddingFeatureVectorizer(
        AggregateEncodableFeatureVectorizer, FeatureDocumentVectorizer):
    """A :class:`~zensols.deepnlp.vectorize.AggregateEncodableFeatureVectorizer`
    that is useful for token level classification (i.e. NER).  It uses a
    delegate to first vectorizer the features, then concatenates in to one
    aggregate.

    In shape terms, this takes the single sentence position.  The additional
    unsqueezed dimensions set with :obj:`n_unsqueeze` is useful when the
    delegate vectorizer encodes booleans or any other value that does not take
    an additional dimension.

    :shape: (1, |tokens|, <delegate vectorizer shape>[, <unsqueeze dimensions])

    """
    DESCRIPTION = 'token aggregate vectorizer'

    level: TextFeatureType = field(default=TextFeatureType.TOKEN)
    """The level at which to take the attribute value, which is ``document``,
    ``sentence`` or ``token``.

    """
    add_dims: int = field(default=0)
    """The number of dimensions to add (see class docs)."""

    def _get_shape(self):
        dim = [1]
        dim.extend(super()._get_shape())
        dim.extend([1] * self.add_dims)
        return tuple(dim)

    @property
    def feature_type(self) -> TextFeatureType:
        return self.level

    def encode(self, doc: Union[Tuple[FeatureDocument], FeatureDocument]) -> \
            FeatureContext:
        return TransformableFeatureVectorizer.encode(self, doc)

    def _decode(self, context: MultiFeatureContext) -> Tensor:
        tensor: Tensor = super()._decode(context)
        for _ in range(self.add_dims):
            return tensor.unsqueeze(-1)
        return tensor


@dataclass
class StatisticsFeatureDocumentVectorizer(FeatureDocumentVectorizer):
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

    :shape: (1, 9,)

    """
    DESCRIPTION = 'statistics'
    FEATURE_TYPE = TextFeatureType.DOCUMENT

    def _get_shape(self) -> Tuple[int, int]:
        return -1, 9

    def _encode(self, doc: FeatureDocument) -> FeatureContext:
        n_toks = len(doc.tokens)
        n_sents = 1
        min_tlen = sys.maxsize
        max_tlen = 0
        ave_tlen = 1
        min_slen = sys.maxsize
        max_slen = 0
        ave_slen = 1
        n_char = 0
        for t in doc.tokens:
            tlen = len(t.norm)
            n_char += tlen
            min_tlen = min(min_tlen, tlen)
            max_tlen = max(max_tlen, tlen)
        ave_tlen = n_char / n_toks
        if isinstance(doc, FeatureDocument):
            n_sents = len(doc.sents)
            ave_slen = n_toks / n_sents
            for s in doc.sents:
                slen = len(s.tokens)
                min_slen = min(min_slen, slen)
                max_slen = max(max_slen, slen)
        stats = (n_char, n_toks, min_tlen, max_tlen, ave_tlen,
                 n_sents, ave_slen, min_slen, max_slen)
        arr = self.torch_config.from_iterable(stats).unsqueeze(0)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'array shape: {arr.shape}')
        return TensorFeatureContext(self.feature_id, arr)


@dataclass
class OverlappingFeatureDocumentVectorizer(MultiDocumentVectorizer):
    """Vectorize the number of normalized and lemmatized tokens (in this order)
    across multiple documents.

    The input to this feature vectorizer are a tuple N of
    :class:`.FeatureDocument` instances.

    :shape: (2,)

    """
    DESCRIPTION = 'overlapping token counts'

    def _get_shape(self) -> Tuple[int, int]:
        return 2,

    @staticmethod
    def _norms(ac: TokenContainer, bc: TokenContainer) -> Tuple[int]:
        a = set(map(lambda s: s.norm.lower(), ac.token_iter()))
        b = set(map(lambda s: s.norm.lower(), bc.token_iter()))
        return a & b

    @staticmethod
    def _lemmas(ac: TokenContainer, bc: TokenContainer) -> Tuple[int]:
        a = set(map(lambda s: s.lemma_.lower(), ac.token_iter()))
        b = set(map(lambda s: s.lemma_.lower(), bc.token_iter()))
        return a & b

    def _encode(self, docs: Tuple[FeatureDocument]) -> FeatureContext:
        norms = reduce(self._norms, docs)
        lemmas = reduce(self._lemmas, docs)
        arr = self.torch_config.from_iterable((len(norms), len(lemmas)))
        return TensorFeatureContext(self.feature_id, arr)


@dataclass
class MutualFeaturesContainerFeatureVectorizer(MultiDocumentVectorizer):
    """Vectorize the shared count of all tokens as a S X M * N tensor, where S
    is the number of sentences, M is the number of token feature ids and N is
    the columns of the output of the :class:`.SpacyFeatureVectorizer`
    vectorizer.

    This uses an instance of :class:`CountEnumContainerFeatureVectorizer` to
    compute across each spacy feature and then sums them up for only those
    features shared.  If at least one shared document has a zero count, the
    features is zeroed.

    The input to this feature vectorizer are a tuple of N
    :class:`.TokenContainer` instances.

    :shape: (|sentences|, |decoded features|,) from the referenced
            :class:`CountEnumContainerFeatureVectorizer` given by
            :obj:`count_vectorizer_feature_id`

    """
    DESCRIPTION = 'mutual feature counts'

    count_vectorizer_feature_id: str = field()
    """The string feature ID configured in the
    :class:`.FeatureDocumentVectorizerManager` of the
    :class:`CountEnumContainerFeatureVectorizer` to use for the count features.

    """
    @property
    def count_vectorizer(self) -> CountEnumContainerFeatureVectorizer:
        """Return the count vectorizer used for the count features.

        :see: :obj:`count_vectorizer_feature_id`

        """
        return self.manager[self.count_vectorizer_feature_id]

    @property
    def ones(self) -> Tensor:
        """Return a tensor of ones for the shape of this instance.

        """
        return self.torch_config.ones((1, self.shape[1]))

    def _get_shape(self) -> Tuple[int, int]:
        return -1, self.count_vectorizer.shape[1]

    def _encode(self, docs: Tuple[FeatureDocument]) -> FeatureContext:
        ctxs = tuple(map(self.count_vectorizer.encode,
                         map(lambda doc: doc.combine_sentences(), docs)))
        return MultiFeatureContext(self.feature_id, ctxs)

    def _decode(self, context: MultiFeatureContext) -> Tensor:
        def decode_context(ctx):
            sents = self.count_vectorizer.decode(ctx)
            return torch.sum(sents, axis=0)

        ones = self.ones
        arrs = tuple(map(decode_context, context.contexts))
        if len(arrs) == 1:
            # return the single document as a mutual count against itself
            return arrs[0]
        else:
            arrs = torch.stack(arrs, axis=0).squeeze(1)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'combined counts (doc/row): {arrs.shape}')
            # clone so the operations of this vectorizer do not effect the
            # tensors from the delegate count vectorizer
            cnts = self.torch_config.clone(arrs)
            # multiple counts of all docs so any 0 count feature will be 0 in
            # the mask
            prod = cnts.prod(axis=0).unsqueeze(0)
            # create 2 X N with count product with ones
            cat_ones = torch.cat((prod, ones))
            # keep 0s for no count features or 1 if there is at least one for
            # the mask
            mask = torch.min(cat_ones, axis=0)[0]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'counts mask: {cat_ones.shape}')
            # use the mask to zero out counts that aren't mutual across all
            # documents, then sum the counts across docuemnts
            return (cnts * mask).sum(axis=0).unsqueeze(0)


@dataclass
class WordEmbeddingFeatureVectorizer(EncodableFeatureVectorizer):
    """Vectorizes string tokens in to word embedded vectors.  This class works
    directly with the string tokens rather than
    :class:`~zensols.nlp.FeatureDocument` instances.  It can be useful when
    there's a need to vectorize tokens outside of a feature document
    (i.e. ``cui2vec``).

    """
    FEATURE_TYPE = TextFeatureType.EMBEDDING
    DESCRIPTION = 'word embedding encoder'

    embed_model: WordEmbedModel = field()
    """The word embedding model that has the string tokens to vector mapping."""

    def _get_shape(self):
        return (-1, self.embed_model.vector_dimension)

    def _encode(self, keys: Iterable[str]) -> FeatureContext:
        em: WordEmbedModel = self.embed_model
        vecs: np.ndarray = tuple(map(lambda k: em.get(k), keys))
        arr: np.ndarray = np.stack(vecs)
        return TensorFeatureContext(self.feature_id, torch.from_numpy(arr))
