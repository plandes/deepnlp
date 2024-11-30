"""The tokenizer object.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple, Dict, Set, ClassVar, Any
from dataclasses import dataclass, field
import logging
from frozendict import frozendict
import torch
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from zensols.nlp import FeatureDocument
from zensols.persist import persisted, PersistableContainer
from zensols.deeplearn import TorchConfig
from zensols.deepnlp.transformer import TransformerResource
from . import TransformerError, TokenizedFeatureDocument

logger = logging.getLogger(__name__)


@dataclass
class TransformerDocumentTokenizer(PersistableContainer):
    """Creates instances of :class:`.TokenziedFeatureDocument` using a
    HuggingFace :class:`~transformers.PreTrainedTokenizer`.

    """
    DEFAULT_PARAMS: ClassVar[Dict[str, Any]] = frozendict({
        'return_offsets_mapping': True,
        'is_split_into_words': True,
        'return_special_tokens_mask': True,
        'padding': 'longest'})
    """Default parameters for the HuggingFace tokenizer.  These get overriden by
    the ``tokenizer_kwargs`` in :meth:`tokenize` and the processing of value
    :obj:`word_piece_token_length`.

    """
    resource: TransformerResource = field()
    """Contains the model used to create the tokenizer."""

    word_piece_token_length: int = field(default=None)
    """The max number of word piece tokens.  The word piece length is always the
    same or greater in count than linguistic tokens because the word piece
    algorithm tokenizes on characters.

    If this value is less than 0, than do not fix sentence lengths.  If the
    value is 0 (default), then truncate to the model's longest max lenght.
    Otherwise, if this value is ``None``, set the length to the model's longest
    max length using the model's ``model_max_length`` value.

    Setting this to a value to 0, making documents multi-length, has the
    potential of creating token spans longer than the model can tolerate
    (usually 512 word piece tokens).  In these cases, this value must be set to
    (or lower) than the model's ``model_max_length``.

    Tokenization padding is on by default.

    :see: `HF Docs <https://huggingface.co/docs/transformers/pad_truncation>`_

    """
    params: Dict[str, Any] = field(default=None)
    """Additional parameters given to the
    :class:`transformers.PreTrainedTokenizer`.

    """
    feature_id: str = field(default='text')
    """The feature ID to use for token string values from
    :class:`~zensols.nlp.tok.FeatureToken`.

    """
    def __post_init__(self):
        super().__init__()
        if self.word_piece_token_length is None:
            self.word_piece_token_length = \
                self.resource.tokenizer.model_max_length

    @property
    def pretrained_tokenizer(self) -> PreTrainedTokenizer:
        """The HuggingFace tokenized used to create tokenized documents."""
        return self.resource.tokenizer

    @property
    def token_max_length(self) -> int:
        """The word piece token maximum length supported by the model."""
        if self.word_piece_token_length is None or \
           self.word_piece_token_length == 0:
            return self.pretrained_tokenizer.model_max_length
        return self.word_piece_token_length

    @property
    @persisted('_id2tok')
    def id2tok(self) -> Dict[int, str]:
        """A mapping from the HuggingFace tokenizer's vocabulary to it's word
        piece equivalent.

        """
        vocab = self.pretrained_tokenizer.vocab
        return {vocab[k]: k for k in vocab.keys()}

    @property
    @persisted('_all_special_tokens')
    def all_special_tokens(self) -> Set[str]:
        """Special tokens used by the model (such BERT's as ``[CLS]`` and
        ``[SEP]`` tokens).

        """
        return frozenset(self.pretrained_tokenizer.all_special_tokens)

    def tokenize(self, doc: FeatureDocument,
                 tokenizer_kwargs: Dict[str, Any] = None) -> \
            TokenizedFeatureDocument:
        """Tokenize a feature document in a form that's easy to inspect and
        provide to :class:`.TransformerEmbedding` to transform.

        :param doc: the document to tokenize

        """
        if not self.resource.tokenizer.is_fast:
            raise TransformerError(
                'only fast tokenizers are supported for needed offset mapping')

        fid: str = self.feature_id
        sents = list(map(lambda sent: list(
            map(lambda tok: tok.get_feature(fid), sent)), doc))
        return self._from_tokens(sents, doc, tokenizer_kwargs)

    def _from_tokens(self, sents: List[List[str]], doc: FeatureDocument,
                     tokenizer_kwargs: Dict[str, Any] = None) -> \
            TokenizedFeatureDocument:
        torch_config: TorchConfig = self.resource.torch_config
        tlen: int = self.word_piece_token_length
        tokenizer: PreTrainedTokenizer = self.pretrained_tokenizer
        params: Dict[str, bool] = dict(self.DEFAULT_PARAMS)
        if self.params is not None:
            params.update(self.params)

        if len(sents) == 0:
            return TokenizedFeatureDocument(
                tensor=torch_config.singleton([[], [], []], dtype=torch.long),
                boundary_tokens=False,
                char_offsets=[],
                feature=doc,
                id2tok=self.id2tok)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'parsing {sents} with token length: {tlen}')

        if tlen > 0:
            params.update({'truncation': True,
                           'max_length': tlen,
                           'padding': 'max_length'})
        else:
            params.update({'truncation': False})

        if tokenizer_kwargs is not None:
            params.update(tokenizer_kwargs)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'using tokenizer parameters: {params}')
        tok_dat: BatchEncoding = tokenizer(sents, **params)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"lengths: {[len(i) for i in tok_dat['input_ids']]}")
            logger.debug(f"inputs: {tok_dat['input_ids']}")

        input_ids: List[List[int]] = tok_dat.input_ids
        char_offsets: List[List[int]] = tok_dat.offset_mapping
        boundary_tokens: bool = (tok_dat.special_tokens_mask[0][0]) == 1
        sent_offsets: Tuple[Tuple[int, Tuple[Tuple[int, int], ...]], ...] = \
            tuple(map(lambda s: tuple(map(lambda x: -1 if x is None else x, s)),
                      map(lambda si: tok_dat.word_ids(batch_index=si),
                          range(len(input_ids)))))

        if logger.isEnabledFor(logging.DEBUG):
            six: int
            tids: Tuple[Tuple[int, int], ...]
            for six, tids in enumerate(sent_offsets):
                logger.debug(f'tok ids: {tids}')
                stix: int
                tix: int
                for stix, tix in enumerate(tids):
                    bid = tok_dat['input_ids'][six][stix]
                    wtok = self.id2tok[bid]
                    if tix >= 0:
                        stok = sents[six][tix]
                    else:
                        stok = '-'
                    logger.debug(
                        f'sent={six}, idx={tix}, id={bid}: {wtok} -> {stok}')

        tok_data = [input_ids, tok_dat.attention_mask, sent_offsets]
        if hasattr(tok_dat, 'token_type_ids'):
            tok_data.append(tok_dat.token_type_ids)
        arr = torch_config.singleton(tok_data, dtype=torch.long)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'tok doc mat: shape={arr.shape}, dtype={arr.dtype}')

        return TokenizedFeatureDocument(
            tensor=arr,
            # needed by expander vectorizer
            boundary_tokens=boundary_tokens,
            char_offsets=char_offsets,
            feature=doc,
            id2tok=self.id2tok)

    def __call__(self, doc: FeatureDocument,
                 tokenizer_kwargs: Dict[str, Any] = None) -> \
            TokenizedFeatureDocument:
        return self.tokenize(doc, tokenizer_kwargs)
