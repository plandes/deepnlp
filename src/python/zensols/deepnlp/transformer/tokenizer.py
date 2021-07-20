"""The tokenizer object.

"""
__author__ = 'Paul Landes'

from typing import List, Dict, Any
from dataclasses import dataclass, field
import logging
import torch
from zensols.nlp import FeatureDocument
from zensols.deepnlp.transformer import TransformerResource
from zensols.persist import persisted, PersistableContainer
from . import TransformerError, TokenizedFeatureDocument

logger = logging.getLogger(__name__)


@dataclass
class TransformerDocumentTokenizer(PersistableContainer):
    MAX_TOKEN_LENGTH = 512
    """The maximum token length to truncate before converting to IDs.  If this
    isn't done, the following error is raised:

      ``error: CUDA error: device-side assert triggered``

    """
    resource: TransformerResource = field()
    """Contains the model used to create the tokenizer."""

    word_piece_token_length: int = field(default=MAX_TOKEN_LENGTH)
    """The max number of word peice tokens.  The word piece length is always the
    same or greater in count than linguistic tokens because the word piece
    algorithm tokenizes on characters.

    If this value is less than 0, than do not fix sentence lengths.

    """

    def __post_init__(self):
        super().__init__()

    @property
    @persisted('_id2tok')
    def id2tok(self) -> Dict[int, str]:
        vocab = self.resource.tokenizer.vocab
        return {vocab[k]: k for k in vocab.keys()}

    def tokenize(self, doc: FeatureDocument,
                 tokenizer_kwargs: Dict[str, Any] = None) -> \
            TokenizedFeatureDocument:
        """Tokenize a feature document in a form that's easy to inspect and provide to
        :class:`.TransformerEmbedding` to transform.

        :param doc: the document to tokenize

        """
        if not self.resource.tokenizer.is_fast:
            raise TransformerError(
                'only fast tokenizers are supported for needed offset mapping')

        sents = list(map(lambda sent: list(
            map(lambda tok: tok.text, sent)), doc))
        return self._from_tokens(sents, doc, tokenizer_kwargs)

    def _from_tokens(self, sents: List[List[str]], doc: FeatureDocument,
                     tokenizer_kwargs: Dict[str, Any] = None) -> \
            TokenizedFeatureDocument:
        torch_config = self.resource.torch_config
        tlen = self.word_piece_token_length
        tokenizer = self.resource.tokenizer
        params = {'return_offsets_mapping': True,
                  'is_split_into_words': True,
                  'return_special_tokens_mask': True}

        for i, sent in enumerate(sents):
            if len(sent) == 0:
                raise TransformerError(
                    f'Sentence {i} is empty: can not tokenize')

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'parsing {sents} with token length: {tlen}')

        if tlen > 0:
            params.update({'padding': 'max_length',
                           'truncation': True,
                           'max_length': tlen})
        else:
            params.update({'padding': 'longest',
                           'truncation': False})

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'using tokenizer parameters: {params}')

        if tokenizer_kwargs is not None:
            params.update(tokenizer_kwargs)
        tok_dat = tokenizer(sents, **params)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"lengths: {[len(i) for i in tok_dat['input_ids']]}")
            logger.debug(f"inputs: {tok_dat['input_ids']}")

        input_ids = tok_dat.input_ids
        char_offsets = tok_dat.offset_mapping
        boundary_tokens = (tok_dat.special_tokens_mask[0][0]) == 1
        sent_offsets = tuple(
            map(lambda s: tuple(map(lambda x: -1 if x is None else x, s)),
                map(lambda si: tok_dat.word_ids(batch_index=si),
                    range(len(input_ids)))))

        if logger.isEnabledFor(logging.DEBUG):
            for six, tids in enumerate(sent_offsets):
                logger.debug(f'tok ids: {tids}')
                for stix, tix in enumerate(tids):
                    bid = tok_dat['input_ids'][six][stix]
                    wtok = self.id2tok[bid]
                    if tix >= 0:
                        stok = sents[six][tix]
                    else:
                        stok = '-'
                    logger.debug(
                        f'sent={six}, idx={tix}, id={bid}: {wtok} -> {stok}')

        arr = torch_config.singleton(
            [input_ids, tok_dat.attention_mask, sent_offsets],
            dtype=torch.long)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'tok doc mat: shape={arr.shape}, dtype={arr.dtype}')

        return TokenizedFeatureDocument(
            tensor=arr,
            # needed by expander vectorizer
            boundary_tokens=boundary_tokens,
            char_offsets=char_offsets,
            feature=doc,
            id2tok=self.id2tok)
