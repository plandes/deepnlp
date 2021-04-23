"""The tokenizer object.

"""
__author__ = 'Paul Landes'

from typing import List, Dict, Any
from dataclasses import dataclass, field
import logging
import torch
from zensols.deepnlp import FeatureDocument
from zensols.deepnlp.transformer import TransformerResource
from zensols.persist import persisted
from . import TokenizedFeatureDocument

logger = logging.getLogger(__name__)


@dataclass
class TransformerDocumentTokenizer(object):
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

    """

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
                  'is_split_into_words': True}

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'parsing {sents} with token length: {tlen}')

        if tlen > 0:
            params.update({'padding': 'max_length',
                           'truncation': True,
                           'max_length': tlen})
        else:
            params.update({'padding': 'longest',
                           'truncation': False})
        if tokenizer_kwargs is not None:
            params.update(tokenizer_kwargs)
        tok_dat = tokenizer(sents, **params)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"lengths: {[len(i) for i in tok_dat['input_ids']]}")

        offsets = tok_dat['offset_mapping']

        # roberta offsets are 1-indexed
        if self.resource._is_roberta():
            offsets = tuple(map(lambda sent: list(
                list(lambda x: (x[0]-1, x[1]), sent)), offsets))

        sent_offsets = []
        for six, six_offsets in enumerate(offsets):
            # start at index 1 if end span > 0, which indicates a [CLS] (or <s>
            # i.e. roberta) token
            off = 1 if six_offsets[0][1] == 0 else 0
            tix = 0
            tok_offsets = []
            sent_offsets.append(tok_offsets)
            pad = False
            for i, (s, e) in enumerate(six_offsets[off:]):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'{i}: s/e={s},{e}, off={(tix-off)}')
                if e == 0:
                    # we get an ending range for each padded token
                    if pad:
                        tok_offsets.append(-1)
                    else:
                        # the first end indicates the padding starts
                        tok_offsets.append(tix-off)
                        pad = True
                else:
                    tok_offsets.append(tix-off)
                    if s == 0:
                        tix += 1
            # if we have a starting token, add an another (invalid) offset for
            # the terminating sentence token (i.e. [SEP])
            if off == 1:
                tok_offsets.append(-1)

        if logger.isEnabledFor(logging.DEBUG):
            for six, tids in enumerate(sent_offsets):
                logger.debug(f'tok ids: {tids}')
                for stix, tix in enumerate(tids):
                    bid = tok_dat['input_ids'][six][stix]
                    if tix >= 0:
                        stok = sents[six][tix]
                    else:
                        stok = '-'
                    logger.debug(
                        f'sent={six}, tok id={tix}, model id={bid}: ' +
                        f'{self.id2tok[bid]} -> {stok}')

        arr = torch_config.singleton(
            [tok_dat['input_ids'], tok_dat['attention_mask'], sent_offsets],
            dtype=torch.long)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'tok doc mat: shape={arr.shape}, dtype={arr.dtype}')

        return TokenizedFeatureDocument(arr, doc, self.id2tok)
