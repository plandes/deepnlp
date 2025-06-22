"""Classes to predict fill-mask tasks.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Iterable, Dict, Any
from dataclasses import dataclass, field
import logging
import sys
from collections import OrderedDict
from io import TextIOBase
import pandas as pd
import torch
from torch import Tensor
from torch.return_types import topk
from transformers import PreTrainedTokenizer, PreTrainedModel
from zensols.config import Dictable
from zensols.nlp import FeatureToken, TokenContainer
from zensols.deeplearn import TorchConfig
from zensols.deepnlp.transformer import TransformerResource
from . import TransformerError

logger = logging.getLogger(__name__)


@dataclass
class TokenPrediction(Dictable):
    """Couples a masked model prediction token to which it belongs and its
    score.

    """
    token: FeatureToken = field()
    prediction: str = field()
    score: float = field()

    def __str__(self) -> str:
        return f"{self.token} -> {self.prediction} ({self.score:.4f})"


@dataclass
class Prediction(Dictable):
    """A container class for masked token predictions produced by
    :class:`.MaskFiller`.  This class offers many ways to get the predictions,
    including getting the sentences as instances of
    :class:`~zensols.nlp.container.TokenContainer` by using it as an iterable.

    The sentences are also available as the ``pred_sentences`` key when using
    :meth:`~zensols.config.dictable.Dictable.asdict`.

    """
    cont: TokenContainer = field()
    """The document, sentence or span to predict masked tokens."""

    masked_tokens: Tuple[FeatureToken] = field()
    """The masked tokens matched."""

    df: pd.DataFrame = field()
    """The predictions with dataframe columns:

      * ``k``: the *k* in the top-*k* highest scored masked token match
      * ``mask_id``: the N-th masked token in the source ordered by position
      * ``token``: the predicted token
      * ``score``: the score of the prediction (``[0, 1]``, higher the better)

    """
    def get_container(self, k: int = 0) -> TokenContainer:
        """Get the *k*-th top scored sentence.  This method should be called
        only once for each instance since it modifies the tokens of the
        container for each invocation.

        A client may call this method as many times as necessary (i.e. for
        multiple values of ``k``) since :obj:``cont`` tokens are modified while
        retaining the original masked tokens :obj:`masked_tokens`.

        :param k: as *k* increases the less likely the mask substitutions, and
                  thus sentence; *k* = 0 is the most likely given the sentence
                  and masks

        """
        cont: TokenContainer = self.cont
        if len(self.df) == 0:
            raise TransformerError(f'No predictions found for <{cont.text}>')
        n_top_k: int = len(self) - 1
        if k > n_top_k:
            raise IndexError(f'Only {n_top_k} predictions but asked for {k}')
        df: pd.DataFrame = self.df
        df = df[df['k'] == k].sort_values('mask_id')
        # iterate over the masked tokens, then for each, populate the prediction
        tok: FeatureToken
        repl: str
        for tok, repl in zip(self.masked_tokens, df['token']):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'{repl} -> {tok.norm}')
            # modify the previously matched token clobbering the norm for each
            # iteration
            tok.norm = repl
        # clear to force a container level norm to be generated
        cont.clear()
        return cont

    def get_tokens(self) -> Iterable[TokenPrediction]:
        """Return an iterable of the prediction coupled with the token it
        belongs to and its score.

        """
        preds: Iterable[Tuple[str, float]] = self.df.\
            sort_values('mask_id')['token score'.split()].\
            itertuples(name=None, index=False)
        return map(lambda t: TokenPrediction(t[0], t[1][0], t[1][1]),
                   zip(self.masked_tokens, preds))

    @property
    def masked_token_dicts(self) -> Tuple[Dict[str, Any]]:
        """A tuple of :class:`.builtins.dict` each having token index, norm and
        text data.

        """
        feats: Tuple[str] = ('i', 'idx', 'i_sent', 'norm', 'text')
        return tuple(map(lambda t: t.get_features(feats), self.masked_tokens))

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_masked_tokens: bool = True,
              include_predicted_tokens: bool = True,
              include_predicted_sentences: bool = True):
        self._write_line(f'source: {self.cont.text}', depth, writer)
        if include_masked_tokens:
            self._write_line('masked:', depth, writer)
            for mt in self.masked_token_dicts:
                self._write_dict(mt, depth + 1, writer, one_line=True)
        if include_predicted_tokens:
            self._write_line('predicted:', depth, writer)
            for k, df in self.df.groupby('k')['mask_id token score'.split()]:
                scs: List[str] = []
                for mid, r in df.groupby('mask_id'):
                    s = f"{r['token'].item()} ({r['score'].item():.4f})"
                    scs.append(s)
                self._write_line(f'k={k}: ' + ', '.join(scs), depth + 1, writer)
        if include_predicted_sentences:
            self._write_line('sentences:', depth, writer)
            self._write_iterable(tuple(map(lambda t: t.norm, self)),
                                 depth + 1, writer)

    def _from_dictable(self, *args, **kwargs):
        return OrderedDict(
            [['source', self.cont.text],
             ['masked_tokens', self.masked_token_dicts],
             ['pred_tokens', self.df.to_dict('records')],
             ['pred_sentences', tuple(map(lambda t: t.norm, self))]])

    def __iter__(self) -> Iterable[TokenContainer]:
        return map(self.get_container, range(len(self)))

    def __getitem__(self, i: int) -> TokenContainer:
        return self.get_container(i)

    def __len__(self) -> int:
        return len(self.df['k'].drop_duplicates())

    def __str__(self) -> str:
        return self.get_container().norm


@dataclass
class MaskFiller(object):
    """The class fills masked tokens with the prediction of the underlying maked
    model.  Masked tokens with attribute :obj:`feature_id` having value
    :obj:`feature_value` (:obj:`~zensols.nlp.tok.FeatureToken.norm` and ``MASK``
    by default respectively) are substituted with model values.

    To use this class, parse a sentence with a
    :class:`~zensols.nlp.parser.FeatureDocumentParser` with masked tokens
    using the string :obj:`feature_value`.

    For example (with class defaults), the sentence::

        Paris is the MASK of France.

    becomes::

        Parise is the <mask> of France.

    The ``<mask>`` string becomes the
    :obj:`~transformers.PreTrainedTokenizer.mask_token` for the model's
    tokenzier.

    """
    resource: TransformerResource = field()
    """A container class with the Huggingface tokenizer and model."""

    k: int = field(default=1)
    """The number of top K predicted masked words per mask.  The total number of
    predictions will be <number of masks> X ``k`` in the source document.

    """
    feature_id: str = field(default='norm')
    """The :class:`~zensols.nlp.FeatureToken` feature ID to match on masked
    tokens.

    :see: :obj:`feature_value`

    """
    feature_value: str = field(default='MASK')
    """The value of feature ID :obj:`feature_id` to match on masked tokens."""

    def _predict(self, text: str) -> pd.DataFrame:
        tc: TorchConfig = self.resource.torch_config

        # models are created in the resource
        tokenizer: PreTrainedTokenizer = self.resource.tokenizer
        model: PreTrainedModel = self.resource.model
        # rows of the dataframe are the k, nth mask tok, token str, score/proba
        rows: List[Tuple[int, int, str, float]] = []

        # tokenization produces the vocabulary wordpiece ids
        input_ids: Tensor = tc.to(tokenizer.encode(text, return_tensors='pt'))
        # get the wordpiece IDs of the masks
        mask_token_index: Tensor = torch.where(
            input_ids == tokenizer.mask_token_id)[1]

        # predict and get the masked wordpiece token logits
        token_logits: Tensor = model(input_ids)[0]
        mask_token_logits: Tensor = token_logits[0, mask_token_index, :]
        mask_token_logits = torch.softmax(mask_token_logits, dim=1)

        # get the top K matches based on the masked token logits
        top: topk = torch.topk(mask_token_logits, k=self.k, dim=1)
        # iterate over masks
        top_ix: Tensor = top.indices
        mix: int
        for mix in range(top_ix.shape[0]):
            top_tokens = zip(top_ix[mix].tolist(), top.values[mix].tolist())
            token_id: int
            score: float
            # iterate over the top K tokens
            for k, (token_id, score) in enumerate(top_tokens):
                token: str = tokenizer.decode([token_id]).strip()
                rows.append((k, mix, token, score))
        return pd.DataFrame(rows, columns='k mask_id token score'.split())

    def predict(self, source: TokenContainer) -> Prediction:
        """Predict subtitution values for token masks.

        **Important:** ``source`` is modified as a side-effect of this method.
        Use :meth:`~zensols.nlp.TokenContainer.clone` on the ``source`` document
        passed to this method to preserve the original if necessary.

        :param source: the source document, sentence, or span for which to
                       substitute values

        """
        mask_tok: PreTrainedTokenizer = self.resource.tokenizer.mask_token
        fid: str = self.feature_id
        fval: str = self.feature_value
        # identify the masked tokens
        masked_tokens: Tuple[FeatureToken] = tuple(filter(
            lambda t: getattr(t, fid) == fval, source.token_iter()))
        # substitute the tokenizer's token mask needed for prediction
        tok: FeatureToken
        for tok in masked_tokens:
            tok.norm = mask_tok
        # clear to force a new norm with the tokenzier mask pattern
        source.clear()
        df: pd.DataFrame = self._predict(source.norm)
        return Prediction(source, masked_tokens, df)
