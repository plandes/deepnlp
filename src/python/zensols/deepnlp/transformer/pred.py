"""Predictions output for transformer models.

"""
__author__ = 'Paul Landes'

from typing import Callable, List, Tuple, Dict, Iterable, Any
from dataclasses import dataclass, field
import logging
import pandas as pd
from torch import Tensor
from zensols.nlp import FeatureToken, FeatureSentence, FeatureDocument
from zensols.deeplearn.batch import Batch, DataPoint
from zensols.deeplearn.result import SequencePredictionsDataFrameFactory
from . import TokenizedDocument

logger = logging.getLogger(__name__)


@dataclass
class TransformerSequencePredictionsDataFrameFactory(
        SequencePredictionsDataFrameFactory):
    """Like the super class but create predictions for transformer sequence
    models.  By default, transformer input is truncated at the model's max token
    length (usually 512 word piece tokens).  It then truncate the tokens that
    are added as the ``text`` column from (configured by default)
    :class:`..classify.TokenClassifyModelFacade`.

    For all predictions where the sequence passed the model's maximum, this
    class maps that last word piece token output to the respective token in the
    :obj:`predictions_dataframe_factory_class` instance's ``transform`` output.

    """
    embedded_document_attribute: str = field(default=None)
    """The :obj:`~zensols.deeplearn.batch.domain.Batch` attribute key for the
    tensor that contains the vectorized document.

    """
    def _trunc_tokens(self, batch: Batch) -> Iterable[Tuple[FeatureToken, ...]]:
        """Return tokens truncated at the length of the last word piece token.

        :param batch: contains the ``data_points`` with sentences to truncate

        :return: the truncated tokens for each data point in ``batch``

        """
        # merge documents from the data points into a document for the batch
        dps_doc: FeatureDocument = FeatureDocument.combine_documents(
            map(lambda dp: dp.doc, batch.data_points))
        # re-hydrate the vectorized document from the batch tensor
        emb: Tensor = batch[self.embedded_document_attribute]
        tdoc: TokenizedDocument = TokenizedDocument.from_tensor(emb)
        # map word piece tokens to feature document tokens
        sent_maps: List[Dict[str, Any]] = tdoc.map_to_word_pieces(
            sentences=dps_doc,
            includes={'map', 'sent'})
        for dpix, dp in enumerate(batch.data_points):
            tmap: Tuple[FeatureToken, Tuple[Tuple[str, int, int], ...]] = \
                sent_maps[dpix]['map']
            yield len(tmap)

    def _calc_len(self, batch: Batch) -> int:
        trunc_lens: Tuple[int] = tuple(self._trunc_tokens(batch))
        batch._trunc_lens = trunc_lens
        return sum(map(lambda tl: tl, trunc_lens))

    def _transform_dataframe(self, batch: Batch, labs: List[str],
                             preds: List[str]):
        dfs: List[pd.DataFrame] = []
        start: int = 0
        transform: Callable = self.data_point_transform
        self._assert_label_pred_batch_size(batch, labs, preds, False)
        dp: DataPoint
        tl: int
        for dp, tl in zip(batch.data_points, batch._trunc_lens):
            end: int = start + tl
            df = pd.DataFrame({
                self.ID_COL: dp.id,
                self.LABEL_COL: labs[start:end],
                self.PREDICTION_COL: preds[start:end]})
            dp_data: Tuple[Tuple[str, ...]] = transform(dp)
            if len(dp_data) != tl and logger.isEnabledFor(logging.WARNING):
                sent_str: str = ''
                if hasattr(dp, 'doc'):
                    doc: FeatureDocument = dp.doc
                    sent_str: str = f' for document: {doc}'
                    logger.warning(
                        f'trimming outcomes from {len(dp_data)} ' +
                        f'to word piece max (equivalent) {tl}{sent_str}')
            dp_data = dp_data[:tl]
            df[list(self.column_names)] = dp_data
            dfs.append(df)
            start = end
        return pd.concat(dfs)
