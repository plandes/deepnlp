"""Sequence result analysis tools.

"""
__author__ = 'Paul Landes'

from typing import Type
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import sklearn.metrics as mt
from sklearn.preprocessing import LabelEncoder
from zensols.persist import persisted
from zensols.deeplearn.batch import Batch, DataPoint
from zensols.deeplearn.model import ModelFacade
from zensols.deeplearn.vectorize import CategoryEncodableFeatureVectorizer
from zensols.deeplearn.result import EpochResult


@dataclass
class EpochResultAnalyzer(object):
    facade: ModelFacade = field()
    vectorizer_name: str = field()
    result: EpochResult = field()

    def _add_data_point(self, batch: Batch, dp: DataPoint, df: pd.DataFrame):
        df['batch_id'] = batch.id

    @property
    @persisted('_predictions')
    def predictions(self) -> pd.DataFrame:
        vec: CategoryEncodableFeatureVectorizer = \
            self.facade.vectorizer_manager_set.\
            get_vectorizer(self.vectorizer_name)
        le: LabelEncoder = vec.label_encoder
        labs: np.ndarray = self.result.labels
        preds: np.ndarray = self.result.predictions
        assert len(labs) == len(preds)
        dfs = []
        start = 0
        for bid in self.result.batch_ids:
            batch: Batch = self.facade.batch_stash[bid]
            for dp in batch.data_points:
                end = start + len(dp)
                doc_labs = le.inverse_transform(labs[start:end])
                doc_preds = le.inverse_transform(preds[start:end])
                df = pd.DataFrame({'label': doc_labs, 'pred': doc_preds})
                self._add_data_point(batch, dp, df)
                dfs.append(df)
                start = end
        df = pd.concat(dfs)
        assert len(df) == len(labs)
        return df

    @property
    def metrics(self) -> pd.DataFrame:
        rows = []
        df = self.predictions
        dfg = df.groupby('label').agg({'label': 'count'}).\
            rename(columns={'label': 'count'})
        for ann_name, dfg in df.groupby('label'):
            acc = mt.accuracy_score(dfg['label'], dfg['pred'])
            rows.append((ann_name, len(dfg), acc))
        dfr = pd.DataFrame(rows, columns='name count acc'.split())
        return dfr.sort_values('count', ascending=False).reset_index(drop=True)


@dataclass
class SequenceAnalyzer(object):
    facade: ModelFacade = field()
    vectorizer_name: str = field()
    epoch_analyzer_class: Type[EpochResultAnalyzer] = field(
        default=EpochResultAnalyzer)

    def get_epoch_analyzer(self, result: EpochResult):
        cls = self.epoch_analyzer_class
        return cls(self.facade, self.vectorizer_name, result)
