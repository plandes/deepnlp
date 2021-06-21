"""Application facade.

"""

from typing import List
from dataclasses import dataclass, field
import logging
from zensols.deeplearn.batch import Batch
from zensols.deepnlp.model import (
    LanguageModelFacade,
    LanguageModelFacadeConfig,
)
from zensols.deepnlp.vectorize import FeatureDocumentVectorizerManager
from zensols.deepnlp.transformer import TransformerEmbedding
from . import NERBatch, SentenceStats

logger = logging.getLogger(__name__)


@dataclass
class NERModelFacade(LanguageModelFacade):
    """A facade for the movie review sentiment analysis task.  See super classes
    for more information on the purprose of this class.

    All the ``set_*`` methods set parameters in the model.

    """
    LANGUAGE_MODEL_CONFIG = LanguageModelFacadeConfig(
        manager_name=NERBatch.LANGUAGE_FEATURE_MANAGER_NAME,
        attribs=set(),
        embedding_attribs=NERBatch.EMBEDDING_ATTRIBUTES)

    sent_stats: SentenceStats = field(default=None)
    """Computes the corpus statistics."""

    def _get_language_model_config(self) -> LanguageModelFacadeConfig:
        return self.LANGUAGE_MODEL_CONFIG

    @property
    def transformer_trainable_embedding_model(self) -> TransformerEmbedding:
        mng: FeatureDocumentVectorizerManager = \
            self.language_vectorizer_manager
        name: str = NERBatch.TRANSFORMER_TRAINABLE_MODEL_NAME
        return mng[name].embed_model

    def _configure_debug_logging(self):
        super()._configure_debug_logging()
        for name in ['ner',
                     #'zensols.deepnlp.vectorize.layer',
                     'zensols.deeplearn.model.module']:
            logging.getLogger(name).setLevel(logging.DEBUG)

    def _configure_cli_logging(self, info_loggers: List[str],
                               debug_loggers: List[str]):
        super()._configure_cli_logging(info_loggers, debug_loggers)
        info_loggers.append('ner')
        info_loggers.remove('zensols.deeplearn.batch.stash')

    def write_corpus_stats(self):
        """Computes the corpus statistics."""
        self.sent_stats.write()

    def assert_label_mapping(self, do_print: bool = False):
        seq_vec = self.language_vectorizer_manager['entlabel_trans']
        vec = seq_vec.delegate
        le = vec.label_encoder
        if do_print:
            print(le.classes_)
        batch_stash = self.batch_stash
        res = self.last_result
        ds = res.train
        ds.write()
        epoch = ds.converged_epoch
        res_labels = epoch.labels
        start = 0
        end = None
        for bid in epoch.batch_ids:
            batch: Batch = batch_stash[bid]
            if do_print:
                batch.write()
            blabs = batch.get_labels().squeeze()
            for six, dp in enumerate(batch.get_data_points()):
                sent = dp.sent
                assert len(sent) == len(sent.annotations)
                end = start + len(sent)
                rs_labs = res_labels[start:end]
                mapped_labs = tuple(vec.get_classes(rs_labs).tolist())
                start = end
                if do_print:
                    print(sent)
                    seq_vec.tokenize(sent.to_document()).write()
                    print(blabs[six].tolist())
                    print('G', sent.annotations)
                    print('L', mapped_labs)
                    print('-' * 10)
                assert len(sent.annotations) == len(mapped_labs)
                assert sent.annotations == mapped_labs
