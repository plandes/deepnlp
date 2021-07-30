"""Example application to demonstrate Transformer sequence classification.

"""
__author__ = 'plandes'

from dataclasses import dataclass
import logging
import itertools as it
from torch import Tensor
from zensols.persist import dealloc
from zensols.config import Settings
from zensols.nlp import FeatureDocument
from zensols.deeplearn.model import ModelFacade
from zensols.deeplearn.cli import FacadeApplication
from zensols.deepnlp.model import (
    BioSequenceAnnotationMapper, SequenceDocumentAnnotation
)

logger = logging.getLogger(__name__)


@dataclass
class NERFacadeApplication(FacadeApplication):
    """Example application to demonstrate Transformer sequence classification.

    """
    CLI_META = {'mnemonic_overrides': {'assert_label_mapping': 'labmap'}} | \
        FacadeApplication.CLI_META
    CLASS_INSPECTOR = {}

    def __post_init__(self):
        super().__post_init__()
        self.sent = "I'm Paul Landes.  I live in the United States."
        self.sent2 = 'West Indian all-rounder Phil Simmons took four for 38 on Friday as Leicestershire beat Somerset by an innings and 39 runs in two days to take over at the head of the county championship.'

    def stats(self):
        """Print out the corpus statistics."""
        with dealloc(self.create_facade()) as facade:
            facade.write_corpus_stats()

    def assert_label_mapping(self):
        """Confirm the the mapping of the labels is correct."""
        with dealloc(self.create_facade()) as facade:
            facade.assert_label_mapping()

    def predict(self, sentence: str):
        """Create NER labeled predictions.

        :param sentence: the sentence to classify

        """
        if sentence is None:
            sents = (self.sent, self.sent2)
        else:
            sents = (sentence,)
        if 0:
            self.clear_cached_facade()
        facade: ModelFacade = self.get_cached_facade()
        res: Settings = facade.predict(sents)
        anoner: BioSequenceAnnotationMapper = facade.config_factory('anon_mapper')
        anon: SequenceDocumentAnnotation
        for anon in anoner.map(res.classes, res.docs):
            #anon.write(short=True)
            print(anon.doc)
            if 0:
                for label, ftok, stok in anon.token_matches:
                    print(ftok, stok, type(ftok), type(stok), label)
            for sanon in anon.sequence_anons:
                print('  ', sanon)

    def _test_transform(self):
        with dealloc(self.create_facade()) as facade:
            model = facade.transformer_trainable_embedding_model
            doc = facade.doc_parser.parse(self.sent)
            tdoc = model.tokenize(doc)
            tdoc.write()
            arr: Tensor = model.transform(tdoc)
            print(arr.shape)

    def _test_decode(self):
        with dealloc(self.create_facade()) as facade:
            sents = tuple(it.islice(facade.feature_stash.values(), 3))
            doc = FeatureDocument(sents)
            vec = facade.language_vectorizer_manager['syn']
            from zensols.util.log import loglevel
            with loglevel('zensols.deepnlp'):
                vec.encode(doc)

    def _write_max_word_piece_token_length(self):
        logger.info('calculatating word piece length on data set...')
        with dealloc(self.create_facade()) as facade:
            mlen = facade.get_max_word_piece_len()
            print(f'max word piece token length: {mlen}')

    def _test(self):
        with dealloc(self.create_facade()) as facade:
            facade.remove_metadata_mapping_field('glove_300_embedding')
            facade.remove_metadata_mapping_field('word2vec_300_embedding')
        self._test_transform()
        self._test_decode()
        # this takes a while since it iterates through the corpus
        self._write_max_word_piece_token_length()

    def proto(self):
        self.predict(None)
