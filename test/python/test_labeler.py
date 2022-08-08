from typing import List
import torch
from torch import Tensor
from zensols.util import loglevel
from zensols.config import ImportIniConfig, ImportConfigFactory
from zensols.nlp import FeatureDocument, FeatureDocumentParser
from zensols.deepnlp.vectorize import TextFeatureType
from zensols.deepnlp.transformer import (
    TransformerNominalFeatureVectorizer
)
from util import TestFeatureVectorization


class TestLabelVectorizer(TestFeatureVectorization):
    """Test not only TransformerNominalFeatureVectorizer, but
    TokenContainerVectorizer.

    """
    def setUp(self):
        config = ImportIniConfig('test-resources/transformer.conf')
        self.fac = ImportConfigFactory(config, shared=True)
        self.sents = ('Fighting and shelling continued Saturday on several fronts. Even as Ukrainian authorities said that Moscow and Kyiv had agreed.',
                      'Ukrainian women are volunteering to fight.')
        self.doc_parser: FeatureDocumentParser = self.fac('lab_doc_parser')
        self.docs = self._set_attributes(list(map(self.doc_parser, self.sents)))
        # concatenate tokens of each document in to singleton sentence
        # documents (single document case)
        self.should_single = \
            [[-100, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -100, -100, -100, -100, -100],
             [-100, 0, 0, 2, 0, 0, 0, 2, 0, 4, 4, 4, 0, 0, 0, -100]]
        # concatenate tokens of each document in to singleton sentence
        # documents
        self.should_concat = \
            [[-100, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 , 0, 2, 0, 0, 0, 2, 0, 4, 4, 4, 0, 0, 0, -100],
             [-100, 3, 0, 0, 0, 0, 0, 0, 0, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]]
        # all sentences of all documents become singleton sentence documents
        self.should_sentence = \
            [[-100, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -100, -100, -100, -100, -100],
             [-100, 0, 0, 2, 0, 0, 0, 2, 0, 4, 4, 4, 0, 0, 0, -100],
             [-100, 3, 0, 0, 0, 0, 0, 0, 0, -100, -100, -100, -100, -100, -100, -100]]
        # every sentence of each document is encoded separately, then the each
        # sentence output is concatenated as the respsective document during
        # decoding
        self.should_separate = \
            [[-100, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -100, -100, 0, 0, 2, 0, 0, 0, 2, 0, 4, 4, 4, 0, 0, 0, -100],
             [-100, 3, 0, 0, 0, 0, 0, 0, 0, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]]

    def _set_attributes(self, docs: List[FeatureDocument]):
        for doc in docs:
            for s in doc.sents:
                s.annotations = tuple(map(lambda t: t.ent_, s))
        return docs

    def _get_vec(self, name: str):
        vec = self.fac(name)
        vec.encode_transformed = self.encode_transformed
        return vec

    def _test_single(self, doc, vec):
        tensor: Tensor = vec.transform(doc)
        self.assertEqual((2, 16), tensor.shape)
        should = self.should_single
        self.assertEqual((2, 16), tensor.shape)
        self.assertTensorEquals(torch.tensor(should), tensor)

    def _test_labeler(self):
        vec: TransformerNominalFeatureVectorizer

        vec = self._get_vec('ent_label_trans_concat_tokens_vectorizer')
        vec.encode_transformed = self.encode_transformed
        self.assertEqual(vec.feature_type, TextFeatureType.NONE)
        self._test_single(self.docs[0], vec)

        tensor: Tensor = vec.transform(self.docs)
        self.assertEqual((2, 26), tensor.shape)
        self.assertTensorEquals(torch.tensor(self.should_concat), tensor)

        vec = self._get_vec('ent_label_trans_sentence_vectorizer')
        self._test_single(self.docs[0], vec)
        tensor: Tensor = vec.transform(self.docs)
        self.assertEqual((3, 16), tensor.shape)
        self.assertTensorEquals(torch.tensor(self.should_sentence), tensor)

        vec = self._get_vec('ent_label_trans_separate_vectorizer')
        tensor: Tensor = vec.transform(self.docs[0])
        self.assertEqual((1, 28), tensor.shape)
        self.assertTensorEquals(torch.tensor(self.should_separate[0]), tensor.squeeze(0))

        with loglevel('zensols.deepnlp.vectorize', init=True, enable=False):
            tensor: Tensor = vec.transform(self.docs)
        self.assertEqual((2, 28), tensor.shape)
        self.assertTensorEquals(torch.tensor(self.should_separate), tensor)

    def test_labeler_non_transformed(self):
        self.encode_transformed = False
        self._test_labeler()

    def test_labeler_transformed(self):
        self.encode_transformed = True
        self._test_labeler()

    def _create_mask(self, lns):
        ml = max(lns)
        arrs = torch.zeros((len(lns), ml), dtype=bool)
        for ix, ln in enumerate(lns):
            arrs[ix, :ln] = torch.ones(ln, dtype=bool)
        return arrs

    def _test_masker(self):
        vec: TransformerNominalFeatureVectorizer

        vec = self._get_vec('ent_mask_trans_concat_tokens_vectorizer')
        tensor: Tensor = vec.transform(self.docs[0])
        self.assertTensorEquals(self._create_mask((12, 16)), tensor)

        tensor: Tensor = vec.transform(self.docs)
        self.assertEqual(torch.bool, tensor.dtype)
        self.assertTensorEquals(self._create_mask((26, 10)), tensor)

        vec = self._get_vec('ent_mask_trans_sentence_vectorizer')
        tensor: Tensor = vec.transform(self.docs)
        self.assertEqual(torch.bool, tensor.dtype)
        self.assertTensorEquals(self._create_mask((12, 16, 10)), tensor)

        vec = self._get_vec('ent_mask_trans_separate_vectorizer')
        tensor: Tensor = vec.transform(self.docs[0])
        self.assertEqual(torch.bool, tensor.dtype)
        self.assertTensorEquals(self._create_mask((28,)), tensor)

        tensor: Tensor = vec.transform(self.docs)
        self.assertEqual(torch.bool, tensor.dtype)
        self.assertTensorEquals(self._create_mask((28, 10)), tensor)

    def test_masker_non_transformed(self):
        self.encode_transformed = False
        self._test_masker()

    def test_masker_transformed(self):
        self.encode_transformed = True
        self._test_masker()
