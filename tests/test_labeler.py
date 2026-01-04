from typing import List
from pathlib import Path
import torch
from torch import Tensor
from zensols.util import loglevel
from zensols.config import ImportIniConfig, ImportConfigFactory
from zensols.nlp import (
    FeatureDocument, FeatureDocumentParser, TokenAnnotatedFeatureDocument
)
from zensols.deepnlp.vectorize import TextFeatureType
from zensols.deepnlp.transformer import (
    TransformerNominalFeatureVectorizer
)
from util import TestFeatureVectorization, Should


class TestLabelVectorizer(TestFeatureVectorization):
    """Test not only TransformerNominalFeatureVectorizer, but
    TokenContainerVectorizer.

    """
    DEBUG: bool = False
    WRITE: bool = 0

    @classmethod
    def setUpClass(cls):
        cls.should = Should(
            Path('test-resources/should/label-vec.json'),
            is_write=cls.WRITE,
            dtype='torch')
        if not cls.WRITE:
            cls.should.load()

    @classmethod
    def tearDownClass(cls):
        if cls.WRITE:
            cls.should.save()

    def setUp(self):
        config = ImportIniConfig('test-resources/transformer.conf')
        self.fac = ImportConfigFactory(config, shared=True)
        self.sents = ('Fighting and shelling continued Saturday on several fronts. Even as Ukrainian authorities said that Moscow and Kyiv had agreed.',
                      'Ukrainian women are volunteering to fight.')
        self.doc_parser: FeatureDocumentParser = self.fac('lab_doc_parser')
        self.docs = self._set_attributes(list(map(self.doc_parser, self.sents)))

    def _set_attributes(self, docs: List[FeatureDocument]):
        for doc in docs:
            for s in doc.sents:
                s.annotations = tuple(map(lambda t: t.ent_, s))
        return docs

    def _get_vec(self, name: str):
        vec = self.fac(name)
        vec.encode_transformed = self.encode_transformed
        return vec

    def _test_single(self, doc: TokenAnnotatedFeatureDocument,
                     vec: TransformerNominalFeatureVectorizer):
        # concatenate tokens of each document in to singleton sentence
        # documents (single document case)
        if self.DEBUG:
            print('vectorizer:')
            vec.write(1)
            print('doc:')
            doc.write(1)
            tdoc = vec.tokenize(doc)
            tdoc.write(1)
        tensor: Tensor = vec.transform(doc)
        self.assertEqual((2, 16), tensor.shape)
        should = self.should('single', tensor)
        self.assertEqual((2, 16), tensor.shape)
        self.assertTensorEquals(should, tensor)

    def _test_labeler(self):
        vec: TransformerNominalFeatureVectorizer

        vec = self._get_vec('ent_label_trans_concat_tokens_vectorizer')
        vec.encode_transformed = self.encode_transformed
        self.assertEqual(vec.feature_type, TextFeatureType.NONE)
        self._test_single(self.docs[0], vec)

        # concatenate tokens of each document in to singleton sentence
        # documents
        tensor: Tensor = vec.transform(self.docs)
        self.assertEqual((2, 26), tensor.shape)
        should = self.should('concat', tensor)
        self.assertTensorEquals(should, tensor)

        # all sentences of all documents become singleton sentence documents
        vec = self._get_vec('ent_label_trans_sentence_vectorizer')
        self._test_single(self.docs[0], vec)
        tensor: Tensor = vec.transform(self.docs)
        self.assertEqual((3, 16), tensor.shape)
        should = self.should('sentence', tensor)
        self.assertTensorEquals(should, tensor)

        # every sentence of each document is encoded separately, then the each
        # sentence output is concatenated as the respsective document during
        # decoding
        vec = self._get_vec('ent_label_trans_separate_vectorizer')
        tensor: Tensor = vec.transform(self.docs[0])
        self.assertEqual((1, 28), tensor.shape)
        should = self.should('separate_sq', tensor.squeeze(0))
        self.assertTensorEquals(should, tensor.squeeze(0))

        with loglevel('zensols.deepnlp.vectorize', init=True, enable=False):
            tensor: Tensor = vec.transform(self.docs)
        should = self.should('separate', tensor.squeeze())
        self.assertEqual((2, 28), tensor.shape)
        self.assertTensorEquals(should, tensor)

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
