from typing import List
import torch
from torch import Tensor
from zensols.util import loglevel
from zensols.config import ImportIniConfig, ImportConfigFactory
from zensols.nlp import FeatureDocument, FeatureDocumentParser
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

    def _set_attributes(self, docs: List[FeatureDocument]):
        for doc in docs:
            for s in doc.sents:
                s.annotations = tuple(map(lambda t: t.ent_, s))
        return docs

    def test_labeler(self):
        def test_single(doc):
            tensor: Tensor = vec.transform(doc)
            tensor = tensor.squeeze(2)
            should = [[-100, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -100, -100, -100, -100, -100],
                      [-100, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, -100]]
            self.assertTensorEquals(torch.tensor(should), tensor)

        doc_parser: FeatureDocumentParser = self.fac('lab_doc_parser')
        docs = self._set_attributes(list(map(doc_parser, self.sents)))

        # concatenate tokens of each document in to singleton sentence
        # documents
        should = [[-100, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, -100],
                  [-100, 2, 0, 0, 0, 0, 0, 0, 0, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]]
        vec: TransformerNominalFeatureVectorizer = \
            self.fac('ent_label_trans_concat_tokens_vectorizer')
        test_single(docs[0])
        tensor: Tensor = vec.transform(docs)
        self.assertEqual((2, 26, 1), tensor.shape)
        tensor = tensor.squeeze(2)
        self.assertTensorEquals(torch.tensor(should), tensor)

        # all sentences of all documents become singleton sentence documents
        should = [[-100, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -100, -100, -100, -100, -100],
                  [-100, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, -100],
                  [-100, 2, 0, 0, 0, 0, 0, 0, 0, -100, -100, -100, -100, -100, -100, -100]]
        vec: TransformerNominalFeatureVectorizer = \
            self.fac('ent_label_trans_sentence_vectorizer')
        test_single(docs[0])
        tensor: Tensor = vec.transform(docs)
        self.assertEqual((3, 16, 1), tensor.shape)
        tensor = tensor.squeeze(2)
        self.assertTensorEquals(torch.tensor(should), tensor)

        # every sentence of each document is encoded separately, then the each
        # sentence output is concatenated as the respsective document during
        # decoding
        should = [[-100, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -100, -100, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, -100.],
                  [-100, 2, 0, 0, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.]]
        vec: TransformerNominalFeatureVectorizer = \
            self.fac('ent_label_trans_separate_vectorizer')
        tensor: Tensor = vec.transform(docs[0])
        self.assertEqual((1, 28), tensor.shape)
        self.assertTensorEquals(torch.tensor(should[0]), tensor.squeeze(0))

        with loglevel('zensols.deepnlp.vectorize', init=True, enable=False):
            tensor: Tensor = vec.transform(docs)
        self.assertEqual((2, 28), tensor.shape)
        self.assertTensorEquals(torch.tensor(should), tensor)

    def _test_labeler(self):
        vec: TransformerNominalFeatureVectorizer = \
            self.fac('ent_label_trans_vectorizer')
        doc_parser = self.fac('doc_parser')
        doc_parser: FeatureDocumentParser = self.fac('lab_doc_parser')
        docs = self._set_attributes(list(map(doc_parser, self.sents)))
        doc = FeatureDocument.combine_documents(docs, concat_tokens=False)
        print()
        for s in doc:
            print(s)
            
