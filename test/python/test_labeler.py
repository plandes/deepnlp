from typing import List
import torch
from torch import Tensor
from zensols.config import ImportIniConfig, ImportConfigFactory
from zensols.nlp import FeatureDocument, FeatureDocumentParser
from zensols.deepnlp.transformer import (
    TransformerNominalFeatureVectorizer
)
from util import TestFeatureVectorization


class TestLabelVectorizer(TestFeatureVectorization):
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
        vec: TransformerNominalFeatureVectorizer = \
            self.fac('ent_label_trans_vectorizer')
        doc_parser: FeatureDocumentParser = self.fac('lab_doc_parser')
        docs = self._set_attributes(list(map(doc_parser, self.sents)))
        doc = FeatureDocument.combine_documents(docs, concat_tokens=False)

        tensor: Tensor = vec.transform(docs[0])
        tensor = tensor.squeeze(2)
        should = [[-100, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -100, -100, -100, -100, -100],
                  [-100, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, -100]]
        self.assertTensorEquals(torch.tensor(should), tensor)

        tensor: Tensor = vec.transform(doc)
        tensor = tensor.squeeze(2)
        should = [[-100, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -100, -100, -100, -100, -100],
                  [-100, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, -100],
                  [-100, 2, 0, 0, 0, 0, 0, 0, 0, -100, -100, -100, -100, -100, -100, -100]]
        self.assertTensorEquals(torch.tensor(should), tensor)

        doc = FeatureDocument.combine_documents(docs, concat_tokens=True)
        tensor: Tensor = vec.transform(doc)
        tensor = tensor.squeeze(2)
        should = [[-100, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, -100],
                  [-100, 2, 0, 0, 0, 0, 0, 0, 0, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]]
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
            
