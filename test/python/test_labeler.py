import torch
from torch import Tensor
from zensols.config import ImportIniConfig, ImportConfigFactory
from zensols.deepnlp.transformer import (
    TransformerNominalFeatureVectorizer
)
from util import TestFeatureVectorization


class TestLabelVectorizer(TestFeatureVectorization):
    def setUp(self):
        config = ImportIniConfig('test-resources/transformer.conf')
        self.fac = ImportConfigFactory(config, shared=True)
        self.sent_text = 'Fighting and shelling continued Saturday on several fronts. Even as Ukrainian authorities said that Moscow and Kyiv had agreed.'

    def test_labeler(self):
        doc_parser = self.fac('doc_parser')
        doc = doc_parser(self.sent_text)
        for s in doc.sents:
            s.annotations = tuple(map(lambda t: t.ent_, s))
        vec: TransformerNominalFeatureVectorizer = \
            self.fac('ent_label_trans_vectorizer')
        tensor: Tensor = vec.transform(doc)
        tensor = tensor.squeeze(2)
        should = [[-100, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -100, -100, -100, -100, -100],
                  [-100, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, -100]]
        self.assertTensorEquals(torch.tensor(should), tensor)
