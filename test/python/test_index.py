import logging
from zensols.deepnlp.index import TopicModelDocumentIndexerVectorizer
from util import TestFeatureVectorization

logger = logging.getLogger(__name__)


class TestFeatureVectorizationParse(TestFeatureVectorization):
    # test just importing for now
    def test_import(self):
        self.assertEqual('latent semantic indexing',
                         TopicModelDocumentIndexerVectorizer.DESCRIPTION)
