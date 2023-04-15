"""A Deerwester latent semantic index vectorizer implementation.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Iterable, Any, Dict
from dataclasses import dataclass, field
import logging
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from zensols.util import time
from zensols.nlp import FeatureDocument, TokenContainer
from zensols.deeplearn import DeepLearnError
from zensols.deeplearn.vectorize import FeatureContext, TensorFeatureContext
from zensols.deepnlp.vectorize import TextFeatureType
from . import DocumentIndexVectorizer

logger = logging.getLogger(__name__)


@dataclass
class LatentSemanticDocumentIndexerVectorizer(DocumentIndexVectorizer):
    """Train a latent semantic indexing (LSI, aka LSA) model from::

      Deerwester, S., Dumais, S.T., Furnas, G.W., Landauer, T.K., and Harshman,
      R. 1990.  Indexing by Latent Semantic Analysis. Journal of the American
      Society for Information Science; New York, N.Y. 41, 6, 391–407.

    This class can be used only to index TF/IDF.  To skip the LSI training, set
    :obj:`iterations` to zero.

    :shape: ``(1,)``

    :see: :class:`sklearn.decomposition.TruncatedSVD`

    """
    DESCRIPTION = 'latent semantic indexing'
    FEATURE_TYPE = TextFeatureType.DOCUMENT

    components: int = field(default=100)
    """The number of components for the output."""

    iterations: int = field(default=10)
    """Number of iterations for randomized SVD solver if greater than 0 (see
    class docs).

    """
    vectorizer_params: Dict[str, Any] = field(default_factory=dict)
    """Additional parameters passed to
    :class:`~sklearn.feature_extraction.text.TfidfVectorizer` when vectorizing
    TF/IDF features.

    """
    def _get_shape(self) -> Tuple[int, int]:
        return 1,

    def _create_model(self, docs: Iterable[FeatureDocument]) -> Dict[str, Any]:
        """Train using a singular value decomposition, then truncate to get the
        most salient terms in a document/term matrics.

        """
        vectorizer = TfidfVectorizer(
            lowercase=False,
            tokenizer=self.feat_to_tokens,
            **self.vectorizer_params,
        )
        model: Dict[str, Any] = {'vectorizer': vectorizer}
        with time('TF/IDF vectorized {X_train_tfidf.shape[0]} documents'):
            X_train_tfidf = vectorizer.fit_transform(docs)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'tfidf shape: {X_train_tfidf.shape}')
        svd = TruncatedSVD(self.components, n_iter=self.iterations)
        if self.iterations > 0:
            lsa: Pipeline = make_pipeline(svd, Normalizer(copy=False))
            with time('SVD complete'):
                X_train_lsa = lsa.fit_transform(X_train_tfidf)
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'created model w/{self.components} components, ' +
                            f'over {self.iterations} iterations with ' +
                            f'TF/IDF matrix shape: {X_train_tfidf.shape}, ' +
                            f'SVD matrix shape: {X_train_lsa.shape}')
            model['lsa'] = lsa
        return model

    @property
    def vectorizer(self) -> TfidfVectorizer:
        """The vectorizer trained on the document set."""
        return self.model['vectorizer']

    @property
    def lsa(self) -> Pipeline:
        """The LSA pipeline trained on the document set."""
        if 'lsa' not in self.model:
            raise DeepLearnError('SVD model was not trained')
        return self.model['lsa']

    def _transform_doc(self, doc: FeatureDocument, vectorizer: TfidfVectorizer,
                       lsa: Pipeline) -> np.ndarray:
        X_test_tfidf: csr_matrix = vectorizer.transform([doc])
        X_test_lsa: csr_matrix = lsa.transform(X_test_tfidf)
        return X_test_lsa

    def similarity(self, a: FeatureDocument, b: FeatureDocument) -> float:
        """Return the semantic similarity between two documents.

        """
        vectorizer: TfidfVectorizer = self.vectorizer
        lsa: Pipeline = self.lsa
        emb_a = self._transform_doc(a, vectorizer, lsa)
        emb_b = self._transform_doc(b, vectorizer, lsa)
        return np.dot(emb_a, emb_b.T)[0][0]

    def _encode(self, containers: Tuple[TokenContainer]) -> FeatureContext:
        measure = self.similarity(*containers)
        arr = self.torch_config.singleton([measure])
        return TensorFeatureContext(self.feature_id, arr)
