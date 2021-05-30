from typing import Tuple, Iterable, Any
from dataclasses import dataclass, field
import logging
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from zensols.util import time
from zensols.nlp import FeatureDocument, TokensContainer
from zensols.deeplearn.vectorize import FeatureContext, TensorFeatureContext
from zensols.deepnlp.vectorize import TextFeatureType
from . import DocumentIndexVectorizer

logger = logging.getLogger(__name__)


@dataclass
class LatentSemanticDocumentIndexerVectorizer(DocumentIndexVectorizer):
    """Train a latent semantic indexing (LSI, aka LSA) model.

    Citation:

    Deerwester, S., Dumais, S.T., Furnas, G.W., Landauer, T.K., and Harshman,
    R. 1990.  Indexing by Latent Semantic Analysis. Journal of the American
    Society for Information Science; New York, N.Y. 41, 6, 391–407.

    :shape: ``(1,)``

    :see: :class:`sklearn.decomposition.TruncatedSVD`

    """
    DESCRIPTION = 'latent semantic indexing'
    FEATURE_TYPE = TextFeatureType.DOCUMENT

    components: int = field(default=100)
    """The number of components for the output."""

    iterations: int = field(default=10)
    """Number of iterations for randomized SVD solver."""

    def _get_shape(self) -> Tuple[int, int]:
        return 1,

    def _create_model(self, docs: Iterable[FeatureDocument]) -> Any:
        """Train using a singular value decomposition, then truncate to get the most
        salient terms in a document/term matrics.

        """
        vectorizer = TfidfVectorizer(
            lowercase=False,
            tokenizer=self.feat_to_tokens
        )
        with time('TF/IDF vectorized {X_train_tfidf.shape[0]} documents'):
            X_train_tfidf = vectorizer.fit_transform(docs)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'tfidf shape: {X_train_tfidf.shape}')
        svd = TruncatedSVD(self.components, n_iter=self.iterations)
        lsa: Pipeline = make_pipeline(svd, Normalizer(copy=False))
        with time('SVD complete'):
            X_train_lsa = lsa.fit_transform(X_train_tfidf)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'created model with {self.components} components, ' +
                        f'over {self.iterations} iterations with ' +
                        f'TF/IDF matrix shape: {X_train_tfidf.shape}, ' +
                        f'SVD matrix shape: {X_train_lsa.shape}')
        return {'vectorizer': vectorizer,
                'lsa': lsa}

    def _transform_doc(self, doc: FeatureDocument, vectorizer: TfidfVectorizer,
                       lsa: Pipeline) -> np.ndarray:
        X_test_tfidf = vectorizer.transform([doc])
        X_test_lsa = lsa.transform(X_test_tfidf)
        return X_test_lsa

    def similarity(self, a: FeatureDocument, b: FeatureDocument) -> float:
        """Return the semantic similarity between two documents.

        """
        model = self.model
        vectorizer = model['vectorizer']
        lsa = model['lsa']
        emb_a = self._transform_doc(a, vectorizer, lsa)
        emb_b = self._transform_doc(b, vectorizer, lsa)
        return np.dot(emb_a, emb_b.T)[0][0]

    def _encode(self, containers: Tuple[TokensContainer]) -> FeatureContext:
        measure = self.similarity(*containers)
        arr = self.torch_config.singleton([measure])
        return TensorFeatureContext(self.feature_id, arr)
