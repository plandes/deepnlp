from typing import Tuple
from dataclasses import dataclass, field
import logging
import torch
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from zensols.deeplearn.vectorize import FeatureContext, TensorFeatureContext
from zensols.deepnlp import TokensContainer
from zensols.deepnlp.vectorize import TokenContainerFeatureType
from . import DocumentIndexVectorizer

logger = logging.getLogger(__name__)


@dataclass
class TopicModelDocumentIndexerVectorizer(DocumentIndexVectorizer):
    DESCRIPTION = 'latent semantic indexing'
    FEATURE_TYPE = TokenContainerFeatureType.DOCUMENT

    topics: int = field(default=30)
    decode_as_flat: bool = field(default=True)
    n_containers: int = field(default=1)

    def _get_shape(self) -> Tuple[int, int]:
        if self.decode_as_flat:
            return self.n_containers * self.topics,
        else:
            return self.n_containers, self.topics

    def _create_model(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'creating {self.topics}')
        docs = self.doc_factory.create_training_docs()
        docs = tuple(map(lambda doc: self.feat_to_tokens(doc), docs))
        id2word = corpora.Dictionary(docs)
        corpus = tuple(map(lambda doc: id2word.doc2bow(doc), docs))
        lda = LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=self.topics,
            random_state=100,
            update_every=1,
            chunksize=100,
            passes=10,
            alpha='auto',
            per_word_topics=True)
        return {'lda': lda, 'corpus': corpus, 'id2word': id2word}

    def query(self, tokens: Tuple[str]) -> Tuple[float]:
        """Return a distribution over the topics for a query set of tokens.

        :param tokens: the string list of tokens to use for inferencing in the
                       model
        :return: a list of tuples in the form ``(topic_id, probability)``

        """
        lda = self.model['lda']
        id2word = self.model['id2word']
        docs_q = [tokens]
        corpus_q = tuple(map(lambda doc: id2word.doc2bow(doc), docs_q))
        return lda.get_document_topics(corpus_q, minimum_probability=0)[0]

    def _encode(self, containers: Tuple[TokensContainer]) -> FeatureContext:
        arrs = []
        for container in containers:
            terms = tuple(map(lambda t: t.lemma, container.tokens))
            arr = self.torch_config.from_iterable(
                map(lambda x: x[1], self.query(terms)))
            arrs.append(arr)
        arrs = torch.stack(arrs)
        return TensorFeatureContext(self.feature_id, arrs)

    def _decode(self, context: FeatureContext) -> torch.Tensor:
        arr = super()._decode(context)
        if self.decode_as_flat:
            shape = arr.shape
            arr = arr.flatten()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'decode shape {shape} -> {arr.shape}')
        return arr
