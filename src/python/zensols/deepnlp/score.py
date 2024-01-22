"""Additional deep learning based scoring methods.

This needs the BERTScore packge; install it with ``pip install bert-score``.

"""
__author__ = 'Paul Landes'

from typing import Iterable, Tuple, Type, Dict, Any
from dataclasses import dataclass, field
from torch import Tensor
from zensols.persist import persisted
from zensols.nlp import TokenContainer
from zensols.nlp.score import ScoreContext, ScoreMethod, FloatScore
from bert_score import BERTScorer
from zensols.deepnlp import transformer


@dataclass
class BERTScoreScoreMethod(ScoreMethod):
    """A scoring method that uses BERTScore.  Sentence pairs are ordered as
    ``(<references>, <candidates>)``.

    Citation:

    .. code:: none

      Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav
      Artzi. 2020. BERTScore: Evaluating Text Generation with BERT. In
      Proceedings of the 8th International Conference on Learning
      Representations, Addis Ababa, Ethopia, March.

    """
    use_norm: bool = field(default=True)
    """Whether to compare with
    :obj:`~zensols.nlp.container.TokenContainer.norm` or
    :obj:`~zensols.nlp.container.TokenContainer.text`.

    """
    bert_score_params: Dict[str, Any] = field(
        default_factory=lambda: dict(
            model_type='microsoft/deberta-xlarge-mnli'))
    """The parameters given to :class:`bert_score.scorer.BERTScorer`."""

    @classmethod
    def _get_external_modules(cls: Type) -> Tuple[str, ...]:
        transformer.suppress_warnings()
        return ('bert_score',)

    @property
    @persisted('_bert_scorer')
    def bert_scorer(self) -> BERTScorer:
        return BERTScorer(**self.bert_score_params)

    def _score(self, meth: str, context: ScoreContext) -> Iterable[FloatScore]:
        def container_to_str(container: TokenContainer) -> str:
            return container.norm if self.use_norm else container.text

        refs: Tuple[str] = tuple(map(
            lambda p: container_to_str(p[0]), context.pairs))
        cands: Tuple[str] = tuple(map(
            lambda p: container_to_str(p[1]), context.pairs))
        scorer: BERTScorer = self.bert_scorer
        scores: Tuple[Tensor] = scorer.score(cands=cands, refs=refs)
        return map(FloatScore, scores[0].tolist())
