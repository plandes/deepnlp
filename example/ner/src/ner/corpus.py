"""Classes to parse the corpus.

"""
__author__ = 'plandes'

from typing import Tuple, Iterable, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path
import logging
import sys
import random
import re
import collections
from io import TextIOBase
from zensols.util import time
from zensols.persist import (
    OneShotFactoryStash, PersistedWork, persisted, PersistableContainer
)
from zensols.config import Dictable
from zensols.nlp import BasicTokenFeatures, FeatureToken, FeatureSentence
from zensols.dataset import AbstractSplitKeyContainer, DatasetSplitStash

logger = logging.getLogger(__name__)


class NERTokenFeatures(BasicTokenFeatures):
    """Contains the data of a row of the NER data.

    Fields:
        1. Word,
        2. Part-of-speech (POS) tag,
        3. A syntactic chunk tag,
        4. The BIO named entity tag.

    """
    WRITABLE_FIELD_IDS = 'i text norm tag_ ent_'.split()
    FIELD_SET = frozenset(set(WRITABLE_FIELD_IDS) | set('syn_'.split()))

    def __init__(self, i: int, text: str, tag_: str, syn_: str, ent_: str):
        super().__init__(text)
        self.i = i
        self.i_sent = i
        self.tag_ = tag_
        self.syn_ = syn_
        self.ent_ = ent_


class NERFeatureToken(FeatureToken):
    """A feature token that uses :class:`.NERTokenFeatures` as the features class.

    """
    def __init__(self, features: NERTokenFeatures):
        super().__init__(features, NERTokenFeatures.FIELD_SET)


@dataclass
class NERFeatureSentence(FeatureSentence):
    """A feature sentence with an identifier.

    """
    sent_id: int = field(default=None)

    def __str__(self) -> str:
        return f'{self.sent_id}: {super().__str__()}'


@dataclass
class SentenceFactoryStash(OneShotFactoryStash, AbstractSplitKeyContainer):
    """A factory stash that creates instances of :class:`.NERFeatureSentence` from
    CoNLL 2003 format.

    """
    DOC_START = re.compile(r'^\s*-DOCSTART- -X- -X- O\n*', re.MULTILINE)

    source_path: Path = field(default=None)
    """The path to the corpus input file."""

    corpus_split_names: Tuple[str] = field(default=None)
    """The names of the splits (i.e. ``train``, ``test``)."""

    def _read_split(self, split_name: str) -> List[NERFeatureSentence]:
        path = self.source_path / f'{split_name}.txt'
        logger.info(f'reading {path}')
        toks = []
        sents = []
        with open(path) as f:
            lines = f.readlines()
            for line in map(lambda s: s.strip(), lines):
                if self.DOC_START.match(line) is not None:
                    continue
                if len(line) > 0:
                    feats = NERTokenFeatures(len(toks), *line.split())
                    toks.append(NERFeatureToken(feats))
                else:
                    sent = NERFeatureSentence(
                        sent_tokens=tuple(toks), sent_id=len(sents))
                    sents.append(sent)
                    toks.clear()
        if len(toks) > 0:
            sent = NERFeatureSentence(
                sent_tokens=tuple(toks), sent_id=len(sents))
            sents.append(sent)
        return sents

    def worker(self) -> Iterable[Tuple[str, FeatureSentence]]:
        corp = []
        split_keys = {}
        start = 0
        for name in self.corpus_split_names:
            with time('parsed {slen} sentences ' + f'from {name}'):
                sents: List[NERFeatureSentence] = self._read_split(name)
                slen = len(sents)
            random.shuffle(sents)
            end = start + len(sents)
            keys = tuple(map(str, range(start, end)))
            assert(len(keys) == len(sents))
            split_keys[name] = keys
            corp.extend(zip(keys, sents))
            start = end
        self._worker_split_keys = split_keys
        return corp

    def prime(self):
        super().prime()
        AbstractSplitKeyContainer.prime(self)

    def _create_splits(self) -> Dict[str, Tuple[str]]:
        self.prime()
        return self._worker_split_keys


@dataclass
class SentenceStats(PersistableContainer, Dictable):
    """Display sentence stats.

    """
    stash: DatasetSplitStash
    path: Path

    def __post_init__(self):
        self._data = PersistedWork(self.path, self, mkdir=True)

    @property
    @persisted('_data')
    def data(self) -> Dict[str, Any]:
        with time('parsed stats data'):
            tag = collections.defaultdict(lambda: 0)
            syn = collections.defaultdict(lambda: 0)
            ent = collections.defaultdict(lambda: 0)
            for sent in self.stash.values():
                for tok in sent:
                    tag[tok.tag_] += 1
                    syn[tok.syn_] += 1
                    ent[tok.ent_] += 1
            return {'features': {'tag': dict(tag),
                                 'syn': dict(syn),
                                 'ent': dict(ent)}}

    def asdict(self, recurse: bool = True, readable: bool = True,
               class_name_param: str = None) -> Dict[str, Any]:
        return self.data

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line('splits:', depth, writer)
        self.stash.write(depth + 1, writer)
        super().write(depth, writer)

    def write_config_section(self):
        from configparser import ConfigParser
        dfeats = self.data['features']
        feats = {}
        for k, v in dfeats.items():
            vals = map(lambda s: s.replace('$', '$$'), dfeats[k].keys())
            vals = str(tuple(vals))
            feats[k] = vals
        conf = ConfigParser()
        conf['category_settings'] = feats
        conf.write(sys.stdout)
