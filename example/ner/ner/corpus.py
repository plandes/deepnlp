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
from zensols.config import Dictable, ConfigFactory
from zensols.install import Installer, Resource
from zensols.nlp import LexicalSpan, FeatureToken, FeatureSentence
from zensols.dataset import AbstractSplitKeyContainer, DatasetSplitStash

logger = logging.getLogger(__name__)


class NERFeatureToken(FeatureToken):
    """Contains the data of a row of the NER data.

    Fields:
        1. Word,
        2. Part-of-speech (POS) tag,
        3. A syntactic chunk tag,
        4. The BIO named entity tag.

    """
    WRITABLE_FEATURE_IDS = 'i norm tag_ ent_'.split()
    FEATURE_SET = frozenset(set(WRITABLE_FEATURE_IDS) | set('syn_'.split()))

    def __init__(self, i: int, text: str, tag_: str, syn_: str, ent_: str):
        super().__init__(i, i, i, text, LexicalSpan(i, i + len(text)))
        # not one to one with token in sentence index, but works for this
        # example
        self.tag_ = tag_
        self.syn_ = syn_
        self.ent_ = ent_


@dataclass
class NERFeatureSentence(FeatureSentence):
    """A feature sentence with an identifier.

    """
    sent_id: int = field(default=None)

    def __str__(self) -> str:
        return f'{self.sent_id}: {super().__str__()}'


@dataclass
class SentenceFactoryStash(OneShotFactoryStash, AbstractSplitKeyContainer):
    """A factory stash that creates instances of :class:`.NERFeatureSentence`
    from CoNLL 2003 format.

    """
    DOC_START = re.compile(r'^\s*-DOCSTART- -X- -X- O\n*', re.MULTILINE)

    source_path: Path = field(default=None)
    """The path to the corpus input file."""

    installer: Installer = field(default=None)
    """The installer used to download and find the corpus."""

    dataset_limit: int = field(default=sys.maxsize)
    """The maximum number of corpus records to process."""

    def _read_split(self, path: Path) -> List[NERFeatureSentence]:
        logger.info(f'reading {path}')
        toks = []
        sents = []
        match = self.DOC_START.match
        with open(path) as f:
            lines = map(lambda s: s.strip(), f.readlines())
            lines = filter(lambda x: match(x) is None, lines)
            for line in lines:
                if len(line) > 0:
                    toks.append(NERFeatureToken(len(toks), *line.split()))
                else:
                    sent = NERFeatureSentence(
                        tokens=tuple(toks), sent_id=len(sents))
                    sents.append(sent)
                    toks.clear()
                    if len(sents) >= self.dataset_limit:
                        break
        if len(toks) > 0:
            sent = NERFeatureSentence(tokens=tuple(toks), sent_id=len(sents))
            sents.append(sent)
        return sents

    def worker(self) -> Iterable[Tuple[str, FeatureSentence]]:
        corp = []
        split_keys = {}
        start = 0
        res: Resource
        for res in self.installer.by_name.values():
            path: Path = self.installer[res]
            name: str = path.stem
            with time('parsed {slen} sentences ' + f'from {res}'):
                sents: List[NERFeatureSentence] = self._read_split(path)
                slen: int = len(sents)
            random.shuffle(sents)
            end = start + len(sents)
            keys = tuple(map(str, range(start, end)))
            assert (len(keys) == len(sents))
            split_keys[name] = keys
            corp.extend(zip(keys, sents))
            start = end
        self._worker_split_keys = split_keys
        return corp

    def prime(self):
        self.installer()
        super().prime()
        AbstractSplitKeyContainer.prime(self)

    def _create_splits(self) -> Dict[str, Tuple[str]]:
        self.prime()
        return self._worker_split_keys


@dataclass
class SentenceStatsCalculator(PersistableContainer, Dictable):
    """Display sentence stats.

    """
    config_factory: ConfigFactory = field()
    stash: DatasetSplitStash = field()
    path: Path = field()

    def __post_init__(self):
        self._data = PersistedWork(self.path, self, mkdir=True)

    @persisted('_data')
    def _from_dictable(self, *args, **kwargs) -> Dict[str, Any]:
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

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line('feature splits:', depth, writer)
        self.stash.write(depth + 1, writer)
        self._write_line('batch splits:', depth, writer)
        self.config_factory('batch_stash').write(depth + 1, writer)
        super().write(depth, writer)

    def write_config_section(self):
        from configparser import ConfigParser
        dfeats = self.asdict()['features']
        feats = {}
        for k, v in dfeats.items():
            vals = map(lambda s: s.replace('$', '$$'), dfeats[k].keys())
            vals = str(tuple(vals))
            feats[k] = vals
        conf = ConfigParser()
        conf['category_settings'] = feats
        conf.write(sys.stdout)
