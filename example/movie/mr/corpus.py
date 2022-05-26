"""Merge the Stanford movie review corpus with the Cornell labels.

"""
__author__ = 'Paul Landes'

from typing import Set
from dataclasses import dataclass, field
import logging
import re
import shutil
from pathlib import Path
import pandas as pd
from zensols.install import Installer, Resource

logger = logging.getLogger(__name__)


@dataclass
class DatasetFactory(object):
    """Creates a dataframe out of the merged dataset (text reviews with labels).

    """
    FILE_NAME = re.compile(r'^(.+)\.csv$')

    installer: Installer
    standford_resource: Resource
    cornell_resource: Resource
    dataset_path: Path
    tok_len: int
    throw_out: Set[str] = field(repr=False)
    repls: dict = field(repr=False)
    split_col: str

    def __post_init__(self):
        self.installer()
        self.stanford_path = self.installer[self.standford_resource]
        self.rt_pol_path = self.installer[self.cornell_resource]

    @staticmethod
    def split_sents(line: str) -> str:
        """In the Cornell corpus, many sentences are joined.  This them
        in to separate sentences.

        """
        toks = re.split(r'([a-zA-Z0-9]{3,}) \. ([^.]{2,})', line)
        if len(toks) == 4:
            lines = [' '.join(toks[0:2]), ' '.join(toks[2:])]
        else:
            lines = [line]
        return lines

    def sent2bow(self, sent: str) -> str:
        """Create an skey from a text that represents a sentence.

        """
        sent = sent.lower().strip()
        sent = sent.encode('ascii', errors='ignore').decode()
        for repl in self.repls:
            sent = sent.replace(*repl)
        sent = re.sub(r"\[([a-zA-Z0-9' ]+)\]", '\\1', sent)
        sent = re.split(r"[\t ,';:\\/.]+", sent)
        sent = filter(lambda x: len(x) > 0, sent)
        sent = filter(lambda x: x not in self.throw_out, sent)

        # tried taking every other word for utterances that start the same, but
        # this brought down matches
        sent = tuple(sent)[0:self.tok_len]
        sent = '><'.join(sent)
        return sent

    def polarity_df(self, path, polarity) -> pd.DataFrame:
        """Create a polarity data frame.

        :param path: the path to the Cornell annotated corpus
        :param polarity: the string used for the polarity column (`p` or `n`)

        """
        lines = []
        rid_pol = []
        with open(path, encoding='latin-1') as f:
            for line in f.readlines():
                line = line.strip()
                lines.append(line)

        for line in lines:
            # tried to split on mulitple sentences that were joined, but that
            # makes things worse
            for sent in self.split_sents(line):
                key = self.sent2bow(sent)
                if len(key) == 0:
                    continue
                rid_pol.append((polarity, key))
        return pd.DataFrame(rid_pol, columns=['polarity', 'skey'])

    def create_corp(self):
        # create a dataframe with the sentence keys and polarity used to match later
        df_polarity = pd.concat(
            [self.polarity_df(self.rt_pol_path / 'rt-polarity.pos', 'p'),
             self.polarity_df(self.rt_pol_path / 'rt-polarity.neg', 'n')])

        # load the sentences from the Stanford corpus
        df_sents = pd.read_csv(
            self.stanford_path / 'datasetSentences.txt', sep='\t')

        # add the skey to the Stanford corpus
        df_sents_keyed = df_sents.copy()
        df_sents_keyed['skey'] = df_sents.sentence.map(self.sent2bow)
        self.df_sents_keyed = df_sents_keyed

        # print out the number of positive and negative comments collated
        df_cnts = df_polarity['polarity'].value_counts()
        logger.debug(df_cnts)
        assert(df_cnts['n'].item() == 5834)
        assert(df_cnts['p'].item() == 5757)
        self.df_polarity = df_polarity

    def match_label_data(self):
        df_sents = self.df_sents_keyed
        df_polarity = self.df_polarity
        df = self.df_sents_keyed.merge(df_polarity, how='left', on='skey')
        df_mismatch = df[df['polarity'].isnull()]
        df_match = df[~df['polarity'].isnull()]
        dups = df_match[df_match.skey.duplicated()]
        logger.info(f'sentences: {df_sents.shape[0]}, polarity found: {df_polarity.shape[0]}, merged: {df.shape[0]}, dups: {dups.shape[0]}')
        logger.info(f'matched {df_match.shape[0]}, mismatched: {df_mismatch.shape[0]} ({df_mismatch.shape[0] / df.shape[0] * 100:.3f}%)')
        # remove duplicate matches
        df_match = df_match[~df_match.sentence_index.isin(dups.sentence_index)]
        df_match = df_match.drop(columns=['skey'])
        logger.info(f'total data set: {df_match.shape[0]} ({df_match.shape[0] / df.shape[0] * 100:.3f}% of Stanford dataaset)')
        self.df_match = df_match

    def create_dataset(self):
        df_match = self.df_match

        def subset_ds(split_label):
            """Create a dataset subset.
            :param split_label: indicates the split type:
                                1 = train
                                2 = test
                                3 = dev
            """
            return df_match[df_match.sentence_index.isin(df_split[df_split.splitset_label == split_label].sentence_index)]

        df_split = pd.read_csv(self.stanford_path / 'datasetSplit.txt')
        df_train = subset_ds(1)
        df_test = subset_ds(2)
        df_dev = subset_ds(3)

        logger.info(f'total split items: {df_split.shape[0]}')
        logger.info(f'train: {df_train.shape[0]}, test: {df_test.shape[0]}, dev: {df_dev.shape[0]}')
        logger.info(f'train to test split: {df_train.shape[0] / float(df_train.shape[0] + df_test.shape[0]):.2f}')

        dataset_path = self.dataset_path
        dataset_path.mkdir(parents=True, exist_ok=True)
        df_train.to_csv(dataset_path / 'train.csv', index=False)
        df_test.to_csv(dataset_path / 'test.csv', index=False)
        df_dev.to_csv(dataset_path / 'dev.csv', index=False)

    def dup_check(self, df: pd.DataFrame):
        sents = set()
        for i, row in df.iterrows():
            sent = row['sentence']
            if sent in sents:
                raise ValueError(f'duplicate sentence: idx={i}: {row}')
            sents.add(sent)

    def clear(self):
        logger.warning(f'deleting {self.dataset_path}')
        if self.dataset_path.exists():
            shutil.rmtree(self.dataset_path)
        self._dataset.clear()

    def assert_dataset(self):
        if not self.dataset_path.exists():
            self.create_corp()
            self.match_label_data()
            self.create_dataset()

    def load_dataset(self) -> pd.DataFrame:
        dfs = []
        for path in self.dataset_path.iterdir():
            m = self.FILE_NAME.match(path.name)
            split_type = m.group(1)
            logger.info(f'parsing {split_type} in {path}')
            df = pd.read_csv(path)
            df[self.split_col] = split_type
            dfs.append(df)
        df = pd.concat(dfs)
        self.dup_check(df)
        df.index = df['sentence_index']
        df.index = df.index.rename('sid').map(str)
        df = df.drop(['sentence_index'], axis=1)
        return df

    @property
    def dataset(self) -> pd.DataFrame:
        logger.info('maybe creating and then loading dataset')
        self.assert_dataset()
        df = self.load_dataset()
        return df
