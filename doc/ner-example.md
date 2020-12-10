# NER Example

This document describes the [named entity task example] to demonstrate
conditional random field and other features in the DeepZenols NLP framework.
Before working through this example, please first read through the
[movie review example].

The first thing you'll notice is that there is not one single configuration
file like there is for the [movie review example] or any test project in the
[deeplearn API].  Instead, all the files in this directory are read as one
contiguous configuration, which is done by giving a directory instead of file
path as the `config_file` parameter to the [IniConfig initializer] (see the
[config.py] source file).


## Configuration Files

Most of the configuration is similar to the [movie review example] and is
summarized below:

* [main.conf]: contains what's usually at the top of a configuration file,
  which are root paths the rest of the configuration builds on, GPU settings
  and some model defaults
* [lang.conf]: has the linguistic vectorizers, managers and embeddings just as
  before
* [vectorizer.conf]: contains the one hot encoded vectorizers for the features
  given in the [CoNLL 2003 data set]; something to note in this configuration
  file is the `mask_vectorizer` configuration entry (see the [Mask](#mask)
  section)
* [batch.conf]: has the batch configuration and settings, such as batch size
* [corpus.conf]: configuration for the code that parses the CoNNL formatted
  corpus
* [model.conf]: contains the network and model configuration.


### BiLSTM-CRF

The [model.conf] is the more interesting file for this project as it defines
the [CRF] used to for sequencing over the NER tags.  For our example, we
configure a BiLSTM-CRF, which is a bi-directional LSTM with a decoding layer
connected to a CRF terminal layer.  This network learns sequences of nominal
labels, which in our case, are the NER tags.  The `recurrent_crf_settings`
entry contains the configuration for this BiLSTM-CRF.

The [EmbeddedRecurrentCRFSettings] given in the [model.conf] shows how
to configure the BiLSTM-CRF with an input embedded layer.  With the reference
to the `recurrent_crf_settings` entry, which uses an instance
[EmbeddedRecurrentCRF], we have a complete neural network without having
to write any code for it.


### Mask

As mentioned in the [Configuration Files](#configuration-files) section, as
mask vectorizer is specified.  This is needed for the [CRF] to mask out blank
tokens for sentences shorter than a max length.  Usually, zeroed tensors are
used for token slots not used, for example in the word embedding layer for deep
learning networks.  This is because the zero vectors are learned for sentences
are shorter.  However, the CRF layer needs to block these as valid state
transitions during training and testing.


### Scored Batch Iteration

The [ScoredBatchIterator] configured in the `model_settings` in [model.conf]
indicates to use a different scoring method.  This class is used in the
framework to calculate a different loss and produce the output, which must be
treated differently than neural float tensor output.  This is because the
Viterbi algorithm is used to determine the lowest cost path through the
elements.  The sum of this path is used as the cost instead of a differential
optimization function.

Because we use a [CRF] as the output layer for [EmbeddedRecurrentCRF], our
output are the NER labels.  Therefore, must also set `reduce_outcomes = none`
to pass the [CRF] output through unaltered.


## Code

As mentioned, no code is necessary for the model is it is already provided in
configuration using the framework.  The code that is necessary includes:
* [corpus.py] to parse the [CoNLL 2003 data set]
* [batch.py] defines data point and batch classes just as seen in the [movie
  review example]
* [facade.py] defines and wires the batch mappings just as seen in the [movie
  review example]


### Corpus Install

To install the corpus:
1. Install [GNU make](https://www.gnu.org/software/make/)
1. Change the working directory to the example: `cd examples/ner`
1. Download and install the corpora: `make corpus`.  This should download all
   the necessary corpora and [Glove] and [Word2Vec] word embeddings.
1. Confirm there are no errors and the corpus directories exist:
   * `corpus/stanfordSentimentTreebank/datasetSentences.txt`
   * `corpus/rt-polaritydata/rt-polarity.{pos,neg}`
   * `corpus/glove/glove.6B.300d.txt`
   * `corpus/word2vec/GoogleNews-vectors-negative300.bin.gz`


### Command Line

To train and test the model invoke: `make modeltraintest`.


### Jupyter Notebook

To run the [Jupyter NER notebook]:
1. Pip install: `pip install notebook`
1. Go to the notebook directory: `cd examples/ner/notebook`
1. Start the notebook: `jupyter notebook`
1. Start the execution in the notebook with `Cell > Run All`.


<!-- links -->

[Glove]: https://nlp.stanford.edu/projects/glove/
[Word2Vec]: https://code.google.com/archive/p/word2vec/
[CoNLL 2003 data set]: https://www.clips.uantwerpen.be/conll2003/ner/

[named entity task example]: https://github.com/plandes/deepnlp/blob/master/example/ner
[movie review example]: movie-example.html

[deeplearn API]: https://plandes.github.io/deeplearn/index.html
[config.py]: https://github.com/plandes/deepnlp/blob/master/example/ner/src/ner/config.py
[corpus.py]: https://github.com/plandes/deepnlp/blob/master/example/ner/src/ner/corpus.py
[batch.py]: https://github.com/plandes/deepnlp/blob/master/example/ner/src/ner/batch.py
[facade.py]: https://github.com/plandes/deepnlp/blob/master/example/ner/src/ner/facade.py

[batch.conf]: https://github.com/plandes/deepnlp/blob/master/example/ner/resources/conf/batch.conf
[model.conf]: https://github.com/plandes/deepnlp/blob/master/example/ner/resources/conf/model.conf
[vectorizer.conf]: https://github.com/plandes/deepnlp/blob/master/example/ner/resources/conf/vectorizer.conf
[main.conf]: https://github.com/plandes/deepnlp/blob/master/example/ner/resources/conf/main.conf
[lang.conf]: https://github.com/plandes/deepnlp/blob/master/example/ner/resources/conf/lang.conf
[corpus.conf]: https://github.com/plandes/deepnlp/blob/master/example/ner/resources/conf/corpus.conf

[Jupyter NER notebook]: https://github.com/plandes/deepnlp/blob/master/example/ner/notebook/ner.ipynb

[ExtendedInterpolationEnvConfig]: https://plandes.github.io/util/api/zensols.config.html#zensols.config.iniconfig.ExtendedInterpolationEnvConfig
[IniConfig initializer]: https://plandes.github.io/util/api/zensols.config.html#zensols.config.iniconfig.IniConfig.__init__
[CRF]: https://plandes.github.io/deeplearn/api/zensols.deeplearn.layer.html#zensols.deeplearn.layer.crf.CRF
[ScoredBatchIterator]: https://plandes.github.io/deeplearn/api/zensols.deeplearn.model.html#zensols.deeplearn.model.batchiter.ScoredBatchIterator
[EmbeddedRecurrentCRFSettings]: ../api/zensols.deepnlp.layer.html#zensols.deepnlp.layer.embrecurcrf.EmbeddedRecurrentCRFSettings
[EmbeddedRecurrentCRF]: ../api/zensols.deepnlp.layer.html#zensols.deepnlp.layer.embrecurcrf.EmbeddedRecurrentCRF
