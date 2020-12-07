# NER Example

This document describes the [named entity task example] to demonstrate
conditional random field and other features in the DeepZenols NLP framework.
Before working through this example, please first read through the
[movie review example].

The first thing you'll notice is that there is not one single configuration
file like there is for the [movie review example] or any test project in the
[deeplearn API].  Instead, all the files in this directory are read as one
contiguous configuration, which is done by giving a directory instead of file
path as the `config_file` parameter to the [IniConfig.__init__] initializer
(see the [config.py] source file).


## Configuration Files

Most of the configuration is similar to the [movie review example] and is
summarized below:

* [main.conf]: contains what's usually at the top of a configuration file,
  which are root paths the rest of the configuration builds on, GPU settings
  and some model defaults
* [lang.conf]: has the linguistic vectorizers, managers and embeddings just as
  before
* [vectorizer.conf]: contains the one hot encoded vectorizers for the features
  given in the CoNLL 2003 data set; something to note in this configuration
  file is the `mask_vectorizer` configuration entry, which we'll discuss later
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

The [EmbeddedRecurrentCRFNetworkSettings] given in the [model.conf] shows how
to configure the BiLSTM-CRF with an input embedded layer.  With the reference
to the `recurrent_crf_settings` entry, we have a complete neural network
without having to write any code for it.


### Scored Batch Iteration

The [ScoredBatchIterator] configured in the `model_settings` in [model.conf]
indicates to use a different scoring method.  This class is used in the
framework to calculate a different loss and produce the output.  Because we use
a [CRF] as the output layer, our output are the NER labels, which must be
treated differently than neural float tensor output.  We must also set
`reduce_outcomes = none` to pass the [CRF] output through unaltered.


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

To run the jupyter notebook:
1. Pip install: `pip install notebook`
1. Go to the notebook directory: `cd examples/ner/notebook`
1. Start the notebook: `jupyter notebook`
1. Start the execution in the notebook with `Cell > Run All`.


<!-- links -->

[Glove]: https://nlp.stanford.edu/projects/glove/

[named entity task example]: https://github.com/plandes/deepnlp/blob/master/example/ner
[movie review example]: movie-example.html
[deeplearn API]: https://plandes.github.io/deeplearn/index.html
[dataset.py]: https://github.com/plandes/deepnlp/blob/master/example/ner/src/ner/config.py
[main.conf]: https://github.com/plandes/deepnlp/blob/master/example/ner/resources/conf/main.conf
[vectorizer.conf]: https://github.com/plandes/deepnlp/blob/master/example/ner/resources/conf/vectorizer.conf

[ExtendedInterpolationEnvConfig]: https://plandes.github.io/util/api/zensols.config.html#zensols.config.iniconfig.ExtendedInterpolationEnvConfig
[IniConfig.__init__]: https://plandes.github.io/util/api/zensols.config.html#zensols.config.iniconfig.IniConfig.__init__
[CRF]: https://plandes.github.io/deeplearn/api/zensols.deeplearn.layer.html#zensols.deeplearn.layer.crf.CRF
[EmbeddedRecurrentCRFNetworkSettings]: ../api/zensols.deepnlp.layer.html#zensols.deepnlp.layer.embrecurcrf.EmbeddedRecurrentCRFNetworkSettings
[ScoredBatchIterator]: https://plandes.github.io/deeplearn/api/zensols.deeplearn.model.html#zensols.deeplearn.model.batchiter.ScoredBatchIterator
