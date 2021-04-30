# DeepZensols Natural Language Processing

[![PyPI][pypi-badge]][pypi-link]
[![Python 3.7][python37-badge]][python37-link]
[![Python 3.8][python38-badge]][python38-link]
[![Python 3.9][python39-badge]][python39-link]

Deep learning utility library for natural language processing that aids in
feature engineering and embedding layers (see the [full documentation]).

Features:
* Configurable layers with little to no need to write code.
* [Natural language specific layers](doc/md/layers.md):
  * Easily configurable word embedding layers for [Glove], [Word2Vec],
    [fastText].
  * Huggingface transformer ([BERT]) context based word vector layer.
  * Full [Embedding+BiLSTM-CRF] implementation using easy to configure
	constituent layers.
* [NLP specific vectorizers] that generate [zensols deeplearn] encoded and
  decoded [batched tensors] for [spaCy] parsed features, dependency tree
  features, overlapping text features and others.
* Easily swapable during runtime embedded layers as [batched tensors] and other
  linguistic vectorized features.
* Support for token, document and embedding level vectorized features.
* Transformer word piece to linguistic token mapping.
* Two full documented examples provided as both command line and [Jupyter
  notebooks](#usage-and-examples).


## Documentation

See the [full documentation].


## Obtaining

The easiest way to install the command line program is via the `pip` installer:
```bash
pip3 install zensols.deepnlp
```

Binaries are also available on [pypi].


## Usage and Examples

If you're in a rush, you can dive right in to the [Movie Review Sentiment]
example, which is a working project that uses this library.  However, you'll
either end up reading up on the [zensols deeplearn] library before or during
the tutorial.

The usage of this library is explained in terms of two examples:
* The [Movie Review Sentiment] trained and tested on the [Stanford movie
  review] and [Cornell sentiment polarity] data sets, which assigns a positive
  or negative score to a natural language movie review by critics.  Also see
  the [Jupyter movie notebook].

* The [Named Entity Recognizer] trained and tested on the [CoNLL 2003 data set]
  to label named entities on natural language text.  Also see the [Jupyter NER
  notebook].


## Attribution

This project, or example code, uses:
* [Gensim] for [Glove], [Word2Vec] and [fastText] word embeddings.
* [Huggingface Transformers] for [BERT] contextual word embeddings.
* [bcolz] for fast read access to word embedding vectors.
* [zensols nlparse] for feature generation from [spaCy] parsing.
* [zensols deeplearn] for deep learning network libraries.

Corpora used include:
* [Stanford movie review]
* [Cornell sentiment polarity]
* [CoNLL 2003 data set]


## Changelog

An extensive changelog is available [here](CHANGELOG.md).


## License

[MIT License](LICENSE.md)

Copyright (c) 2020 Paul Landes


<!-- links -->
[pypi]: https://pypi.org/project/zensols.deepnlp/
[pypi-link]: https://pypi.python.org/pypi/zensols.deepnlp
[pypi-badge]: https://img.shields.io/pypi/v/zensols.deepnlp.svg
[python37-badge]: https://img.shields.io/badge/python-3.7-blue.svg
[python37-link]: https://www.python.org/downloads/release/python-370
[python38-badge]: https://img.shields.io/badge/python-3.8-blue.svg
[python38-link]: https://www.python.org/downloads/release/python-380
[python39-badge]: https://img.shields.io/badge/python-3.9-blue.svg
[python39-link]: https://www.python.org/downloads/release/python-390

[Gensim]: https://radimrehurek.com/gensim/
[Huggingface Transformers]: https://huggingface.co
[Glove]: https://nlp.stanford.edu/projects/glove/
[Word2Vec]: https://code.google.com/archive/p/word2vec/
[fastText]: https://fasttext.cc
[BERT]: https://huggingface.co/transformers/model_doc/bert.html
[bcolz]: https://pypi.org/project/bcolz/
[spaCy]: https://spacy.io
[Pandas]: https://pandas.pydata.org

[Stanford movie review]: https://ai.stanford.edu/~amaas/data/sentiment/
[Cornell sentiment polarity]: https://www.cs.cornell.edu/people/pabo/movie-review-data/
[CoNLL 2003 data set]: https://www.clips.uantwerpen.be/conll2003/ner/

[zensols deeplearn]: https://github.com/plandes/deeplearn
[zensols nlparse]: https://github.com/plandes/nlparse

[full documentation]: https://plandes.github.io/deepnlp/index.html
[Movie Review Sentiment]: doc/movie-example.md
[Named Entity Recognizer]: doc/ner-example.md
[Embedding+BiLSTM-CRF]: https://plandes.github.io/deepnlp/doc/ner-example.html#bilstm-crf
[batched tensors]: https://plandes.github.io/deeplearn/doc/preprocess.html#batches
[deep convolution layer]: https://plandes.github.io/deepnlp/api/zensols.deepnlp.layer.html#zensols.deepnlp.layer.conv.DeepConvolution1d
[NLP specific vectorizers]: doc/vectorizers.md
[Jupyter NER notebook]: https://github.com/plandes/deepnlp/blob/master/example/ner/notebook/ner.ipynb
[Jupyter movie notebook]: https://github.com/plandes/deepnlp/blob/master/example/movie/notebook/movie.ipynb
