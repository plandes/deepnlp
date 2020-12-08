# DeepZensols Natural Language Processing

[![PyPI][pypi-badge]][pypi-link]
[![Python 3.7][python37-badge]][python37-link]

Deep learning utility library for natural language processing that aids in
feature engineering and embedding layers (see the [full documentation]).

Features:
* Configurable layers with no need to write code.
* Integration with [Pandas] data frames from data ingestion.


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
  or negative score to a natural language movie review by critics.
* The [Named Entity Recognizer] trained and tested on the [CoNLL 2003 data set]
  to label named entities on natural language text.


## Attribution

This project, or example code, uses:
* [Gensim] for [Glove] and [Word2Vec] word embeddings.
* [Huggingface Transformers] for BERT contextual word embeddings.
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

[Gensim]: https://radimrehurek.com/gensim/
[Huggingface Transformers]: https://huggingface.co
[Glove]: https://nlp.stanford.edu/projects/glove/
[Word2Vec]: https://code.google.com/archive/p/word2vec/
[bcolz]: https://pypi.org/project/bcolz/
[spaCy]: https://spacy.io
[Pandas]: https://pandas.pydata.org

[Stanford movie review]: https://ai.stanford.edu/~amaas/data/sentiment/
[Cornell sentiment polarity]: https://www.cs.cornell.edu/people/pabo/movie-review-data/
[CoNLL 2003 data set]: https://www.clips.uantwerpen.be/conll2003/ner/

[full documentation]: https://plandes.github.io/deepnlp/index.html
[Movie Review Sentiment]: doc/movie-example.md
[Named Entity Recognizer]: doc/ner-example.md

[zensols deeplearn]: https://github.com/plandes/deeplearn
[zensols nlparse]: https://github.com/plandes/nlparse
