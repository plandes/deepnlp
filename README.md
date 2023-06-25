# DeepZensols Natural Language Processing

[![PyPI][pypi-badge]][pypi-link]
[![Python 3.9][python39-badge]][python39-link]
[![Python 3.10][python310-badge]][python310-link]
[![Build Status][build-badge]][build-link]

Deep learning utility library for natural language processing that aids in
feature engineering and embedding layers.

* See the [full documentation].
* Paper on [arXiv](http://arxiv.org/abs/2109.03383).

Features:
* Configurable layers with little to no need to write code.
* [Natural language specific layers]:
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
* Two full documented reference models provided as both command line and
  [Jupyter notebooks](#usage-and-reference-models).
* Command line support for training, testing, debugging, and creating
  predictions.


## Documentation

* [Full documentation](https://plandes.github.io/deepnlp/index.html)
* [Layers](https://plandes.github.io/deepnlp/doc/layers.html): NLP specific
  layers such as embeddings and transformers
* [Vectorizers](https://plandes.github.io/deepnlp/doc/vectorizers.html):
  specific vectorizers that digitize natural language text in to tensors ready
  as [PyTorch] input
* [API reference](https://plandes.github.io/install/api.html)
* [Reference Models](#usage-and-reference-models)


## Obtaining

The easiest way to install the command line program is via the `pip` installer:
```bash
pip3 install --use-deprecated=legacy-resolver zensols.deepnlp
```

Binaries are also available on [pypi].


## Usage and Reference Models

If you're in a rush, you can dive right in to the [Clickbate Text
Classification] reference model, which is a working project that uses this
library.  However, you'll either end up reading up on the [zensols deeplearn]
library before or during the tutorial.

The usage of this library is explained in terms of the reference models:

* The [Clickbate Text Classification] is the best reference model to start with
  because the only code consists of is the corpus reader and a module to remove
  sentence segmentation (corpus are newline delimited headlines).  It was also
  uses [resource libraries], which greatly reduces complexity, where as the
  other reference models do not.  Also see the [Jupyter clickbate
  classification notebook].

* The [Movie Review Sentiment] trained and tested on the [Stanford movie
  review] and [Cornell sentiment polarity] data sets, which assigns a positive
  or negative score to a natural language movie review by critics.  Also see
  the [Jupyter movie sentiment notebook].

* The [Named Entity Recognizer] trained and tested on the [CoNLL 2003 data set]
  to label named entities on natural language text.  Also see the [Jupyter NER
  notebook].

The unit test cases are also a good resource for the more detailed programming
integration with various parts of the library.


## Attribution

This project, or reference model code, uses:
* [Gensim] for [Glove], [Word2Vec] and [fastText] word embeddings.
* [Huggingface Transformers] for [BERT] contextual word embeddings.
* [h5py] for fast read access to word embedding vectors.
* [zensols nlparse] for feature generation from [spaCy] parsing.
* [zensols deeplearn] for deep learning network libraries.

Corpora used include:
* [Stanford movie review]
* [Cornell sentiment polarity]
* [CoNLL 2003 data set]


## Citation

If you use this project in your research please use the following BibTeX entry:
```bibtex
@article{Landes_DiEugenio_Caragea_2021,
  title={DeepZensols: Deep Natural Language Processing Framework},
  url={http://arxiv.org/abs/2109.03383},
  note={arXiv: 2109.03383},
  journal={arXiv:2109.03383 [cs]},
  author={Landes, Paul and Di Eugenio, Barbara and Caragea, Cornelia},
  year={2021},
  month={Sep}
}
```

## Community

Please star the project and let me know how and where you use this API.
Contributions as pull requests, feedback and any input is welcome.


## Changelog

An extensive changelog is available [here](CHANGELOG.md).


## License

[MIT License](LICENSE.md)

Copyright (c) 2020 - 2023 Paul Landes


<!-- links -->
[pypi]: https://pypi.org/project/zensols.deepnlp/
[pypi-link]: https://pypi.python.org/pypi/zensols.deepnlp
[pypi-badge]: https://img.shields.io/pypi/v/zensols.deepnlp.svg
[python39-badge]: https://img.shields.io/badge/python-3.9-blue.svg
[python39-link]: https://www.python.org/downloads/release/python-390
[python310-badge]: https://img.shields.io/badge/python-3.10-blue.svg
[python310-link]: https://www.python.org/downloads/release/python-310
[build-badge]: https://github.com/plandes/util/workflows/CI/badge.svg
[build-link]: https://github.com/plandes/deepnlp/actions

[PyTorch]: https://pytorch.org
[Gensim]: https://radimrehurek.com/gensim/
[Huggingface Transformers]: https://huggingface.co
[Glove]: https://nlp.stanford.edu/projects/glove/
[Word2Vec]: https://code.google.com/archive/p/word2vec/
[fastText]: https://fasttext.cc
[BERT]: https://huggingface.co/transformers/model_doc/bert.html
[h5py]: https://www.h5py.org
[spaCy]: https://spacy.io
[Pandas]: https://pandas.pydata.org

[Stanford movie review]: https://nlp.stanford.edu/sentiment/
[Cornell sentiment polarity]: https://www.cs.cornell.edu/people/pabo/movie-review-data/
[CoNLL 2003 data set]: https://www.clips.uantwerpen.be/conll2003/ner/

[zensols deeplearn]: https://github.com/plandes/deeplearn
[zensols nlparse]: https://github.com/plandes/nlparse

[full documentation]: https://plandes.github.io/deepnlp/index.html
[resource libraries]: https://plandes.github.io/util/doc/config.html#resource-libraries
[Natural language specific layers]: https://plandes.github.io/deepnlp/doc/layers.html
[Clickbate Text Classification]: https://plandes.github.io/deepnlp/doc/clickbate-example.html
[Movie Review Sentiment]: https://plandes.github.io/deepnlp/doc/movie-example.html
[Named Entity Recognizer]: https://plandes.github.io/deepnlp/doc/ner-example.html
[Embedding+BiLSTM-CRF]: https://plandes.github.io/deepnlp/doc/ner-example.html#bilstm-crf
[batched tensors]: https://plandes.github.io/deeplearn/doc/preprocess.html#batches
[deep convolution layer]: https://plandes.github.io/deepnlp/api/zensols.deepnlp.layer.html#zensols.deepnlp.layer.conv.DeepConvolution1d
[NLP specific vectorizers]: https://plandes.github.io/deepnlp/doc/vectorizers.html
[Jupyter NER notebook]: https://github.com/plandes/deepnlp/blob/master/example/ner/notebook/ner.ipynb
[Jupyter movie sentiment notebook]: https://github.com/plandes/deepnlp/blob/master/example/movie/notebook/movie.ipynb
[Jupyter clickbate classification notebook]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/notebook/clickbate.ipynb
