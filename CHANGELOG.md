# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## [Unreleased]


## [1.6.0] - 2023-01-23
### Added
- Word piece API to map to non-word-piece tokens.
- Add word piece embeddings.


## [1.5.0] - 2022-11-06
### Added
- Sentence BERT (sbert) resource library and tested.
- Add HuggingFace local download model files resource library defaults.

### Changed
- Switched additional columns from tuple to as dictionary to solve ordering in
  `DataframeDocumentFeatureStash`.
- Fix `OneHotEncodedFeatureDocumentVectorizer` for document use case.
- Fix model `ClassifyNetwork` linear input size calculation so transformers (or
  models that do not use a terminal CRF layer) can add document level features.


## [1.4.1] - 2022-10-02
### Changed
- Transformer model fetch configuration.


## [1.4.0] - 2022-10-01
### Added
- Add a token embedding feature vectorizer.

### Changes
- Replace `None` shape component with -1 in `EnumContainer` vectorizer.


## [1.3.0] - 2022-08-08
Update dependent libraries release.

### Changed
- Upgrade torch 1.12.
- Upgraded to spaCy 3.2
- Upgrade resource library with `zensols.util` changes.


## [1.2.0] - 2022-06-14
This is primarily a refactoring release to simplify the API.

### Added
- Resource library configuration taken from examples and made generic for
  reuse.
- Resource library and example documentation.

### Changed
- Simplification of the API and examples.
- Added option to tokenize only during encoding for transformer components.
- Fixed transformer expander vectorizer bugs.
- Fixed deallocation issues in test notebook.

### Removed
- Replaced example model configuration with `--override` option semantics.


## [1.1.2] - 2022-05-15
### Changed
- Fixed YML resource library configuration files not found.


## [1.1.1] - 2022-05-15
### Changed
- Retrofit resource library and examples with batch metadata changes from
  `zensols.deeplearn`.


## [1.1.0] - 2022-05-04
### Added
- A recurrent CRF and default classify facade to the resource library.
- Tokenized transformer document truncation.
- Token classification resource library.
- More huggingface support, models and tests.
- Facebook fastText embeddings.

### Changed
- Recurrent embedded CRF uses a new network settings factory method.
- Update examples.
- Pin `zensols.nlp` version dependency to minor (second component) release.
- All deep NLP vectorizers inherit from `TransformableFeatureVectorizer` to
  simplify class hierarchy.  This change now requires `encode_transformed` in
  respective vectorizer configurations.
- Embedded Bi{LSTM,GRU,RNN}-CRF}: utilize `recurcrf` module decode over
  re-implementation.
- Change default dropout, activation order (that use them) in all layers per
  the literature.


## [1.0.1] - 2022-02-12
### Added
- Runtime bench marking.
- Missing batch configuration in resource library from `zensols.deeplearn`.
- Add observer pattern for logging and Pandas data frame / CSV output.

### Changed
- Word embedding model now compatible with gensim 4.


## [1.0.0] - 2022-01-25
Major stable release.

### Added
- DistilBERT pooler output.
- The `word2vec` model is installed programmatically.
- Clickbate example now also includes RoBERTa and DistilBERT.

### Changed
- Upgrade to transformers 4.12.5.
- Fix duplicate word embeddings matrix copied to GPU, which saves space and
  time.
- Other efficiencies such as log guards and data structure creation checks.
- Notebook example fixes and cleanup.

### Removed
- PyTorch init call in nlp package init so the client can do it before other
  modules are loaded.


## [0.0.8] - 2021-10-22
### Added
- A factory method in `zensols.deepnlp.WordEmbedModel` to create a Gensim
  `KeyedVectors` instance to provide word vector operations for all embedding
  model types.
- Make sub directory in text embedding models configurable.
- Glove model automatically downloads embeddings if not present on the file
  system using `zensols.install`.

### Changed
- `FeatureDocumentVectorizerManager.token_feature_ids` default to its owned
  `doc_parser`'s token features.
- Pin dependencies to working huggingface transformers as new version breaks
  this version.
- Fix glove embedding factory create functionality.


## [0.0.7] - 2021-09-22
### Changed
- Refactored downstream renaming of files from `zensols.deeplearn`.
- Moved `ClassificationPredictionMapper` class to new `classify` module.

### Added
- Classification module and classes now fully implement text classification
  with RNN/LSTM/GRU network types or any HuggingFace transformer with pooler
  output.  This means there is no coding necessary for text classification with
  the exception of writing a data loader if not in a supported format like
  Pandas dataframe (i.e. CSV file).
- Configuration resource library.
- Clickbate corpus example and documentation.


## [0.0.6] - 2021-09-07
### Changed
- Revert to version 3.8.3 of gensim and support back/forward comparability.
- Upgrade zensols libraries.
- Documentation and clean up.


## [0.0.5] - 2021-08-07
### Changed
- Update dependencies.


## [0.0.4] - 2021-08-07
### Added
- Sequence/token classification for BiLSTM+CRF and HuggingFace transformers.
  This has been tested with BERT/DistilBERT/RoBERTa and the large BERT models.
- The HuggingFace transformers optimizer for `AdamW` and scheduler for
  functionality such as fine tuning warm up.
- More NLP facade specific support such as easier embedding model access.
- Better support for Jupyter notebook rapid prototyping and experimentation.
- Jupyter integration tests in review movie example.

### Changed
- Upgrade to spaCy 3 via the [zensols.nlparse] dependency.

### Removed
- Move feature containers and parser to [zensols.nlparse], including test
  cases.
- The dependency on [bcolz] as it is no longer maintained.  The caching of
  binary word vectors was replaced with [H5PY].


## [0.0.3] - 2021-04-30
### Added
- BERT/DistilBERT/RoBERTa transformer word piece tokenizer to linguistic token
  mapping.
- Upgraded to `gensum` 4.0.1.
- Upgraded to [zensols.deeplearn] 0.1.2, which is upgraded to use [PyTorch] 1.8.
- Added simple vectorizer example.
- Multiprocessing vectorization now supports GPU access via torch
  multiprocessing subsystem.

### Changed
- Refactored word embedding (sub) modules.
- Moved BERT transformer embeddings to separate `transformer` module.
- Refactored vectorizers to standardize around `FeatureDocument` rather token
  collection instances.
- Standardize vectorizer shapes.
- Updated examples to use new vectorizer API and [zensols.util] application
  CLI.


## [0.0.2] - 2020-12-29
Maintenance release.
### Changed
- Updated dependencies and tested across Python 3.7, 3.8, 3.9.


## [0.0.1] - 2020-05-04
### Added
- Initial version.


<!-- links -->
[Unreleased]: https://github.com/plandes/deepnlp/compare/v1.6.0...HEAD
[1.6.0]: https://github.com/plandes/deepnlp/compare/v1.5.0...v1.6.0
[1.5.0]: https://github.com/plandes/deepnlp/compare/v1.4.1...v1.5.0
[1.4.1]: https://github.com/plandes/deepnlp/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/plandes/deepnlp/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/plandes/deepnlp/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/plandes/deepnlp/compare/v1.1.2...v1.2.0
[1.1.2]: https://github.com/plandes/deepnlp/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/plandes/deepnlp/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/plandes/deepnlp/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/plandes/deepnlp/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/plandes/deepnlp/compare/v0.0.8...v1.0.0
[0.0.8]: https://github.com/plandes/deepnlp/compare/v0.0.7...v0.0.8
[0.0.7]: https://github.com/plandes/deepnlp/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/plandes/deepnlp/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/plandes/deepnlp/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/plandes/deepnlp/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/plandes/deepnlp/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/plandes/deepnlp/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/plandes/deepnlp/compare/v0.0.0...v0.0.1

[zensols.util]: https://github.com/plandes/util
[zensols.nlparse]: https://github.com/plandes/nlparse
[zensols.deeplearn]: https://github.com/plandes/deeplearn
[PyTorch]: https://pytorch.org
[bcolz]: https://github.com/Blosc/bcolz
[H5PY]: https://www.h5py.org
