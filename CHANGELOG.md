# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## [Unreleased]


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
[Unreleased]: https://github.com/plandes/deepnlp/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/plandes/deepnlp/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/plandes/deepnlp/compare/v0.0.0...v0.0.1

[zensols.util]: https://github.com/plandes/util
[zensols.deeplearn]: https://github.com/plandes/deeplearn
[PyTorch]: https://pytorch.org
