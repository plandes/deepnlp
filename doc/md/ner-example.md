# NER Example

This document describes the [named entity task example] to demonstrate
conditional random field and other features in the DeepZenols NLP framework.
Before working through this example, please first read through the
[movie review example].  The difference between this and the movie review
sentiment example is this project classifies at the token level instead of
sentence level.  For this reason, only those parts that differ from the movie
review example are documented.


## Configuration

The [app.conf] is nearly identical with the [movie review example] except that
it adds `--override` option defaults for the [HuggingFace] transformer model
name `bert-base-cased`.  The `app_imp_conf` maps [YAML] file extensions to
[ConditionalYamlConfig] with the `type_map` property.

This NER example is similar to the [movie review example] [obj.yml] file, but
slightly differs in the following ways:

* Corpus resources are defined and downloaded the first time it is accessed.
  However, this downloads three separate files that are not compressed.
* This example does not import [feature resource library] as it is different
  enough for it to be easier to redefine the configuration found in the
  *Corpus/feature creation* section.
* The *Language parsing* section overrides the [FeatureDocumentParser] to
  remove all space tokens, empty sentences and not add named entities (that's
  what this project classifies) and keep only the features parsed from the
  CoNLL corpus.
* The *Vectorization* section has vectorizers for the CoNLL corpus features and
  adds them to the language vectorizer manager.
* The *Batch* configuration differs on a per section basis in the following
  ways:
  * `conll_lang_batch_mappings`: [batch mapping](reslib.html#batch-stash) are
  added for the CoNLL corpus features.  We must also add a transformer specific
  label since there is not a one-to-one mapping from tokens to word piece
  tokens.
  * `ner_batch_mappings`: this transformer label is then only used when the
  embeddings name (`ner_default:name`) from the aforementioned `--override`
  configuration.  See [YAML conditionals] for more information on how this
  if/then/else logic is utilized.  Note that what mappings we add and keep
  closely resembles that of the [movie review example].
  * `batch_dir_stash`: Which features to group together also resembles that of
    the [movie review example]
  * `batch_stash`: we refer to the `ner_batch_mappings` we defined earlier, use
    our own custom [DataPoint] class, set number of sub-processes to 2 (memory
    constraint on large feature sets) and mini-batch size will have 32
    sentences per batch.
* The *Model* shows how the `exectuor` is configured with the `net_settings`,
  which tells the framework which network model to use.  For our example, we
  configure a BiLSTM-CRF, which is a bi-directional LSTM with a decoding layer
  connected to a CRF terminal layer.  This network learns sequences of nominal
  labels, which in our case, are the NER tags.  The `recurrent_crf_settings`
  entry contains the configuration for this BiLSTM-CRF.


## Code

As mentioned, no code is necessary for the model is it is already provided in
configuration using the framework.  The code that is necessary includes:
* [corpus.py] to parse the [CoNLL 2003 data set]
* [domain.py] defines data point class, and the overridden prediction mapper to
  set the `is_pred` flag
* [app.py] a small a small application to demonstrate how to prototype


### Command Line

Everything can be done with the harness script:
```bash
# get the command line help using a thin wrapper around the framework
./harness.py -h
# the executor tests and trains the model, use it to get the stats used to train
./harness.py info
# print a sample Glove 50 (default) batch of what the model will get during training
./harness.py info -i batch
# print a sample transformer batch of what the model will get during training
./harness.py info -i batch -c models/transformer-trainable.conf 
# train and test the model but switch to model profile with optmizied 
./harness.py traintest -p
# all model, its (hyper)parameters, metadata and results are stored in subdirectory of files
./harness.py result
```

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
[HuggingFace]: https://github.com/huggingface/transformers
[YAML]: https://yaml.org
[Jupyter NER notebook]: https://github.com/plandes/deepnlp/blob/master/example/ner/notebook/ner.ipynb

[named entity task example]: https://github.com/plandes/deepnlp/blob/master/example/ner
[movie review example]: movie-example.html
[YAML conditionals]: https://plandes.github.io/util/doc/config.html#yaml-conditionals

[deeplearn API]: https://plandes.github.io/deeplearn/index.html
[app.conf]: https://github.com/plandes/deepnlp/blob/master/example/ner/resources/app.conf
[obj.yml]: https://github.com/plandes/deepnlp/blob/master/example/ner/resources/obj.yml
[config.py]: https://github.com/plandes/deepnlp/blob/master/example/ner/src/ner/config.py
[corpus.py]: https://github.com/plandes/deepnlp/blob/master/example/ner/src/ner/corpus.py
[batch.py]: https://github.com/plandes/deepnlp/blob/master/example/ner/src/ner/batch.py
[facade.py]: https://github.com/plandes/deepnlp/blob/master/example/ner/src/ner/facade.py
[feature resource library]: https://github.com/plandes/deepnlp/blob/master/resources/feature.conf

[batch.conf]: https://github.com/plandes/deepnlp/blob/master/example/ner/resources/batch.conf
[model.conf]: https://github.com/plandes/deepnlp/blob/master/example/ner/resources/model.conf
[vectorizer.conf]: https://github.com/plandes/deepnlp/blob/master/example/ner/resources/vectorizer.conf
[main.conf]: https://github.com/plandes/deepnlp/blob/master/example/ner/resources/main.conf
[lang.conf]: https://github.com/plandes/deepnlp/blob/master/example/ner/resources/lang.conf
[corpus.conf]: https://github.com/plandes/deepnlp/blob/master/example/ner/resources/corpus.conf

[FeatureDocumentParser]: ../api/zensols.deepnlp.html#zensols.deepnlp.parse.FeatureDocumentParser
[DataPoint]: https://plandes.github.io/deeplearn/api/zensols.deeplearn.batch.html?highlight=datapoint#zensols.deeplearn.batch.domain.DataPoint
[ConditionalYamlConfig]: https://plandes.github.io/util/api/zensols.config.html#zensols.config.condyaml.ConditionalYamlConfig
[ExtendedInterpolationEnvConfig]: https://plandes.github.io/util/api/zensols.config.html#zensols.config.iniconfig.ExtendedInterpolationEnvConfig
[IniConfig initializer]: https://plandes.github.io/util/api/zensols.config.html#zensols.config.iniconfig.IniConfig.__init__
[CRF]: https://plandes.github.io/deeplearn/api/zensols.deeplearn.layer.html#zensols.deeplearn.layer.crf.CRF
[ScoredBatchIterator]: https://plandes.github.io/deeplearn/api/zensols.deeplearn.model.html#zensols.deeplearn.model.batchiter.ScoredBatchIterator
[EmbeddedRecurrentCRFSettings]: ../api/zensols.deepnlp.layer.html#zensols.deepnlp.layer.embrecurcrf.EmbeddedRecurrentCRFSettings
[EmbeddedRecurrentCRF]: ../api/zensols.deepnlp.layer.html#zensols.deepnlp.layer.embrecurcrf.EmbeddedRecurrentCRF
