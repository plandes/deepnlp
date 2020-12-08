# Movie Review Sentiment Example

This document describes the [movie review task example] to demonstrate the
DeepZenols NLP framework on the sentiment analysis task using the [Stanford
movie review] and [Cornell sentiment polarity] data sets.

As before, we'll incrementally go through the [configuration file] section by
section skipping those we have already covered in the [deeplearn API] and [deep
NLP] APIs.  It is assumed you have read the [deeplearn API] documentation and
are reading this in parallel with the [deep NLP] documentation.

Note that there is quite a bit of inline documentation in the [configuration
file] so it is recommended the reader follow it while reading this tutorial.


## Embedding

The Zensols [deep NLP] library supports word embeddings for [Glove],
[Word2Vec], [fastText] and [BERT].  The `embedding` section of the
[configuration file] specifies which word vector models and layers that use
them.  This defines the 6 billion token (400K vocab) 50 dimension [Glove] model
with a [GloveWordEmbedModel] instance:
```ini
# glove embeddding model (not layer)
[glove_50_embedding]
class_name = zensols.deepnlp.embed.GloveWordEmbedModel
path = path: ${default:corpus_dir}/glove
desc = 6B
dimension = 50
lowercase = True
```
with the `lowercase` property telling the framework to down case all queries to
the model since the word vectors were trained on a down cased corpus.

Next, the feature vectorizer [WordVectorSentenceFeatureVectorizer] that uses
the above embedding is defined.  This converts the word vector indexes
(depending on the configuration) to a tensor of the word embedding representing
the corresponding sentence:
```ini
# a vectorizer that turns tokens (TokensContainer) in to indexes given to the
# embedding layer
[glove_50_feature_vectorizer]
class_name = zensols.deepnlp.vectorize.WordVectorSentenceFeatureVectorizer
# the feature id is used to connect instance data with the vectorizer used to
# generate the feature at run time
feature_id = wvglove50
embed_model = instance: glove_50_embedding
# treat the document as a stream of tokens generating a flat set of indexes
as_document = True
```

Finally, the last step is to define a [WordVectorEmbeddingLayer], which extends
a `torch.nn.Module`, and used by the [PyTorch] framework to utilize the word
embedding:
```ini
# a torch.nn.Module implementation that uses the an embedding model
[glove_50_embedding_layer]
class_name = zensols.deepnlp.vectorize.WordVectorEmbeddingLayer
embed_model = instance: glove_50_embedding
feature_vectorizer = instance: language_feature_manager
```
This module uses the glove embedding model to forward using a
`torch.nn.Embedding` as input at the beginning of the forward [PyTorch] process.
The reference to `language_feature_manager` is covered later.

The next three entries have similar definitions for [Glove] 300 dimension,
Google's pre-trained 300 dimension word2vec, and [BERT] embeddings.  The
`as_document`, when true, parameter tells the framework to treat the embedding
as a document using all tokens as one long stream as apposed to "stacking" each
as a sentence.  When `decode_embedding` is set to true, the embedding is
created during decode time, rather than at the time the batch is processed.
The `model_name` for the [BERT] embeddings tells it which model to use, which
can be `bert`, `distilbert`, `roberta` as defined by the [huggingface]
transformers API.


## Linguistic Configuration

Linguistic features are vectorized at one of the following levels:
* **token**: token level with a shape congruent with the number of tokens,
  typically concatenated with the ebedding layer
* **document**: document level, typically added to a join layer
* **embedding**: embedding layer, typically used as the input layer

Each [TokenContainerFeatureVectorizer], which extends the [deeplearn API]
[EncodableFeatureVectorizer] class defines a `FEATURE_TYPE` of type
[TokenContainerFeatureType] that indicates this level.  We'll see examples of
these later in the configuration.  See the [deeplearn API] for more information
on the base class [deeplearn vectorizers].


### Natural Language Parsing

The `language_defaults` section is a shared configuration.  The
`lower_case_token_mapper` to `langres` sections describe how to parse the
natural language text with [spaCy] and pertain to the [nlparse] API (see that
API for more information).

The `doc_parser` is used by the deep NLP framework code to parse the text in to
[FeatureDocument] instances:
```ini
# creates features from documents by invoking by using SpaCy to parse the text
[doc_parser]
class_name = zensols.deepnlp.FeatureDocumentParser
langres = instance: langres
token_feature_ids = eval: set('norm ent dep tag children i dep_ is_punctuation'.split())
```
Note that the configuration indicates a `LanguageResource` to use, which was
defined above this configuration using the [nlparse] API.  The
`token_feature_ids` tell the parser which token level features to keep.


### Vectorizer Configuration

The next configuration defines an [EnumContainerFeatureVectorizer], which
vectorizes [spaCy] features in to one hot encoded vectors at the *token* level.
In this example, POS tags, NER tags and dependency head tree is vectorized.
See [SpacyFeatureVectorizer] for more information.
```ini
[enum_feature_vectorizer]
class_name = zensols.deepnlp.vectorize.EnumContainerFeatureVectorizer
feature_id = enum
# train time tweakable
decoded_feature_ids = eval: set('ent tag dep'.split())
```

Similarly, the [CountEnumContainerFeatureVectorizer] encodes counts of each
feature in the text at the *document* level.
```ini
[count_feature_vectorizer]
class_name = zensols.deepnlp.vectorize.CountEnumContainerFeatureVectorizer
feature_id = count
decoded_feature_ids = eval: set('ent tag dep'.split())
```

Several other linguistic feature vectorizers are defined until we get to the
`language_feature_manager` entry:
```ini
[language_feature_manager]
class_name = zensols.deepnlp.vectorize.TokenContainerFeatureVectorizerManager
torch_config = instance: torch_config
# word embedding vectorizers can not be class level since each instance is
# configured
configured_vectorizers = eval: [
  'word2vec_300_feature_vectorizer',
  'glove_50_feature_vectorizer',
  'glove_300_feature_vectorizer',
  'bert_feature_vectorizer',
  'enum_feature_vectorizer',
  'count_feature_vectorizer',
  'language_stats_feature_vectorizer',
  'depth_token_feature_vectorizer']
# used for parsing `FeatureDocument` instances
doc_parser = instance: doc_parser
# the number of tokens in the document to use
token_length = ${language_defaults:token_length}
# features to use
token_feature_ids = ${doc_parser:token_feature_ids}
```
This configuration creates a [TokenContainerFeatureVectorizerManager], which is
a language specific vectorizer manager that uses the [FeatureDocumentParser] we
defined earlier with the `doc_parser` entry.  This class extends from
[FeatureVectorizerManager] as an NLP specific manager that creates and encodes
the word embeddings and the other linguistic feature vectorizers configured.
The `token_length` parameter are the lengths of sentences or documents in numbers of
tokens.


## Project Specific

The next configuration set defines the vectorizer for the label itself, which
is a binary either *positive* or *negative* review of the movie:
```ini
# vectorize the labels from text to PyTorch tensors
[label_vectorizer]
class_name = zensols.deeplearn.vectorize.NominalEncodedEncodableFeatureVectorizer
categories = eval: '${class:labels}'.split()
feature_id = rvlabel
```

We define a manager and manager set separate from the linguistic configuration
since the package space is different:
```ini
[label_vectorizer_manager]
class_name = zensols.deeplearn.vectorize.FeatureVectorizerManager
torch_config = instance: torch_config
configured_vectorizers = eval: 'label_vectorizer'.split()

[vectorizer_manager_set]
class_name = zensols.deeplearn.vectorize.FeatureVectorizerManagerSet
names = eval: 'language_feature_manager label_vectorizer_manager'.split()
```


## Data Set and Corpus

This example utilizes much of the [deeplearn API] framework code.  The main
thrust is to create a [Pandas] data frame, which is then used to provide the
natural language text and labels.  All features are taken only from the text.


### Feature Stash

The [dataset.py] file provides a class that merges the Stanford movie review
corpus with the Cornell labels.  The `dataset` property is called by the
framework to get a [Pandas] data frame that is used by `ReviewRowStash`, which
inherits and gets it's functionality from a [DataframeStash].  This dataframe
contains the label and the natural language text used to test and train the
model.  The `dataset_factory` entry in the [configuration file] configures this
class.  However, since this is very implementation specific to the movie review
corpus, we will not cover it in this tutorial.

The [domain.py] file defines a `Review` domain container class that has the
polarity positive or negative feed back for each review.  All we need to do is
to extend [FeatureDocument] and add the label to get the complete domain
feature document for our application.

Also defined in this file is the `ReviewFeatureStash`, which extends a
[DocumentFeatureStash] that handles the work of parsing the text.  In the
initializer, we have to tell the framework which class to create
(i.e. instances of `Review` rather than the default `FeatureDocument`) for each
document parsed:
```python
    def __post_init__(self):
        super().__post_init__()
        self.vec_manager.doc_parser.doc_class = Review
```

Next we override a method invoke the parsing with the text and set the label
from the [Pandas] data frame:
```python
    def _parse_document(self, id: int, row: pd.Series) -> Review:
        # text to parse with SpaCy
        text = row['sentence']
        # the class label
        polarity = row['polarity']
        return self.vec_manager.parse(text, polarity)
```
Note the `parse` call is on the vectorizer manager we configured in the
[configuration file] `language_feature_manager` entry we saw in the [Linguistic
Configuration](#linguistic-configuration) section.


### Data Frame Stash

So far in this section we've only covered the implementation.  Here is the
configuration for the data frame stash:
```ini
[dataframe_stash]
class_name = movie.ReviewRowStash
dataset_factory = instance: dataset_factory
# location of pickled cache data to avoid recreating the dataframe each time
dataframe_path = path: ${default:data_dir}/df.dat
split_col = ${dataset_factory:split_col}
key_path = path: ${default:data_dir}/keys.dat
```

See the [deeplearn API] more documentation on [data frame stashes].


### Data Point

So far we've defined that base feature class `Review`, the stash that keeps
track of them, `ReviewFeatureStash`.  Now we need to extend the [deeplearn API
batch] classes.  We'll start with the data point class:
```python
@dataclass
class ReviewDataPoint(FeatureDocumentDataPoint):
    @property
    def label(self) -> str:
        return self.doc.polarity
```
which extends from the linguistic specific data point class
[FeatureDocumentDataPoint].  There's not much more to this than returning the
label from the `Review` class that's retrieved from the `ReviewFeatureStash`
that's set as the `doc` attribute on the [FeatureDocumentDataPoint] by
`ReviewFeatureStash` (via the super class [DocumentFeatureStash]).


### Feature Vectorizer to Batch Mapping

The last bit of implementation that's needed is the binding between the domain
classes and the vectorizers.  First we define constants of the [configuration
file] entries in the [Batch] class `ReviewBatch` to reduce duplication in our
code:
```python
@dataclass
class ReviewBatch(Batch):
    LANGUAGE_FEATURE_MANAGER_NAME = 'language_feature_manager'
    GLOVE_50_EMBEDDING = 'glove_50_embedding'
    GLOVE_300_EMBEDDING = 'glove_300_embedding'
    WORD2VEC_300_EMBEDDING = 'word2vec_300_embedding'
    BERT_EMBEDDING = 'bert_embedding'
    EMBEDDING_ATTRIBUTES = {GLOVE_50_EMBEDDING, GLOVE_300_EMBEDDING,
                            WORD2VEC_300_EMBEDDING, BERT_EMBEDDING}
    STATS_ATTRIBUTE = 'stats'
    ENUMS_ATTRIBUTE = 'enums'
    COUNTS_ATTRIBUTE = 'counts'
    DEPENDENCIES_ATTRIBUTE = 'dependencies'
    LANGUAGE_ATTRIBUTES = {STATS_ATTRIBUTE, ENUMS_ATTRIBUTE, COUNTS_ATTRIBUTE,
                           DEPENDENCIES_ATTRIBUTE}
```

Next we define the mapping of the label class and vectorizer of the label:
```python
    MAPPINGS = BatchFeatureMapping(
        'label',
        [ManagerFeatureMapping(
            'label_vectorizer_manager',
            (FieldFeatureMapping('label', 'rvlabel', True),)),
```

Now we provide the linguistic feature mapping to vectorizers, again using the
`feature_id` given in the [configuration file]:
```python
         ManagerFeatureMapping(
             LANGUAGE_FEATURE_MANAGER_NAME,
             (FieldFeatureMapping(GLOVE_50_EMBEDDING, 'wvglove50', False, 'doc'),
              FieldFeatureMapping(GLOVE_300_EMBEDDING, 'wvglove300', False, 'doc'),
              FieldFeatureMapping(WORD2VEC_300_EMBEDDING, 'w2v300', False, 'doc'),
              FieldFeatureMapping(BERT_EMBEDDING, 'bert', False, 'doc'),
              FieldFeatureMapping(STATS_ATTRIBUTE, 'stats', False, 'doc', 0),
              FieldFeatureMapping(ENUMS_ATTRIBUTE, 'enum', False, 'doc', 0),
              FieldFeatureMapping(COUNTS_ATTRIBUTE, 'count', False, 'doc', 0),
              FieldFeatureMapping(DEPENDENCIES_ATTRIBUTE, 'dep', False, 'doc', 0)))])
```

Finally, we return the given `MAPPINGS` class level constant in our class as
the single abstract method of the [Batch] super class:
```python
    def _get_batch_feature_mappings(self) -> BatchFeatureMapping:
        return self.MAPPINGS
```


### Batch Stash

The batch stash configuration should look familiar if you have read through the
[deeplearn API batch stash] documentation.  It bears mention the utility of the
[BatchDirectoryCompositeStash] configuration that splits data in separate files
across features for each batch.

In this configuration, we split the label, embeddings, and linguistic features
in their own groups so that we can experiment using different embeddings for
each test.  Using BERT will take the longest since each sentence will be
computed during decoding.

However, a much faster set, will be [Glove] 50D embeddings as only the indexes
are stored and quickly retrieved in the [PyTorch] API on demand.  Our caching
strategy also changes as we can (with most graphics cards) fit the entire
[Glove] 50D embedding in GPU memory.  Our composition stash configuration
follows:
```ini
[batch_dir_stash]
class_name = zensols.deeplearn.batch.BatchDirectoryCompositeStash
# top level directory to store each feature sub directory
path = path: ${default:batch_dir}/data
groups = eval: (
       # there will be N (batch_stash:batch_size) batch labels in one file in a
       # directory of just label files
       set('label'.split()),
       # because we might want to switch between embeddings, separate them
       set('glove_50_embedding'.split()),
       set('glove_300_embedding'.split()),
       set('word2vec_300_embedding'.split()),
       set('bert_embedding'.split()),
       # however, natural language features are optional for this task
       set('enums stats counts dependencies'.split()))
```


## Facade

The [facade.py] file contains the implementation of the facade used for the
movie reviews.  This extends from [LanguageModelFacade], which supports natural
language model feature updating and sets up logging.  This class is used both
from the [command line interface](#command-line) and from the [Jupyter
notebook](#jupyter-notebook).

The feature updating wiring happens with the [LanguageModelFacadeConfig]:
```python
@dataclass
class ReviewModelFacade(LanguageModelFacade):
    LANGUAGE_MODEL_CONFIG = LanguageModelFacadeConfig(
        manager_name=ReviewBatch.LANGUAGE_FEATURE_MANAGER_NAME,
        attribs=ReviewBatch.LANGUAGE_ATTRIBUTES,
        embedding_attribs=ReviewBatch.EMBEDDING_ATTRIBUTES)
```
and used by the framework by overriding:
```python
    def _get_language_model_config(self) -> LanguageModelFacadeConfig:
        return self.LANGUAGE_MODEL_CONFIG
```

Setting the dropout triggers property setters to propagate (linear and
recurrent layers) the setting when set on the facade:
```python
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self.dropout = self.executor.net_settings.dropout
```

We also override the embeddings setter so we get a long stream of tokens for
BERT embeddings:
```python
    def _set_embedding(self, embedding: str):
        needs_change = super()._set_embedding(embedding)
        if needs_change and embedding == 'bert':
            # m/m F1 814, 811
            vec_mng = self.language_vectorizer_manager
            vec_mng.token_length = 100
```

We can also override the `get_predictions` method to include the review text
and it's length when creating the data frame and respective CSV export:
```python
    def get_predictions(self) -> pd.DataFrame:
        return super().get_predictions(
            ('text', 'len'),
            lambda dp: (dp.review.text, len(dp.review.text)))
```


## Running

The movie review data set example can be run from the command line or as a
Jupyter notebook.  First you must download and install the corpus.


### Corpus Install

To install the corpus:
1. Install [GNU make](https://www.gnu.org/software/make/)
1. Change the working directory to the example: `cd examples/movie`
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

To run the [Jupyter movie notebook]:
1. Pip install: `pip install notebook`
1. Go to the notebook directory: `cd examples/movie/notebook`
1. Start the notebook: `jupyter notebook`
1. Start the execution in the notebook with `Cell > Run All`.


<!-- links -->

[PyTorch]: https://pytorch.org
[Pandas]: https://pandas.pydata.org
[huggingface]: https://github.com/huggingface/transformers
[spaCy]: https://spacy.io
[Glove]: https://nlp.stanford.edu/projects/glove/
[Word2Vec]: https://code.google.com/archive/p/word2vec/
[fastText]: https://fasttext.cc
[BERT]: https://huggingface.co/transformers/model_doc/bert.html
[Stanford movie review]: https://ai.stanford.edu/~amaas/data/sentiment/
[Cornell sentiment polarity]: https://www.cs.cornell.edu/people/pabo/movie-review-data/

[nlparse]: https://github.com/plandes/nlparse
[deeplearn API]: https://plandes.github.io/deeplearn/index.html
[data frame stashes]: https://plandes.github.io/deeplearn/doc/preprocess.html#data-as-a-pandas-data-frame
[deeplearn vectorizers]: https://plandes.github.io/deeplearn/doc/preprocess.html#vectorizers
[deeplearn API batch]: https://plandes.github.io/deeplearn/doc/preprocess.html#batches
[deeplearn API batch stash]: https://plandes.github.io/deeplearn/doc/preprocess.html#batch-stash

[deep NLP]: https://plandes.github.io/deepnlp/index.html
[movie review task example]: https://github.com/plandes/deepnlp/blob/master/example/movie
[configuration file]: https://github.com/plandes/deepnlp/blob/master/example/movie/resources/movie.conf
[dataset.py]: https://github.com/plandes/deepnlp/blob/master/example/movie/src/movie/dataset.py
[domain.py]: https://github.com/plandes/deepnlp/blob/master/example/movie/src/movie/domain.py
[facade.py]: https://github.com/plandes/deepnlp/blob/master/example/movie/src/movie/facade.py
[Jupyter movie notebook]: https://github.com/plandes/deepnlp/blob/master/example/movie/notebook/movie.ipynb

[DataframeStash]: https://plandes.github.io/deeplearn/api/zensols.dataframe.html#zensols.dataframe.stash.DataframeStash
[Batch]: https://plandes.github.io/deeplearn/api/zensols.deeplearn.batch.html#zensols.deeplearn.batch.domain.Batch
[EncodableFeatureVectorizer]: https://plandes.github.io/deeplearn/api/zensols.deeplearn.vectorize.html#zensols.deeplearn.vectorize.manager.EncodableFeatureVectorizer
[FeatureVectorizerManager]: https://plandes.github.io/deeplearn/api/zensols.deeplearn.vectorize.html?highlight=featurevectorizermanager#zensols.deeplearn.vectorize.manager.FeatureVectorizerManager
[BatchDirectoryCompositeStash]: https://plandes.github.io/deeplearn/api/zensols.deeplearn.batch.html?highlight=batchdirectorycompositestash#zensols.deeplearn.batch.stash.BatchDirectoryCompositeStash

[WordVectorSentenceFeatureVectorizer]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.layer.WordVectorSentenceFeatureVectorizer
[GloveWordEmbedModel]: ../api/zensols.deepnlp.embed.html#zensols.deepnlp.embed.glove.GloveWordEmbedModel
[WordVectorEmbeddingLayer]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.layer.WordVectorEmbeddingLayer
[FeatureDocument]: ../api/zensols.deepnlp.html#zensols.deepnlp.domain.FeatureDocument
[EnumContainerFeatureVectorizer]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.vectorizers.EnumContainerFeatureVectorizer
[SpacyFeatureVectorizer]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.spacy.SpacyFeatureVectorizer
[TokenContainerFeatureType]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.manager.TokenContainerFeatureType
[TokenContainerFeatureVectorizer]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.manager.TokenContainerFeatureVectorizer
[CountEnumContainerFeatureVectorizer]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.vectorizers.CountEnumContainerFeatureVectorizer
[TokenContainerFeatureVectorizerManager]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.manager.TokenContainerFeatureVectorizerManager
[FeatureDocumentParser]: ../api/zensols.deepnlp.html#zensols.deepnlp.parse.FeatureDocumentParser
[DocumentFeatureStash]: ../api/zensols.deepnlp.feature.html#zensols.deepnlp.feature.stash.DocumentFeatureStash
[FeatureDocumentDataPoint]: ../api/zensols.deepnlp.batch.html#zensols.deepnlp.batch.domain.FeatureDocumentDataPoint
[LanguageModelFacade]: ../api/zensols.deepnlp.model.html#zensols.deepnlp.model.facade.LanguageModelFacade
[LanguageModelFacadeConfig]: ../api/zensols.deepnlp.model.html#zensols.deepnlp.model.facade.LanguageModelFacadeConfig
