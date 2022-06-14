# Resource Library

DeepZenols NLP framework has a comprehensive [resource library] that configures
popular models that enable little to no code written for many standard language
models.  This document provides a highlight of the available configuration of
the API and [deepnlp resource library] available with this package.


## Embedding

The models configured by the [deepnlp resource library] files include
non-contextual word embeddings (i.e. GloVE), a frozen transformer (i.e. [BERT])
transformer model and a fine-tune trainable transformer model.

The Zensols Deep NLP library supports word embeddings for [GloVE], [word2Vec],
[fastText] and [BERT].  The `embedding` section of the [GloVE resource library]
specifies which word vector models and layers that use them:
```ini
[glove_50_embedding]
class_name = zensols.deepnlp.embed.GloveWordEmbedModel
path = path: ${default:corpus_dir}/glove
desc = 6B
dimension = 50
lowercase = True
```
which defines the 6 billion token (400K vocab) 50 dimension [GloVE] model with a
[GloveWordEmbedModel] instance.  The `lowercase` property telling the framework
to down case all queries to the model since the word vectors were trained on a
down cased corpus.

The feature vectorizer [WordVectorSentenceFeatureVectorizer] that uses
the above embedding is defined.  This converts the word vector indexes
(depending on the configuration) to a tensor of the word embedding representing
the corresponding sentence:
```ini
[glove_50_feature_vectorizer]
class_name = zensols.deepnlp.vectorize.WordVectorSentenceFeatureVectorizer
feature_id = wvglove50
embed_model = instance: glove_50_embedding
```

The last configuration needed is a [WordVectorEmbeddingLayer], which extends
a `torch.nn.Module`, and used by the [PyTorch] framework to utilize the word
embedding:
```ini
[glove_50_embedding_layer]
class_name = zensols.deepnlp.vectorize.WordVectorEmbeddingLayer
embed_model = instance: glove_50_embedding
feature_vectorizer = instance: language_feature_manager
```
This module uses the glove embedding model to forward using a
`torch.nn.Embedding` as input at the beginning of the forward [PyTorch] process.
The reference to `language_feature_manager` is covered later.

The embedding resource libraries have a similar definition for the [GloVE] 300
dimension, the [word2vec resource library] for the Google's pre-trained 300
dimension, the [fasttext resource library] for Facebook's pre-trained News and
Crawl pre-trained embeddings, and the [transformer resource library] contains
[BERT] embeddings.  When `decode_embedding` is set to true, the embedding are
created during decode time, rather than at the time the batch is processed.
The `transformer_trainable_resource:model_id` is the [HuggingFace] model
identifier to use, such as `bert-base-cased`, `bert-large-cased`,
`distilbert-base-cased`, `roberta-base`.


### Vectorizer Configuration

Linguistic features are vectorized at one of the following levels:
* **token**: token level with a shape congruent with the number of tokens,
  typically concatenated with the ebedding layer
* **document**: document level, typically added to a join layer
* **embedding**: embedding layer, typically used as the input layer

Each [FeatureDocumentVectorizer], which extends the [deeplearn API]
[EncodableFeatureVectorizer] class defines a `FEATURE_TYPE` of type
[TextFeatureType] that indicates this level.  We'll see examples of
these later in the configuration.  See the [deeplearn API] for more information
on the base class [deeplearn vectorizers].

The next configuration defines an [EnumContainerFeatureVectorizer] in the
[vectorizer resource library], which vectorizes [spaCy] features in to one hot
encoded vectors at the *token* level.  In this configuration, POS tags, NER
tags and dependency head tree is vectorized.  See [SpacyFeatureVectorizer] for
more information.
```ini
[enum_feature_vectorizer]
class_name = zensols.deepnlp.vectorize.EnumContainerFeatureVectorizer
feature_id = enum
decoded_feature_ids = set: ent, tag, dep
```

Similarly, the [CountEnumContainerFeatureVectorizer] encodes counts of each
feature in the text at the *document* level.
```ini
[count_feature_vectorizer]
class_name = zensols.deepnlp.vectorize.CountEnumContainerFeatureVectorizer
feature_id = count
decoded_feature_ids = eval: set('ent tag dep'.split())
```

The `language_feature_manager` configuration is used to create a
[FeatureDocumentVectorizerManager], which is a language specific vectorizer
manager that uses the [FeatureDocumentParser] we defined earlier with the
`doc_parser` entry.  This class extends from [FeatureVectorizerManager] as an
NLP specific manager that creates and encodes the word embeddings and the other
linguistic feature vectorizers configured.  The `token_length` parameter are
the lengths of sentences or documents in numbers of tokens.
```ini
[language_feature_manager]
class_name = zensols.deepnlp.vectorize.FeatureDocumentVectorizerManager
torch_config = instance: torch_config
configured_vectorizers = eval: [
  'word2vec_300_feature_vectorizer',
  'glove_50_feature_vectorizer',
  'glove_300_feature_vectorizer',
  'transformer_feature_vectorizer',
  'enum_feature_vectorizer',
  'count_feature_vectorizer',
  'language_stats_feature_vectorizer',
  'depth_token_feature_vectorizer']
doc_parser = instance: doc_parser
token_length = ${language_defaults:token_length}
token_feature_ids = ${doc_parser:token_feature_ids}
```


## Text Classification

The [text classification resource library] provides configuration for components and
models used to classify tokens and text.

See the [Clickbate example] of how this resource library is used.


### Vectorization (Text)

This configuration set defines the vectorizer for the label itself, which uses
option `categories` as the labels and provided in the [application context]:
```ini
[classify_label_vectorizer]
class_name = zensols.deeplearn.vectorize.NominalEncodedEncodableFeatureVectorizer
#categories = y, n
feature_id = lblabel
```

We define a manager and manager set separate from the linguistic configuration
since the package space is different:
```ini
# the vectorizer for labels is not language specific and lives in the
# zensols.deeplearn.vectorize package, so it needs it's own instance
[classify_label_vectorizer_manager]
class_name = zensols.deeplearn.vectorize.FeatureVectorizerManager
torch_config = instance: torch_config
configured_vectorizers = list: classify_label_vectorizer

[vectorizer_manager_set]
names = list: language_vectorizer_manager, classify_label_vectorizer_manager
```


### Batch Stash

The batch stash configuration should look familiar if you have read through the
[deeplearn API batch stash] documentation.  The configuration below is for a
[BatchDirectoryCompositeStash], which splits data in separate files across
features for each batch.

In this configuration, we split the label, embeddings, and linguistic features
in their own groups so that we can experiment using different embeddings for
each test.  Using [BERT] will take the longest since each sentence will be
computed during decoding.

However, [GloVE] 50D embeddings vectorize much quicker as only the indexes
are stored and quickly retrieved in the [PyTorch] API on demand.  Our caching
strategy also changes as we can (with most graphics cards) fit the entire
[GloVE] 50D embedding in GPU memory.  Our composition stash configuration
follows:
```ini
[batch_dir_stash]
groups = eval: (
       set('label'.split()),
       set('glove_50_embedding'.split()),
...
       set('transformer_enum_expander transformer_dep_expander'.split()))
```

The batch stash is configured next.  This configuration uses dynamic batch
mappings, which map feature attribute names used in the code with the feature
IDs used in vectorizers:
```ini
[batch_stash]
data_point_type = eval({'import': ['zensols.deepnlp.classify']}): zensols.deepnlp.classify.LabeledFeatureDocumentDataPoint
batch_feature_mappings = dataclass(zensols.deeplearn.batch.ConfigBatchFeatureMapping): classify_batch_mappings
```
[LabeledFeatureDocumentDataPoint] is a subclass of [DataPoint] class that
contains a [FeatureDocument], and the `classify_batch_mappings` is a reference
to the batch binding in [classify-batch.yml], which is defined as:
```yaml
classify_batch_mappings:
  batch_feature_mapping_adds:
    - 'dataclass(zensols.deeplearn.batch.BatchFeatureMapping): classify_label_batch_mappings'
    - 'dataclass(zensols.deeplearn.batch.BatchFeatureMapping): lang_batch_mappings'
```

The root defines a section, the second level adds classification and language
specific mappings.  The classify batch mappings are:
```yaml
classify_label_batch_mappings:
  label_attribute_name: label
  manager_mappings:
    - vectorizer_manager_name: classify_label_vectorizer_manager
      fields:
        - attr: label
          feature_id: lblabel
          is_agg: true
```

This says to use the singleton `label` mapping under `fields` for the label and
used by the framework to calculate performance metrics.


### Facade

The facade is configured as a [ClassifyModelFacade]:
```ini
[facade]
class_name = zensols.deepnlp.classify.ClassifyModelFacade
```

This class extends [LanguageModelFacade], which supports natural
language model feature updating and sets up logging.  This class is used both
from the command line and the Jupyter notebook via the CLI facade applications.

This facade class adds classification specific functionality, including
feature updating from a Jupyter notebook or Python REPL.
```python
@dataclass
class ClassifyModelFacade(LanguageModelFacade):
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
	settings: NetworkSettings = self.executor.net_settings
	if hasattr(settings, 'dropout'):
		# set to trigger writeback through to sub settings (linear, recur)
		self.dropout = self.executor.net_settings.dropout
```

We can also override the [get_predictions method] to include the review text
and it's length when creating the data frame and respective CSV export:
```python
def get_predictions(self, *args, **kwargs) -> pd.DataFrame:
	return super().get_predictions(
		('text', 'len'),
		lambda dp: (dp.doc.text, len(dp.doc.text)),
		*args, **kwargs)
```


### Model (Text)

The model section configures the [ClassifyNetworkSettings], which is either a
BiLSTM with an optional CRF output layer or a transformer (see the [movie
review sentiment example] for how this can be configured in both settings.

```ini
[classify_net_settings]
class_name = zensols.deepnlp.classify.ClassifyNetworkSettings
#embedding_layer = instance: ${deepnlp_default:embedding}_layer
recurrent_settings = None
linear_settings = instance: linear_settings
batch_stash = instance: batch_stash
dropout = None
```
The `batch_stash` instance is configured on this model so it has access to the
dynamic batch metadata for the embedding layer.  The commented out
`embedding_layer` has to be overridden and set as the instance of the embedding
layer instance use that create the input embeddings from the input text.  The
`linear_settings` is the network between the recurrent network and the output
CRF (if there is one configured).


### Prediction (Text)

The prediction mapper uses the model to classify text from the command line.
For text classification, the [ClassificationPredictionMapper] is used and takes
text given from the command line and predicts a label:
```ini
[classify_feature_prediction_mapper]
class_name = zensols.deepnlp.classify.ClassificationPredictionMapper
vec_manager = instance: language_vectorizer_manager
label_feature_id = classify_label_vectorizer_manager.lblabel
```

This component needs the vectorizer manager that creates the vectorized label
and the nominal vectorizer to reverse map using a scikit-learn [LabelEncoder]
back to the human readable label.


## Token Classification

Token classification refers to labeling tokens instead of a string of text as
with [text classification](#text-classification).  However, there is some cross
over functionality between these two tasks, so the [token classification
resource library] resource library uses some of the same components (not
configuration) defined in the [text classification resource library].  For
example, we reuse the [ClassifyModelFacade] by overriding the class in the
`facade` section.

*Note*: despite this overlap, either import only the [text classification
resource library] for text classification projects and only [token classification
resource library] for token classification projects, but not both.

Only the notable differences compared to the [text
classification](#text-classification) section are documented.

See the [NER example] of how this resource library is used.


### Vectorization (Token)

This section has the token label vectorizers and mask vectorizers.  The mask is
needed for the [CRF] (when configured) to mask out blank tokens for sentences
shorter than a max length.  Usually, zeroed tensors are used for token slots
not used, for example in the word embedding layer for deep learning networks.
This is because the zero vectors are learned for sentences are shorter.
However, the CRF layer needs to block these as valid state transitions during
training and testing.
```yaml
tok_label_1_vectorizer:
  class_name: zensols.deeplearn.vectorize.NominalEncodedEncodableFeatureVectorizer
  feature_id: tclabel1

tok_label_vectorizer:
  class_name: zensols.deeplearn.vectorize.AggregateEncodableFeatureVectorizer
  feature_id: tclabel
  size: -1
  delegate_feature_id: tclabel1

tok_mask_vectorizer:
  class_name: zensols.deeplearn.vectorize.MaskFeatureVectorizer
  feature_id: tmask
  size: -1

tok_label_batch_mappings:
  manager_mappings:
    - vectorizer_manager_name: tok_label_vectorizer_manager
      fields:
        - attr: tok_labels
          feature_id: tclabel
          is_agg: true
          is_label: True
        - attr: tok_mask
          feature_id: tmask
          is_agg: true
          attr_access: tok_labels

tok_label_vectorizer_manager:
  class_name: zensols.deeplearn.vectorize.FeatureVectorizerManager
  torch_config: 'instance: torch_config'
  configured_vectorizers:
    - tok_label_1_vectorizer
    - tok_label_vectorizer
    - tok_mask_vectorizer

# add new feature vectorizer managers
vectorizer_manager_set:
  names:
    - language_vectorizer_manager
    - tok_label_vectorizer_manager
```


### Model (Token)

The [SequenceBatchIterator] configured in the `model_settings` indicates to use
a different scoring method.  This class is used in the framework to calculate a
different loss and produce the output, which must be treated differently than
neural float tensor output.  This is because the Viterbi algorithm is used to
determine the lowest cost path through the elements.  The sum of this path is
used as the cost instead of a differential optimization function.

```yaml
model_settings:
  batch_iteration_class_name: zensols.deeplearn.model.SequenceBatchIterator
  reduce_outcomes: none
  prediction_mapper_name: feature_prediction_mapper

recurrent_crf_net_settings:
  mask_attribute: tok_mask
```

Because we use a [CRF] as the output layer for [EmbeddedRecurrentCRF], our
output are the NER labels.  Therefore, must also set `reduce_outcomes = none`
to pass the [CRF] output through unaltered.

In the `recurrent_crf_net_settings` section, we override the `mask_attribute`,
which tells recurrent [CRF] to use the `tok_mask` attribute when masking the
label output.


### Prediction (Token)

The section is the same, but we instead use the sequence base version
([SequencePredictionMapper]) where the token stream is used as that sequence.

```yaml
feature_prediction_mapper:
  class_name: zensols.deepnlp.classify.SequencePredictionMapper
  vec_manager: 'instance: language_vectorizer_manager'
  label_feature_id: tok_label_vectorizer_manager.tclabel1
```


<!-- links -->
[spaCy]: https://spacy.io
[PyTorch]: https://pytorch.org
[HuggingFace]: https://github.com/huggingface/transformers

[GloVE]: https://nlp.stanford.edu/projects/glove/
[BERT]: https://huggingface.co/transformers/model_doc/bert.html
[word2Vec]: https://code.google.com/archive/p/word2vec/
[fastText]: https://fasttext.cc

[deeplearn API]: https://plandes.github.io/deeplearn/index.html
[deeplearn API batch stash]: https://plandes.github.io/deeplearn/doc/preprocess.html#batch-stash
[deeplearn vectorizers]: https://plandes.github.io/deeplearn/doc/preprocess.html#vectorizers

[deepnlp resource library]: https://github.com/plandes/deepnlp/tree/master/resources
[resource library]: https://plandes.github.io/util/doc/config.html#resource-libraries
[application context]: https://plandes.github.io/util/doc/config.html#application-context
[NER example]: ner-example.html
[Clickbate example]: clickbate-example.html

[GloVE resource library]: https://github.com/plandes/deepnlp/blob/master/resources/glove.conf
[fasttext resource library]: https://github.com/plandes/deepnlp/blob/master/resources/fasttext.conf
[word2vec resource library]: https://github.com/plandes/deepnlp/blob/master/resources/word2vec.conf
[transformer resource library]: https://github.com/plandes/deepnlp/blob/master/resources/transformer.conf
[vectorizer resource library]: https://github.com/plandes/deepnlp/blob/master/resources/vectorizer.conf
[text classification resource library]: https://github.com/plandes/deepnlp/blob/master/resources/classify.conf
[classify-batch.yml]: https://github.com/plandes/deepnlp/blob/master/resources/classify-batch.yml
[token classification resource library]: https://github.com/plandes/deepnlp/blob/master/resources/token-classify.conf
[movie review sentiment example]: https://plandes.github.io/deepnlp/doc/movie-example.html
[get_predictions method]: ../api/zensols.deepnlp.classify.html#zensols.deepnlp.classify.facade.ClassifyModelFacade.get_predictions

[ClassifyModelFacade]: ../api/zensols.deepnlp.classify.html#zensols.deepnlp.classify.facade.ClassifyModelFacade
[ClassifyNetworkSettings]: ../api/zensols.deepnlp.classify.html#zensols.deepnlp.classify.model.ClassifyNetworkSettings
[CountEnumContainerFeatureVectorizer]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.vectorizers.CountEnumContainerFeatureVectorizer
[EncodableFeatureVectorizer]: https://plandes.github.io/deeplearn/api/zensols.deeplearn.vectorize.html#zensols.deeplearn.vectorize.manager.EncodableFeatureVectorizer
[EnumContainerFeatureVectorizer]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.vectorizers.EnumContainerFeatureVectorizer
[FeatureDocumentParser]: ../api/zensols.deepnlp.html#zensols.deepnlp.parse.FeatureDocumentParser
[FeatureDocumentVectorizerManager]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.manager.FeatureDocumentVectorizerManager
[FeatureDocumentVectorizer]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.manager.FeatureDocumentVectorizer
[FeatureDocumentVectorizer]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.manager.FeatureDocumentVectorizer
[FeatureDocument]: ../api/zensols.deepnlp.html#zensols.deepnlp.domain.FeatureDocument
[FeatureVectorizerManager]: https://plandes.github.io/deeplearn/api/zensols.deeplearn.vectorize.html#zensols.deeplearn.vectorize.manager.FeatureVectorizerManager
[GloveWordEmbedModel]: ../api/zensols.deepnlp.embed.html#zensols.deepnlp.embed.glove.GloveWordEmbedModel
[LabeledFeatureDocumentDataPoint]: ../api/zensols.deepnlp.classify.html#zensols.deepnlp.classify.domain.LabeledFeatureDocumentDataPoint
[LanguageModelFacade]: ../api/zensols.deepnlp.model.html#zensols.deepnlp.model.facade.LanguageModelFacade
[SpacyFeatureVectorizer]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.spacy.SpacyFeatureVectorizer
[TextFeatureType]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.manager.TextFeatureType
[WordVectorEmbeddingLayer]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.layer.WordVectorEmbeddingLayer
[WordVectorSentenceFeatureVectorizer]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.layer.WordVectorSentenceFeatureVectorizer

[BatchDirectoryCompositeStash]: https://plandes.github.io/deeplearn/api/zensols.deeplearn.batch.html#zensols.deeplearn.batch.stash.BatchDirectoryCompositeStash
[DataPoint]: https://plandes.github.io/deeplearn/api/zensols.deeplearn.batch.html#zensols.deeplearn.batch.domain.DataPoint
[CRF]: https://plandes.github.io/deeplearn/api/zensols.deeplearn.layer.html#zensols.deeplearn.layer.crf.CRF
[SequenceBatchIterator]: https://plandes.github.io/deeplearn/api/zensols.deeplearn.model.html#zensols.deeplearn.model.sequence.SequenceBatchIterator
[EmbeddedRecurrentCRF]: ../api/zensols.deepnlp.layer.html#zensols.deepnlp.layer.embrecurcrf.EmbeddedRecurrentCRF
[LabelEncoder]: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
[ClassificationPredictionMapper]: ../api/zensols.deepnlp.classify.html#zensols.deepnlp.classify.pred.ClassificationPredictionMapper
[SequencePredictionMapper]: ../api/zensols.deepnlp.classify.html#zensols.deepnlp.classify.pred.SequencePredictionMapper
