# Movie Review Example

This document describes the [movie review task example] to demonstrate the
DeepZenols NLP framework on the sentiment analysis task using the Stanford
sentiment analysis corpus.  It is highly recommended to first read through the
[clickbate example], which contains concepts assumed that are understood for
this example.  For this reason, only new configuration and concepts will be
provided.


## Corpus

The corpus used for this example is fairly small so the models train fast.  It
is the Stanford movie review dataset with Cornell labels:
* [Stanford movie review]
* [Cornell sentiment polarity]

The corpus is automatically downloaded to the `corpus` directory the first time
the model is trained or the batch set accessed.


## Model Configuration

The model specific configuration is located in the `models` directory.  Each
has a file that's given with the `--config` command line option to the
[harness.py] entry point Python file and contains configuration that overrides
on a per model basis.


## Data Set and Corpus

This example utilizes much of the [deeplearn API] framework code.  The main
thrust is to create a [Pandas] data frame, which is then used to provide the
natural language text and labels.  All features are taken only from the text.


## Application Configuration

Like the [clickbate example], the [app.conf] contains command line
configuration used by the entry point [harness.py] script to invoke the example
application.  The model and other application configuration is given in the
[obj.yml] resource library file.  Also like the clickbate example, we will
detail each section that was not already covered since these two projects are
text classification.  Instead, this document will focus on those more advanced
areas such as extending the feature creation aspect of the application.

Generally, the term *section* refers to both a configuration section (like
those described in the [INI format]).  However, for the remainder of this
document, *section group* refers to a grouping of sections that are demarcated
by two hashes (`##`) in the configuration file such as `## Install the corpus`.

The [obj.yml] contains the application specific configuration for reading the
corpus files and parsing it in to features that will later be vectorized.  It
also contains the model.  All of this is described in each sub section in this
document with the respective named group section (root YAML nodes) in the
[obj.yml] application configuration file.


### Install the Corpus

As in the [clickbate example] we download several files: the corpus and the
labels.  The [corpus.py] file provides the `DatasetFactory` class that merges
the Stanford movie review corpus with the Cornell labels and populated with the
install resources to locate the local corpus files.  This class is configured
as the `dataset_factory` in the [obj.yml] application configuration file.


### Natural language parsing

Like in the [clickbate example], we configure a [FeatureDocumentParser].
However, this time, we'll use the one we provide in the `MovieReview` project
described more in the next section.


### Feature Creation

Most of the feature creation code comes with the package and the respective
configuration with the [feature resource library], which is why the project has
only three Python source code files, two resource library files, and three
model configuration files.

The `MovieReviewRowStash` (configured by overriding the [feature resource
library] `dataframe_stash` section in [obj.yml]) in the [domain.py] source file
is used by framework to get a [Pandas] data frame and inherits
[ResourceFeatureDataframeStash] to first download the corpus.  Afterward, it
uses the `DatasetFactory` to create a dataframe containing the label and the
natural language text used to test and train the model.

The [domain.py] file also defines a `MovieReview` container class that has the
polarity positive or negative feed back for each review.  All we need to do is
to extend [FeatureDocument] and add the label to get the complete domain
feature document for our application.  This class is used by
`MovieReviewFeatureStash`, that handles the work of parsing the text in to
features and extends ([DocumentFeatureStash]) to use parser to create instances
of `MovieReview` feature documents with the polarity (positive or negative
sentiment) labels.  Note that we could have used the default
[LabeledFeatureDocument] from the [classify resource library], but this example
shows how to create our own specific labels by overriding a method invoke the
parsing with the text and set the label from the [Pandas] data frame:
```python
def _parse_document(self, id: int, row: pd.Series) -> Review:
	# text to parse with SpaCy
	text = row['sentence']
	# the class label
	polarity = row['polarity']
	return self.vec_manager.parse(text, polarity)
```

See the [deeplearn API] more documentation on [data frame stashes].

So far we've defined that base feature class `MovieReview`, the stash that keeps
track of them, `MovieReviewFeatureStash`.  Now we need to extend the [deeplearn API
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
label from the `MovieReview` class that's retrieved from the `MovieReviewFeatureStash`
that's set as the `doc` attribute on the [FeatureDocumentDataPoint] by
`MovieReviewFeatureStash` (via the super class [DocumentFeatureStash]).


### Vectorization

We still have configure the polarity labels (`n`, and `p`) for the label
vectorizer so it knows what to use for nominal values during the batch
processing.  We also declare the vectorizer managers to include language
features, the label vectorizer (`classify_label_vectorizer_manager` provided in
the [feature resource library]) and the transformer expanders that allow for
language features to be concatenated to BERT embeddings.


### Batch

We configure our custom `MovieReviewDataPoint` class in the `batch_stash`
section.  The rest of this section is self explanatory if the [clickbate
example] has been reviewed


## Running

The movie review data set example can be run from the command line or as a
Jupyter notebook.


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
# train and test the model but switch to model profile with optimized
./harness.py traintest -p
# all model, its (hyper)parameters, metadata and results are stored in subdirectory of files
./harness.py result
# predict and write the test set to a CSV file
./harness.py -c models/glove50.conf preds
# predict ad-hoc a new sentence
./harness.py predtext 'Great movie'
```

Note the `run.sh` script in the same directory provides a simpler API and more
prediction examples as a way of calling the [harness.py] entry point.  It also
serves as an example of how one might simplify a command line for a specific
model.


### Jupyter Notebook

To run the [Jupyter movie notebook]:
1. Pip install: `pip install notebook`
1. Go to the notebook directory: `cd examples/movie/notebook`
1. Start the notebook: `jupyter notebook`
1. Start the execution in the notebook with `Cell > Run All`.


<!-- links -->

[Pandas]: https://pandas.pydata.org
[Stanford movie review]: https://nlp.stanford.edu/sentiment/
[Cornell sentiment polarity]: https://www.cs.cornell.edu/people/pabo/movie-review-data/

[deeplearn API]: https://plandes.github.io/deeplearn/index.html
[data frame stashes]: https://plandes.github.io/deeplearn/doc/preprocess.html#data-as-a-pandas-data-frame
[deeplearn API batch]: https://plandes.github.io/deeplearn/doc/preprocess.html#batches
[classify resource library]: https://github.com/plandes/deepnlp/blob/master/resources/classify.conf
[INI format]: https://plandes.github.io/util/doc/config.html#ini-format

[movie review task example]: https://github.com/plandes/deepnlp/blob/master/example/movie
[obj.yml]: https://github.com/plandes/deepnlp/blob/master/example/movie/resources/obj.yml
[corpus.py]: https://github.com/plandes/deepnlp/blob/master/example/movie/mr/corpus.py
[domain.py]: https://github.com/plandes/deepnlp/blob/master/example/movie/mr/domain.py
[Jupyter movie notebook]: https://github.com/plandes/deepnlp/blob/master/example/movie/notebook/movie.ipynb

[ResourceFeatureDataframeStash]: https://plandes.github.io/deeplearn/api/zensols.dataframe.html#zensols.dataframe.stash.ResourceFeatureDataframeStash
[DocumentFeatureStash]: ../api/zensols.deepnlp.feature.html#zensols.deepnlp.feature.stash.DocumentFeatureStash
[FeatureDocument]: ../api/zensols.deepnlp.html#zensols.deepnlp.domain.FeatureDocument
[FeatureDocumentParser]: ../api/zensols.deepnlp.html#zensols.deepnlp.parse.FeatureDocumentParser
[FeatureDocumentDataPoint]: ../api/zensols.deepnlp.batch.html#zensols.deepnlp.batch.domain.FeatureDocumentDataPoint
[LabeledFeatureDocument]: ../api/zensols.deepnlp.classify.html#zensols.deepnlp.classify.domain.LabeledFeatureDocument

[clickbate example]: clickbate.md
[harness.py]: https://github.com/plandes/deepnlp/blob/master/example/movie/harness.py
[app.conf]: https://github.com/plandes/deepnlp/blob/master/example/movie/resources/app.conf
[obj.yml]: https://github.com/plandes/deepnlp/blob/master/example/movie/resources/app.conf
[feature resource library]: https://github.com/plandes/deepnlp/blob/master/resources/feature.conf
