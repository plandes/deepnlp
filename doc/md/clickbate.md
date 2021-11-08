# Clickbate Example

This example provides a good starting point since it only contains code to
parse the [clickbate corpus].  There is also a [spaCy] pipe component to remove
sentence chunking since the corpus are newline delimited headlines that should
remain as a single sentence.

The example shows how to create, train, validate and test a model that
determines if a headline is clickbate or not (see the corpus for details).  It
comes with two models: one that uses word vectors (GloVE 50 dimension) with
additional language features, and a non-contextual BERT word embedding example
with no extra features.


## Command Line Interface

The example is written as an Zensols CLI application.  The entry point program
is [run.py].  However, the application is configured in [app.conf], which is
where most of what makes the application is set up.  In this configuration file
in the `default` section, the application is configured with file system paths
used in the project all based from a root path supplied by the entry point
[run.py] application, or a different relative directory for its [Jupyter
notebook example].

Note this project is different from the NER and movie review sentiment analysis
examples in that it uses [resource libraries], which is why the configuration
is much smaller and more manageable.  The following imports the
`zensols.deeplearn` and `zensols.deepnlp` packages' [resource libraries]:
```ini
# import the `imp_conf` section, but make the `default` section available as
# configuration during processing of other files' configuration properties
[import]
references = default
sections = imp_conf

# import overrides, resource libraries, and configuration to create features
# from parse natural language text
[imp_conf]
type = importini
config_files = list:
    ${default:resources_dir}/default.conf,
    resource(zensols.deeplearn): resources/obj.conf,
    resource(zensols.deepnlp): resources/obj.conf,
    ${default:resources_dir}/feature.conf
```

The following configuration adds default applications, which is invoked from
the command line by the `CliHarness` defined in the [run.py] entry point:
```ini
# application to provide information about the model
[fac_info_app]
class_name = zensols.deeplearn.cli.FacadeInfoApplication

# application to provide training, testing and other functionality to excercise
# the model
[fac_model_nlp_app]
class_name = zensols.deeplearn.cli.FacadeModelApplication

# application to provide NLP specific funtionality such as text prediction
# (classification for our example)
[fac_model_app]
class_name = zensols.deepnlp.cli.NLPFacadeModelApplication
```

These applications are simply Python [dataclasses] who's class and method
docstrings as help for the command line interface.  Each method is mapped to an
[action with positional and optional parameters].


## Configuration

The [feature.conf] has very application specific configuration for reading the
corpus files and parsing it in to features that will later be vectorized.
First the configuration defines where to download the corpus and where to
uncompress the files to make it available to the program.

The rest defines a series of [Stash] instances that cache work it goes (see
*batch encoding* in [the paper]).  For this example application, the process
follows as such:
1. Download the corpus and uncompress if it isn't already.
2. Parse the corpus from the sentence text files (`dataframe_stash`).
3. Randomly split the dataset in to train, validation and test set, then store
   the data from the dataframe as a picked file on the file system
   (`dataframe_stash`).
4. Parse the English sentences from the `dataframe_stash` using spaCy across as
   many processes as the CPU has cores (`feature_factory_stash`) and persisting
   them to the file system in directories by feature (`feature_dir_stash`).
5. Read only certain files (based on feature selection for the particular
   model) from the file system to reconstruct batches (see *batch decoding* in
   [the paper]).
6. Train, validate and test the model using the same ordering and splits
   sampled by the `dataframe_stash` from step 1.

The [default.conf] file contains configuration that overrides the default
configuration given in the [resource libraries] (with some exception).  For
example, the `doc_parser` section tells the parser to create instances of a
different class that what was defined in its resource library
(`FeatureDocument`).  The class we provide for this example contains an
attribute to carry a label for our text classification task.

Because this is a text classification model, we declare the classes and
override the configuration necessary to vectorize them and add them to the
output decoder network.  We declare the `ClassifyModelFacade` to be used since
it has the CLI predict action allowing ad-hoc text to be classified from the
command line.


## Model Definition

The model specific configuration is located in the `models` directory.  Each
has a file that's given with the `--config` flag to the [run.py] entry point
Python file and contains configuration that overrides on a per model basis.

The [glove.conf] defines an LSTM model that uses a fully connected decoder
network to provide the output label.  The [transformer.conf] defines a
HuggingFace BERT transformer model that is fine tuned during the training.

In each of these files, the `clickbate_default` section is used for settings
just in this configuration file.  The `batch_stash` section tells what features
we want to use for this mode, which is how the *batch decoding* process knows
which vectorized features to read from the file system.  The
`classify_net_settings` tells it which embedding (layer) to and any additional
network parameters such as the dropout.

For the [glove.conf] model, we use the RNN settings defined in the [resource
library] defined for the [zensols.deeplearn] package that has the network code
and configuration.  For the [transformer.conf] the `recurrent_settings`
property is left out, which defaults to `None` indicating not to use a
recurrent network and rely on just the embedding layer has only the
transformer networks.

Finally the model settings provide configuration to the model, which is where
to store the output of the model (i.e. weights) in the `path` property.  Number
of epoch to train and other model specific parameters such as the learning rate
and scheduler parameters can be given here.


## Code

The code is in the [cb] directory, and given its name to fit nicely as a Python
module.  This directory only has the spaCy pipeline component for removing
sentence boundaries and another file to read the corpus.  These files have
inline comments that explain the simple tasks they do.


## Notebook

There is a [Jupyter notebook] that executes the entire download, train,
validate, test and report process for both models.  In [notebook directory] is
the notebook, a Python source `harness.py` file that "glues" the CLI to the
notebook API, and the output of a previous run of the notebook.

The `harness.py` file contains a convenience class used by the notebook to add
directories to the Python path, which is useful for debugging when the package
isn't installed.  It also has life cycle methods to manage instances of
[ModelFacade] and configure the Jupyter notebook for things such as logging and
page width.


<!-- links -->
[clickbate corpus]: https://github.com/bhargaviparanjape/clickbait/tree/master/dataset

[resource libraries]: https://plandes.github.io/util/doc/config.html#resource-libraries
[resource library]: https://plandes.github.io/util/doc/config.html#resource-libraries
[cb]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/cb
[run.py]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/run.py
[app.conf]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/resources/app.conf
[dataclasses]: https://docs.python.org/3/library/dataclasses.html
[action with positional and optional parameters]: https://plandes.github.io/util/doc/command-line.html#application-class-and-actions
[default.conf]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/resources/default.conf
[feature.conf]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/resources/feature.conf
[glove.conf]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/models/glove.conf
[transformer.conf]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/models/transformer.conf
[Stash]: https://plandes.github.io/util/api/zensols.persist.html#zensols.persist.domain.Stash
[spaCy]: https://spacy.io
[the paper]: https://arxiv.org/pdf/2109.03383.pdf
[zensols.deeplearn]: https://github.com/plandes/deeplearn
[Jupyter notebook example]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/notebook/clickbate.ipynb
[Jupyter notebook]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/notebook/clickbate.ipynb
[notebook directory]: https://github.com/plandes/deepnlp/tree/master/example/clickbate/notebook
[ModelFacade]: https://plandes.github.io/deeplearn/api/zensols.deeplearn.model.html#zensols.deeplearn.model.facade.ModelFacade
