# Clickbate Example

This example provides a good starting point since it only contains code to
parse a [corpus].  It also shows how to use your own model your own
data by synthesizing a positive and negative dataset sources in to one, which
is provided in the [only source code] file for the project (excluding the entry
point [harness.py] script).

The example shows how to create, train, validate and test a model that
determines if a headline is clickbate or not (see the corpus for details).  It
comes with two models: one that uses word vectors (GloVE 50 dimension and the
fasttext news pre-trained embeddings) with additional language features, and a
BERT word embedding example.

Note that there is quite a bit of inline documentation in the [app.conf] and
[obj.yml] configuration files. so it is recommended the reader follow it while
reading this tutorial.


## Command Line Interface

All of the examples for this package are written as an Zensols CLI
applications.  The entry point program is [harness.py].  However, the command
line is configured in [app.conf] and the application in [obj.yml], which is
where almost all of the example.  These files configure the file system paths,
tell where to load other [resource libraries], point to corpus resources and
are also used by the [Jupyter notebook example] to allow for more rapid
prototyping and experimentation.

Because the examples (including this one) use [resource libraries], the
configuration is much smaller and more manageable.  First we start with adding
the application defaults allowing `name` to be [overridden] with the
`--override` command line option:
```ini
[cb_default]
lang_features = dependencies, enums,
embedding = ${name}_embedding
```

The `--override` command line option takes a string (or file) containing any
configuration in a comma delimited `<section>.<option>` and given on the
command line to specify which word embeddings to use.  For example: `--override
cb_default.name=glove_50`, would specify the 50 dimension [GloVE resource
library].

Now we add defaults for the deep learning package that set the model name,
appears in results and file system naming.  In this example, we simply set the
model name as the embeddings we'll use:
```ini
[deeplearn_default]
model_name = ${cb_default:embedding}
```

The following configuration adds default applications, which is invoked from
the command line by the [CliHarness] defined in the [harness.py] entry point
and imported from [resource libraries] as [first pass actions]:
```ini
[cli]
apps = list: ${cli_config_default:apps}, ${cli_deeplearn_default:apps}, ${cli_deepnlp_default:apps},
  deepnlp_fac_text_classify_app, cleaner_cli
cleanups = list: ${cli_config_default:cleanups}, ${cli_deeplearn_default:cleanups},
  ${cli_deepnlp_default:cleanups}, deepnlp_fac_text_classify_app, cleaner_cli
cleanup_removes = set: log_cli
```
The application defined in the sections loaded are simply Python [dataclasses]
who's class and method docstrings as help for the command line interface.  Each
method is mapped to an [action with positional and optional parameters].

Note the `log_cli` section is mentioned because it is listed as a clean up in a
resource library.  However, we must keep this section because it is useful to
configure child processes when batches are created to keep a consistent logging
configuration.

We can also configure a default for the `--override` flag that indicates the
word embedding with:
```ini
[override_cli_decorator]
option_overrides = dict: {'override': {'default': 'cb_default.name=glove_50'}}
```

While a user can create a model specific configuration file specified with the
`--config` option (such as in the other examples), this example is so simple as
to not need it.  For this reason, we make it optional:
```ini
[config_cli]
expect = False
```

The configured actions and their options for the CLI in the `cli` section
described earlier must be imported from their respective [resource libraries],
which is done with:
```ini
[import]
config_files = list:
    resource(zensols.util): resources/default.conf,
...
    resource(zensols.deepnlp): resources/cleaner.conf
```


Finally we import the model configuration from the [resource libraries] with a
special section used by the `--config` option's [first pass action].  This
provides special directives for loading the [overridden] `--override` and the
configuration file.  We reference the `default` and `cb_default` as they are
utilized in the loaded in the subordinate configuration files:
```ini
[config_import]
references = list: default, cb_default
sections = list: app_imp_conf

[app_imp_conf]
type = import
config_files = list:
    ^{override},
    ^{config_path},
    resource(zensols.deeplearn): resources/default.conf,
... 
    resource(cb): resources/obj.yml,
    ^{config_path}
```

This first loads the [override] with `--override`, then the [configuration
importer] from a user provided configuration file with `--config` (if
provided).  Then the defaults and model configuration is loaded.  Finally the
configuration file is loaded again providing the user the option to override
anything clobbered by the [resource libraries] as everything loaded is either
added over overwritten in order.  Nested in this list of resource files
includes the [obj.yml] file, which is this application example's specific
configuration.


## Application Configuration

As mentioned in the previous section, the [app.conf] specifies [resource
libraries] to load allowing the [obj.yml] to add and modify existing
configuration.  This file could have been written as an `ini` file (like
[app.conf]).  However, it contains the hierarchical vectorizer to batch
mappings lending itself better to a hierarchical data format such as YAML.

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

First the configuration defines where to download the corpus uncompress the
files to make it available to the program.  The `Install the corpus` sections
tell where the resources are on the Internet, and the file naming, which by
default takes the naming from the URL.

```yaml
cb_data_resource:
  class_name: zensols.install.Resource
  url: 'https://github.com/bhargaviparanjape/clickbait/raw/master/dataset/clickbait_data.gz'
non_cb_data_resource:
  class_name: zensols.install.Resource
  url: 'https://github.com/bhargaviparanjape/clickbait/raw/master/dataset/non_clickbait_data.gz'

feature_installer:
  resources: 'instance: list: cb_data_resource, non_cb_data_resource'
```

The installer has a list of resources it uses to download the files and
uncompress them on the file system.  This local directory is set in the
[feature resource library].


### Feature Creation

This section contains several sections that describe [Stash] instances that
cache the vectorized batches to the file system (see *batch encoding* in [the
paper]).  This process as it relates to this section includes:

1. Parse the downloaded corpus from the sentence text files
   (`dataframe_stash`).
1. Randomly split the dataset in to train, validation and test set, then store
   the data from the dataframe as a picked file on the file system
   (`dataframe_stash`).
1. Parse the English sentences from the `dataframe_stash` using spaCy across as
   many processes as the CPU has cores (`feature_factory_stash`) and persisting
   them to the file system in directories by feature (`feature_dir_stash` only
   found in the `deeplearn` resource library).
1. Read only certain files (based on feature selection for the particular
   model) from the file system to reconstruct batches (see *batch decoding* in
   [the paper]).
1. Train, validate and test the model using the same ordering and splits
   sampled by the `dataframe_stash` from step 1.


### Natural Language Processing

The `doc_parser` section tells the parser to create instances of a different
class that what was defined in its resource library (`FeatureDocument`) using
the classification resource library set up (loaded by the [classify resource
library] by [app.conf]).  The class we provide for this example contains an
attribute to carry a label for our text classification task.

```yaml
doc_parser:
  doc_class: 'class: zensols.deepnlp.classify.LabeledFeatureDocument'
  components: 'instance: list: remove_sent_boundaries_component'

classify_label_vectorizer:
  categories: ${dataframe_stash:labels}
```

The `classify_label_vectorizer` comes from the [feature resource library],
which needs the output nominal label names for encoding/vectorization.


### Batch

The `Batch` section provides all the configuration necessary to batch the
vectorized data in to chunks usable by the model.  Specifically, the
`batch_stash` section describes how to map between the vectorized output to
entries in the batches and their grouping.  It also gives the default set of
attributes to test with at experimentation and the number of sub-process
workers to use during batching.

```yaml
batch_stash:
  batch_feature_mappings: 'dataclass(zensols.deeplearn.batch.ConfigBatchFeatureMapping): cb_batch_mappings'
  decoded_attributes: 'set: label, ${cb_default:lang_features} ${cb_default:embedding}'
  workers: -2

cb_batch_mappings:
  batch_feature_mapping_adds:
    - 'dataclass(zensols.deeplearn.batch.BatchFeatureMapping): classify_label_batch_mappings'
    - 'dataclass(zensols.deeplearn.batch.BatchFeatureMapping): lang_batch_mappings'
  field_keep: [label, enums, dependencies, glove_50_embedding, fasttext_news_300_embedding]
```

The `cb_batch_mappings` section indicates to reuse the text classification
mappings from the [classify resource library], and the more general language
features (such as spaCy parsed vectorized data) from the [lang-batch resource
library].  The `workers: -2` says to use all but two cores for the number of
sub-processes for vectorization/batch creation.


### Model

The executor in the `Model` section sets `net_settings` to
`classify_net_settings` to provide the top level text classification for the
application using a BiLSTM-CRF.  This model is provided in the [classify
resource library] in `deepnlp` (this project), with little left to specify.
These remaining portions of the model that are specified are:

* Dense output layer that connects the LSTM to the CRF and specifies the output
  label cardinality, which is set to two (either clickbate or not).
* The embedding layer to use, which is string substituted with the
  `cb_default:embedding` section/option injected with the [overridden]
  (`--override`) option.
* The section containing the LSTM configuration (`recurrent_settings`).
* Model settings that include the model name, learning rate, default epoch
  count and the component that decodes model output to labels and softmaxes
  (confidence like scores).

```yaml
executor:
  net_settings: 'instance: classify_net_settings'

linear_settings:
  out_features: "eval: '${dataframe_stash:labels}'.count(',') + 1"

classify_net_settings:
  embedding_layer: 'instance: ${cb_default:embedding}_layer'
  recurrent_settings: 'instance: recurrent_settings'
  dropout: 0.2

model_settings:
  model_name: 'clickbate: ${cb_default:name}'
  learning_rate: 1e-3
  epochs: 35
```


## Imported from Resource Libraries

Other important components of the application not specified in the [obj.yml]
but present from being imported from resource libraries include:

* The facade class (`ClassifyModelFacade`) provided in the [classify resource
  library], which is used by a second pass CLI application to predict ad-hoc
  text.
* Model events (i.e. when training or validation starts/end) to track model
  train/test time consumption using an observer pattern in [observer resource
  library].
* Vectorizer configuration, vectorizer manager and manager sets, which take
  data (in our case English text) and vectorize in to binary form usable by the
  model.  See [the paper] for more information.
* A [Stash] that stratifies each dataset by label and other components that
  enable batching from the [feature resource library].


## Code

All the code for this example is in the is in the [cb.py] that merges the
corpus CSV files and the [harness.py] entry point application that invokes the
command line interface API.


## Running

The movie review data set example can be run from the command line or as a
Jupyter notebook.


### Command Line

Everything can be done with the harness script:

Get the command line help using a thin wrapper around the framework:
```bash
$ ./harness.py -h

Usage: harness.py <actions> [options]:

Options:
  -h, --help                                     show this help message and exit
  --version                                      show the program version and exit
  --level X                                      the level to set the application logger, X is one of: debug, err, info, warn
  -c, --config FILE                              the path to the configuration file
  --override <FILE|DIR|STRING>  cb_default.n...  a config file/dir or a comma delimited section.key=value string that overrides
                                                 configuration with default cb_default.name=glove_50

Actions:
list                                             list all actions and help
  --lstfmt <json|name|text>     text             the output format for the action listing
...
```

The executor tests and trains the model, use it to get the stats used to train:
```bash
$ ./harness.py info
clickbate: glove_50:
executor:
    model: clickbate: glove_50
    feature splits:
        split stash splits:
            train: 25598 (80.0%)
            validation: 3201 (10.0%)
            test: 3201 (10.0%)
            total: 32000
        total this instance: 32000
        keys consistent: True
...
```

Print a sample Glove 50 (default) batch of what the model will get during training:
```bash
$ ./harness.py info -i batch
clickbate: glove_50:
DefaultBatch
    size: 200
        label: torch.Size([200])
        glove_50_embedding: torch.Size([200, 20])
        enums: torch.Size([200, 20, 174])
        dependencies: torch.Size([200, 20, 1])
...
```

Train and test the model but switch to model profile with optimized:
```bash
$ ./harness.py traintest -p
2022-06-14 13:55:06,094 resetting executor
...
2022-06-14 13:55:06,947 training model <class 'zensols.deepnlp.classify.model.ClassifyNetwork'> on cpu for 35 epochs using learning rate 0.001
```

All model, its (hyper)parameters, metadata and results are stored in subdirectory of files:
```bash
$ ./harness.py result
Name: clickbate: glove_50: 1
Run index: 2
Learning rate: 0.001
    train:
        started: 06/14/2022 13:55:06:957329
...
```

Predict and write the test set to a CSV file:
```bash
$ ./harness.py preds
2022-06-14 13:58:57,186 wrote predictions: clickbate-glove_50.csv
```

Predict ad-hoc a few sentences:
```
$ ./harness.py predtext "Can't Wait For Summer?  You've Got To See These Pics"
pred=y, logit=0.9994343519210815: Can't Wait For Summer?  You've Got To See These Pics

$ ./harness.py predtext "Biden is fired up over inflation."
pred=n, logit=0.9951714277267456: Biden is fired up over inflation.
```

Note the `run.sh` script in the same directory provides a simpler API and more
prediction examples as a way of calling the [harness.py] entry point.  It also
serves as an example of how one might simplify a command line for a specific
model.


### Jupyter Notebook

There is a [Jupyter notebook] that executes the entire download, train,
validate, test and report process for both models.  In [notebook directory] is
the notebook, a Python source [mngfac.py] (facade manager factory) file that
"glues" the CLI to the notebook API, and the output of a previous run of the
notebook.

The [mngfac.py] file contains a convenience class used by the notebook to add
directories to the Python path, which is useful for debugging when the package
isn't installed.  It also has life cycle methods to manage instances of
[ModelFacade] and configure the Jupyter notebook for things such as logging and
page width.

To run the [Jupyter notebook]:
1. Pip install: `pip install notebook`
1. Go to the notebook directory: `cd examples/clickbate/notebook`
1. Start the notebook: `jupyter notebook`
1. Start the execution in the notebook with `Cell > Run All`.


<!-- links -->
[corpus]: https://github.com/bhargaviparanjape/clickbait/tree/master/dataset
[the paper]: https://arxiv.org/pdf/2109.03383.pdf
[GloVE resource library]: https://github.com/plandes/deepnlp/blob/master/resources/glove.conf

[resource libraries]: https://plandes.github.io/util/doc/config.html#resource-libraries
[INI format]: https://plandes.github.io/util/doc/config.html#ini-format
[cb.py]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/cb.py
[harness.py]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/harness.py
[app.conf]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/resources/app.conf
[dataclasses]: https://docs.python.org/3/library/dataclasses.html
[action with positional and optional parameters]: https://plandes.github.io/util/doc/command-line.html#application-class-and-actions
[obj.yml]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/resources/obj.yml

[Stash]: https://plandes.github.io/util/api/zensols.persist.html#zensols.persist.domain.Stash
[CliHarness]: https://plandes.github.io/util/api/zensols.cli.html#zensols.cli.harness.CliHarness
[ModelFacade]: https://plandes.github.io/deeplearn/api/zensols.deeplearn.model.html#zensols.deeplearn.model.facade.ModelFacade

[Jupyter notebook example]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/notebook/clickbate.ipynb
[Jupyter notebook]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/notebook/clickbate.ipynb
[classify resource library]: https://github.com/plandes/deepnlp/blob/master/resources/classify.conf
[configuration importer]: https://plandes.github.io/util/api/zensols.cli.lib.html?#zensols.cli.lib.config.ConfigurationImporter
[feature resource library]: https://github.com/plandes/deepnlp/blob/master/resources/feature.conf
[first pass action]: https://plandes.github.io/util/doc/command-line.html#user-configuration
[first pass actions]: https://plandes.github.io/util/doc/command-line.html#user-configuration
[lang-batch resource library]: https://github.com/plandes/deepnlp/blob/master/resources/lang-batch.yml
[mngfac.py]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/notebook/mngfac.py
[notebook directory]: https://github.com/plandes/deepnlp/tree/master/example/clickbate/notebook
[observer resource library]: https://github.com/plandes/deeplearn/blob/master/resources/observer.conf
[only source code]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/cb.py
[overridden]: https://plandes.github.io/util/api/zensols.cli.lib.html#zensols.cli.lib.config.ConfigurationOverrider
[override]: https://plandes.github.io/util/api/zensols.cli.lib.html#zensols.cli.lib.config.ConfigurationOverrider
[resource libraries]: https://plandes.github.io/util/doc/config.html#resource-libraries
