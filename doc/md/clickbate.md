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


<!-- links -->
[clickbate corpus]: https://github.com/bhargaviparanjape/clickbait/tree/master/dataset

[Jupyter notebook example]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/notebook/clickbate.ipynb
[resource libraries]: https://plandes.github.io/util/doc/config.html#resource-libraries
[run.py]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/run.py
[app.conf]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/resources/app.conf
[dataclasses]: https://docs.python.org/3/library/dataclasses.html
[action with positional and optional parameters]: https://plandes.github.io/util/doc/command-line.html#application-class-and-actions
[default.conf]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/resources/default.conf
[feature.conf]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/resources/feature.conf
[Stash]: https://plandes.github.io/util/api/zensols.persist.html#zensols.persist.domain.Stash
[the paper]: https://arxiv.org/pdf/2109.03383.pdf
