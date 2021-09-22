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


## Application

The example is written as an Zensols CLI application.  The entry point program
is `run.py`.  However, the application is configured in `app.conf`, which is
where most of what makes the application is set up.  In this configuraiton
file, the application is configured with file system paths used in the project
all based from a root path supplied by the entry point `run.py` application, or
a different relative directory for its [Jupyter notebook example].


<!-- links -->
[clickbate corpus]: https://github.com/bhargaviparanjape/clickbait/tree/master/dataset

[Jupyter notebook example]: https://github.com/plandes/deepnlp/blob/master/example/clickbate/notebook/clickbate.ipynb
