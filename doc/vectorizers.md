# Vectorizers

Please first read the [vectorizers] section first.

The set of vectorizers included with this package are listed below:

* [EnumContainerFeatureVectorizer]: Encode tokens found in the container by
  aggregating the spaCy vectorizers output.
* [CountEnumContainerFeatureVectorizer]: Return the count of all tokens as a *1
  X M * N* tensor where *M* is the number of token feature ids and *N* is the
  columns of the output of the [SpacyFeatureVectorizer] vectorizer.
* [DepthFeatureDocumentVectorizer]: Return the depths of tokens based on
  how deep they are in a head dependency tree.
* [StatisticsFeatureDocumentVectorizer]: Return basic statics including
  token and sentence count for [FeatureDocument] instances.
* [OverlappingFeatureDocumentVectorizer]: Return the number of normalized
  and lemmatized tokens across multiple documents.
* [MutualFeaturesContainerFeatureVectorizer]: Return the shared count of all
  tokens as a *1 X M * N* tensor where *M* is the number of token feature ids
  and *N* is the columns of the output of the the [SpacyFeatureVectorizer]
  vectorizer.


<!-- links -->

[vectorizers]: movie-example.html#vectorizer-configuration

[EnumContainerFeatureVectorizer]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.vectorizers.EnumContainerFeatureVectorizer
[CountEnumContainerFeatureVectorizer]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.vectorizers.CountEnumContainerFeatureVectorizer
[DepthFeatureDocumentVectorizer]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.vectorizers.DepthFeatureDocumentVectorizer
[StatisticsFeatureDocumentVectorizer]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.vectorizers.StatisticsFeatureDocumentVectorizer
[OverlappingFeatureDocumentVectorizer]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.vectorizers.OverlappingFeatureDocumentVectorizer
[MutualFeaturesContainerFeatureVectorizer]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.vectorizers.MutualFeaturesContainerFeatureVectorizer

[SpacyFeatureVectorizer]: ../api/zensols.deepnlp.vectorize.html#zensols.deepnlp.vectorize.spacy.SpacyFeatureVectorizer
[FeatureDocument]: ../api/zensols.deepnlp.html#zensols.deepnlp.domain.FeatureDocument
