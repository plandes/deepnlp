# Layers

The set of layers, layer settings and layer factories included with this
package are listed below:

* Convolution (1D):
  * [DeepConvolution1dNetworkSettings]: Configurable repeated series of
    1-dimension convolution, pooling, batch norm and activation layers.
  * [DeepConvolution1d]: Configurable repeated series of 1-dimension
    convolution, pooling, batch norm and activation layers.
* Word Embedding Layer:
  * [EmbeddingNetworkSettings]: A utility container settings class for models
    that use an embedding input layer that inherit from
    [EmbeddingNetworkModule].
  * [EmbeddingNetworkModule]: An module that uses an embedding as the input
    layer.
* Conditional Random Field:
  * [EmbeddedRecurrentCRFSettings]: A utility container settings class for
    convulsion network models.
  * [EmbeddedRecurrentCRF]: A recurrent neural network composed of an embedding
    input, an recurrent network, and a linear conditional random field output
    layer.
* Transformers:
  * [TransformerEmbeddingLayer]: A transformer (i.e. BERT) embedding layer that
    allows for direct access to output embeddings.  Sentence and token
    classification is supported out of the box and selected with the
    [TransformerEmbedding.output] attribute.  The
    [TransformerResource.model_id] attribute gives a list of tested models.
  * [TransformerSequence]: A sequence based model for token classification use
    HuggingFace transformers.  Named entity recognition (NER) is one use case
  * of this model.
  * See [ner embedding.conf] and [movie embedding.conf] for examples of how to
    configure transformer layers.

<!-- links -->
[DeepConvolution1dNetworkSettings]: ../api/zensols.deepnlp.layer.html#zensols.deepnlp.layer.conv.DeepConvolution1dNetworkSettings
[DeepConvolution1d]: ../api/zensols.deepnlp.layer.html#zensols.deepnlp.layer.conv.DeepConvolution1d
[EmbeddingNetworkSettings]: ../api/zensols.deepnlp.layer.html#zensols.deepnlp.layer.embed.EmbeddingNetworkSettings
[EmbeddingNetworkModule]: ../api/zensols.deepnlp.layer.html#zensols.deepnlp.layer.embed.EmbeddingNetworkModule
[EmbeddedRecurrentCRFSettings]: ../api/zensols.deepnlp.layer.html#zensols.deepnlp.layer.embrecurcrf.EmbeddedRecurrentCRFSettings
[EmbeddedRecurrentCRF]: ../api/zensols.deepnlp.layer.html#zensols.deepnlp.layer.embrecurcrf.EmbeddedRecurrentCRF
[TransformerEmbeddingLayer]: ../api/zensols.deepnlp.transformer.html#zensols.deepnlp.transformer.layer.TransformerEmbeddingLayer
[TransformerResource.model_id]: ../api/zensols.deepnlp.transformer.html#zensols.deepnlp.transformer.resource.TransformerResource.model_id
[TransformerEmbedding.output]: ../api/zensols.deepnlp.transformer.html#zensols.deepnlp.transformer.embed.TransformerEmbedding.output
[TransformerSequence]: ../api/zensols.deepnlp.transformer.html#zensols.deepnlp.transformer.layer.TransformerSequence
[ner embedding.conf]: https://github.com/plandes/deepnlp/blob/master/example/ner/resources/embedding.conf
[movie embedding.conf]: https://github.com/plandes/deepnlp/blob/master/example/movie/resources/embedding.conf
