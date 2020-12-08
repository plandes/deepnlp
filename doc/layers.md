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

<!-- links -->
[DeepConvolution1dNetworkSettings]: ../api/zensols.deepnlp.layer.html#zensols.deepnlp.layer.conv.DeepConvolution1dNetworkSettings
[DeepConvolution1d]: ../api/zensols.deepnlp.layer.html#zensols.deepnlp.layer.conv.DeepConvolution1d
[EmbeddingNetworkSettings]: ../api/zensols.deepnlp.layer.html#zensols.deepnlp.layer.embed.EmbeddingNetworkSettings
[EmbeddingNetworkModule]: ../api/zensols.deepnlp.layer.html#zensols.deepnlp.layer.embed.EmbeddingNetworkModule
[EmbeddedRecurrentCRFSettings]: ../api/zensols.deepnlp.layer.html#zensols.deepnlp.layer.embrecurcrf.EmbeddedRecurrentCRFSettings
[EmbeddedRecurrentCRF]: ../api/zensols.deepnlp.layer.html#zensols.deepnlp.layer.embrecurcrf.EmbeddedRecurrentCRF
