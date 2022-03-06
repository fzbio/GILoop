from keras.layers import Layer
from keras import activations
import tensorflow as tf


class InnerProduct(Layer):
    def __init__(self, activation=None, **kwargs):
        super(InnerProduct, self).__init__(**kwargs)
        self.activation = activations.get(activation)

    # def compute_output_shape(self, input_shape):
    #     return (input_shape[0], input_shape[0])

    def call(self, inputs, **kwargs):
        return self.activation(tf.linalg.matmul(inputs[0], inputs[1], transpose_b=True))

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation)
        }
        base_config = super(InnerProduct, self).get_config()
        return {**config, **base_config}
