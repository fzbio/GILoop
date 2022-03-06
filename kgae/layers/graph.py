from __future__ import print_function

import tensorflow as tf
from keras import activations, initializers, constraints
from keras import regularizers
from tensorflow.keras.layers import Layer
import keras.backend as K


class GraphConvolution(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    """
    This layer only supports 1 graph as input; multiple-graph input has not been
    implemented yet. 
    """
    def __init__(self, units,
                 activation=None,
                 featureless=False,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.featureless = featureless
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    # def compute_output_shape(self, input_shapes):
    #     features_shape = input_shapes[0]
    #     output_shape = (features_shape[0], self.units)
    #     return output_shape  # (batch_size, output_dim)

    def build(self, input_shapes):
        self.batch_size = input_shapes[0][0]
        if not self.featureless:
            features_shape = input_shapes[0]
            assert len(features_shape) == 3
            batch_size = features_shape[0]
            node_count = features_shape[1]
            input_dim = features_shape[2]
            self.kernel = self.add_weight(shape=(input_dim, self.units),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
            if self.use_bias:
                self.bias = self.add_weight(shape=(node_count, self.units),
                                            initializer=self.bias_initializer,
                                            name='bias',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
            else:
                self.bias = None
            self.built = True
        else:
            assert len(input_shapes) == 1
            basis_shape = input_shapes[0]
            self.kernel = self.add_weight(shape=(basis_shape[1], self.units),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
            if self.use_bias:
                self.bias = self.add_weight(shape=(basis_shape[1], self.units),
                                            initializer=self.bias_initializer,
                                            name='bias',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
            else:
                self.bias = None
            self.built = True

    def call(self, inputs, mask=None):
        if not self.featureless:
            assert len(inputs) == 2
            features = inputs[0]
            basis = inputs[1]
            output = tf.matmul(basis, features)
            # output = tf.tensordot(output, self.kernel, axes=1)
            output = tf.matmul(output, self.kernel)
        else:
            assert len(inputs) == 1
            basis = inputs[0]
            # output = tf.tensordot(basis, self.kernel, axes=1)
            output = tf.matmul(basis, self.kernel)
        if not (self.bias is None):
            output += self.bias
        return self.activation(output)


    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
        }

        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
