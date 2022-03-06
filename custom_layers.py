import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Rescaling


class BilinearFusion(Layer):
    def __init__(self, **kwargs):
        super(BilinearFusion, self).__init__(**kwargs)

    def build(self, input_shapes):
        self.pixel_num = input_shapes[0][1]
        self.t1_feature_dim = input_shapes[0][2]
        self.t2_feature_dim = input_shapes[1][2]

    def call(self, inputs):
        t1 = inputs[0]
        t2 = inputs[1]
        t1 = tf.expand_dims(t1, axis=-1)
        t2 = tf.expand_dims(t2, axis=-2)
        outer = tf.matmul(t1, t2)
        outer = tf.reshape(outer, [-1, self.pixel_num, self.t1_feature_dim * self.t2_feature_dim])
        outer = tf.sign(outer) * tf.sqrt(tf.abs(outer + 1e-9))
        outer = tf.math.l2_normalize(outer, axis=-1)
        return outer

    def get_config(self):
        #         config = {'pooling_num': self.pooling_num}
        base_config = super(BilinearFusion, self).get_config()
        return {**base_config}


class ClipByValue(Layer):
    def __init__(self, max_hic_value, **kwargs):
        super(ClipByValue, self).__init__(**kwargs)
        self.max_hic_value = max_hic_value

    def call(self, inputs):
        clipped = tf.clip_by_value(inputs, 0, self.max_hic_value)
        return clipped

    def get_config(self):
        config = {
            'max_hic_value': self.max_hic_value
        }
        base_config = super(ClipByValue, self).get_config()
        return {**config, **base_config}


class HiCScale(Layer):
    def __init__(self, max_hic_value, **kwargs):
        super(HiCScale, self).__init__(**kwargs)
        self.max_hic_value = max_hic_value
        self.rescale = Rescaling(1 / max_hic_value)

    def call(self, inputs):
        clipped = tf.clip_by_value(inputs, 0, self.max_hic_value)
        rescaled = self.rescale(clipped)
        return rescaled

    def get_config(self):
        config = {
            'max_hic_value': self.max_hic_value,
            'rescale': self.rescale
        }
        base_config = super(HiCScale, self).get_config()
        return {**config, **base_config}


class CombineConcat(Layer):
    def __init__(self, node_num, **kwargs):
        super(CombineConcat, self).__init__(**kwargs)
        self.node_num = node_num

    def call(self, inputs):
        size = self.node_num
        # Make 2D grid of indices
        r = tf.range(size)
        ii, jj = tf.meshgrid(r, r, indexing='ij')
        ii = tf.reshape(ii, [size * size, 1])
        jj = tf.reshape(jj, [size * size, 1])

        g1 = tf.map_fn(lambda nodes: tf.gather_nd(nodes, ii), inputs[0], fn_output_signature=tf.float32)
        g2 = tf.map_fn(lambda nodes: tf.gather_nd(nodes, jj), inputs[1], fn_output_signature=tf.float32)
        #         g1 = tf.gather_nd(inputs[0], ii)
        #         g2 = tf.gather_nd(inputs[1], jj)
        return tf.concat([g1, g2], 2)

    def get_config(self):
        config = {'node_num': self.node_num}
        base_config = super(CombineConcat, self).get_config()
        return {**config, **base_config}


class Edge2Node(Layer):
    def __init__(self, pooling_num, **kwargs):
        super(Edge2Node, self).__init__(**kwargs)
        self.pooling_num = pooling_num

    def build(self, input_shape):
        self.node_num = tf.math.floordiv(input_shape[1], self.pooling_num)
        self.feature_dim = input_shape[2]

    def call(self, edge_tensor):
        t1 = tf.reshape(edge_tensor, [-1, self.node_num, self.pooling_num, self.feature_dim])
        t1 = tf.reduce_sum(t1, axis=2)
        t2 = tf.reshape(edge_tensor, [-1, self.pooling_num, self.node_num, self.feature_dim])
        t2 = tf.reduce_sum(t2, axis=1)
        return tf.concat([t1, t2], 1)

    def get_config(self):
        config = {'pooling_num': self.pooling_num}
        base_config = super(Edge2Node, self).get_config()
        return {**config, **base_config}