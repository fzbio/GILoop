import tensorflow as tf


def normalized_bce(norm, pos_weight):
    """

    """
    def loss_function(y_true, y_pred):
        y_true_list = tf.unstack(y_true)
        y_pred_list = tf.unstack(y_pred)
        cost_list = []
        for i in range(len(y_true_list)):
            cost_list.append(norm * tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(labels=y_true_list[i], logits=y_pred_list[i], pos_weight=pos_weight)
            ))
        # cost = norm * tf.reduce_mean(
        #     tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=pos_weight)
        # )
        assert len(cost_list) == 1
        return tf.stack(cost_list)
    return loss_function

def gvae_loss():
    pass