#!/usr/bin/env python3
import random
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Activation, Reshape, Dense
from tensorflow.keras.models import Model

import tensorflow_addons as tfa
import json

from custom_layers import CombineConcat, Edge2Node, BilinearFusion
from utils import IMAGE_SIZE, scale_hic, normalise_graphs, get_split_dataset
from metrics import compute_auc


def finetune_run(chroms, run_id, seed, dataset_name, epoch=50):
    dataset_dir = os.path.join('dataset', dataset_name)
    print('#' * 10 + ' Fine-tuning ' + '#' * 10)
    # seed = hash(run_id)
    train_images, train_graphs, train_features, train_y, val_images, val_graphs, val_features, val_y, test_images, \
    test_graphs, test_features, test_y = get_split_dataset(dataset_dir, IMAGE_SIZE, seed, chroms)

    graph_upper_bound = np.quantile(train_graphs, 0.996)

    train_graphs = normalise_graphs(scale_hic(train_graphs, graph_upper_bound))
    val_graphs = normalise_graphs(scale_hic(val_graphs, graph_upper_bound))
    test_graphs = normalise_graphs(scale_hic(test_graphs, graph_upper_bound))
    FEATURE_DIM = train_features.shape[2]


    def crop_and_mutual_concat(input_tensor, graph_size, image_size, feature_num):
        t = Reshape((graph_size, feature_num, 1))(input_tensor)
        t1 = tf.keras.layers.Cropping2D(cropping=((0, image_size), (0, 0)))(t)
        t1 = Reshape((image_size, feature_num))(t1)
        t2 = tf.keras.layers.Cropping2D(cropping=((image_size, 0), (0, 0)))(t)
        t2 = Reshape((image_size, feature_num))(t2)
        t = CombineConcat(image_size)([t1, t2])
        return t


    Complete_METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', threshold=0.5),
        tf.keras.metrics.AUC(curve="ROC", name='ROC_AUC'),
        tf.keras.metrics.AUC(curve="PR", name='PR_AUC')
    ]
    complete_learning_rate = 0.0001

    # Data preparation (convert to tensors)

    train_images_tensor = tf.convert_to_tensor(train_images, dtype=tf.float32)
    train_features_tensor = tf.convert_to_tensor(train_features, dtype=tf.float32)
    train_graphs_tensor = tf.convert_to_tensor(train_graphs, dtype=tf.float32)

    val_images_tensor = tf.convert_to_tensor(val_images, dtype=tf.float32)
    val_features_tensor = tf.convert_to_tensor(val_features, dtype=tf.float32)
    val_graphs_tensor = tf.convert_to_tensor(val_graphs, dtype=tf.float32)

    train_x_tensors = [train_images_tensor, train_features_tensor, train_graphs_tensor]
    val_x_tensors = [val_images_tensor, val_features_tensor, val_graphs_tensor]

    flatten_train_y = train_y.reshape((-1, IMAGE_SIZE * IMAGE_SIZE))[..., np.newaxis]
    flatten_val_y = val_y.reshape((-1, IMAGE_SIZE * IMAGE_SIZE))[..., np.newaxis]

    # Batch size setup
    bs = 8

    GNN = tf.keras.models.load_model(
        'models/{}_GNN'.format(run_id)
    )
    GNN = Model(inputs=GNN.inputs, outputs=GNN.get_layer('gnn_embedding').output, name='GNN')

    CNN = tf.keras.models.load_model(
        'models/{}_CNN'.format(run_id)
    )
    CNN = Model(inputs=CNN.inputs, outputs=CNN.get_layer('cnn_embedding').output, name='CNN')

    I = Input(CNN.inputs[0].get_shape()[1:])
    F = Input(GNN.inputs[0].get_shape()[1:])
    A = Input(GNN.inputs[1].get_shape()[1:])

    combined_decoded = BilinearFusion()([CNN([I]), GNN([F, A])])

    combined_decoded = Dropout(0.3)(combined_decoded)
    combined_decoded = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        combined_decoded)
    combined_decoded = Dropout(0.3)(combined_decoded)
    combined_decoded = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        combined_decoded)
    combined_decoded = Dropout(0.3)(combined_decoded)
    combined_decoded = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        combined_decoded)
    combined_decoded = Dropout(0.3)(combined_decoded)
    combined_decoded = Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        combined_decoded)
    combined_decoded = Dropout(0.3)(combined_decoded)
    combined_decoded = Dense(1, name='logits_flattened', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        combined_decoded)

    flattened_decoded = combined_decoded

    sig_flattened = Activation('sigmoid', name='sigmoid_flattened')(flattened_decoded)
    model = Model(inputs=[I, F, A], outputs=[flattened_decoded, sig_flattened])
    model.compile(
        loss={
            'sigmoid_flattened': tfa.losses.SigmoidFocalCrossEntropy(from_logits=False, alpha=0.5, gamma=1.2,
                                                                     reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        },
        loss_weights={'sigmoid_flattened': IMAGE_SIZE * IMAGE_SIZE},
        optimizer=tf.keras.optimizers.Adam(learning_rate=complete_learning_rate),
        metrics={
            'sigmoid_flattened': Complete_METRICS
        }
    )

    inputs = train_x_tensors
    history = model.fit(
        inputs, y=[flatten_train_y, flatten_train_y],
        batch_size=bs, epochs=epoch,
        validation_data=(val_x_tensors, [flatten_val_y, flatten_val_y]),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_sigmoid_flattened_' + 'PR_AUC',  # use validation AUC of precision-recall for stopping
                min_delta=0.0001, patience=5,
                verbose=1, mode='max')
        ],
        verbose=2
    )

    train_y_pred = np.asarray(model.predict(train_x_tensors)[1])
    val_y_pred = np.asarray(model.predict(val_x_tensors)[1])
    test_y_pred = np.asarray(model.predict([test_images, test_features, test_graphs])[1])

    train_auc, train_ap = compute_auc(train_y_pred, train_y.astype('bool'))
    val_auc, val_ap = compute_auc(val_y_pred, val_y.astype('bool'))
    test_auc, test_ap = compute_auc(test_y_pred, test_y.astype('bool'))

    print('=' * 30)
    print('*******Finetune**********')
    print('Train AUC is {}. Train AP is {}.'.format(train_auc, train_ap))
    print('Validation AUC is {}. Validation AP is {}.'.format(val_auc, val_ap))
    print('Test AUC is {}. Test AP is {}.'.format(test_auc, test_ap))

    model.save(os.path.join('models', '{}_Finetune'.format(run_id)))
