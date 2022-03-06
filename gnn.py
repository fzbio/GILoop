#!/usr/bin/env python3
import random
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, ReLU, Flatten, Activation, Conv2D, MaxPooling2D, Reshape, \
    UpSampling2D, GaussianNoise, Dense
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.models import Model

import tensorflow_addons as tfa
import json

from kgae.layers.graph import GraphConvolution
from custom_layers import CombineConcat, Edge2Node
from utils import IMAGE_SIZE, GRAPH_SIZE, get_split_graphset, scale_hic, normalise_graphs
from metrics import compute_auc


def gnn_run(chroms, run_id, seed, dataset_name, epoch=50):
    # seed = hash(run_id)
    dataset_dir = os.path.join('dataset', dataset_name)

    print('#' * 10 + ' Start training GCN ' + '#'*10)

    train_graphs, train_features, train_y, val_graphs, val_features, val_y, test_graphs, test_features, test_y = \
        get_split_graphset(dataset_dir, IMAGE_SIZE, seed, chroms)
    graph_upper_bound = np.quantile(train_graphs, 0.996)
    extra_settings = {'graph_upper_bound': graph_upper_bound}
    with open('configs/{}_extra_settings.json'.format(run_id), 'w') as fp:
        json.dump(extra_settings, fp)

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

    GNN_METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', threshold=0.5),
        tf.keras.metrics.AUC(curve="ROC", name='ROC_AUC'),
        tf.keras.metrics.AUC(curve="PR", name='PR_AUC')
    ]

    gnn_learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
        0.0005,
        2000 * 20,
        end_learning_rate=0.00001,
        power=3.0
    )
    # Data preparation (convert to tensors)

    train_features_tensor = tf.convert_to_tensor(train_features, dtype=tf.float32)
    train_graphs_tensor = tf.convert_to_tensor(train_graphs, dtype=tf.float32)

    val_features_tensor = tf.convert_to_tensor(val_features, dtype=tf.float32)
    val_graphs_tensor = tf.convert_to_tensor(val_graphs, dtype=tf.float32)

    train_x_tensors = [train_features_tensor, train_graphs_tensor]
    val_x_tensors = [val_features_tensor, val_graphs_tensor]

    flatten_train_y = train_y.reshape((-1, IMAGE_SIZE * IMAGE_SIZE))[..., np.newaxis]
    flatten_val_y = val_y.reshape((-1, IMAGE_SIZE * IMAGE_SIZE))[..., np.newaxis]

    # Batch size setup
    bs = 8

    A = Input(shape=(GRAPH_SIZE, GRAPH_SIZE), sparse=False)
    F = Input(shape=(GRAPH_SIZE, FEATURE_DIM))
    Ft = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(F)
    Ft = Dropout(0.2)(Ft)
    Ft = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(Ft)
    Ft = Dropout(0.2)(Ft)
    Ft = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(Ft)
    Ft = Dropout(0.2)(Ft)

    G = [A]
    H = GraphConvolution(1024, featureless=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))([Ft] + G)
    H = ReLU()(H)
    H = Dropout(0.3)(H)
    H = GraphConvolution(256, kernel_regularizer=tf.keras.regularizers.l2(0.0001))([H] + G)
    H = ReLU()(H)
    # H = Dropout(0.3)(H)

    b = crop_and_mutual_concat(H, GRAPH_SIZE, IMAGE_SIZE, 256)
    b = Dense(384, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(b)
    b = Dropout(0.2)(b)
    b = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(b)
    b = Dropout(0.2)(b)
    b = Edge2Node(IMAGE_SIZE)(b)
    b = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(b)
    b = Dropout(0.2)(b)

    b = crop_and_mutual_concat(b, GRAPH_SIZE, IMAGE_SIZE, 256)
    b = Dense(384, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(b)
    b = Dropout(0.2)(b)
    b = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(b)
    b = Dropout(0.2)(b)
    b = Edge2Node(IMAGE_SIZE)(b)
    b = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(b)
    b = Dropout(0.2)(b)

    b = crop_and_mutual_concat(b, GRAPH_SIZE, IMAGE_SIZE, 256)
    b = Dense(384, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(b)
    b = Dropout(0.2)(b)
    b = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(b)
    b = Dropout(0.2)(b)
    b = Edge2Node(IMAGE_SIZE)(b)
    b = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(b)
    b = Dropout(0.2)(b)

    b = crop_and_mutual_concat(b, GRAPH_SIZE, IMAGE_SIZE, 256)
    b = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(b)
    b = Dropout(0.25)(b)
    b = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(b)
    b = Dropout(0.25)(b)
    b = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(b)
    b = Dropout(0.25)(b)
    graph_embedding = Dense(16, name='gnn_embedding', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(b)

    graph_decode = ReLU()(graph_embedding)
    graph_decode = Dropout(0.3)(graph_decode)
    graph_decode = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(graph_decode)
    graph_decode = Dropout(0.3)(graph_decode)
    graph_decode = Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(graph_decode)
    graph_decode = Dropout(0.3)(graph_decode)
    graph_decode = Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(graph_decode)
    graph_decode = Dropout(0.3)(graph_decode)
    gnn_logits = Dense(1, name='gnn_logits', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(graph_decode)
    gnn_sigmoid = Activation('sigmoid', name='gnn_sigmoid')(gnn_logits)
    GNN = Model(inputs=[F, A], outputs=[gnn_logits, gnn_sigmoid])
    GNN.compile(
        loss={
            'gnn_sigmoid': tfa.losses.SigmoidFocalCrossEntropy(from_logits=False, alpha=0.5, gamma=1.2,
                                                               reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        },
        loss_weights={'gnn_sigmoid': IMAGE_SIZE * IMAGE_SIZE},
        optimizer=tf.keras.optimizers.Adam(learning_rate=gnn_learning_rate),
        metrics={
            'gnn_sigmoid': GNN_METRICS
        }
    )
    inputs = train_x_tensors

    history = GNN.fit(
        inputs, y=[flatten_train_y, flatten_train_y],
        batch_size=bs, epochs=epoch,
        validation_data=(val_x_tensors, [flatten_val_y, flatten_val_y]),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_gnn_sigmoid_' + 'PR_AUC',  # use validation AUC of precision-recall for stopping
                min_delta=0.0001, patience=7,
                verbose=1, mode='max'),
        ],
        verbose=2
    )

    train_y_pred = np.asarray(GNN.predict(train_x_tensors)[1])
    val_y_pred = np.asarray(GNN.predict(val_x_tensors)[1])
    test_y_pred = np.asarray(GNN.predict([test_features, test_graphs])[1])

    train_auc, train_ap = compute_auc(train_y_pred, train_y.astype('bool'))
    val_auc, val_ap = compute_auc(val_y_pred, val_y.astype('bool'))
    test_auc, test_ap = compute_auc(test_y_pred, test_y.astype('bool'))

    print('=' * 30)
    print('*******GNN**********')
    print('Train AUC is {}. Train AP is {}.'.format(train_auc, train_ap))
    print('Validation AUC is {}. Validation AP is {}.'.format(val_auc, val_ap))
    print('Test AUC is {}. Test AP is {}.'.format(test_auc, test_ap))

    GNN.save(os.path.join('models', '{}_GNN'.format(run_id)))
