#!/usr/bin/env python3
import random
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, ReLU, Flatten, Activation, Conv2D, MaxPooling2D, Reshape, \
    UpSampling2D, GaussianNoise, Dense, Rescaling
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.models import Model

import tensorflow_addons as tfa
import json

from custom_layers import HiCScale, CombineConcat, ClipByValue
from utils import IMAGE_SIZE, get_split_imageset
from metrics import compute_auc


if __name__ == '__main__':
    chroms = [str(i) for i in range(1, 23)] + ['X']
    run_time = int(sys.argv[1])
    downsampling_ratio = str(sys.argv[2])
    seed = run_time + 1000

    # os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    cell_line_name = 'gm12878'
    protein = 'ctcf'
    model_name = 'pretrainFinetune'
    dataset_dir = 'dataset/{}_{}_{}'.format(cell_line_name, protein, downsampling_ratio)

    run_id = '_'.join([model_name, cell_line_name, protein, str(downsampling_ratio)])

    print('#'*10)
    print('**CNN** Downsampling rate: {}; Running time: {}'.format(downsampling_ratio, run_time))

    train_images, train_y, val_images, val_y, test_images, test_y = \
        get_split_imageset(dataset_dir, IMAGE_SIZE, seed, chroms)

    image_upper_bound = np.quantile(train_images, 0.996)

    CNN_METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', threshold=0.5),
        tf.keras.metrics.AUC(curve="ROC", name='ROC_AUC'),
        tf.keras.metrics.AUC(curve="PR", name='PR_AUC')
    ]

    cnn_learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
        0.001,
        2000 * 20,
        end_learning_rate=0.00005,
        power=2.0
    )
    # Data preparation (convert to tensors)

    train_images_tensor = tf.convert_to_tensor(train_images, dtype=tf.float32)
    val_images_tensor = tf.convert_to_tensor(val_images, dtype=tf.float32)
    train_x_tensors = [train_images_tensor]
    val_x_tensors = [val_images_tensor]
    flatten_train_y = train_y.reshape((-1, IMAGE_SIZE * IMAGE_SIZE))[..., np.newaxis]
    flatten_val_y = val_y.reshape((-1, IMAGE_SIZE * IMAGE_SIZE))[..., np.newaxis]

    # Batch size setup
    bs = 8

    I = Input(shape=(IMAGE_SIZE, IMAGE_SIZE))
    # x = HiCScale(image_upper_bound)(I)
    x = ClipByValue(image_upper_bound)(I)
    x = Rescaling(1 / image_upper_bound)(x)
    x = Reshape((IMAGE_SIZE, IMAGE_SIZE, 1))(x)
    x = GaussianNoise(0.05)(x)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv2)
    drop2 = Dropout(0.3)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        conv3)
    drop3 = Dropout(0.3)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        pool3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        pool4)
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        merge6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        conv6)

    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        merge7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        conv7)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        merge8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv8)

    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
        merge9)
    conv9 = Conv2D(16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv9)
    conv10 = conv9
    image_embedding = Reshape((IMAGE_SIZE * IMAGE_SIZE, -1), name='cnn_embedding')(conv10)

    image_decode = ReLU()(image_embedding)
    image_decode = Dropout(0.3)(image_decode)
    image_decode = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(image_decode)
    image_decode = Dropout(0.3)(image_decode)
    image_decode = Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(image_decode)
    image_decode = Dropout(0.3)(image_decode)
    image_decode = Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(image_decode)
    image_decode = Dropout(0.3)(image_decode)
    cnn_logits = Dense(1, name='cnn_logits', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(image_decode)
    cnn_sigmoid = Activation('sigmoid', name='cnn_sigmoid')(cnn_logits)

    CNN = Model(inputs=[I], outputs=[cnn_logits, cnn_sigmoid])
    CNN.compile(
        loss={
            'cnn_sigmoid': tfa.losses.SigmoidFocalCrossEntropy(from_logits=False, alpha=0.5, gamma=1.2,
                                                               reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        },
        loss_weights={'cnn_sigmoid': IMAGE_SIZE * IMAGE_SIZE},
        optimizer=tf.keras.optimizers.Adam(learning_rate=cnn_learning_rate),
        metrics={
            'cnn_sigmoid': CNN_METRICS
        }
    )

    inputs = [train_x_tensors[0]]
    history = CNN.fit(
        inputs, y=[flatten_train_y, flatten_train_y],
        batch_size=bs, epochs=50,
        validation_data=([val_x_tensors[0]], [flatten_val_y, flatten_val_y]),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_cnn_sigmoid_' + 'PR_AUC',  # use validation AUC of precision-recall for stopping
                min_delta=0.0001, patience=5,
                verbose=1, mode='max'),
        ],
        verbose=2
    )

    train_y_pred = np.asarray(CNN.predict([train_x_tensors[0]])[1])
    val_y_pred = np.asarray(CNN.predict([val_x_tensors[0]])[1])
    test_y_pred = np.asarray(CNN.predict([test_images])[1])

    train_auc, train_ap = compute_auc(train_y_pred, train_y.astype('bool'))
    val_auc, val_ap = compute_auc(val_y_pred, val_y.astype('bool'))
    test_auc, test_ap = compute_auc(test_y_pred, test_y.astype('bool'))

    print('=' * 30)
    print('*******CNN**********')
    print('Train AUC is {}. Train AP is {}.'.format(train_auc, train_ap))
    print('Validation AUC is {}. Validation AP is {}.'.format(val_auc, val_ap))
    print('Test AUC is {}. Test AP is {}.'.format(test_auc, test_ap))

    CNN.save(os.path.join('models', '{}_CNN_{}'.format(run_id, str(run_time))))

