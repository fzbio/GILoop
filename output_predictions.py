import pandas as pd
import numpy as np
import os
from hickit.reader import get_headers, get_chrom_sizes
import tensorflow as tf
import json
import tensorflow_addons as tfa
from utils import *
import gc
from sklearn.metrics import f1_score, average_precision_score


if __name__ == '__main__':
    resolution = 10000
    protein = 'ctcf'
    chrom_size_path = 'hg19.chrom.sizes'
    threshold = 0.48
    # mode = 'transfer'
    base_dir = 'D:\\research_projects\\loop-bulk-run'
    IMAGE_SIZE = 64
    models_path = os.path.join(base_dir, 'models')

    # if mode == 'transfer':
    source_ratio = '20'
    run_time = '5'
    source_cell_line = 'gm12878'
    model_name = 'pretrainFinetune'
    run_id = '_'.join([model_name, source_cell_line, protein, str(source_ratio)])
    model_stage = 'Finetune'
    chroms = [str(i) for i in range(1, 9)] + [str(i) for i in range(10, 23)] + ['X']

    target_ratio = '100'
    target_cell_line = 'k562'

    output_dir = os.path.join(
        'predictions',
        'manual_{}_{}_rt{}_{}_{}'.format(source_cell_line, source_ratio, run_time, target_cell_line, target_ratio)
    )

    dataset_dir = os.path.join(base_dir, 'dataset', '{}_{}_{}'.format(target_cell_line, protein, target_ratio))
    source_dataset_dir = os.path.join(base_dir, 'dataset', '{}_{}_{}'.format(source_cell_line, protein, source_ratio))
    extra_config_path = os.path.join(base_dir, 'configs', '{}_{}_extra_settings.json'.format(run_id, run_time))
    with open(extra_config_path) as fp:
        saved_upper_bound = json.load(fp)['graph_upper_bound']

    seed = int(run_time) + 1000

    for chrom in chroms:
        model = tf.keras.models.load_model(os.path.join(models_path, '_'.join([run_id, model_stage, run_time])))
        indicator_path = os.path.join(dataset_dir, 'indicators.{}.csv'.format(chrom))
        identical_path = os.path.join(dataset_dir, 'graph_identical.{}.npy'.format(chrom))
        images, graphs, y, features = read_data_with_motif([chrom], dataset_dir, IMAGE_SIZE)
        graphs = normalise_graphs(scale_hic(graphs, saved_upper_bound))
        test_y_pred = np.asarray(model.predict([images, features, graphs])[1])
        print('Chromosome {} F1-score is {}'.format(chrom, f1_score(y.flatten(), test_y_pred.flatten() > threshold)))
        print('Chromosome {} AP is {}'.format(chrom, average_precision_score(y.flatten(), test_y_pred.flatten())))
        chrom_proba, chrom_gt = get_chrom_proba(
            chrom,
            get_chrom_sizes(chrom_size_path),
            resolution,
            test_y_pred,
            y,
            indicator_path,
            identical_path,
            IMAGE_SIZE
        )
        output_chrom_pred_to_bedpe(
            chrom, chrom_proba, threshold,
            get_headers([chrom], get_chrom_sizes(chrom_size_path), resolution),
            output_dir, resolution
        )
        del model
        gc.collect()
        tf.keras.backend.clear_session()
        print('Chromosome {} has been predicted.'.format(chrom))
