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


def run_output_predictions(run_id, model_stage, threshold, target_dataset_name, target_assembly, chroms, output_path, mode):
    """

    :param run_id: String - The string that specifies the run of experiment
    :param model_stage: String - can only be 'GNN', 'CNN', or 'Finetune'
    :param threshold: Float - The probability threshold
    :param target_dataset_name: String - The name of dataset you want to predict on
    :param chroms: List - Chromosome list we want to predict on. e.g. ['1', '2', 'X']
    :param target_assembly: String - 'hg19' or 'hg38'
    :param output_path: String - The path to the output file
    :param mode: String - 'test' or 'realworld'. Test mode means the target cell line has the ground truth ChIA-PET
    data and the program will calculate the PRAUC for it. 'realworld' mode does not print PRAUC because the target
    dataset does not have label.
    :return: Pandas dataframe contains the genome-wide annotations
    """
    dataset_dir = os.path.join('dataset', target_dataset_name)
    model_path = os.path.join('models', run_id + '_' + model_stage)
    chrom_size_path = '{}.chrom.sizes'.format(target_assembly)
    extra_config_path = os.path.join('configs', '{}_extra_settings.json'.format(run_id))
    with open(extra_config_path) as fp:
        saved_upper_bound = json.load(fp)['graph_upper_bound']
    pred_dfs = []
    ys = []
    y_preds = []
    for chrom in chroms:
        model = tf.keras.models.load_model(model_path)
        indicator_path = os.path.join(dataset_dir, 'indicators.{}.csv'.format(chrom))
        identical_path = os.path.join(dataset_dir, 'graph_identical.{}.npy'.format(chrom))
        images, graphs, y, features = read_data_with_motif([chrom], dataset_dir, IMAGE_SIZE)
        graphs = normalise_graphs(scale_hic(graphs, saved_upper_bound))
        test_y_pred = np.asarray(model.predict([images, features, graphs])[1])
        ys.append(y.flatten())
        y_preds.append(test_y_pred.flatten())
        chrom_proba, chrom_gt = get_chrom_proba(
            chrom,
            get_chrom_sizes(chrom_size_path),
            10000,
            test_y_pred,
            y,
            indicator_path,
            identical_path,
            IMAGE_SIZE
        )
        current_df = get_chrom_pred_df(
            chrom, chrom_proba, threshold,
            get_headers([chrom], get_chrom_sizes(chrom_size_path), 10000),
        )
        pred_dfs.append(current_df)
        del model
        gc.collect()
        tf.keras.backend.clear_session()
    if mode == 'test':
        print('PRAUC on the target cell line is {}'.format(
            average_precision_score(np.concatenate(ys), np.concatenate(y_preds))
        ))
    full_pred_df = pd.concat(pred_dfs)
    full_pred_df.to_csv(output_path, sep='\t', index=False, header=False)
    return full_pred_df


if __name__ == '__main__':
    run_output_predictions(
        'gm12878_ctcf_50',                              # Specify the ID of a pre-trained model
        'Finetune',                                     # Specify using which stage of the model to make prediction
        0.48,                                           # Set the probability threshold
        'hela_100',                                     # Specify the name of the dataset you want to predict on
        'hg38',                                         # The genome assembly of the target dataset
        ['1'],                                          # Annotate on which Chromosomes
        'predictions/hela_test.bedpe',                  # The output file path
        'test'                                          # Test mode means the target dataset has label; 'realworld' mode
                                                        # means the target cell line does not have label
    )

