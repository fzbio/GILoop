import numpy as np
import pandas as pd
import os
from hickit.reader import get_headers, get_chrom_sizes


def create_kmer_feature_df(the_headers: pd.DataFrame, gw_kmer_df):
    the_headers = the_headers.copy()
    kmer_df = the_headers.merge(right=gw_kmer_df, how='left', on=['chrom', 'locus'])
    assert len(kmer_df) == len(the_headers)
    return kmer_df


def get_chrom_kmer_features(chrom_name, dataset_dir, gw_kmer_df, graph_size):
    indicators = pd.read_csv(
        os.path.join(dataset_dir, 'indicators.{}.csv'.format(chrom_name)),
        sep=',', index_col=0, dtype={'chrom': 'str'}
    )
    features = create_kmer_feature_df(indicators, gw_kmer_df).drop(columns=['chrom', 'locus']).values
    features[np.isnan(features)] = 0
    assert features.shape[0] == len(indicators)
    return features.reshape((-1, graph_size, features.shape[1]))


def create_motif_feature_df(full_headers: pd.DataFrame, gw_motif_df):
    full_headers = full_headers.copy()
    feature_df = full_headers.merge(right=gw_motif_df, how='left', on=['chrom', 'locus'])
    assert len(full_headers) == len(feature_df)
    return feature_df


def get_chrom_motif_features(chrom_name, dataset_dir, gw_motif_df, graph_size):
    indicators = pd.read_csv(
        os.path.join(dataset_dir, 'indicators.{}.csv'.format(chrom_name)),
        sep=',', index_col=0, dtype={'chrom': 'str'}
    )
    features = create_motif_feature_df(indicators, gw_motif_df).drop(columns=['chrom', 'locus']).values
    features[np.isnan(features)] = -1
    assert features.shape[0] == len(indicators)
    return features.reshape((-1, graph_size, features.shape[1]))


def run_generate_node_features(run_id, chroms, assembly_name):
    print('='*10 + ' Start generating node features ' + '='*10)
    dataset_path = os.path.join('dataset', run_id)
    kmer_csv_path = os.path.join('dataset', 'kmer_{}.csv'.format(assembly_name))
    motif_csv_path = os.path.join('dataset', 'fimo_{}.csv'.format(assembly_name))
    gw_kmer_df = pd.read_csv(kmer_csv_path, dtype={'chrom': 'str'}, sep=',', index_col=False)
    for cn in chroms:
        kmer_features = get_chrom_kmer_features(
            cn, dataset_path, gw_kmer_df, 128
        )
        np.save(os.path.join(dataset_path, 'node_features.{}.npy'.format(cn)),
                kmer_features.astype('float32'))
    gw_motif_df = pd.read_csv(motif_csv_path, dtype={'chrom': 'str'}, sep=',', index_col=False)
    for cn in chroms:
        motif_features = get_chrom_motif_features(
            cn, dataset_path, gw_motif_df, 128
        )
        np.save(
            os.path.join(dataset_path, 'motif_features.{}.npy'.format(cn)),
            motif_features.astype('float32')
        )
    print('=' * 10 + ' Finished generating node features ' + '=' * 10)


# if __name__ == '__main__':
#     cell_line_name = 'hela'
#     protein_name = 'ctcf'
#     chroms = [str(i) for i in range(1, 18)] + [str(i) for i in range(19, 23)] + ['X']
#     ratio = '100_100'
#     dataset_name = '_'.join([cell_line_name, protein_name, ratio])
#     dataset_path = os.path.join('dataset', dataset_name)
#
#     gw_kmer_df = pd.read_csv('data/kmer_hg38.csv', dtype={'chrom': 'str'}, sep=',', index_col=False)
#     for cn in chroms:
#         # print('Generating k-mer feature for chromosome {}...'.format(cn))
#         features = get_chrom_kmer_features(
#             cn, dataset_path, gw_kmer_df, 128
#         )
#         np.save(os.path.join(dataset_path, 'node_features.{}.npy'.format(cn)),
#                 features.astype('float32'))
#
#     gw_motif_df = pd.read_csv('data/fimo_hg38.csv', dtype={'chrom': 'str'}, sep=',', index_col=False)
#     for cn in chroms:
#         motif_features = get_chrom_motif_features(
#             cn, dataset_path, gw_motif_df, 128
#         )
#         np.save(
#             os.path.join(dataset_path, 'motif_features.{}.npy'.format(cn)),
#             motif_features.astype('float32')
#         )

