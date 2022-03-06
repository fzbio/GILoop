import os.path

import numpy as np
import pandas as pd

import gutils
from gutils import get_raw_graph, block_sampling, parsebed
import random


def get_segment_count(cropped_header_length, patch_size):
    if cropped_header_length % patch_size != 0:
        return int(cropped_header_length / patch_size) + 1
    else:
        return int(cropped_header_length / patch_size)


def get_patches_different_downsampling_rate(chrom_name, patch_size, graph_txt_dir, image_txt_dir, resolution,
                                            chrom_sizes_path, bedpe_list):
    image_matrix = get_raw_graph(chrom_name, image_txt_dir, resolution, chrom_sizes_path, filter_by_nan=False)
    graph_matrix = get_raw_graph(chrom_name, graph_txt_dir, resolution, chrom_sizes_path, filter_by_nan=False)

    graph_matrix.filter_by_nan_percentage(0.9999)
    unified_cropped_headers = graph_matrix.get_cropped_headers()
    unified_loci_existence = graph_matrix.get_loci_existence_vector()
    image_matrix.mat = image_matrix.mat[unified_loci_existence, :][:, unified_loci_existence]
    image_matrix._filtered = True
    image_matrix._cropped_headers = unified_cropped_headers
    image_matrix._loci_existence = unified_loci_existence

    segment_count = get_segment_count(len(image_matrix.get_cropped_headers()), patch_size)
    start_tuples = get_start_tuples(segment_count, patch_size, resolution)

    image_set = np.zeros((len(start_tuples), patch_size, patch_size), dtype='float32')
    graph_set = np.zeros((len(start_tuples), 2*patch_size, 2*patch_size), dtype='float32')
    labels = np.zeros((len(start_tuples), patch_size, patch_size), dtype='bool')
    indicators = []
    print('Chromosome {}'.format(chrom_name))
    for i, tuple in enumerate(start_tuples):
        print('Sampling... {}/{}'.format(i+1, len(start_tuples)))
        if tuple[0] == tuple[1]:
            g, l, p = block_sampling(graph_matrix, (tuple[0],), patch_size, bedpe_list)
            graph = np.zeros((patch_size*2, patch_size*2))
            graph[:patch_size, :patch_size] = g
            graph[patch_size:, patch_size:] = g
            graph[:patch_size, patch_size:] = g
            graph[patch_size:, :patch_size] = g
            graph_set[i, :, :] = graph
            g, _, p = block_sampling(image_matrix, (tuple[0],), patch_size, bedpe_list)
            image_set[i, :, :] = g
            labels[i, :, :] = l
            p = gutils.autofill_indicators([p], patch_size)[0]
            p = pd.concat([p, p])
        else:
            graph_set[i, :, :], l, p = block_sampling(graph_matrix, tuple, patch_size, bedpe_list)
            g, _, p = block_sampling(image_matrix, tuple, patch_size, bedpe_list)
            image_set[i, :, :] = g[:patch_size, patch_size:]
            labels[i, :, :] = l[:patch_size, patch_size:]
            p = gutils.autofill_indicators([p], 2 * patch_size)[0]
        assert len(p) == 2 * patch_size
        indicators.append(p)
    indicators = pd.concat(indicators)
    for i, graph in enumerate(graph_set):
        graph_set[i, :patch_size, :patch_size] = 0
        graph_set[i, patch_size:, patch_size:] = 0
    assert len(indicators) == len(graph_set) * 2 * patch_size
    return image_set, graph_set, labels, indicators


def get_boolean_graph_property(chrom_name, patch_size, txt_dir, resolution, chrom_sizes_path):
    print('Chromosome {}'.format(chrom_name))
    matrix = get_raw_graph(chrom_name, txt_dir, resolution, chrom_sizes_path)
    segment_count = get_segment_count(len(matrix.get_cropped_headers()), patch_size)
    # segment_count = int(len(matrix.get_cropped_headers())/patch_size) + 1
    start_tuples = get_start_tuples(segment_count, patch_size, resolution)
    graph_nodes_identical = np.zeros((len(start_tuples),))
    for i, tup in enumerate(start_tuples):
        if tup[0] == tup[1]:
            graph_nodes_identical[i] = 1
    return graph_nodes_identical


def get_start_tuples(segment_count, patch_size, resolution):
    start_tuples = []
    for i in range(segment_count):
        right_edge = min(segment_count, int(i+(2000000/resolution/patch_size))+2)
        for j in range(i, right_edge):
            start_tuples.append((i*patch_size, j*patch_size))
    return start_tuples


def run_sample_patches(
        dataset_name, assembly, bedpe_path,
        image_txt_dir, graph_txt_dir,
        chroms
):
    dataset_path = os.path.join('dataset', dataset_name)
    RES = 10000
    chrom_size_path = '{}.chrom.sizes'.format(assembly)
    bedpe_list = parsebed(bedpe_path, valid_threshold=1)
    os.makedirs(dataset_path, exist_ok=False)
    for cn in chroms:
        image_set, graph_set, labels, indicators = \
            get_patches_different_downsampling_rate(
                cn, 64, graph_txt_dir, image_txt_dir, RES, chrom_size_path,
                bedpe_list
            )
        indicators.to_csv(os.path.join(dataset_path, 'indicators.{}.csv'.format(cn)))
        np.save(os.path.join(dataset_path, 'imageset.{}.npy'.format(cn)), image_set.astype('float32'))
        np.save(os.path.join(dataset_path, 'graphset.{}.npy'.format(cn)), graph_set.astype('float32'))
        np.save(os.path.join(dataset_path, 'labels.{}.npy'.format(cn)), labels.astype('int'))

        graph_nodes_identical = np.zeros((len(graph_set),), dtype='bool')
        for idx in range(len(graph_set)):
            if indicators.iloc[idx * (2 * 64)]['locus'] == indicators.iloc[idx * (2 *64) + 64]['locus']:
                graph_nodes_identical[idx] = True
        np.save(os.path.join(dataset_path, 'graph_identical.{}.npy'.format(cn)), graph_nodes_identical.astype('int'))

