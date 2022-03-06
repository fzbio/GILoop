from collections import defaultdict, Counter
import numpy as np
import struct
import io
import os
from subprocess import run
import pandas as pd
from hickit.reader import get_chrom_sizes, get_headers
from hickit.matrix import CisMatrix


def parsebed(chiafile, res=10000, lower=1, upper=5000000, valid_threshold=1):
    """
    Read the ChIA-PET bed file and generate a distionary of positive center points.
    """
    coords = defaultdict(list)
    upper = upper // res
    with open(chiafile) as o:
        for line in o:
            s = line.rstrip().split()
            a, b = float(s[1]), float(s[4])
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            a //= res
            b //= res
            # all chromosomes including X and Y
            if (b - a > lower) and (b - a < upper) and 'M' not in s[0]:
                # always has prefix "chr", avoid potential bugs
                chrom = 'chr' + s[0].lstrip('chr')
                coords[chrom].append((a, b))
    valid_coords = dict()
    for c in coords:
        current_set = set(coords[c])
        valid_set = set()
        for coord in current_set:
            if coords[c].count(coord) >= valid_threshold:
                valid_set.add(coord)
        valid_coords[c] = valid_set
    return valid_coords


def autofill_indicators(indicators, full_size):
    for i, indicator in enumerate(indicators):
        if len(indicator) < full_size:
            padded_size = full_size - len(indicator)
            padder = {
                'chrom': [indicator['chrom'].unique()[0]] * padded_size,
                'locus': [-1] * padded_size
            }
            padder = pd.DataFrame(padder)
            padded_indicator = pd.concat([indicator, padder]).reset_index(drop=True)
            indicators[i] = padded_indicator
    return indicators


def block_sampling(matrix, starts_tuple, continuous_len, bedpe_list, filter=True):
    full_size = matrix.mat.shape[0]
    sampled_indices = []
    regions_tuples = []
    for start in starts_tuple:
        if start + continuous_len <= full_size:
            sampled_indices += list(range(start, start + continuous_len))
            regions_tuples.append((start, start + continuous_len))
        else:
            sampled_indices += list(range(start, full_size))
            regions_tuples.append((start, full_size))
    subgraph = matrix.mat[sampled_indices, :][:, sampled_indices]
    if filter:
        position_indicator = matrix.get_cropped_headers().copy().reset_index(drop=True)
    else:
        position_indicator = matrix.headers.copy().reset_index(drop=True)
    position_indicator = position_indicator.iloc[sampled_indices].reset_index(drop=True)
    label = get_label_for_continuous_subgraph(
        position_indicator, bedpe_list, continuous_len,
        pd.unique(matrix.headers['chrom'])[0]
    )
    if subgraph.shape[0] < len(starts_tuple) * continuous_len:
        subgraph, label = padding(subgraph, label, len(starts_tuple) * continuous_len)
    return subgraph.astype('float32'), label.astype('bool'), position_indicator


def get_label_for_continuous_subgraph(position_indicator, bedpe_list, continuous_len, chrom_name):
    """
    Generate label matrix and delete the marked label points from the bedpe list.

    :param position_indicator: DataFrame - Marginal headers of length [1, continuous_len] for a symmetric matrix;
    or headers of length [continuous_len+1, 2*continuous_len] for both margins of an unsymmetric one
    :param bedpe_list: list of tuple -
    :param continuous_len: int - I.e., the image size
    :param chrom_name: str - E.g., '1', '2'
    :return: Unpadded label matrix of shape (len(position_indicator), len(position_indicator))
    """
    position_indicator = position_indicator[position_indicator['locus'] >= 0]
    subgraph_size = len(position_indicator)
    label = np.zeros((subgraph_size, subgraph_size), dtype='bool')
    current_chrom = 'chr' + chrom_name
    current_set = bedpe_list[current_chrom] if current_chrom in bedpe_list else {}
    edge_list_row = []
    edge_list_col = []
    if len(position_indicator) > continuous_len:
        row_locus_span = (position_indicator['locus'].iloc[0], position_indicator['locus'].iloc[continuous_len-1] + 1)
        col_locus_span = (position_indicator['locus'].iloc[continuous_len], position_indicator['locus'].iloc[-1] + 1)
    else:
        row_locus_span = (position_indicator['locus'].iloc[0], position_indicator['locus'].iloc[-1] + 1)
        col_locus_span = row_locus_span
    region = [row_locus_span, col_locus_span]
    marked_truth = []
    for truth in current_set:
        if is_entry_in_genomic_region(truth, region):
            if is_entry_valid_in_cropped_map(truth, position_indicator):
                edge_list_row.append(list(position_indicator['locus']).index(truth[0]))
                edge_list_col.append(list(position_indicator['locus']).index(truth[1]))
            marked_truth.append(truth)
    for marked in marked_truth:
        current_set.remove(marked)
    label[edge_list_row, edge_list_col] = True
    label = np.triu(label) + np.tril(label.T, 1)
    return label.astype('bool')


def is_entry_in_genomic_region(entry, genomic_region):
    if (genomic_region[0][0] <= entry[0] < genomic_region[0][1]) and \
            (genomic_region[1][0] <= entry[1] < genomic_region[1][1]):
        return True
    else:
        return False


def is_entry_valid_in_cropped_map(entry, position_indicator):
    if entry[0] in list(position_indicator['locus']) and entry[1] in list(position_indicator['locus']):
        return True
    else:
        return False


def is_ascent_order(the_list):
    the_max = None
    for item in the_list:
        if the_max is None:
            the_max = item
        elif item > the_max:
            the_max = item
        else:
            return False
    return True


def padding(subgraph, label, subgraph_size):
    padding_len = subgraph_size - subgraph.shape[0]
    subgraph = np.pad(subgraph, [(0, padding_len), (0, padding_len)], mode='constant')
    label = np.pad(label, [(0, padding_len), (0, padding_len)], mode='constant')
    return subgraph, label


def get_raw_graph(chrom_name, txt_dir, resolution, chrom_sizes_path, filter_by_nan=True):
    print("Generating the chromosome {}".format(chrom_name))
    X = create_interaction_matrix(chrom_name, txt_dir, chrom_sizes_path)
    A = np.asarray(X)
    headers = get_headers([chrom_name], get_chrom_sizes(chrom_sizes_path), resolution)
    matrix = CisMatrix(headers[headers['chrom'] == chrom_name], A, resolution)
    if filter_by_nan:
        matrix.filter_by_nan_percentage(0.9999)
    return matrix


def initialise_mat(chr_index, resolution, chrom_size_path):
    chrom_sizes = get_chrom_sizes(chrom_size_path)
    nloci = int((chrom_sizes[chr_index] / resolution) + 1)
    mat = np.zeros((nloci, nloci))
    return mat


def read_txt_data(txt_dir, i):
    path_to_txt = os.path.join(txt_dir, 'chr{}.contact.txt'.format(i))
    txt_data = pd.read_csv(path_to_txt, sep='\t', header=None).values
    return txt_data


def create_interaction_matrix(chr_index, txt_dir, chrom_size_path, resolution=10000):
    txt_data = read_txt_data(txt_dir, chr_index)
    mat = initialise_mat(chr_index, resolution, chrom_size_path)
    rows = (txt_data[:, 0] / resolution).astype(int)
    cols = (txt_data[:, 1] / resolution).astype(int)
    data = txt_data[:, 2]
    mat[rows, cols] = data

    mat = np.triu(mat) + np.tril(mat.T, 1)
    # re-scale KR matrix to ICE-matrix range

    return mat


def hic_to_intra_txt(juicer_path, file_path, out_dir, chrom, norm='KR', resolution=10000, hic_type='oe'):
    """
    This function extract information from the .hic file and
    dump it to .txt files.
    """
    print('Start extracting matrix from the hic file...')
    print('Extracting intra-chromosomal interactions from chr {} ...'.format(chrom))
    out_path = os.path.join(out_dir, 'chr{}.contact.txt'.format(chrom))
    cmd = [
        'java', '-jar', juicer_path, 'dump', hic_type,
        norm, file_path, chrom, chrom, 'BP', str(resolution), out_path
    ]
    run(cmd, shell=False)


if __name__ == '__main__':
    a = parsebed('bedpe/gm12878.tang.ctcf-chiapet.hg19.bedpe', valid_threshold=1)
    b = parsebed('bedpe/gm12878.tang.ctcf-chiapet.hg19.bedpe', valid_threshold=3)
    print(a['chr1'])
    print(len(b['chr1']))
