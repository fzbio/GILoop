import re
import numpy as np
import pandas as pd


def get_chrom_sizes(file_path):
    sizes = {}
    with open(file_path, 'r') as fp:
        for line in fp:
            line_split = line.split()
            line_split[0] = line_split[0][3:]
            sizes[line_split[0]] = int(line_split[1])
    return sizes


def save_matrix(arr, path_to_save):
    np.save(path_to_save, arr)


# def get_reference_locus_seq(ref_dir, chrom, locus, res):
#     """
#     Read reference sequence of a certain locus. The reference must be saved
#     chromosome-wise.
#
#     Args:
#         ref_dir: str
#             The path of the reference directory.
#         chrom: str
#             The chromosome index. e.g., '1', '2', '3', 'X'
#         locus: int
#             The starting location of the locus. e.g., 0, 100000, 200000
#         res: int
#             The desired resolution of Hi-C data.
#
#     Returns:
#         seq: str
#             The sequence of the specified locus. All digits are uppercase.
#     """
#     seq = ''
#
#     return seq


def calculate_nan_percent_in_seq(seq):
    count = 0
    for c in seq:
        if c == 'N':
            count += 1
    return count / len(seq)


def kth_diag_indices(a, k):
    if k == 0:
        return np.diag_indices_from(a)
    rowidx, colidx = np.diag_indices_from(a)
    colidx = colidx.copy()  # rowidx and colidx share the same buffer
    if k > 0:
        colidx += k
    else:
        rowidx -= k
    k = np.abs(k)
    return rowidx[:-k], colidx[:-k]


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def full2cropped(full_df, loci_existence_vec):
    cropped_df = full_df.copy()[loci_existence_vec]
    return cropped_df


def cropped2full(cropped_df, full_headers, absent_loci_placeholders):
    full_df = full_headers.copy()
    cropped_headers_tuples = [_[1:] for _ in cropped_df[['chrom', 'locus']].itertuples(name=None)]
    for i, col in enumerate(cropped_df.columns[2:]):
        full_vec = np.full((len(full_headers),), absent_loci_placeholders[i])
        for j, row in enumerate(full_headers.itertuples(name=None)):
            row = row[1:]
            if row in cropped_headers_tuples:
                # print(row)
                full_vec[j] = cropped_df[col].tolist()[cropped_headers_tuples.index(row)]
        full_df[col] = full_vec
    return full_df

# def construct_wg_annotation(any_df_list, loci_existence_vecs, absent_loci=-1):
#     for i, df in enumerate(any_df_list):
#
#     wg_df = pd.concat(any_df_list, ignore_index=True)


def output_to_bed(any_df, output_path):
    any_df.to_csv(path_or_buf=output_path, sep='\t')


if __name__ == '__main__':
    from hickit import reader
    data = {'chrom': ['1', '1', '1', '1'], 'locus': [0, 2, 3, 4], 'test1': [1, 2, 1, 3], 'test2': ['y', 'n', 'n', 'y']}
    test_df = pd.DataFrame.from_dict(data)
    headers = reader.get_headers(['1'], get_chrom_sizes('data/hg19.chrom.sizes'), 100000)
    fulldf = cropped2full(test_df, headers, absent_loci_placeholders=[-1, None])
    print(fulldf)