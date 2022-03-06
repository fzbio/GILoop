from abc import ABC, abstractmethod
from .utils import *
import numpy as np
import pandas as pd
from .matrix import GenomeWideMatrix, CisMatrix


def get_headers(chrom_indices, chrom_sizes, res):
    headers = []
    for chrom in chrom_indices:
        for bin_start in range(int(chrom_sizes[chrom] / res + 1)):
            headers.append([chrom, bin_start])
    return pd.DataFrame(headers, columns=['chrom', 'locus'])


def construct_symmetric_array(chrom_indices, chrom_sizes, res):
    loci_count = 0
    for i in chrom_indices:
        loci_count += int(chrom_sizes[i] / res) + 1
    return np.zeros((loci_count, loci_count))


class BasicReader(ABC):
    def __init__(self, data_path, res, chromsize):
        self.data_path = data_path
        self.res = res
        if isinstance(chromsize, str):
            self.chrom_sizes = get_chrom_sizes(chromsize)
        elif isinstance(chromsize, dict):
            self.chrom_sizes = chromsize
        else:
            raise NotImplementedError('Only string and dict are supported for chromsize.')

    @abstractmethod
    def get_whole_genome_matrix(self, chrom_indices):
        pass

    @abstractmethod
    def get_region_matrix(self, chrom1, chrom2):
        pass


class ProMatrixReader(BasicReader):
    def __init__(self, data_path, res, chromsize, annotation_path):
        super().__init__(data_path, res, chromsize)
        self.annotation_path = annotation_path
        self.bin_annotation = self.read_bin_annotation()
        self.data = self.read_data_from_file()

    def get_whole_genome_matrix(self, chrom_indices):
        return self._get_symmetric_matrix(chrom_indices)

    def get_region_matrix(self, chrom1, chrom2):
        if chrom1 != chrom2:
            return self._get_trans_matrix(chrom1, chrom2)
        else:
            return self._get_symmetric_matrix([chrom1])

    def _get_symmetric_matrix(self, chrom_indices):
        sym_array = construct_symmetric_array(
            chrom_indices, self.chrom_sizes, self.res
        )
        converted_data = self.convert_idx_to_pos(chrom_indices)
        sym_array[converted_data['loc1'].values, converted_data['loc2'].values] = \
            converted_data['value'].values
        sym_array = np.triu(sym_array) + np.tril(sym_array.T, 1)
        headers = get_headers(chrom_indices, self.chrom_sizes, self.res)
        if len(chrom_indices) > 1:
            return GenomeWideMatrix(headers, sym_array, self.res)
        elif len(chrom_indices) == 1:
            return CisMatrix(headers, sym_array, self.res)
        else:
            raise NotImplementedError('Invalid chrom list.')

    def _get_trans_matrix(self, chrom1, chrom2):
        pass

    def convert_idx_to_pos(self, chrom_indices):
        """
        Convert the original HiCPro indices to the positions of the desired matrix
        matrix. Delete the entries that do not exist in the target region.
        :return: The converted dataframe.
        """
        data_df = self.data.copy()
        annotation_df = self.bin_annotation.copy()
        annotation_df = annotation_df.loc[annotation_df['chrom'].isin(chrom_indices)]
        annotation_df['pos'] = annotation_df.apply(
            self.ann2pos_factory(chrom_indices), axis=1
        )
        annotation_arr = annotation_df.drop(columns=['chrom']).values
        annotation_dict = {ann[2]: ann[3] for ann in annotation_arr}
        data_df = data_df[
            (data_df['loc1'].isin(annotation_dict.keys())) &
            (data_df['loc2'].isin(annotation_dict.keys()))
        ]
        loc_arr = data_df[['loc1', 'loc2']].values
        assert loc_arr.dtype == 'int64'
        for i in range(loc_arr.shape[0]):
            loc_arr[i, 0] = annotation_dict[loc_arr[i, 0]]
            loc_arr[i, 1] = annotation_dict[loc_arr[i, 1]]
        data_df[['loc1', 'loc2']] = loc_arr
        return data_df

    def ann2pos_factory(self, chrom_indices):
        def ann2pos(row):
            pos = 0
            for chr in chrom_indices:
                if chr != row['chrom']:
                    pos += (int(self.chrom_sizes[chr] / self.res) + 1)
                else:
                    pos += int(row['start'] / self.res)
                    break
            return pos
        return ann2pos

    def read_bin_annotation(self):
        annotation = pd.read_csv(
            self.annotation_path, sep="\t",
            names=['chrom', 'start', 'end', 'index']
        )
        annotation['chrom'] = annotation['chrom'].map(lambda x: x[3:])
        return annotation

    def read_data_from_file(self):
        data = pd.read_csv(
            self.data_path,
            sep='\t', names=['loc1', 'loc2', 'value']
        )
        return data
