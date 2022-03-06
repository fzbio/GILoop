from .hic_exception import *
import numpy as np
from .utils import *
from abc import ABC, abstractmethod
from scipy.linalg import block_diag
from .interfaces import *


class BaseHicMatrix(ABC):
    def __init__(self, the_array, res):
        self._oe = False
        self.mat = the_array
        self.res = res
        # The following fields will be assigned after calling self.generate_oe()
        self._oe_mat = None
        self._expected_mat = None
        # The following field is a user-defined field, which allows users to save their own
        # matrix in this data structure.
        self._arbitrary_mat = {}

    def has_oe_mat(self):
        return self._oe

    def get_oe_mat(self):
        if not self._oe:
            raise NoOEException(
                'The OE matrix has not been generated. Try calling generate_oe to create it.'
            )
        return self._oe_mat

    def get_expected_mat(self):
        if not self._oe:
            raise NoOEException(
                'The expected matrix has not been generated. Try calling generate_oe to create it.'
            )
        return self._expected_mat

    def set_arbitrary_mat(self, name, arb_mat):
        self._arbitrary_mat[name] = arb_mat

    def get_arbitrary_mat(self, name):
        return self._arbitrary_mat[name]


class BaseSymmetricMatrix(BaseHicMatrix, FlexSymmetricHeaded, Filterable, ABC):
    def __init__(self, headers, the_array, res):
        BaseHicMatrix.__init__(self, the_array=the_array, res=res)
        FlexSymmetricHeaded.__init__(self, headers)
        Filterable.__init__(self)
    # def filter_by_reference(self, ref_dir_path, percentage):
    #     if self._filtered:
    #         raise AlreadyFilteredException('The matrix has already been filtered.')
    #     self._filtered = True
    #     rows_to_keep = self.headers.apply(lambda row: row, axis=1)

    def filter_by_nan_percentage(self, percentage):
        """
        The loci where more than percentage of entries are 0 or nan will be removed.
        """
        if self._filtered:
            raise AlreadyFilteredException('The matrix has already been filtered.')
        self._filtered = True
        # Remove rows
        rows_to_keep = np.sum(
            np.logical_or(np.isnan(self.mat), self.mat == 0), 1
        ).astype(float) / len(self.mat[0, :]) <= percentage
        self.mat = self.mat[rows_to_keep, :]
        self.mat = self.mat[:, rows_to_keep]
        if self._oe:
            self.filter_expected_and_oe(rows_to_keep)
        # Update headers
        self._cropped_headers = self.headers[rows_to_keep].copy().reset_index(drop=True)
        self._loci_existence = rows_to_keep

    def filter_expected_and_oe(self, rows_to_keep):
        self._oe_mat = self._oe_mat[rows_to_keep, :]
        self._oe_mat = self._oe_mat[:, rows_to_keep]
        self._expected_mat = self._expected_mat[rows_to_keep, :]
        self._expected_mat = self._expected_mat[:, rows_to_keep]

    def _calculate_chrom_expected_mat(self, region):
        for i in range(region.shape[0]):
            diag_indices = kth_diag_indices(region, i)
            dist_sum = region[diag_indices].sum()
            region[diag_indices] = dist_sum / diag_indices[0].shape[0]
        region = np.triu(region) + np.tril(region.T, -1)
        return region

    def generate_oe(self):
        if self._filtered:
            raise AlreadyFilteredException(
                'generate_oe should be called before filtering the matrix.'
            )
        if self._oe:
            raise AlreadyOEException('An OE field has already been existing.')
        self._generate_expected_mat()
        divider = self._expected_mat.copy()
        divider[divider == 0] = 1
        self._oe_mat = self.mat / divider
        self._oe = True

    @abstractmethod
    def _generate_expected_mat(self):
        pass


# TODO: Add a field indicating the type of the matrix (e.g., hic/cool/hicpro) and specify which types are allowed to
#  call generate_oe.
class GenomeWideMatrix(BaseSymmetricMatrix):
    def __init__(self, headers, gw_array, res):
        super(GenomeWideMatrix, self).__init__(headers, gw_array, res)

    def _generate_expected_mat(self):
        chrom_indices = self.headers['chrom'].unique()
        self._expected_mat = np.ones(self.mat.shape)
        for i in range(len(chrom_indices)):
            for j in range(len(chrom_indices)):
                if i != j:
                    self.__calculate_trans_expected(chrom_indices[i], chrom_indices[j])
                else:
                    self.__calculate_cis_expected(chrom_indices[i])

    def __calculate_cis_expected(self, chrom):
        """
        Directly modify self.__expected_mat in place.
        """
        cis_pos = np.asarray(self.headers.index[self.headers['chrom'] == chrom])
        assert cis_pos.dtype == 'int64'
        start_pos = cis_pos[0]
        end_pos = cis_pos[-1] + 1
        current_region = self.mat[start_pos:end_pos, start_pos:end_pos].copy()
        current_region = self._calculate_chrom_expected_mat(current_region)
        # assert check_symmetric(current_region)
        self._expected_mat[start_pos:end_pos, start_pos:end_pos] = current_region

    def __calculate_trans_expected(self, chrom1, chrom2):
        positions1 = np.asarray(self.headers.index[self.headers['chrom'] == chrom1])
        positions2 = np.asarray(self.headers.index[self.headers['chrom'] == chrom2])
        row_start = positions1[0]
        row_end = positions1[-1] + 1
        col_start = positions2[0]
        col_end = positions2[-1] + 1
        current_region = self.mat[row_start:row_end, col_start:col_end].copy()
        region_mean = np.mean(current_region)
        if region_mean == 0:
            region_mean = 1
        current_region = region_mean
        self._expected_mat[row_start:row_end, col_start:col_end] = current_region


class CisMatrix(BaseSymmetricMatrix):
    def __init__(self, headers, cis_array, res):
        super(CisMatrix, self).__init__(headers, cis_array, res)

    def _generate_expected_mat(self):
        self._expected_mat = np.ones(self.mat.shape)
        current_region = self.mat.copy()
        self._expected_mat = self._calculate_chrom_expected_mat(current_region)


class DiagonalBlockMatrix(BaseHicMatrix, FlexSymmetricHeaded):
    def __init__(self, cis_matrices):
        BaseHicMatrix.__init__(self, the_array=None, res=None)
        FlexSymmetricHeaded.__init__(self, None)
        self.cis_matrices = cis_matrices
        self.chrom_names = [
            matrix.headers['chrom'].unique()[0] for matrix in self.cis_matrices
        ]
        self._check_matrices_homogeneous()
        self._deduplicate_names()
        # super(DiagonalBlockMatrix, self).__init__(None, None)
        self._set_container_meta()

    def _set_container_meta(self):
        self._filtered = self.cis_matrices[0].is_filtered()
        self._oe = self.cis_matrices[0].has_oe_mat()
        self.res = self.cis_matrices[0].res
        self.mat = block_diag(*[matrix.mat for matrix in self.cis_matrices])
        if self._oe:
            self._oe_mat = block_diag(*[m.get_oe_mat() for m in self.cis_matrices])
            self._expected_mat = block_diag(*[
                m.get_expected_mat() for m in self.cis_matrices
            ])
        self.headers = pd.concat(
            [m.headers for m in self.cis_matrices], ignore_index=True
        )
        if self._filtered:
            self._cropped_headers = pd.concat(
                [m.get_cropped_headers() for m in self.cis_matrices], ignore_index=True
            )
            self._loci_existence = np.concatenate(
                [m.get_loci_existence_vector() for m in self.cis_matrices]
            )

    def _check_matrices_homogeneous(self):
        for matrix in self.cis_matrices:
            assert isinstance(matrix, CisMatrix)
            assert matrix.is_filtered() == self.cis_matrices[0].is_filtered()
            assert matrix.has_oe_mat() == self.cis_matrices[0].has_oe_mat()
            assert matrix.res == self.cis_matrices[0].res

    def _deduplicate_names(self):
        name_dict = {}
        for i, name in enumerate(self.chrom_names):
            if name not in name_dict:
                name_dict[name] = 1
            else:
                self._change_matrix_chrom_name(i, name_dict[name])
                name_dict[name] += 1

    def _change_matrix_chrom_name(self, matrix_idx, dup_time):
        new_name = self.chrom_names[matrix_idx] + '_' + str(dup_time)
        self.chrom_names[matrix_idx] = new_name
        self.cis_matrices[matrix_idx].headers['chrom'] = new_name
        if self.is_filtered():
            cropped_headers = \
                self.cis_matrices[matrix_idx].get_cropped_headers()
            cropped_headers['chrom'] = new_name
            self.cis_matrices[matrix_idx].set_cropped_headers(cropped_headers)

    def get_chrom_names(self):
        return self.chrom_names

    def get_matrix_by_name(self, name):
        return self.cis_matrices[self.chrom_names.index(name)]

    def get_matrix_by_index(self, index):
        return self.cis_matrices[index]

    def co_filter_matrices(self):
        pass
