from .hic_exception import *
import numpy as np
from .utils import *
from abc import ABC, abstractmethod
from scipy.linalg import block_diag


class FlexHeaded(ABC):
    def __init__(self):
        self._filtered = False

    def is_filtered(self):
        return self._filtered


class SymmetricHeaded(ABC):
    def __init__(self, headers, **kwargs):
        self.headers = headers


class AsymmetricHeaded(ABC):
    pass


class FlexSymmetricHeaded(SymmetricHeaded, FlexHeaded, ABC):
    def __init__(self, headers, **kwargs):
        SymmetricHeaded.__init__(self, headers=headers)
        FlexHeaded.__init__(self)
        self._cropped_headers = None
        self._loci_existence = None

    def get_cropped_headers(self):
        if not self._filtered:
            raise NotFilteredException('The matrix has not yet been filtered.')
        return self._cropped_headers

    def set_cropped_headers(self, new_cropped_headers):
        assert len(self._cropped_headers) == len(new_cropped_headers)
        self._cropped_headers = new_cropped_headers

    def get_loci_existence_vector(self):
        if not self._filtered:
            raise NotFilteredException('The matrix has not yet been filtered.')
        return self._loci_existence


class FlexAsymmetricHeaded(AsymmetricHeaded, FlexHeaded, ABC):
    pass


class Filterable(FlexHeaded, ABC):
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def filter_by_nan_percentage(self, percentage):
        pass

    @abstractmethod
    def filter_expected_and_oe(self, rows_to_keep):
        pass