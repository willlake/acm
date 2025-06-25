from .data import BaseDataObservable
from .model import BaseModelObservable

class Observable(BaseDataObservable, BaseModelObservable):
    """
    Default class to handle the loading and filtering of the data and models in the ACM pipeline.
    """
    # Redefine common methods that exist in the two classes
    @property
    def unfiltered_bin_values(self):
        """
        Unfiltered bin values for the statistic. (e.g. separation bins for the correlation function)
        """
        return super().unfiltered_bin_values # Taken from BaseDataObservable because of inheritance order
    
    @property
    def bin_values(self):
        """
        Bin values for the statistic, with filters applied. (e.g. separation bins for the correlation function)
        """
        return super().bin_values # Taken from BaseDataObservable because of inheritance order