from .base import BaseObservable
from sunbird.data.data_utils import convert_to_summary
import logging


class MinkowskiFunctionals(BaseObservable):
    """
    Class for the Emulator's Mock Challenge Minkowski functionals.
    """
    def __init__(self, phase_correction=False, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stat_name = 'minkowski'
        self.sep_name = 'delta'

        if phase_correction and hasattr(self, 'compute_phase_correction'):
            self.logger.info('Computing phase correction.')
            self.phase_correction = self.compute_phase_correction()

        super().__init__(**kwargs)

    @property
    def lhc_indices(self):
        """
        Indices of the Latin hypercube samples, including variations in cosmology and HOD parameters.
        """
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(200)),
        }

    @property
    def test_set_indices(self):
        """
        Indices of the test set samples, including variations in cosmology and HOD parameters.
        """
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)),
            'hod_idx': list(range(200)),
        }


    @property
    def coordinates(self):
        """
        Coordinates of the data and model vectors.
        """
        return{
            'delta': self.separation,
        }
    
    @property
    def coordinates_indices(self):
        """
        Indices of the (flat) coordinates of the data and model vectors.
        """
        return{'bin_idx': list(range(len(self.separation)))}


    @property
    def model_fn(self):
        # return f'/pscratch/sd/e/epaillas/emc/v1.1/trained_models/minkowski/cosmo+hod/best-model-epoch=132-val_loss=0.0319.ckpt'
        return f'/pscratch/sd/e/epaillas/emc/v1.1/trained_models/best/MinkowskiFunctionals/last.ckpt'

    def get_emulator_error(self, select_filters=None, slice_filters=None):
        from sunbird.data.data_utils import convert_to_summary
        from pathlib import Path
        import numpy as np
        error_dir = '/pscratch/sd/e/epaillas/emc/v1.1/emulator_error/'
        error_fn = Path(error_dir) / 'minkowski_apr11.npy'
        error = np.load(error_fn, allow_pickle=True).item()['emulator_error']
        coords = self.coordinates_indices if self.select_indices else self.coordinates
        coords_shape = tuple(len(v) for k, v in coords.items())
        dimensions = list(coords.keys())
        error = error.reshape(*coords_shape)
        select_filters = self.select_coordinates if self.select_coordinates else self.select_indices
        slice_filters = self.slice_coordinates
        return convert_to_summary(
            data=error, dimensions=dimensions, coords=coords,
            select_filters=select_filters, slice_filters=slice_filters
        ).values.reshape(-1)

    def create_lhc(self, n_hod=20, cosmos=None, phase_idx=0, seed_idx=0):
        """
        Create the Latin hypercube samples for the emulator (both input and output features).
        """
        x, x_names = self.create_lhc_x(cosmos=cosmos, n_hod=n_hod)
        sep, y = self.create_lhc_y(n_hod=n_hod, cosmos=cosmos, phase_idx=phase_idx, seed_idx=seed_idx)
        return sep, x, x_names, y

    def create_lhc_y(self, n_hod=100, cosmos=None, phase_idx=0, seed_idx=0):
        """
        Create the output features for the emulator.
        """
        import numpy as np
        from pathlib import Path
        if cosmos is None:
            cosmos = list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))
        y = []
        data_dir = '/pscratch/sd/e/epaillas/emc/v1.1/abacus/training_sets/cosmo+hod/raw/'
        data_fn = Path(data_dir) / 'Minkowski_Combine_4Rgs_85cos_lhc.npy'
        data = np.load(data_fn, allow_pickle=True).item()
        s = data['delta']
        for cosmo_idx in cosmos:
            for hod_idx in range(n_hod):
                if hod_idx in data[f'c{cosmo_idx:03}_ph000_index']:
                    where = np.where(data[f'c{cosmo_idx:03}_ph000_index'] == hod_idx)[0][0]
                    y.append(data[f'c{cosmo_idx:03}_ph000_y'][where])
                else:
                    y.append(np.zeros_like(data['lhc_train_y'][0]))
        return s, np.array(y)

    def create_lhc_x(self, cosmos=None, n_hod=100):
        """
        Create the input features for the emulator (the cosmological and HOD parameters).
        """
        import pandas
        import numpy as np
        if cosmos is None:
            cosmos = list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))
        lhc_x = []
        for cosmo_idx in cosmos:
            data_dir = '/pscratch/sd/e/epaillas/emc/cosmo+hod_params/'
            data_fn = data_dir + f'AbacusSummit_c{cosmo_idx:03}.csv'
            lhc_x_i = pandas.read_csv(data_fn)
            lhc_x_names = list(lhc_x_i.columns)
            lhc_x_names = [name.replace(' ', '').replace('#', '') for name in lhc_x_names]
            lhc_x.append(lhc_x_i.values[:n_hod, :])
        lhc_x = np.concatenate(lhc_x)
        return lhc_x, lhc_x_names

    def create_small_box_y(self):
        """
        Create the output features for the emulator (the galaxy correlation function multipoles)
        from the small AbacusSummit box.
        """
        from pathlib import Path
        import numpy as np
        data_dir = '/pscratch/sd/e/epaillas/emc/v1.1/abacus/training_sets/cosmo+hod/raw/'
        data_fn = Path(data_dir) / 'Minkowski_Combine_4Rgs_85cos_lhc.npy'
        data = np.load(data_fn, allow_pickle=True).item()
        y = data['y_cov']
        s = data['delta']
        return s, np.array(y)