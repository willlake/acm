from .base import BaseObservable
import logging


class DTVoidGalaxyCorrelationFunctionMultipoles(BaseObservable):
    """
    Class for the Emulator's Mock Challenge void-galaxy correlation
    function multipoles using the DT void finder.
    """
    def __init__(self, phase_correction=False, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stat_name = 'dt_gv_may8'
        self.sep_name = 's'

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
            'hod_idx': list(range(100)),
        }

    @property
    def test_set_indices(self):
        """
        Indices of the test set samples, including variations in cosmology and HOD parameters.
        """
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)),
            'hod_idx': list(range(100)),
        }

    @property
    def small_box_indices(self):
        """
        Indices of the covariance samples, including variations in phase and HOD parameters.
        """
        return {
            # 'phase_idx': list(range(1786)),
            'phase_idx': [i for i in list(range(1786)) if i not in [3319, 3325, 3377, 3569, 3679, 3879, 3902, 4206, 4261, 4300, 4326, 4334, 4416, 4556, 4558, 4650]],
        }

    @property
    def coordinates(self):
        """
        Coordinates of the data and model vectors.
        """
        return{
            'multipoles': [0, 2],
            's': self.separation,
        }
    
    @property
    def coordinates_indices(self):
        """
        Indices of the (flat) coordinates of the data and model vectors.
        """
        return{'bin_idx': list(range(2 * len(self.separation)))}

    @property
    def model_fn(self):
        return '/pscratch/sd/d/dforero/emc/trained_models/DTvoids_gv/cosmo+hod/optuna/last.ckpt'

    def create_lhc(self, n_hod=20, cosmos=None, phase_idx=0, seed_idx=0):
        """
        Create the Latin hypercube samples for the emulator (both input and output features).
        """
        x, x_names = self.create_lhc_x(cosmos=cosmos, n_hod=n_hod)
        sep, y = self.create_lhc_y(n_hod=n_hod, cosmos=cosmos, phase_idx=phase_idx, seed_idx=seed_idx)
        return sep, x, x_names, y

    def create_lhc_y(self, n_hod=100, cosmos=None, phase_idx=0, seed_idx=0):
        """
        Create the output features for the emulator (the galaxy correlation function multipoles).
        """
        import numpy as np
        from pycorr import TwoPointCorrelationFunction
        base_dir = f'/pscratch/sd/d/dforero/projects/ac_emc/data/training_sets/dtfe6/tpcf/z0.5/yuan23_prior2/'
        if cosmos is None:
            cosmos = list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))
        y = []
        for cosmo_idx in cosmos:
            print(cosmo_idx)
            data_dir = base_dir + f'c{cosmo_idx:03}_ph{phase_idx:03}/seed0/'
            for hod_idx in range(n_hod):
                data_fn = f"{data_dir}/gv_tpcf_hod{hod_idx:03d}.npy"
                data = TwoPointCorrelationFunction.load(data_fn)[::4].select((0, 150))
                s, multipoles = data(ells=(0, 2), return_sep=True)
                y.append(np.concatenate(multipoles))
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
        from pycorr import TwoPointCorrelationFunction
        import numpy as np
        import glob
        y = []
        data_dir = f'/pscratch/sd/d/dforero/projects/ac_emc/data/training_sets/dtfe6/tpcf/z0.5/full_prior2/small_onelos/'
        data_fns = glob.glob(f"{data_dir}/gv_tpcf_ph*.npy")
        for data_fn in data_fns:
            data = TwoPointCorrelationFunction.load(data_fn)[::4].select((0, 150))
            s, multipoles = data(ells=(0, 2), return_sep=True)
            y.append(np.concatenate(multipoles))
        return s, np.array(y)