import scipy.io
import numpy as np
from ReadMeta import ReadMeta


class ReadData:
    def __init__(self, datadir, DF_neu):
        self.datadir = datadir
        self.DF_neu = DF_neu

    def get_data(self):

        n_images = 640 + 2560 + 2560
        n_neurons = 168 + 128
        n_trials = 29 + 51 + 47

        # 2017.08.16_hvmdata_mats
        Data_image = np.zeros((n_neurons, 39, n_images))
        Data_trial_V0 = np.zeros((n_neurons, 39, 640, 29))
        Data_trial_V3 = np.zeros((n_neurons, 39, 2560, 51))
        Data_trial_V6 = np.zeros((n_neurons, 39, 2560, 47))

        for b in range(39):
            fname = 'hvm_allrep_t=%02d.mat' % b
            mat = scipy.io.loadmat(self.datadir + '/2017.08.16_hvmdata_mats/' + fname)
            v0 = np.mean(mat['repdata'][0][0][0], 0)
            v1 = np.mean(mat['repdata'][0][0][1], 0)
            v2 = np.mean(mat['repdata'][0][0][2], 0)

            Data_image[:, b, :] = np.concatenate((v0, v1, v2)).T

            v0 = mat['repdata'][0][0][0]
            v1 = mat['repdata'][0][0][1]
            v2 = mat['repdata'][0][0][2]

            Data_trial_V0[:, b, :, :] = np.swapaxes(v0, 0, 2)
            Data_trial_V3[:, b, :, :] = np.swapaxes(v1, 0, 2)
            Data_trial_V6[:, b, :, :] = np.swapaxes(v2, 0, 2)

        IT_trial_V0 = Data_trial_V0[np.where(self.DF_neu['region'] == 'IT')[0], :, :, :]
        IT_trial_V3 = Data_trial_V3[np.where(self.DF_neu['region'] == 'IT')[0], :, :, :]
        IT_trial_V6 = Data_trial_V6[np.where(self.DF_neu['region'] == 'IT')[0], :, :, :]

        V4_trial_V0 = Data_trial_V0[np.where(self.DF_neu['region'] == 'V4')[0], :, :, :]
        V4_trial_V3 = Data_trial_V3[np.where(self.DF_neu['region'] == 'V4')[0], :, :, :]
        V4_trial_V6 = Data_trial_V6[np.where(self.DF_neu['region'] == 'V4')[0], :, :, :]

        return [IT_trial_V0, IT_trial_V3, IT_trial_V6], [V4_trial_V0, V4_trial_V3, V4_trial_V6]