

resultdir = '/home/tahereh/Documents/Research/Results/Mapping_unit_test/'
neuralfeaturesdir = '/home/tahereh/Documents/Research/features/neural_features/'
datadir = '/home/tahereh/Documents/Research/Data/DiCarlo/'


import numpy as np
import h5py
import importlib
import MappingV36
importlib.reload(MappingV36)
from MappingV36 import MappingV36 as Mapping
Mapping = Mapping()
from ReadMeta import ReadMeta
from ReadData import ReadData


class DataMapModel:
    def __init__(self, ni, nf, nt):
        self.ni = ni
        self.nf = nf
        self.nt = nt

    def get_syntheic(self, sds, splitfract, Collinearity, noise_dist):

        D = np.zeros((self.ni, self.nf, self.nt))

        hf = h5py.File(resultdir + 'HvM_stats.h5', 'r')
        mu = np.array(hf.get('mu')).T
        # sds = np.array(hf.get('sd')).T
        hf.close()

        Dtruth = mu[:self.ni, :self.nf]  # np.random.rand(ni, nf)  # model M: nf feat x ni images

        if Collinearity:
            collinearity_r = np.random.uniform(low=0.8, high=1, size=self.nf - 1)
            # collinearity_r = np.random.uniform(low=0., high=0.1, size=nf-1)

            for ir, r in enumerate(collinearity_r):
                Dtruth[ir + 1] = Dtruth[0] * r + Dtruth[ir + 1] * np.sqrt(1 - r ** 2)
                # print(r, ss.pearsonr(M[0], M[ir+1])[0], end="")

        for tr in range(self.nt):
            D[:, :, tr] = Dtruth

        noise1 = np.zeros((self.ni, self.nf, int(self.nt * splitfract)))
        noise2 = np.zeros((self.ni, self.nf, int(self.nt * splitfract)))
        for i in range(self.ni):
            if noise_dist == 'normal':
                n = np.random.rand()
                n1 = np.array([np.random.normal(0, sdf, size=int(self.nt * splitfract)) for sdf in sds])
                n2 = np.array([np.random.normal(0, sdf, size=int(self.nt * splitfract)) for sdf in sds])
                noise1[i] = n1  # (n1 - n1.min()) / (n1.max() - n1.min())
                noise2[i] = n2  # (n2 - n2.min()) / (n2.max() - n2.min())
            elif noise_dist == 'poisson':
                n = np.random.rand()
                n1 = np.array([np.random.poisson(sdf + n, size=int(self.nt * splitfract)) for sdf in sds])
                n2 = np.array([np.random.poisson(sdf + n, size=int(self.nt * splitfract)) for sdf in sds])
                noise1[i] = n1  # (n1-n1.min())/(n1.max()-n1.min())
                noise2[i] = n2  # (n2-n2.min())/(n2.max()-n2.min())
            elif noise_dist == 'HvM_normal':
                n1 = np.array([np.random.normal(sdf, size=int(self.nt * splitfract)) for sdf in sds[i]])
                n2 = np.array([np.random.normal(sdf, size=int(self.nt * splitfract)) for sdf in sds[i]])
                noise1[i] = n1  # (n1-n1.min())/(n1.max()-n1.min())
                noise2[i] = n2  # (n2-n2.min())/(n2.max()-n2.min())

        D[:, :, :int(self.nt * splitfract)] = D[:, :, :int(self.nt * splitfract)] + noise1
        D[:, :, int(self.nt * splitfract):] = D[:, :, int(self.nt * splitfract):] + noise2

        return D, Dtruth

    def get_HvM(self, ):

        # Read Meta
        Meta = ReadMeta(neuralfeaturesdir)
        DF_img = Meta.get_DF_img()
        DF_neu = Meta.get_DF_neu()
        times = Meta.get_times()

        # Read Neural data
        Data = ReadData(datadir, DF_neu)
        IT, V4 = Data.get_data()

        D = Mapping.get_Neu_trial_V36(IT[1:], [70, 170], times)
        image_indices = np.random.randint(low=0, high=D.shape[1], size=ni)
        D = D[:, image_indices, :]
        D = np.swapaxes(D, 0, 1)
        nf = D.shape[1]
        nt = D.shape[2]

        mu = np.zeros((self.nf, self.ni))
        sd = np.zeros((self.nf, self.ni))
        for f in range(self.nf):
            for i in range(self.ni):
                mu[f, i] = D[i, f, :].mean()
                sd[f, i] = D[i, f, :].std()
        hf = h5py.File(resultdir + 'HvM_stats.h5', 'w')
        hf.create_dataset('mu', data=mu)
        hf.create_dataset('sd', data=sd)
        hf.close()

        # #test synthetic as HvM
        # nf = 168
        # nt = 46
        # noise_dist = 'poisson'
        # sds = np.logspace(-1, 1, num=int(nf))
        # D = np.zeros((ni, nf, nt))
        # D_mean = np.random.rand(ni, nf)
        # for tr in range(nt):
        #     D[:, :, tr] = D_mean
        #
        # noise1 = np.zeros((ni, nf, int(nt * splitfract)))
        # noise2 = np.zeros((ni, nf, int(nt * splitfract)))
        # for i in range(ni):
        #     if noise_dist == 'normal':
        #         n = np.random.rand()
        #         noise1[i] = np.array([np.random.normal(0, sd + n, size=int(nt * splitfract)) for sd in sds])
        #         noise2[i] = np.array([np.random.normal(0, sd + n, size=int(nt * splitfract)) for sd in sds])
        #     elif noise_dist == 'poisson':
        #         n = np.random.rand()
        #         noise1[i] = np.array([np.random.poisson(sd + n, size=int(nt * splitfract)) for sd in sds])
        #         noise2[i] = np.array([np.random.poisson(sd + n, size=int(nt * splitfract)) for sd in sds])
        #
        #     D[:, :, :int(nt * splitfract)] = D[:, :, :int(nt * splitfract)] + noise1
        #     D[:, :, int(nt * splitfract):] = D[:, :, int(nt * splitfract):] + noise2

        # to test  HvM as syntheic
        # hf = h5py.File(resultdir+'D.h5', 'w')
        # hf.create_dataset('D', data=D)
        # hf.close()

        sds = []
        Collinearity = 'HvM'
        noise_dist = 'HvM'

        return D

    def get_model(self, D, A, PCA_ncomponents):

        M = np.matmul(D, A)  # D_mean

        if PCA_ncomponents:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=PCA_ncomponents)
            pca.fit(M)
            M = pca.transform(M)

        return M

