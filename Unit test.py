import pickle
import numpy as np
from DataMapModel import DataMapModel as DataMapModelClass
from MappingUnitTest import MappingUnitTest as MappingUnitTestClass
import time
start_time = time.time()

resultdir = '/home/tahereh/Documents/Research/Results/Mapping_unit_test/'
neuralfeaturesdir = '/home/tahereh/Documents/Research/features/neural_features/'
datadir = '/home/tahereh/Documents/Research/Data/DiCarlo/'

# ------------------------------------------------------------
# Data Parameters
# ------------------------------------------------------------
nf = 20

load_saved_data = True
data_unit_indices = range(20)  #[0,4,8,12,16]#np.random.permutation(20)[0:5]

purpose_of_this_run = 'ridgeCV20-10-10-sds-1to1-%dsites'%len(data_unit_indices)

for nc in [5, 8, 11, 14, 17, 20]:

    for imag_feat_ratio in [1, 2, 4]:

        ni = imag_feat_ratio*nf  # # of features
        nt = 46

        trainfraci = 0.8  # image trainfrac
        splitfract = 0.5  # trial splitfrac
        nfoldi = 10
        nfoldt = 5
        noisy_map = False

        Collinearity = False

        Data_type = 'synthetic'  # 'synthetic'#'HvMlike'
        stats_from_data = False
        noise_dist = 'normal'#'normal' # 'HvM_poisson'
        sds = np.logspace(-1, 1, num=int(nf))  # np.arange(0.5, 10, 1)


        print(data_unit_indices)

        # ------------------------------------------------------------
        # Regression Parameters
        # ------------------------------------------------------------

        spearman_brown = False
        corr_method_for_inv = 'pearson'  # 'spearman'

        # regularization parameters

        # if reg_method == 'PLS':
        #     n_components = nf
        #     reg_params = n_components
        #     report_popfit = True
        #
        # elif reg_method == 'ridge':
        #     reg_params = []
        #     report_popfit = True
        #
        # elif reg_method == 'OMP':
        #     reg_params = []
        #     report_popfit = False
        #     report_sitefit = True

        reg_methods = ['OMP', 'PLS', 'ridge', 'ridge']
        reg_params_list = [[], nc, [20, -10, 10], [20, -10, 10]] # for ridge [n_alpha, alpha0, alpha1]
        report_popfit = [False, True, True, True]  # [False, True, True]
        report_sitefit = [True, True, True, True]  # [False, False, False, False]  # [True, True, True]

        # ------------------------------------------------------------
        # Map Parameters
        # ------------------------------------------------------------
        # PCA_ncomponents = -1 means no PCA will be applied on the model,
        # PCA_ncomponents = 0 means refer to the explained_var_ratio to calculate the number of components for PCA
        PCA_ncomponents_list = [-1, -1, -1, -1, nc]  # The first one is for pinv and the rest for the regressions
        explained_var_ratio_list = [0, 0, 0, 0, 0]
        # ------------------------------------------------------------
        #
        # ------------------------------------------------------------
        if load_saved_data is False:

            DataMapModel = DataMapModelClass(ni, nf, nt)

            if Data_type == 'HvMlike':
                D = DataMapModel.get_HvM()
                Dmu = D.mean(2)

            elif Data_type == 'synthetic':
                D, Dtruth = DataMapModel.get_syntheic(sds, splitfract, Collinearity, noise_dist, stats_from_data)

                if noisy_map:
                    Dmu = D.mean(2)
                else:
                    Dmu = Dtruth

            A = np.random.rand(nf, nf)

            pickle.dump([A, D, Dmu, sds], open(resultdir + 'DataandMap_ni%d_nf%d_nt%d_%d_%d_%d_%d_collinearity%s_%s_noisymap%s_statsfromHvM%s.pickle' % (
                     ni, nf, nt, nfoldi, nfoldt, trainfraci, splitfract, Collinearity, noise_dist, noisy_map, stats_from_data), 'wb'))
        else:

            file = open(resultdir + 'DataandMap_ni%d_nf%d_nt%d_%d_%d_%d_%d_collinearity%s_%s_noisymap%s_statsfromHvM%s.pickle' % (
                     ni, nf, nt, nfoldi, nfoldt, trainfraci, splitfract, Collinearity, noise_dist, noisy_map, stats_from_data), 'rb')

            A, D, Dmu, sds = pickle.load(file)
            file.close()

            MappingUnitTest = MappingUnitTestClass(D, Dmu, A, PCA_ncomponents_list, explained_var_ratio_list)

            Data_params = [ni, nf, nt, nfoldi, nfoldt, trainfraci, splitfract, noise_dist,  sds, Collinearity, corr_method_for_inv, noisy_map, data_unit_indices]

            data_list = MappingUnitTest.get_mappings_unit_test(Data_params, reg_methods, reg_params_list, spearman_brown, report_sitefit, report_popfit)

            pickle.dump(data_list, open(resultdir + 'unit_test_%s_%s_%s_%s, ni%d_nf%d_nt%d_collinearity%s_%s_SB%s_noisymap%s_statsfromHvM%s_%dcmp_%s.pickle' % (
                load_saved_data, reg_methods, reg_params_list, PCA_ncomponents_list, ni, nf, nt, Collinearity, noise_dist, spearman_brown, noisy_map,stats_from_data,nc,purpose_of_this_run), 'wb'))

print(time.time()-start_time)