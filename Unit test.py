import pickle
import numpy as np
from DataMapModel import DataMapModel as DataMapModelClass
from MappingUnitTest import MappingUnitTest as MappingUnitTestClass


resultdir = '/home/tahereh/Documents/Research/Results/Mapping_unit_test/'
neuralfeaturesdir = '/home/tahereh/Documents/Research/features/neural_features/'
datadir = '/home/tahereh/Documents/Research/Data/DiCarlo/'

# ------------------------------------------------------------
# Data Parameters
# ------------------------------------------------------------
nf = 20
nc = nf
purpose_of_this_run = 'ridgeCV40-10-10'
for imag_feat_ratio in [1, 2, 4, 8, 16, 64, 128]:  #4, 8, 16, 64, 128

    ni = imag_feat_ratio*nf  # # of features
    nt = 46

    trainfraci = 0.8  # image trainfrac
    splitfract = 0.5  # trial splitfrac
    nfoldi = 10
    nfoldt = 5
    noisy_map = False
    various_unit_stds = True
    Collinearity = False

    stats_from_data = False
    noise_dist = 'normal'#'normal' # 'HvM_poisson'

    if various_unit_stds:
        sds = np.logspace(-1, 0, num=int(nf))  # np.arange(0.5, 10, 1)
    else:
        sds = np.ones((nf))*sd

    Data_type = 'synthetic'  # 'synthetic'#'HvM'

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
    reg_params_list = [[], nc, [40, -10, 10], [40, -10, 10]] # for ridge [n_alpha, alpha0, alpha1]
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
    DataMapModel = DataMapModelClass(ni, nf, nt)

    if Data_type == 'HvM':
        D = DataMapModel.get_HvM()
        Dmu = D.mean(2)

    elif Data_type == 'synthetic':
        D, Dtruth = DataMapModel.get_syntheic(sds, splitfract, Collinearity, noise_dist, stats_from_data)

        if noisy_map:
            Dmu = D.mean(2)
        else:
            Dmu = Dtruth

    A = np.random.rand(nf, nf)

    MappingUnitTest = MappingUnitTestClass(D, Dmu, A, PCA_ncomponents_list, explained_var_ratio_list)

    Data_params = [ni, nf, nt, nfoldi, nfoldt, trainfraci, splitfract, noise_dist,  sds, Collinearity, various_unit_stds, corr_method_for_inv, noisy_map]

    data_list = MappingUnitTest.get_mappings_unit_test(Data_params, reg_methods, reg_params_list, spearman_brown, report_sitefit, report_popfit)

    pickle.dump(data_list, open(resultdir + 'unit_test_%s_%s_%s, ni%d_nf%d_nt%d_collinearity%s_%s_SB%s_VariousSd%s_noisymap%s_statsfromHvM%s_%dcmp_%s.pickle' % (
            reg_methods, reg_params_list, PCA_ncomponents_list, ni, nf, nt, Collinearity, noise_dist, spearman_brown, various_unit_stds, noisy_map,stats_from_data,nc,purpose_of_this_run), 'wb'))